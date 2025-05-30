# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Optional, Dict
import re

import fire
from transformers import Seq2SeqTrainingArguments
import pdb

from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.extras.misc import get_device_count
from llamafactory.extras.packages import is_vllm_available
from llamafactory.hparams import get_infer_args
from llamafactory.model import load_tokenizer


if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest


def process_caption(caption: str) -> str:
    """优化后的文本清洗函数"""
    # 清理图像标签和路径
    caption = re.sub(r'<image>.*?(</image>|$)', '', caption, flags=re.DOTALL)
    caption = re.sub(r'\bimg/\d+\.?\w*\b', '', caption, flags=re.IGNORECASE)

    # 提取有效内容
    bullets = re.findall(r'-\s*(.+?)(?=\n-|\n\s*\d+\.|\Z)', caption, flags=re.DOTALL)
    cleaned = ' '.join(b.strip() for b in bullets) if bullets else caption.strip()

    # 标准化输出
    return re.sub(r'\s+', ' ', cleaned).strip('^[^a-zA-Z0-9]*')


def _parse_qa(response: str) -> Dict[str, str]:
    """优化QA解析，支持多种答案格式"""
    # 提取问题部分
    question_match = re.search(
        r"\*\*Question:\*\*\s*(.+?)(?=\n\*\*|\Z)",
        response,
        re.DOTALL
    )
    question = question_match.group(1).strip() if question_match else ""

    # 提取答案部分（同时匹配 Answer 和 Final Answer）
    answer_section = re.search(
        r"\*\*(?:Final )?Answer:\*\*\s*([\s\S]*?)(?=\n\*\*|\Z)",
        response,
        re.DOTALL
    )
    if not answer_section:
        return {"question": question, "answer": ""}
    
    answer_text = answer_section.group(1).strip()

    # 新的优先匹配规则（修正正则表达式顺序）
    patterns = [
        # 1. 匹配 boxed 内容
        (r'\\boxed{([^}]+)}', 1),
        # 2. 匹配加粗数字或分数（支持 LaTeX 符号）
        (r'\*\*((?:[\d.]+|\\dfrac{.*?}|\\sqrt{.*?}|\\dfrac{\\sqrt{.*?}}{.*?}|\\d+.*?\\d+))\*\*', 1),
        # 3. 匹配行内公式 \( ... \)
        (r'\\\((.*?)\\\)', 1),  # 非贪婪匹配所有内容
        # 4. 匹配纯数字+单位（新增 degrees 和符号支持）
        (r'([$\d.+\-\\/√()\\dfrac{}^a-zA-Z]+)(?:\s*(?:units|°|degrees|square units|等)|$)', 1),
        # 5. 直接提取 "is" 后的数字（新增 degrees 支持）
        (r'(?:The\s+)?(?:length|measure|area|perimeter|degrees)\s+is\s+([\d.]+)', 1),
        # 6. 匹配带符号的表达式（如 60^\circ）
        (r'(\\d+\.?\\d*(?:\\d+)?\\^\\w+)', 0),
        # 7. 匹配纯数字或分数（兜底）
        (r'([\d.]+|\\dfrac{.*?}|\d+/\d+)', 0),
    ]

    answer = ""
    for pattern, group_idx in patterns:
        matches = list(re.finditer(pattern, answer_text, re.IGNORECASE))
        if matches:
            # 取最长匹配结果
            match = max(matches, key=lambda m: len(m.group(group_idx)))
            answer = match.group(group_idx).strip()
            break

    # 新的清理逻辑：保留更多数学符号
    answer = re.sub(r'^[^$\d\\]+', '', answer)  # 移除开头非数字/符号字符
    answer = re.sub(r'[^$\d.+\-\\/√()\\dfrac{}^a-zA-Z]', '', answer)  # 保留必要符号
    answer = answer.replace(' ', '')  # 去除空格

    return {
        "question": question,
        "answer": answer
    }

def vllm_infer(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    dataset: str = "alpaca_en_demo",
    dataset_dir: str = "data",
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: Optional[int] = None,
    vllm_config: str = "{}",
    save_path: str = None,
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    skip_special_tokens: bool = True,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    sample_num: int = 8,
    data_path: str = None,
    questions_path: str = None,
    image_path: str = None
):
    r"""Perform batch generation using vLLM engine, which supports tensor parallelism.

    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    # pipeline_parallel_size = get_device_count()
    
    if not save_path:
        save_path = f'./{dataset}-test.json'

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            adapter_name_or_path=adapter_name_or_path,
            dataset=dataset,
            dataset_dir=dataset_dir,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False  # for vllm generate
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    with open(data_path) as f:
        ori_dataset = json.load(f)

    if questions_path:
        with open(questions_path) as f:
            qa_db = {item["id"]: _parse_qa(item["response"]) for item in json.load(f)}

    inputs, prompts, labels, items = [], [], [], []
    index = 0
    for sample in dataset_module["train_dataset"]:
        if sample["images"]:
            #todo find out the corresponding sample in ori_dataset
            ori_index = sample['images'][0].split('/')[-1].split('.')[0]
            if ori_dataset[index]['id'] == ori_index:
                ori_sample = ori_dataset[index]
            else:
                ori_sample = [sp for sp in ori_dataset if sp['id'] == ori_index].pop()
            multi_modal_data = {
                "image": template_obj.mm_plugin._regularize_images(
                    sample["images"], image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                )["images"]
            }
        elif sample["videos"]:
            multi_modal_data = {
                "video": template_obj.mm_plugin._regularize_videos(
                    sample["videos"], image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels
                )["videos"]
            }
        elif sample["audios"]:
            audio_data = template_obj.mm_plugin._regularize_audios(
                sample["audios"],
                sampling_rate=16000,
            )
            multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
        else:
            multi_modal_data = None
        
        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        prompts.append(tokenizer.decode(sample["input_ids"], skip_special_tokens=skip_special_tokens))
        labels.append(
            tokenizer.decode(
                list(filter(lambda x: x != IGNORE_INDEX, sample["labels"])), skip_special_tokens=skip_special_tokens
            )
        )
        items.append(ori_sample)
        index += 1

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k or -1,  # top_k must > 0
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
        n=sample_num,
    )
    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "gpu_memory_utilization": 0.85,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }
    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)
    
    results = []
    batch_size_manual = 1000
    for i in range(0, len(inputs), batch_size_manual):
        batch = inputs[i:i+batch_size_manual]
        results_batch = LLM(**engine_args).generate(batch, sampling_params, lora_request=lora_request)
        results.extend(results_batch)

    final_results = []
    for item, result in zip(items, results):
        final_results.append({
            **item,
            "captions": [res.text for res in result.outputs],
            "question": qa_db.get(item.get("id"), {}).get("question", ""),
            "ground_truth": qa_db.get(item.get("id"), {}).get("answer", "")
        })

    with open(save_path, 'w', encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
        f.flush()

    print("*" * 70)
    print(f"{len(prompts)} generated results have been saved at {save_path}.")
    print("*" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True, type=str, help="模型路径")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default='data')
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--template", type=str, default='default')
    parser.add_argument("--cutoff_len", type=int, default=2048)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--sample_num", type=int, default=8, help="每个样本生成数量")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--questions_path", required=True, help="QA数据路径")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8, help="推理批次大小")
    parser.add_argument("--main_gpus", nargs='+', type=int, default=[0], help="使用GPU列表")

    
    args = parser.parse_args()
    # fire.Fire(vllm_infer)
    vllm_infer(
        model_name_or_path=args.model_name_or_path,
        adapter_name_or_path=None,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        template=args.template,
        cutoff_len=args.cutoff_len,
        max_samples=args.max_samples,
        vllm_config="{}",
        save_path=args.save_path,
        temperature=0.95,
        top_p=0.7,
        top_k=50,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        seed=None,
        pipeline_parallel_size=args.pipeline_parallel_size,
        image_max_pixels=768 * 768,
        image_min_pixels=32 * 32,
        sample_num=args.sample_num,
        data_path=args.data_path,
        questions_path=args.questions_path,
        image_path=None
    )
