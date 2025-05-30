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
from typing import Optional, Dict, List
import re
import ray
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
    caption = re.sub(r'<image>.*?(</image>|$)', '', caption, flags=re.DOTALL)
    caption = re.sub(r'\bimg/\d+\.?\w*\b', '', caption, flags=re.IGNORECASE)
    bullets = re.findall(r'-\s*(.+?)(?=\n-|\n\s*\d+\.|\Z)', caption, flags=re.DOTALL)
    cleaned = ' '.join(b.strip() for b in bullets) if bullets else caption.strip()
    return re.sub(r'\s+', ' ', cleaned).strip('^[^a-zA-Z0-9]*')

def _parse_qa(response: str) -> Dict[str, str]:
    """优化QA解析，支持多种答案格式"""
    question_match = re.search(r"\*\*Question:\*\*\s*(.+?)(?=\n\*\*|\Z)", response, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    answer_section = re.search(r"\*\*(?:Final )?Answer:\*\*\s*([\s\S]*?)(?=\n\*\*|\Z)", response, re.DOTALL)
    if not answer_section:
        return {"question": question, "answer": ""}
    answer_text = answer_section.group(1).strip()

    patterns = [
        (r'\\boxed{([^}]+)}', 1),
        (r'\*\*((?:[\d.]+|\\dfrac{.*?}|\\sqrt{.*?}|\\dfrac{\\sqrt{.*?}}{.*?}|\\d+.*?\\d+))\*\*', 1),
        (r'\\\((.*?)\\\)', 1),
        (r'([$\d.+\-\\/√()\\dfrac{}^a-zA-Z]+)(?:\s*(?:units|°|degrees|square units|等)|$)', 1),
        (r'(?:The\s+)?(?:length|measure|area|perimeter|degrees)\s+is\s+([\d.]+)', 1),
        (r'(\\d+\.?\\d*(?:\\d+)?\\^\\w+)', 0),
        (r'([\d.]+|\\dfrac{.*?}|\d+/\d+)', 0),
    ]

    answer = ""
    for pattern, group_idx in patterns:
        matches = list(re.finditer(pattern, answer_text, re.IGNORECASE))
        if matches:
            match = max(matches, key=lambda m: len(m.group(group_idx)))
            answer = match.group(group_idx).strip()
            break

    answer = re.sub(r'^[^$\d\\]+', '', answer)
    answer = re.sub(r'[^$\d.+\-\\/√()\\dfrac{}^a-zA-Z]', '', answer)
    answer = answer.replace(' ', '')
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
    r"""Perform batch generation using vLLM engine with Ray-based data parallelism."""
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    if not save_path:
        save_path = f'./{dataset}-test.json'

    model_args, data_args, _, generating_args = get_infer_args(dict(
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
    ))

    training_args = Seq2SeqTrainingArguments(output_dir="dummy_dir")
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template_obj = get_template_and_fix_tokenizer(tokenizer, data_args)
    template_obj.mm_plugin.expand_mm_tokens = False
    dataset_module = get_dataset(template_obj, model_args, data_args, training_args, "ppo", **tokenizer_module)

    with open(data_path) as f:
        ori_dataset = json.load(f)

    qa_db = {}
    if questions_path:
        with open(questions_path) as f:
            qa_db = {item["id"]: _parse_qa(item["response"]) for item in json.load(f)}

    inputs, items = [], []
    index = 0
    for sample in dataset_module["train_dataset"]:
        if sample["images"]:
            ori_index = sample['images'][0].split('/')[-1].split('.')[0]
            if ori_dataset[index]['id'] == ori_index:
                ori_sample = ori_dataset[index]
            else:
                ori_sample = [sp for sp in ori_dataset if sp['id'] == ori_index].pop()
        else:
            ori_sample = ori_dataset[index]

        if sample["images"]:
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
                sample["audios"], sampling_rate=16000
            )
            multi_modal_data = {"audio": zip(audio_data["audios"], audio_data["sampling_rates"])}
        else:
            multi_modal_data = None

        inputs.append({"prompt_token_ids": sample["input_ids"], "multi_modal_data": multi_modal_data})
        items.append(ori_sample)
        index += 1

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,
        top_k=generating_args.top_k or -1,
        stop_token_ids=template_obj.get_stop_token_ids(tokenizer),
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=skip_special_tokens,
        seed=seed,
        n=sample_num
    )

    if model_args.adapter_name_or_path is not None:
        lora_request = LoRARequest("default", 1, model_args.adapter_name_or_path[0])
    else:
        lora_request = None

    # 数据分片准备
    num_gpus = get_device_count()
    chunk_size = (len(inputs) + num_gpus - 1) // num_gpus
    input_chunks = [inputs[i:i+chunk_size] for i in range(0, len(inputs), chunk_size)]
    item_chunks = [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

    # 启动 Ray 并行任务
    futures = []
    for i, gpu_id in enumerate(range(num_gpus)):
        futures.append(
            _ray_inference.remote(
                model_name_or_path=model_name_or_path,
                inputs=input_chunks[i],
                items=item_chunks[i],
                model_args=model_args,
                data_args=data_args,
                generating_args=generating_args,
                template_obj=template_obj,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                lora_request=lora_request,
                cutoff_len=cutoff_len,
                max_new_tokens=max_new_tokens,
                image_max_pixels=image_max_pixels,
                image_min_pixels=image_min_pixels,
                qa_db=qa_db
            )
        )

    # 收集所有结果
    results = []
    for future in ray.get(futures):
        results.extend(future)

    # 保存结果
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()

    print("*" * 70)
    print(f"{len(results)} generated results have been saved at {save_path}.")
    print("*" * 70)

@ray.remote(num_gpus=1)
def _ray_inference(
    model_name_or_path: str,
    inputs: List[Dict],
    items: List[Dict],
    model_args,
    data_args,
    generating_args,
    template_obj,
    tokenizer,
    sampling_params,
    lora_request: Optional[LoRARequest] = None,
    cutoff_len: int = 2048,
    max_new_tokens: int = 1024,
    image_max_pixels: int = 768 * 768,
    image_min_pixels: int = 32 * 32,
    qa_db: Dict = {}
):
    """每个 Ray 任务独立运行推理"""
    engine_args = {
        "model": model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "gpu_memory_utilization": 0.6,
        "max_model_len": cutoff_len + max_new_tokens,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
    }

    if template_obj.mm_plugin.__class__.__name__ != "BasePlugin":
        engine_args["limit_mm_per_prompt"] = {"image": 4, "video": 2, "audio": 2}

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    llm = LLM(**engine_args)
    results = llm.generate(inputs, sampling_params, lora_request=lora_request)
    final_results = []

    for item, result in zip(items, results):
        final_results.append({
            **item,
            "captions": [res.text for res in result.outputs],
            "question": qa_db.get(item.get("id"), {}).get("question", ""),
            "ground_truth": qa_db.get(item.get("id"), {}).get("answer", "")
        })

    return final_results

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
    parser.add_argument("--sample_num", type=int, default=8)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--questions_path", required=True, help="QA数据路径")
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--main_gpus", nargs='+', type=int, default=[0], help="使用GPU列表")
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True)

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
        temperature=0.0,
        top_p=1.0,
        top_k=1,
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