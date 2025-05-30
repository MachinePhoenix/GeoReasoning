import json
import re
from vllm import LLM, SamplingParams
import numpy as np
from typing import Dict
import pdb
import argparse
import os
from caption_reward import *
from reasoning_reward import *


# ================== 核心处理模块 ==================
def process_caption(caption: str) -> str:
    """增强的文本清洗函数"""
    # 第一步：彻底清除所有<image>标签及其内容（包含未闭合情况）
    caption = re.sub(r'<image>.*?(</image>|$)', '', caption, flags=re.DOTALL)
    
    # 第二步：移除残留的图片路径（如img/180等）
    caption = re.sub(r'\bimg/\d+\.?(jpg|png|jpeg)?\b', '', caption, flags=re.IGNORECASE)
    
    # 第三步：清理步骤编号和项目符号
    caption = re.sub(r'\d+\.\s*\*\*.*?\*\*:', '', caption)
    bullet_points = re.findall(r'-\s*(.*?)(?=\n-|\n\s*\d+\.|\n$|$)', caption, flags=re.DOTALL)
    
    # 第四步：合并有效内容
    cleaned = ' '.join([bp.strip() for bp in bullet_points]) if bullet_points else caption.strip()
    
    # 第五步：移除多余空格和特殊符号
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return re.sub(r'^[^a-zA-Z0-9]+', '', cleaned)  # 移除开头的非字母数字字符


def rollout(
    model_path: str,
    data_path: str,
    sample_num: int,
    topK: int,
    batch_size: int,
    qa_model_path: str,
    questions_path: str,
    caption_weight: float = 0.5,
    reasoning_weight: float = 0.5,
    main_gpus: list = [0],
    qa_gpus: list = [1]
):
    # 初始化模型
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, main_gpus))
    llm = LLM(
        model=model_path,
        tensor_parallel_size=len(main_gpus),
        max_num_seqs=batch_size,
        trust_remote_code=True,
        device='cuda',
        gpu_memory_utilization=0.8
    )
    
    # 初始化QA验证器
    qa_validator = QAValidator(qa_model_path, questions_path, qa_gpus)

    # 加载数据集
    with open(data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    ref_captions = []
    prompts = []
    for item in original_data:
        # 提取参考标注（保持原有逻辑）
        gt_caption = next(m["value"] for m in item["conversations"] if m["from"] == "gpt")
        ref_captions.append(process_caption(gt_caption))
        
        # 构建prompt（保持原有逻辑）
        image_path = item["image"]
        human_prompt = next(m["value"] for m in item["conversations"] if m["from"] == "human")
        prompts.append(f"<image>{image_path}</image>\n{human_prompt.replace('<image>', '').strip()}")

    # 配置采样参数（保持原有逻辑）
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
        n=sample_num
    )

    # 批量推理
    outputs = llm.generate(prompts, sampling_params)

    # 处理结果
    new_data = []
    for idx, output in enumerate(outputs):
        original_item = original_data[idx]
        current_ref = ref_captions[idx]
        item_id = original_item["id"]
        
        # 处理生成结果
        captions = [process_caption(out.text.strip()) for out in output.outputs]
        
        # 计算双奖励分数
        scored_captions = []
        for cap in captions:
            if not cap:
                scored_captions.append((cap, 0.0, 0.0, 0.0))
                continue
                
            try:
                # 计算caption相关分数
                caption_scores = caption_compute_score(cap, current_ref)
                caption_score = caption_scores['overall']
            except:
                caption_score = 0.0
                
            try:
                # 计算推理分数
                reasoning_score = qa_validator.reasoning_compute_score(cap, item_id)
            except:
                reasoning_score = 0.0
                
            # 加权总分
            total_score = (
                caption_weight * caption_score +
                reasoning_weight * reasoning_score
            )
            
            scored_captions.append((
                cap,
                total_score,
                caption_score,
                reasoning_score
            ))
        
        # 选择topK
        scored_captions.sort(key=lambda x: x[1], reverse=True)
        best_captions = [{
            "text": cap,
            "total_score": ts,
            "caption_score": cs,
            "reasoning_score": rs
        } for cap, ts, cs, rs in scored_captions[:topK]]
        
        # 构建新数据项
        new_item = {
            **original_item,
            "conversations": [
                original_item["conversations"][0],
                {
                    "from": "gpt",
                    "value": best_captions[0]["text"],
                    "scores": best_captions[0]
                }
            ]
        }
        new_data.append(new_item)
    
    return new_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 原有参数
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--sample_num", type=int, default=5)
    parser.add_argument("--topK", type=int, default=3)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    
    # 新增参数
    parser.add_argument("--qa_model_path", type=str, required=True)
    parser.add_argument("--questions_path", type=str, required=True)
    parser.add_argument("--caption_weight", type=float, default=0.5)
    parser.add_argument("--reasoning_weight", type=float, default=0.5)
    parser.add_argument("--main_gpus", nargs='+', type=int, default=[0], help="main model gpu list")
    parser.add_argument("--qa_gpus", nargs='+', type=int, default=[1], help="qa model gpu list")
    
    args = parser.parse_args()

    new_data = rollout(
        model_path=args.model_path,
        data_path=args.data_path,
        sample_num=args.sample_num,
        topK=args.topK,
        batch_size=args.batch_size,
        qa_model_path=args.qa_model_path,
        questions_path=args.questions_path,
        caption_weight=args.caption_weight,
        reasoning_weight=args.reasoning_weight,
        main_gpus=args.main_gpus,
        qa_gpus=args.qa_gpus
    )
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)