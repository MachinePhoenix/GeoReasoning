from vllm import LLM, SamplingParams
import re
import json
from typing import Dict
import pdb
import os

FORMAT_PROMPT = """You are a geometry problem solver. 
Follow these steps strictly:
1. Think through the problem step by step inside <think> tags
2. Put your final answer in \boxed{} 
3. Use exact values from the description

Example response:
<think>
- Identify rectangle side lengths
- Perimeter formula: 2*(length + width)
- Calculate 2*(1.89 + 2.07)
</think>
The perimeter is \boxed{7.92}."""


def extract_answer(response: str) -> str:
    """从模型响应中提取答案"""
    # 匹配 \boxed{} 格式
    boxed_match = re.search(r"\\boxed{([^}]+)}", response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # 匹配最后一行数值
    last_number = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    return last_number[-1] if last_number else ""


class QAValidator:
    def __init__(self, qa_model_path: str, questions_path: str, qa_gpus: list):
        self.qa_llm = LLM(
            model=qa_model_path,
            tensor_parallel_size=len(qa_gpus),
            device='cuda',
            gpu_memory_utilization=0.8,
            max_num_seqs=64,
            trust_remote_code=True,
            enforce_eager=True
        )
        
        with open(questions_path, 'r') as f:
            self.questions_db = {}
            for item in json.load(f):
                # 解析问题和参考答案
                qa_pair = self._parse_response(item["response"])
                self.questions_db[item["id"]] = {
                    "question": qa_pair["question"],
                    "ground_truth": qa_pair["answer"]
                }
        
        # 配置QA采样参数
        self.qa_sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=200,
            n=1,
            stop=["</think>"]
        )

    def _parse_response(self, response: str) -> Dict[str, str]:
        """增强的response解析方法"""
        # 优化Question提取：匹配到下一个**标签前的内容
        question_match = re.search(
            r"\*\*Question:\*\*\s*(.+?)(?=\n\s*\*\*Inference Process:\*\*|\n\s*\*\*Answer:\*\*|\Z)",
            response,
            re.DOTALL
        )
        
        # 优化Answer提取：支持两种格式
        answer_match = re.search(
            r"\*\*Answer:\*\*\s*(?:\\?boxed{)?([^}\n]+)(?:})?\s*",  # 匹配boxed{7.92}和直接7.92两种格式
            response
        )
        
        # 后处理提取结果
        question = question_match.group(1).strip() if question_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""
        
        # 清理答案中的特殊符号
        answer = re.sub(r"[^0-9.+-]", "", answer)
        
        return {
            "question": question,
            "answer": answer
        }

    def reasoning_compute_score(self, caption: str, item_id: str) -> float:
        """基于问答正确性的评分"""
        # 获取问题信息
        qa_info = self.questions_db.get(item_id, None)
        if not qa_info or not qa_info["question"]:
            return 0.0
        
        prompt = f"""Geometry Problem Solving
            {FORMAT_PROMPT}

            [Image Description]
            {caption}

            [Question]
            {qa_info["question"]}
            """

        try:
            outputs = self.qa_llm.generate([prompt], self.qa_sampling_params)
            response = outputs[0].outputs[0].text
        except Exception as e:
            print(f"QA推理失败: {str(e)}")
            return 0.0
        
        # 评估答案
        return self._evaluate_response(response, qa_info["ground_truth"])

    def _evaluate_response(self, response: str, ground_truth: str) -> float:
        # 格式检查
        format_ok = bool(re.search(r"<think>.*</think>.*\\boxed{", response, re.DOTALL))
        
        # 答案提取
        pred_answer = extract_answer(response)
        gt_answer = ground_truth.split("=")[-1].strip()  # 处理参考答案可能有等号的情况
        
        try:
            pred_val = float(pred_answer)
            gt_val = float(gt_answer)
            accuracy = 1.0 if abs(pred_val - gt_val) < 1e-3 else 0.0
        except:
            accuracy = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
        
        return 0.9 * accuracy + 0.1 * float(format_ok)