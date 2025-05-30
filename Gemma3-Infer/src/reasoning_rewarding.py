import json
from vllm import LLM, SamplingParams
from caption_reward import caption_compute_score
import argparse
import os
import pdb
import re
from collections import defaultdict



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

class Reasoning_Rewarder:
    def __init__(self, qa_model_path, qa_gpus):
        self.qa_llm = LLM(
            model=qa_model_path,
            tensor_parallel_size=len(qa_gpus),
            # max_num_seqs=256,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            # enforce_eager=True
            # dtype="bfloat16",
        )
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            n=1
        )

    def evaluate(self, original_data_path, intermediate_path, output_path, 
                caption_weight, reasoning_weight, topK):
        # 加载原始数据和中间结果
        with open(original_data_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        with open(intermediate_path, 'r', encoding='utf-8') as f:
            intermediate_data = json.load(f)

        # 创建中间数据映射表
        intermediate_map = {item["id"]: item for item in intermediate_data}

        # 验证数据一致性
        if len(original_data) != len(intermediate_data):
            raise ValueError("原始数据与中间数据数量不一致")
        
        # 批量处理所有QA
        all_prompts = []
        item_map = []
        for orig_item in original_data:
            inter_item = intermediate_map[orig_item["id"]]
            for cap in inter_item["captions"]:
                all_prompts.append(self._build_qa_prompt(cap, inter_item["question"]))
                item_map.append((orig_item, inter_item, cap))

        batch_size_manual = len(all_prompts)
        outputs = []
        for i in range(0, len(all_prompts), batch_size_manual):
            batch = all_prompts[i:i+batch_size_manual]
            result_batch = self.qa_llm.generate(batch, self.sampling_params)
            outputs.extend(result_batch)
        
        del self.qa_llm  # 显式删除模型实例
        import gc
        gc.collect()
        
        # 处理结果
        results = defaultdict(list)
        for (orig_item, inter_item, cap), output in zip(item_map, outputs):
            response = output.outputs[0].text
            score = self._calc_score(
                response, cap, 
                inter_item["ground_truth"], 
                orig_item["conversations"][1]["value"],
                caption_weight, 
                reasoning_weight
            )
            results[orig_item["id"]].append((cap, score))

        # 更新原始数据
        updated_data = []
        for orig_item in original_data:
            item_id = orig_item["id"]
            candidates = results.get(item_id, [])
            
            # 选择topK
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_caption = candidates[0][0] if candidates else orig_item["conversations"][1]["value"]
            
            # 克隆原始数据结构
            new_item = json.loads(json.dumps(orig_item))
            
            # 更新对话内容
            new_item["conversations"][1]["value"] = best_caption
            
            # 添加评分信息
            new_item["scores"] = {
                "selected": {
                    "text": best_caption,
                    "total_score": candidates[0][1] if candidates else 0.0,
                    "caption_weight": caption_weight,
                    "reasoning_weight": reasoning_weight
                },
                "candidates": [{
                    "text": c[0],
                    "total_score": c[1],
                    "caption_score": self._get_caption_score(
                        c[0], 
                        orig_item["conversations"][1]["value"]  # 关键修改点
                    ),
                    "reasoning_score": (c[1] - caption_weight * self._get_caption_score(
                        c[0], 
                        orig_item["conversations"][1]["value"]  # 关键修改点
                    )) / reasoning_weight
                } for c in candidates[:topK]]
            }
            
            updated_data.append(new_item)
        
        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
            f.flush()

    def _get_caption_score(self, caption, ref_caption):
        try:
            return caption_compute_score(caption, ref_caption)['overall']
        except:
            return 0.0

    def _build_qa_prompt(self, caption, question):
        return f"""Geometry Problem Solving
        {FORMAT_PROMPT}

        [Image Description]
        {caption}

        [Question]
        {question}
        """

    def _calc_score(self, response, caption, gt_answer, ref_caption, cw, rw):
        # 计算推理分数
        reasoning_score = self._evaluate_response(response, gt_answer)
        
        # 计算caption分数
        caption_score = self._get_caption_score(caption, ref_caption)
        
        return cw * caption_score + rw * reasoning_score

    def _evaluate_response(self, response, gt_answer):
        # 格式检查
        format_ok = bool(re.search(r"<think>.*</think>.*\\boxed{", response, re.DOTALL))
        
        # 答案提取
        pred_answer = self.extract_answer(response)
        gt_answer = gt_answer.split("=")[-1].strip()  # 处理参考答案可能有等号的情况
        
        try:
            pred_val = float(pred_answer)
            gt_val = float(gt_answer)
            accuracy = 1.0 if abs(pred_val - gt_val) < 1e-3 else 0.0
        except:
            accuracy = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
        
        return 0.9 * accuracy + 0.1 * float(format_ok)

    @staticmethod
    def extract_answer(response: str) -> str:
        boxed_match = re.search(r"\\boxed{([^}]+)}", response)
        if boxed_match:
            return boxed_match.group(1).strip()
        last_number = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        return last_number[-1] if last_number else ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_model_path", type=str, required=True)
    parser.add_argument("--original_data_path", type=str, required=True) 
    parser.add_argument("--intermediate_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--caption_weight", type=float, default=0.5)
    parser.add_argument("--reasoning_weight", type=float, default=0.5)
    parser.add_argument("--topK", type=int, default=3)
    parser.add_argument("--qa_gpus", nargs='+', type=int, default=[0])
    args = parser.parse_args()

    evaluator = Reasoning_Rewarder(args.qa_model_path, args.qa_gpus)
    evaluator.evaluate(
        args.original_data_path,
        args.intermediate_path,
        args.output_path,
        args.caption_weight,
        args.reasoning_weight,
        args.topK
    )