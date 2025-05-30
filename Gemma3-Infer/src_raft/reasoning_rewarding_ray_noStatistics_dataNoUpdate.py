import json
from vllm import LLM, SamplingParams
from caption_reward import caption_compute_score
import argparse
import os
import re
from collections import defaultdict
import ray
import torch
import pdb
import numpy as np
import wandb

# Ray 初始化（需在脚本最开始处）
ray.init(ignore_reinit_error=True)

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
    def __init__(self, qa_model_path, qa_gpus, current_epoch):
        self.qa_model_path = qa_model_path
        self.qa_gpus = qa_gpus
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            n=1
        )
        self.current_epoch = current_epoch

    def evaluate(self, original_data_path, intermediate_path, output_path, caption_weight, reasoning_weight, topK):
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

        if abs(reasoning_weight - 0) > 1e-3:
            # 使用 Ray 实现数据并行
            num_gpus = len(self.qa_gpus)
            chunk_size = (len(all_prompts) + num_gpus - 1) // num_gpus
            prompt_chunks = [all_prompts[i:i+chunk_size] for i in range(0, len(all_prompts), chunk_size)]
            item_map_chunks = [item_map[i:i+chunk_size] for i in range(0, len(item_map), chunk_size)]

            # 分发任务到各个 GPU
            futures = []
            for idx, gpu_id in enumerate(self.qa_gpus):
                futures.append(
                    _ray_inference.remote(
                        model_path=self.qa_model_path,
                        prompts=prompt_chunks[idx],
                        item_map=item_map_chunks[idx],
                        sampling_params=self.sampling_params
                    )
                )

            # 收集所有结果
            outputs = []
            item_maps = []
            for future in ray.get(futures):
                batch_outputs, batch_item_map = future
                outputs.extend(batch_outputs)
                item_maps.extend(batch_item_map)

            # 处理结果
            results = defaultdict(list)
            for (orig_item, inter_item, cap), output in zip(item_maps, outputs):
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
                candidates.sort(key=lambda x: x[1], reverse=True)
                #! compare with the last score
                # ori_score = orig_item['scores']['selected']['total_score']
                # best_caption = candidates[0][0] if candidates[0][1] > ori_score else orig_item["conversations"][1]["value"]
                best_caption = candidates[0][0] if candidates else orig_item["conversations"][1]["value"]
                new_item = json.loads(json.dumps(orig_item))
                #! do not update the original caption
                # new_item["conversations"][1]["value"] = best_caption
                new_item["scores"] = {
                    "selected": {
                        "text": best_caption,
                        # "total_score": candidates[0][1] if candidates[0][1] > ori_score else ori_score,
                        "total_score": candidates[0][1] if candidates else 0.0,
                        "caption_weight": caption_weight,
                        "reasoning_weight": reasoning_weight
                    },
                    "candidates": [{
                        "text": c[0],
                        "total_score": c[1],
                        "caption_score": self._get_caption_score(c[0], orig_item["conversations"][1]["value"]),
                        "reasoning_score": (c[1] - caption_weight * self._get_caption_score(c[0], orig_item["conversations"][1]["value"])) / reasoning_weight
                    } for c in candidates[:topK]]
                }
                updated_data.append(new_item)
        
        else:
            results = defaultdict(list)
            for (orig_item, inter_item, cap) in item_map:
                response = ''
                score = self._calc_score_only_caption(
                    response, cap, 
                    inter_item["ground_truth"], 
                    orig_item["conversations"][1]["value"],
                    caption_weight, 
                    reasoning_weight
                )
                results[orig_item["id"]].append((cap, score))

            updated_data = []
            for orig_item in original_data:
                item_id = orig_item["id"]
                candidates = results.get(item_id, [])
                candidates.sort(key=lambda x: x[1], reverse=True)
                #! compare with the last score
                # ori_score = orig_item['scores']['selected']['total_score']
                # best_caption = candidates[0][0] if candidates[0][1] > ori_score else orig_item["conversations"][1]["value"]
                best_caption = candidates[0][0] if candidates else orig_item["conversations"][1]["value"]
                new_item = json.loads(json.dumps(orig_item))
                #! do not update the original caption
                # new_item["conversations"][1]["value"] = best_caption
                new_item["scores"] = {
                    "selected": {
                        "text": best_caption,
                        # "total_score": candidates[0][1] if candidates[0][1] > ori_score else ori_score,
                        "total_score": candidates[0][1] if candidates else 0.0,
                        "caption_weight": caption_weight,
                        "reasoning_weight": reasoning_weight
                    },
                    "candidates": [{
                        "text": c[0],
                        "total_score": c[1],
                        "caption_score": self._get_caption_score(c[0], orig_item["conversations"][1]["value"]),
                        "reasoning_score": 0.0
                    } for c in candidates[:topK]]
                }
                updated_data.append(new_item)


        # 保存结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, ensure_ascii=False, indent=2)
            f.flush()
        
        total_scores, caption_scores, reasoning_scores = [], [], []
        for item in updated_data:
            total_score, caption_score, reasoning_score = item['scores']['candidates'][0]['total_score'], item['scores']['candidates'][0]['caption_score'], item['scores']['candidates'][0]['reasoning_score']
            total_scores.append(total_score)
            caption_scores.append(caption_score)
            reasoning_scores.append(reasoning_score)
        reasoning_acc = sum([1 for item in reasoning_scores if item > 0.15]) / len(reasoning_scores)
        
        wandb.log({
            "epoch": self.current_epoch,
            "total_score/max": max(total_scores),
            "total_score/mean": sum(total_scores)/len(total_scores),
            "total_score/median": np.median(total_scores),
            "total_score/min": min(total_scores),
            
            "caption_score/max": max(caption_scores),
            "caption_score/mean": sum(caption_scores)/len(caption_scores),
            "caption_score/median": np.median(caption_scores),
            "caption_score/min": min(caption_scores),
            
            "reasoning_score/max": max(reasoning_scores),
            "reasoning_score/mean": sum(reasoning_scores)/len(reasoning_scores),
            "reasoning_score/median": np.median(reasoning_scores),
            "reasoning_score/min": min(reasoning_scores),

            "reasoning_acc": reasoning_acc
        })
        
        # with open(score_path, 'w', encoding='utf-8') as f:
        #     f.write(f"total_scores\nmax: {max(total_scores)}\t mean: {sum(total_scores)/len(total_scores)}\t medium: {np.median(total_scores)}\t min: {min(total_scores)}\n")
        #     f.write(f"caption_scores\nmax: {max(caption_scores)}\t mean: {sum(caption_scores)/len(caption_scores)}\t medium: {np.median(caption_scores)}\t min: {min(caption_scores)}\n")
        #     f.write(f"reasoning_scores\nmax: {max(reasoning_scores)}\t mean: {sum(reasoning_scores)/len(reasoning_scores)}\t medium: {np.median(reasoning_scores)}\t min: {min(reasoning_scores)}\n")

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
        reasoning_score = self._evaluate_response(response, gt_answer)
        caption_score = self._get_caption_score(caption, ref_caption)
        return cw * caption_score + rw * reasoning_score
    
    def _calc_score_only_caption(self, response, caption, gt_answer, ref_caption, cw, rw):
        reasoning_score = 0.0
        caption_score = self._get_caption_score(caption, ref_caption)
        return cw * caption_score + rw * reasoning_score

    def _evaluate_response(self, response, gt_answer):
        format_ok = bool(re.search(r"</think>.*\\boxed{", response, re.DOTALL))
        pred_answer = self.extract_answer(response)
        gt_answer = gt_answer.split("=")[-1].strip()
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


# Ray 远程推理任务
@ray.remote(num_gpus=1)
def _ray_inference(model_path, prompts, item_map, sampling_params):
    # 每个 GPU 单独加载模型
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
    )
    # 推理
    outputs = llm.generate(prompts, sampling_params)
    return outputs, item_map  # 返回结果和 item_map 用于后续处理


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_model_path", type=str, required=True)
    parser.add_argument("--original_data_path", type=str, required=True) 
    parser.add_argument("--intermediate_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--caption_weight", type=float, default=0.5)
    parser.add_argument("--reasoning_weight", type=float, default=0.5)
    parser.add_argument("--topK", type=int, default=3)
    parser.add_argument("--qa_gpus", nargs='+', type=int, default=[0])
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--run_id_path", type=str, default=None, help="WandB run ID path to resume")
    args = parser.parse_args()
    
    run_id_path = os.path.join(args.run_id_path, args.project)
    if not os.path.exists(run_id_path):
        os.mkdir(run_id_path)
    run_id_file = os.path.join(run_id_path, ".run_id")
    if args.current_epoch == 1:
        if os.path.exists(run_id_path):
            import shutil
            shutil.rmtree(run_id_path)
        os.makedirs(run_id_path, exist_ok=True)
        run_id = wandb.util.generate_id()
        with open(run_id_file, "w") as f:
            f.write(run_id)
        wandb.init(project=args.project, id=run_id, resume='allow')
    
    else:
        if not os.path.exists(run_id_file):
            raise FileNotFoundError(
                f"Run ID file not found at {run_id_file}. "
                f"Cannot resume run for current_epoch={args.current_epoch}"
            )
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        wandb.init(project=args.project, id=run_id, resume='must')
    
    
    # wandb.log({"step": args.current_step})

    evaluator = Reasoning_Rewarder(args.qa_model_path, args.qa_gpus, args.current_epoch)
    evaluator.evaluate(
        args.original_data_path,
        args.intermediate_path,
        args.output_path,
        # args.score_path,
        args.caption_weight,
        args.reasoning_weight,
        args.topK
    )