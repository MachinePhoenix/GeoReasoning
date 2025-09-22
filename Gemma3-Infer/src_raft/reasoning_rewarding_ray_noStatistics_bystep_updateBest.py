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
torch._dynamo.config.cache_size_limit = 64

FORMAT_PROMPT = """You are a mathmetical problem solver. 
Next I will give you the description of a geometric image and the corresponding problem. Here is what you should do: 
1. You should answer the problem based on the description.
2. The final answer should be rounded to two decimal places.
3. Put the final answer in \\boxed{{}}.
"""

class Reasoning_Rewarder:
    def __init__(self, qa_model_path, current_epoch, step, step_size, step_num):
        self.qa_model_path = qa_model_path
        self.qa_gpus = list(range(torch.cuda.device_count()))
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=1024,
            n=1
        )
        self.current_epoch = current_epoch
        self.step=step
        self.step_size=step_size
        self.step_num=step_num

    def evaluate(self, original_data_path, intermediate_path, output_path, caption_weight, reasoning_weight, correct_weight, topK):
        with open(original_data_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)[self.step * self.step_size: (self.step+1) * self.step_size]
        
        with open(intermediate_path, 'r', encoding='utf-8') as f:
            intermediate_data = json.load(f)[self.step * self.step_size: (self.step+1) * self.step_size]

        intermediate_map = {item["id"]: item for item in intermediate_data}

        if len(original_data) != len(intermediate_data):
            raise ValueError("Inconsistent between the original data and the intermediate data")
        
        all_prompts = []
        item_map = []
        for orig_item in original_data:
            inter_item = intermediate_map[orig_item["id"]]
            for cap in inter_item["captions"]:
                all_prompts.append(self._build_qa_prompt(cap, inter_item["question"]))
                item_map.append((orig_item, inter_item, cap))
        
        if abs(reasoning_weight - 0) > -100:
            num_gpus = len(self.qa_gpus)
            chunk_size = (len(all_prompts) + num_gpus - 1) // num_gpus
            prompt_chunks = [all_prompts[i:i+chunk_size] for i in range(0, len(all_prompts), chunk_size)]
            item_map_chunks = [item_map[i:i+chunk_size] for i in range(0, len(item_map), chunk_size)]

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

            outputs = []
            item_maps = []
            for future in ray.get(futures):
                batch_outputs, batch_item_map = future
                outputs.extend(batch_outputs)
                item_maps.extend(batch_item_map)

            results = defaultdict(list)
            for (orig_item, inter_item, cap), output in zip(item_maps, outputs):
                response = output.outputs[0].text
                scores = self._calc_score(
                    response, cap,
                    inter_item["gt_answer"], 
                    orig_item['caption_ori'],
                    caption_weight, 
                    reasoning_weight,
                    correct_weight
                )
                results[orig_item["id"]].append((cap, scores, response))

            updated_data = []
            update_ratio = 0
            for orig_item in original_data:
                item_id = orig_item["id"]
                candidates = results.get(item_id, [])
                candidates.sort(key=lambda x: x[1]['total'], reverse=True)
                #! compare with the last score, only update data when it has a higher score
                ori_score = orig_item['scores']['selected']['total_score'] if 'scores' in orig_item.keys() else 0.
                if_update = 0
                if candidates[0][1]['total'] >= ori_score:
                    best_score = candidates[0][1]['total']
                    best_caption_score = candidates[0][1]['caption']
                    best_reasoning_score = candidates[0][1]['reasoning']
                    best_caption = candidates[0][0]
                    if_update = 1
                    update_ratio += 1
                else:
                    best_score = ori_score
                    best_caption_score = orig_item['scores']['selected']['caption_score']
                    best_reasoning_score = orig_item['scores']['selected']['reasoning_score']
                    best_caption = orig_item['conversations'][1]['value']
                #! update at any time
                # best_caption = candidates[0][0] if candidates else orig_item["conversations"][1]["value"]
                new_item = json.loads(json.dumps(orig_item))
                new_item["conversations"][1]["value"] = best_caption
                new_item["scores"] = {
                    "selected": {
                        "text": best_caption,
                        "total_score": best_score,
                        # "total_score": candidates[0][1] if candidates else 0.0,
                        "caption_score": best_caption_score,
                        "reasoning_score": best_reasoning_score,
                        "if_update": 'yes' if if_update else 'no',
                        "caption_weight": caption_weight,
                        "reasoning_weight": reasoning_weight
                    },
                    "candidates": [{
                        "text": c[0],
                        "total_score": c[1]['total'],
                        "caption_score": c[1]['caption'],
                        "reasoning_score": c[1]['reasoning'],
                        "response": c[2]
                    } for c in candidates[:topK]]
                }
                new_item['scores']['candidates'][0]['mean_score'] = sum(scores['total'] for text, scores, response in candidates) / len(candidates)
                updated_data.append(new_item)
        
        if self.step == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(updated_data, f, ensure_ascii=False, indent=2)
                f.flush()
        else:
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"The output file {output_path} does not exist, can not add items to it.")
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_data.extend(updated_data)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
                f.flush()
        
        total_scores, caption_scores, reasoning_scores = [], [], []
        candidates_mean_scores = []
        for item in updated_data:
            total_score, caption_score, reasoning_score = item['scores']['selected']['total_score'], item['scores']['selected']['caption_score'], item['scores']['selected']['reasoning_score']
            total_scores.append(total_score)
            caption_scores.append(caption_score)
            reasoning_scores.append(reasoning_score)
            candidates_mean_scores.append(item['scores']['candidates'][0]['mean_score'])
        reasoning_acc = sum([1 for item in reasoning_scores if item > 1-correct_weight+0.05]) / len(reasoning_scores)
        
        wandb.log({
            "epoch": self.current_epoch,
            "real_step": self.current_epoch*self.step_num + self.step,
            "update_ratio": float(update_ratio) / len(original_data),
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

            "candidates_mean_scores/max": max(candidates_mean_scores),
            "candidates_mean_scores/mean": sum(candidates_mean_scores)/len(candidates_mean_scores),
            "candidates_mean_scores/median": np.median(candidates_mean_scores),
            "candidates_mean_scores/min": min(candidates_mean_scores),

            "reasoning_acc": reasoning_acc,
            
        })
        wandb.finish()

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
        [Answer]
        """

    def _calc_score(self, response, caption, gt_answer, ref_caption, cw, rw, correct_weight):
        reasoning_score = self._evaluate_response(response, gt_answer, correct_weight)
        caption_score = self._get_caption_score(caption, ref_caption)
        return {'total': cw * caption_score + rw * reasoning_score, 
                'caption': caption_score, 
                'reasoning': reasoning_score}
    
    def _calc_score_only_caption(self, response, caption, gt_answer, ref_caption, cw, rw, correct_weight):
        reasoning_score = 0.0
        caption_score = self._get_caption_score(caption, ref_caption)
        return {'total': cw * caption_score + rw * reasoning_score, 
                'caption': caption_score, 
                'reasoning': reasoning_score}

    def _evaluate_response(self, response, gt_answer, correct_weight):
        format_ok = bool(re.search(r"\\boxed{", response))
        pred_answer = self.extract_answer(response)
        try:
            pred_val = float(pred_answer)
            gt_val = float(gt_answer)
            accuracy = 1.0 if abs(pred_val - gt_val) < 1e-3 else 0.0
        except:
            accuracy = 1.0 if pred_answer.lower() == gt_answer.lower() else 0.0
        return correct_weight * accuracy + (1-correct_weight) * float(format_ok)

    @staticmethod
    def extract_answer(response: str) -> str:
        boxed_match = re.search(r"\\boxed{([^}]+)}", response)
        if boxed_match:
            return boxed_match.group(1).strip()
        last_number = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        return last_number[-1] if last_number else ""


@ray.remote(num_gpus=1)
def _ray_inference(model_path, prompts, item_map, sampling_params):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        max_num_seqs=1024,
    )
    outputs = llm.generate(prompts, sampling_params)
    return outputs, item_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_model_path", type=str, required=True)
    parser.add_argument("--original_data_path", type=str, required=True) 
    parser.add_argument("--intermediate_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    # parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--caption_weight", type=float, default=0.5)
    parser.add_argument("--reasoning_weight", type=float, default=0.5)
    parser.add_argument("--correct_weight", type=float, default=0.9)
    parser.add_argument("--topK", type=int, default=3)
    # parser.add_argument("--qa_gpus", nargs='+', type=int, default=[0])
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--current_epoch", type=int, default=0)
    parser.add_argument("--run_id_path", type=str, default=None, help="WandB run ID path to resume")
    parser.add_argument("--step", type=int, default=0, help="the rank of the current step in an epoch")
    parser.add_argument("--step_size", type=int, default=500, help="the batchsize of each step")
    parser.add_argument("--step_num", type=int, default=4, help="the number of steps in an epoch")
    args = parser.parse_args()
    
    run_id_path = args.run_id_path
    if not os.path.exists(run_id_path):
        os.mkdir(run_id_path)
    run_id_file = os.path.join(run_id_path, ".run_id")
    #! epoch=0 indicates the primary model
    if args.current_epoch == 0 and args.step == 0:
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
    
    ray.init(ignore_reinit_error=True)
    
    evaluator = Reasoning_Rewarder(args.qa_model_path, args.current_epoch, args.step, args.step_size, args.step_num)
    evaluator.evaluate(
        args.original_data_path,
        args.intermediate_path,
        args.output_path,
        # args.score_path,
        args.caption_weight,
        args.reasoning_weight,
        args.correct_weight,
        args.topK
    )
