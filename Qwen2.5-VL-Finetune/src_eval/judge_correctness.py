import json
import re
from vllm import LLM, SamplingParams
import argparse
import torch

class LLMAnswerEvaluator:
    def __init__(self, model_path, main_gpus):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_model_len=2048,
            dtype=torch.bfloat16
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.95,
            max_tokens=1024
        )

    def _build_prompt(self, question: str, answer: str, response: str) -> str:
        return (f"Task: Check if the model response matches the correct answer.\n"
            f"[Question]\n{question}"
            f"[Correct Answer]\n{answer}"
            f"[Model Response]\n{response}"
            f"[Instruction]\nOutput 'Correct' if the response matches the answer exactly.\nOtherwise, output 'Error'. Do NOT include any explanation or extra text.")

    def evaluate_correctness(self, results_data: dict) -> tuple:
        dataset = results_data["results"]
        correct_count = 0
        
        # 构建所有prompt输入
        inputs = []
        for item in dataset:
            prompt = self._build_prompt(
                item["question"],
                item["ground_truth"],
                item["response"]
            )
            inputs.append(prompt)
        
        # 批量推理
        outputs = self.llm.generate(inputs, self.sampling_params)
        
        # 处理结果
        for i, output in enumerate(outputs):
            result = output.outputs[0].text.strip()
            is_correct = "Correct" in result or "correct" in result
            
            # 更新结果
            dataset[i]["correct_llm"] = is_correct
            if is_correct:
                correct_count += 1
                
        # 计算准确率
        accuracy = correct_count / len(dataset) if dataset else 0
        
        # 更新元数据
        results_data["accuracy_llm"] = accuracy
        results_data["correct_count_llm"] = correct_count
        
        return results_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="原始结果文件路径")
    parser.add_argument("--output_path", help="更新后的结果文件路径")
    parser.add_argument("--model_path", required=True, help="Qwen7B-Instruct模型路径")
    parser.add_argument("--main_gpus", nargs='+', type=int, default=[0], help="使用GPU列表")
    
    args = parser.parse_args()
    # if not args.output_path:
    #     args.output_path = args.input_path
    
    # 加载现有结果
    with open(args.input_path, 'r') as f:
        results_data = json.load(f)
    
    # 创建评估器并执行评估
    evaluator = LLMAnswerEvaluator(args.model_path, args.main_gpus)
    updated_data = evaluator.evaluate_correctness(results_data)
    
    # 保存更新后的结果
    with open(args.output_path, 'w') as f:
        json.dump(updated_data, f, indent=2, ensure_ascii=False)
        f.flush()
    
    print(f"LLM evaluation completed, Accuracy: {updated_data['accuracy_llm']:.2%} ({updated_data['correct_count_llm']}/{len(updated_data['results'])})")

if __name__ == "__main__":
    main()