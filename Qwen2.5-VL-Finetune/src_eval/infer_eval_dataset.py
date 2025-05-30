import json
import re
from vllm import LLM, SamplingParams
from PIL import Image
import argparse
import os
from typing import Dict, List
import torch

def process_caption(caption: str) -> str:
    """优化后的文本清洗函数"""
    caption = re.sub(r'<image>.*?(</image>|$)', '', caption, flags=re.DOTALL)
    caption = re.sub(r'\bimg/\d+\.?\w*\b', '', caption, flags=re.IGNORECASE)
    bullets = re.findall(r'-\s*(.+?)(?=\n-|\n\s*\d+\.|\Z)', caption, flags=re.DOTALL)
    cleaned = ' '.join(b.strip() for b in bullets) if bullets else caption.strip()
    return re.sub(r'\s+', ' ', cleaned).strip('^[^a-zA-Z0-9]*')

def extract_answer(response: str) -> str:
    """通用答案提取函数（复用原代码逻辑）"""
    boxed_match = re.search(r"\\boxed{([^}]+)}", response)
    if boxed_match:
        return boxed_match.group(1).strip()
    
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
        matches = list(re.finditer(pattern, response, re.IGNORECASE))
        if matches:
            match = max(matches, key=lambda m: len(m.group(group_idx)))
            answer = match.group(group_idx).strip()
            break
    
    answer = re.sub(r'^[^$\d\\]+', '', answer)
    answer = re.sub(r'[^$\d.+\-\\/√()\\dfrac{}^a-zA-Z]', '', answer)
    answer = answer.replace(' ', '')
    return answer

class QwenVLEvaluator:
    def __init__(self, image_root, model_path, main_gpus):
        self.image_root = image_root
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=len(main_gpus),
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            enforce_eager=True,
            # max_num_seqs=64,
            max_model_len=2048,
            dtype=torch.bfloat16
        )

    def _build_prompt(self, question: str) -> str:
        """构建标准化的多模态提示模板"""
        return (
            "<|im_start|>system\nYou are a geometric analysis expert. Please answer the following question based on the given image. Please provide the final answer in \\boxed{}. <|im_end|>\n"
                "<|im_start|>user\n"
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                f"{question}<|im_end|>\n"
                "<|im_start|>assistant\n"
        )

    def evaluate_dataset(self, data_path: str, output_path: str, batch_size: int = 8):
        """执行数据集评估"""
        # 加载JSONL数据集
        with open(data_path, 'r') as f:
            dataset = [json.loads(line) for line in f]
        
        correct_count = 0
        results = []
        
        # 分批处理
        batch_size = len(dataset)
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            inputs = []
            
            # 构建输入
            for item in batch:
                img_path = os.path.join(self.image_root, item["image"])
                image = Image.open(img_path).convert("RGB")
                
                prompt = self._build_prompt(item["question"])
                inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                    "id": item["id"],
                    "answer": item["answer"]
                })

            # 执行推理
            outputs = self.llm.generate(
                inputs,
                SamplingParams(
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                    n=1,
                    stop=["<|im_end|>"]
                )
            )

            # 处理结果
            for item, output in zip(batch, outputs):
                response = output.outputs[0].text
                pred_answer = extract_answer(response)
                gt_answer = extract_answer(item["answer"])
                
                is_correct = self._compare_answers(pred_answer, gt_answer)
                if is_correct:
                    correct_count += 1
                
                results.append({
                    "id": item["id"],
                    "question": item["question"],
                    "answer": item['answer'],
                    "ground_truth": gt_answer,
                    "predicted": pred_answer,
                    "response": response,
                    "correct_quick_judge": is_correct
                })

        # 计算正确率
        accuracy = correct_count / len(results) if results else 0
        print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        
        # 保存结果
        with open(output_path, 'w') as f:
            json.dump({
                "accuracy_quick_judge": accuracy,
                "correct_count_quick_judge": correct_count,
                "total": len(results),
                "results": results
            }, f, indent=2, ensure_ascii=False)
            f.flush()
        
        return accuracy

    # def _standardize_answer(self, answer: str) -> str:
    #     """标准化参考答案格式"""
    #     # 移除单位和上下文
    #     answer = re.sub(r'([0-9.]+).*$', r'\1', answer)
    #     # 提取数学符号
    #     answer = re.sub(r'^.*?([0-9.]+(?:\s*[\^\w+]?)?)$', r'\1', answer)
    #     return answer.strip()

    def _compare_answers(self, pred: str, gt: str) -> bool:
        """比较预测答案与参考答案"""
        # 直接文本匹配
        if pred == gt:
            return True
            
        # 尝试数值比较
        try:
            pred_val = float(pred.replace('$', '').replace('\\', ''))
            gt_val = float(gt.replace('$', '').replace('\\', ''))
            return abs(pred_val - gt_val) < 1e-3
        except:
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--data_path", required=True, help="输入数据路径（JSONL格式）")
    parser.add_argument("--image_path", required=True, help="图片根目录")
    parser.add_argument("--output_path", required=True, help="输出文件路径")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批次大小")
    parser.add_argument("--main_gpus", nargs='+', type=int, default=[0], help="使用GPU列表")
    
    args = parser.parse_args()
    
    evaluator = QwenVLEvaluator(
        image_root=args.image_path,
        model_path=args.model_path,
        main_gpus=args.main_gpus
    )
    
    accuracy = evaluator.evaluate_dataset(
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size
    )
    
    print(f"Final Accuracy: {accuracy:.2%}")