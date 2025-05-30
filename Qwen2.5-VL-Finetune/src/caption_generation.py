import json
import re
from vllm import LLM, SamplingParams
from PIL import Image
import argparse
import os
from typing import Dict
import torch

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

class CaptionGenerator:
    def __init__(self, image_root, model_path, main_gpus):
        self.image_root = image_root
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=len(main_gpus),
            trust_remote_code=True,
            gpu_memory_utilization=0.7,
            enforce_eager=True,
            max_num_seqs=64,
            max_model_len=2048,
            dtype=torch.bfloat16
        )

    def _build_prompt(self, user_prompt: str) -> str:
        """构建标准化的多模态提示模板"""
        return (
            "<|im_start|>system\nYou are a geometric analysis expert.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>\n"
            f"{user_prompt.replace('<image>', '').strip()}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate(self, data_path, sample_num, output_path, questions_path):
        # 加载数据
        with open(data_path) as f:
            dataset = json.load(f)
        with open(questions_path) as f:
            qa_db = {item["id"]: self._parse_qa(item["response"]) for item in json.load(f)}

        # 分批处理
        batch_size = 1000
        results = []
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            inputs = []
            
            for item in batch:
                # 加载图像
                img_path = os.path.join(self.image_root, item["image"])
                image = Image.open(img_path).convert("RGB")
                
                # 构建输入
                user_prompt = next(
                    m["value"] for m in item["conversations"] 
                    if m["from"] == "human"
                )
                inputs.append({
                    "prompt": self._build_prompt(user_prompt),
                    "multi_modal_data": {"image": image}
                })

            # 执行推理
            outputs = self.llm.generate(
                inputs,
                SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=1024,
                    n=sample_num,
                    stop=["<|im_end|>"]
                )
            )
            
            # 处理结果
            for item, output in zip(batch, outputs):
                results.append({
                    **item,
                    "captions": [o.text for o in output.outputs],
                    "question": qa_db.get(item["id"], {}).get("question", ""),
                    "ground_truth": qa_db.get(item["id"], {}).get("answer", "")
                })

        # 保存结果
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            f.flush()

    def _parse_qa(self, response: str) -> Dict[str, str]:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--data_path", required=True, help="输入数据路径")
    parser.add_argument("--image_path", required=True, help="图片根目录") 
    parser.add_argument("--sample_num", type=int, default=8, help="每个样本生成数量")
    parser.add_argument("--output_path", required=True, help="输出文件路径")
    parser.add_argument("--questions_path", required=True, help="QA数据路径")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批次大小")
    parser.add_argument("--main_gpus", nargs='+', type=int, default=[0], help="使用GPU列表")
    
    args = parser.parse_args()
    
    generator = CaptionGenerator(
        image_root=args.image_path,
        model_path=args.model_path,
        main_gpus=args.main_gpus
    )
    generator.generate(
        data_path=args.data_path,
        sample_num=args.sample_num,
        output_path=args.output_path,
        questions_path=args.questions_path
    )