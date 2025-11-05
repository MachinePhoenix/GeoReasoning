import json
from google import genai
from google.genai import types
import argparse
import os
import re
from collections import defaultdict
import numpy as np
import pdb

parser = argparse.ArgumentParser(description='处理索引参数')
parser.add_argument('--batch', type=int, required=True)

args = parser.parse_args()

def extract_content(s):
    # 判断是否全为数字
    if re.fullmatch(r'^\d+$', s):
        return s
    match = re.search(r'\\boxed{([^}]*)', s)
    return match.group(1) if match else None


# 初始化客户端
client = genai.Client(api_key="AIzaSyB3JMWlcsNZI2SjQvycj1IRZJyZfT3kSrM")

# 提示词模板
prompt_template = """You are a helpful dataset processor. Please:
1. Generate a mathemetical question according to the given description of a geometric image with the following requirements:
    1.1 The problem should base on the given description, i.e., you must **NOT** generate problems that are unrelated to the given description. 
    1.2 The problem difficulty should not be too low, such as determining some information in the description. 
    1.3 It should also not be too hard, like introducing too much extra information, but anyway you can introduce some extra information to form a good geometric problem. 
    1.4 You should **NOT** include or repeat any information in the description, and just contain the real question. For example, if the description says: `Line segment AB is present.The length of BA is 1.24.', then when you generate the question, you should not include the length of AB.
    1.5 If the question is inconsistent with the given description, the final answer should be `None'.
2. Answer the question you just provided, and express the final answer to two decimal places. The final answer should be in \\boxed{{answer}}.

Description: 
{description}
Generated Question:
{{question}}
Generated Response:
{{response}}
Final Answer:
\\boxed{{answer}}
"""

# 读取原始 JSON 文件
with open("questions_10k_1026.json", "r") as f:
    questions = json.load(f)[args.batch*1000: (args.batch+1)*1000]

if os.path.exists(f"questions_10k_1026_s1_{args.batch}.json"):
    with open(f"questions_10k_1026_s1_{args.batch}.json", "r") as f:
        results = json.load(f)
else:
    results = []

questions = questions[len(results):]

batch_size = 10  # 每batch_size个样本写入一次

for idx, item in enumerate(questions):
    # 构造提示词
    # difficulty = np.random.randint(low=1, high=5, size=1)[0]
    prompt = prompt_template.format(description=item["caption"])
    
    # 调用 Gemini API
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2,
            maxOutputTokens=4096
        )
    )

    # 提取答案
    ques_resp = response.candidates[0].content.parts[0].text
    results.append({
        "id": item["id"],
        'caption': item['caption'],
        # 'gpt_question': item['response'],
        # 'difficulty': str(difficulty),
        'new_question_response': ques_resp,
    })

    # 每处理batch_size个样本，写入文件一次
    if (idx + 1) % batch_size == 0:
        print(f"Processing item {idx + 1}, writing to file...")
        with open(f"questions_10k_1026_s1_{args.batch}.json", "w") as f:
            json.dump(results, f, indent=2)

# 处理剩余的样本
if results:
    print(f"Processing final batch (total items: {len(results)})")
    with open(f"questions_10k_1026_s1_{args.batch}.json", "w") as f:
        json.dump(results, f, indent=2)