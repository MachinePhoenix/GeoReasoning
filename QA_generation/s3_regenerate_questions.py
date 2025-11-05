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
    1.2 Yyou can introduce some extra information to form a good geometric problem. 
    1.3 If you find that it is hard to generate some difficult questions, just give a simple question. For example, when the lengths of all four sides of a quadrilateral are given, you can no longer assume it is a parallelogram or rectangle. In such cases, the problem may only allow for questions like asking for the perimeter, or determining the length of a segment when a certain point divides a side into an n-equal part, etc.
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
with open("questions_10k_1026_s1_total.json", "r") as f:
    questions = json.load(f)[args.batch*1000: (args.batch+1)*1000]

if os.path.exists(f"questions_10k_1026_s3_{args.batch}.json"):
    with open(f"questions_10k_1026_s3_{args.batch}.json", "r") as f:
        results = json.load(f)
else:
    results = []
batch_size = 10

# here listing the incomplete ids from s2_filter_questions
incomplete_items = []


def get_ques_resp_ans(ques_resp):
    if_incomplete = 0
    if (not 'uestion:\n' in ques_resp) or (not 'esponse:\n' in ques_resp):
        if_incomplete = 1
        return 'none', 'none', 'none', if_incomplete
    ques_resp = ques_resp.split('uestion:\n')[1]
    ques, resp = ques_resp.split('\nGenerated Response')[0], ques_resp.split('esponse:\n')[1]
    if not resp:
        return ques, 'none', 'none', if_incomplete
    ans = extract_content(resp)
    if not ans or ans == 'None':
        if_incomplete = 1
    return ques, resp, ans, if_incomplete

idx = 0
for item in questions[len(results):]:
    if not item['id'] in incomplete_items:
        results.append(item)
        continue
    # difficulty = np.random.randint(low=1, high=5, size=1)[0]
    print(item['id'])
    prompt = prompt_template.format(description=item["caption"])
    if_incomplete = 1
    while if_incomplete:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.8, # increase the temperature
                maxOutputTokens=4096
            )
        )
        ques_resp = response.candidates[0].content.parts[0].text
        ques, resp, ans, if_incomplete = get_ques_resp_ans(ques_resp)
        # pdb.set_trace()

    results.append({
        "id": item["id"],
        'caption': item['caption'],
        # 'gpt_question': item['response'],
        # 'difficulty': str(difficulty),
        'new_question_response': ques_resp,
    })
    idx += 1

    # 每处理batch_size个样本，写入文件一次
    if (idx + 1) % batch_size == 0:
        print(f"Processing item {idx + 1}, writing to file...")
        with open(f"questions_10k_1026_s3_{args.batch}.json", "w") as f:
            json.dump(results, f, indent=2)

# 处理剩余的样本
if results:
    print(f"Processing final batch (total items: {len(results)})")
    with open(f"questions_10k_1026_s3_{args.batch}.json", "w") as f:
        json.dump(results, f, indent=2)