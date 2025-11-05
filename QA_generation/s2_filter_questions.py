import json
from google import genai
from google.genai import types
import argparse
import os
import re
from collections import defaultdict
import numpy as np
import pdb


def extract_content(s):
    # 判断是否全为数字
    if re.fullmatch(r'^\d+$', s):
        return s
    match = re.search(r'\\boxed{([^}]*)', s)
    return match.group(1) if match else None


def get_ques_resp_ans(item):
    if_incomplete = 0
    ques_resp = item['new_question_response']
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



with open("questions_10k_1026_s1_total.json", "r") as f:
    questions = json.load(f)


# pdb.set_trace()
results = []

incomplete_items = []
for idx, item in enumerate(questions):
    ques, resp, ans, if_incomplete = get_ques_resp_ans(item)
    item['question'] = ques
    item['response'] = resp
    item['answer'] = ans
    # del item['gpt_question']
    results.append(item)
    if if_incomplete:
        incomplete_items.append(item['id'])

print(incomplete_items, len(incomplete_items))

# print(f"Processing final batch (total items: {len(results)})")
# with open("new_questions_s2.json", "w") as f:
#     json.dump(results, f, indent=2)