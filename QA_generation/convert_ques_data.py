import os
import json
import pdb
import re

with open('questions_10k_1026_s3_total.json', 'r') as f:
    questions = json.load(f)

with open('10k_1026.json', 'r') as f:
    data = json.load(f)

def get_exact_question(q):
    if 'Generated Response:' in q['new_question_response']:
        ques = q['new_question_response'].split('Generated Response:')[0]
        ques = ques.split('Generated Question:\n')[1]
    elif '\nResponse' in q['new_question_response']:
        ques = q['new_question_response'].split('\nResponse')[0]
        if 'Question:\n' in ques:
            ques = ques.split('Question:\n')[1]
            # print(q['id'], ques)
        elif 'Question' in ques:
            ques = ques.split('Question:')[1]
            # print(q['id'], ques)
    return ques


# for q in questions:
    # if 'Generated Response:' in q['new_question_response']:
    #     ques = q['new_question_response'].split('Generated Response:')[0]
    #     ques = ques.split('Generated Question:\n')[1]
    # elif '\nResponse' in q['new_question_response']:
    #     ques = q['new_question_response'].split('\nResponse')[0]
    #     if 'Question:\n' in ques:
    #         ques = ques.split('Question:\n')[1]
    #         # print(q['id'], ques)
    #     elif 'Question' in ques:
    #         ques = ques.split('Question:')[1]
    #         # print(q['id'], ques)
    # else:
    #     print(q['id'])
    
    # pattern = r'\\boxed\{([^}]*)\}'
    # match = re.search(pattern, q['new_question_response'])
    # if match:
    #     extracted_content = match.group(1)
    #     print(f"{extracted_content}")
    # else:
    #     print(q['id'])


new_data = []
for item in data:
    # print(item)
    assert len(questions) == len(data)
    for q in questions:
        if q['id'] == item['id']:
            ques = get_exact_question(q)
            pattern = r'\\boxed\{([^}]*)\}'
            match = re.search(pattern, q['new_question_response'])
            if match:
                ans = match.group(1)
                # print(f"{extracted_content}")
            else:
                print(q['id'])
            break
    item['caption_ori'] = item['conversations'][1]['value']
    item['question'] = ques
    item['gt_answer'] = ans
    new_data.append(item)

with open(f"10k_1026.json", "w") as f:
    json.dump(new_data, f, indent=2)