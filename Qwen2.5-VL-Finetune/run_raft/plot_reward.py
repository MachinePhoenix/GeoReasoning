import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import pdb

epoch_list = np.arange(1, 11, 1)

scores = {}

for epoch in epoch_list:
    scores[epoch] = {}
    reward_path = f'/data/yanshi.xy/MLLM_RL/EasyR1/GeoReasoning/train/data{epoch}.json'
    with open(reward_path) as f:
        data = json.load(f)

    total_scores, caption_scores, reasoning_scores = [], [], []
    for item in data:
        total_score, caption_score, reasoning_score = item['scores']['candidates'][0]['total_score'], item['scores']['candidates'][0]['caption_score'], item['scores']['candidates'][0]['reasoning_score']
        total_scores.append(total_score)
        caption_scores.append(caption_score)
        reasoning_scores.append(reasoning_score)
    scores[epoch]['total'] = {'max': max(total_scores), 'mean': sum(total_scores)/len(total_scores), 'medium': np.median(total_scores), 'min': min(total_scores)}
    scores[epoch]['caption'] = {'max': max(caption_scores), 'mean': sum(caption_scores)/len(caption_scores), 'medium': np.median(caption_scores), 'min': min(caption_scores)}
    scores[epoch]['reasoning'] = {'max': max(reasoning_scores), 'mean': sum(reasoning_scores)/len(reasoning_scores), 'medium': np.median(reasoning_scores), 'min': min(reasoning_scores)}
    correct = [1 if sco > 0.1 else 0 for sco in reasoning_scores]
    scores[epoch]['reasoning_acc'] = sum(correct) / len(correct)

for category in ['total', 'caption', 'reasoning']:
    plt.figure(figsize=(10, 6))
    
    # 为每个统计量绘制曲线
    for stat in ['max', 'mean', 'medium', 'min']:
        # 提取对应统计量的数值序列
        y_values = [scores[epoch][category][stat] for epoch in epoch_list]
        
        plt.plot(epoch_list, y_values, marker='o', label=stat.capitalize())

    # plt.title(f'{category.capitalize()} Score', fontsize=14)
    plt.xlabel('Training epochs', fontsize=12)
    plt.ylabel(f'{category.capitalize()} Score', fontsize=12)
    plt.xticks(epoch_list, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Statistics')
    plt.tight_layout()
    plt.savefig(f'{category}_scores.jpg')

plt.figure(figsize=(10, 6))
acc = [scores[epoch]['reasoning_acc'] for epoch in epoch_list]
plt.plot(epoch_list, acc)
# plt.title(f'Accuracy', fontsize=14)
plt.xlabel('Training epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(epoch_list, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Statistics')
plt.tight_layout()
plt.savefig(f'reasoning_acc_scores.jpg')