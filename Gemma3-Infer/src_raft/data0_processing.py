import json
import os
import argparse

def add_caption_to_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for item in data:
        if len(item.get('conversations', [])) > 1:
            item['caption_ori'] = item['conversations'][1]['value']
        else:
            print(f"Warning: Invalid conversations format in item {item.get('id')}")
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"successfully processing {len(data)} data and covering the original json file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True)
    args = parser.parse_args()
    add_caption_to_json(args.file_path)
