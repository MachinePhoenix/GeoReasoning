import json
import os
import argparse

def add_dataset_to_json(data_name, data_info_path):
    with open(data_info_path, 'r') as f:
        data_info = json.load(f)
    data_info[data_name] = {
        "file_name": f"{data_name}.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "image",
            "id": "id",
            "clause": "clause"
        }
    }
    with open(data_info_path, 'w') as f:
        json.dump(data_info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--data_info_path", required=True)
    args = parser.parse_args()
    add_dataset_to_json(args.data_name, args.data_info_path)


