import os
import argparse
import json
from glob import glob
import re
from tqdm import tqdm

def extract_judge_results(text):
    pattern = r"(?:Decision:)\s+(YES|Yes|NO|No)"
    match = re.findall(pattern, text)
    
    if match:
        if match[-1] == "YES" or match[-1] == "Yes":
            return True
        elif match[-1] == "NO" or match[-1] == "No":
            return False
    
    yes_idx = -1
    no_idx = -1
    if "YES" in text:
        yes_idx = text.rfind("YES")
    if "NO" in text:
        no_idx = text.rfind("NO")
    if "Yes" in text:
        yes_idx = max(text.rfind("Yes"), yes_idx)
    if "No" in text:
        no_idx = max(text.rfind("No"), no_idx)
        
    if yes_idx == -1 and no_idx == -1:
        return None
    if yes_idx > no_idx:
        return True
    if no_idx > yes_idx:
        return False

def process_jsonl_files(root_dir):
    # List to store all data
    all_data = []
    
    # Find all jsonl files in the root directory and its subdirectories
    jsonl_files = glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Process each file
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Parse the JSON object
                        item = json.loads(line.strip())
                        
                        # Add to our list of all data
                        all_data.append(item)
                    except json.JSONDecodeError:
                        print(f"Error parsing JSON in file {file_path}")
                        continue
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    print(f"Loaded {len(all_data)} total items from all files")
    
    # Filter data based on "Result: Yes"
    cot_data = []
    no_cot_data = []
    no_cot_number = 0
    unresolve_number = 0
    for item in tqdm(all_data):
        # 获取 extract_reference_answer_output 的值
        raw_output = item.get("judge_cot_output")
        judge_cot_res = extract_judge_results(raw_output)
        item["judge_cot_result"] = judge_cot_res
        if judge_cot_res == None:
            unresolve_number += 1
            refined_question = item.get("refined_question", "")
            if refined_question == "" or refined_question == "<refined question above>":
                continue
            print("="*50 + "raw_output" + "="*50)
            print(raw_output)
            print("="*100)
            print("="*50 + "item" + "="*50)
            print(item)
            print("="*100)
            continue
            # print("="*50 + "answer" + "="*50)
            # print(item.get(refined_answer))
            # print("="*100)
        if not judge_cot_res:
            no_cot_number += 1
            no_cot_data.append(item)
        else:
            cot_data.append(item)
    print(f"unresolve_number: {unresolve_number}")
    print(f"no_cot_number: {no_cot_number}")
    print(f"cot_number: {len(cot_data)}")
    return cot_data, no_cot_data

def save_filtered_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files and split into CoT and non-CoT data")
    
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Input JSONL file path containing the data to be processed"
    )
    
    parser.add_argument(
        "--output_cot", 
        type=str, 
        required=True,
        help="Output file path for CoT (Chain of Thought) data (JSONL format)"
    )
    
    parser.add_argument(
        "--output_no_cot", 
        type=str, 
        required=True,
        help="Output file path for non-CoT data (JSONL format)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Create output directories if they don't exist
    for output_path in [args.output_cot, args.output_no_cot]:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Process the input file
    cot_data, no_cot_data = process_jsonl_files(args.input_file)
    
    # Save the filtered data
    save_filtered_data(cot_data, args.output_cot)
    save_filtered_data(no_cot_data, args.output_no_cot)
    
    print(f"Processing completed.")
    print(f"CoT data saved to: {args.output_cot}")
    print(f"Non-CoT data saved to: {args.output_no_cot}")