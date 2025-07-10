import os
import json
import argparse
from glob import glob
import re
from tqdm import tqdm

def extract_judge_results(text):
    # Find all scores
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
    no_similar_data = []
    similar_number = 0
    unresolve_number = 0
    for item in tqdm(all_data):
        # Get the value of extract_reference_answer_output
        judge_res_list = []
        for similar_data in item["retrieved_benchmark"]:
            raw_output = similar_data.get("Llama3.3-70B-judge")
            judge_res = extract_judge_results(raw_output)
            similar_data["llm_judge_similarity_result"] = judge_res
            if judge_res == None:
                unresolve_number += 1
                continue
            judge_res_list.append(judge_res)
        item["decontamination_res"] = any(judge_res_list)
        if len(judge_res_list) > 0 and any(judge_res_list):
            similar_number += 1
        elif len(judge_res_list) == 0:
            print("no judge_res_list")
            continue
        else:
            no_similar_data.append(item)
    print(f"unresolve_number: {unresolve_number}")
    print(f"similar_number: {similar_number}")
    print(f"no_similar_number: {len(no_similar_data)}")
    return no_similar_data

def save_filtered_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files for decontamination")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Directory containing the input data")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for the filtered data")
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process all the files
    new_data = process_jsonl_files(args.input_data_dir)
    
    # Save the filtered data
    save_filtered_data(new_data, args.output_path)