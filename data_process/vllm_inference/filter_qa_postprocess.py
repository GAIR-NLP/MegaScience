import os
import json
from glob import glob
import re
import argparse

def extract_filtering_results(text):
    # Find all score
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
    new_data = []
    
    unresolve_number = 0
    for item in all_data:
        raw_output = item.get("filtering_qa_output")
        filtering_res = extract_filtering_results(raw_output)
        if filtering_res:
            new_data.append(item)
        else:
            continue
    return new_data

def save_filtered_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files and filter data")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory path containing the input JSONL files"
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Output file path for the filtered data"
    )
    
    args = parser.parse_args()
    
    # Check if output directory exists, create if it doesn't
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Process all the files in the input directory
    new_data = process_jsonl_files(args.input_dir)
    
    # Save the filtered data to the output file
    save_filtered_data(new_data, args.output_file)