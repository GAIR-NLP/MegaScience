import os
import json
from glob import glob
import re
import argparse

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
    
    for item in all_data:
        # Get the value of extract_reference_answer_output
        raw_output = item.get("extract_reference_answer_output", "")
        
        # Check if there is \boxed{} content - only take the last one
        boxed_pattern = r'\$\\boxed\{(.*?)\}\$'
        boxed_matches = list(re.finditer(boxed_pattern, raw_output))
        
        if boxed_matches:
            # Take the last \boxed{} content
            reference_answer = boxed_matches[-1].group(1)
        elif "Reference Answer:" in raw_output:
            # If there are multiple "Reference Answer:", take the last one
            parts = raw_output.split("Reference Answer:")
            reference_answer = parts[-1].strip()
        elif "final answer is:" in raw_output.lower():
            # If there are multiple "final answer is:", take the last one
            parts = re.split(r"final answer is:", raw_output, flags=re.IGNORECASE)
            reference_answer = parts[-1].strip()
        else:
            # If none of the above, use the entire content
            reference_answer = raw_output
        
        # Create a new dictionary containing original data and extracted reference answer
        item["reference_answer"] = reference_answer
        new_data.append(item)
    
    return new_data

def save_filtered_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files and extract reference answers")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Directory containing the input JSONL files")
    parser.add_argument("--output_path", type=str, required=True, help="Output file path for the processed data")
    
    args = parser.parse_args()
    
    # Create output directories if they don't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process all the files
    new_data = process_jsonl_files(args.input_data_dir)
    
    # Save the filtered data
    save_filtered_data(new_data, args.output_path)