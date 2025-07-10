import os
import argparse
import json
from glob import glob
import re
from tqdm import tqdm

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
    return all_data

def save_filtered_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files and split into CoT and non-CoT data")
    
    parser.add_argument(
        "--input_no_cot_dir", 
        type=str, 
        required=True,
        help="Input JSONL file path containing the data to be processed"
    )
    
    parser.add_argument(
        "--input_cot_file", 
        type=str, 
        required=True,
        help="Input JSONL file path containing the data to be processed"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output file path for CoT (Chain of Thought) data (JSONL format)"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_no_cot_dir):
        print(f"Error: Input file '{args.input_no_cot_dir}' does not exist.")
        sys.exit(1)
        
    if not os.path.exists(args.input_cot_file):
        print(f"Error: Input file '{args.input_cot_file}' does not exist.")
        sys.exit(1)
    
    
    # Create output directories if they don't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Process the input file
    no_cot_data = process_jsonl_files(args.input_no_cot_dir)
    
    cot_data = []
    with open(args.input_cot_file, 'r', encoding='utf-8') as f:
        for line in f:
            cot_data.append(json.loads(line))
    print(f"Load {len(cot_data)} cot data.")
    
    output_data = no_cot_data + cot_data
    
    # Save the filtered data
    save_filtered_data(output_data, args.output)
    
    print(f"Processing completed.")
    print(f"output {len(output_data)} data.")
    print(f"data saved to: {args.output}")