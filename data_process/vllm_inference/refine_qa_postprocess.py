import os
import argparse
import json
from glob import glob
import nltk
from tqdm import tqdm

def remove_empty_boxes(text):
    """
    Removes sentences containing empty box notation ($\\boxed{}$) and any sentences following them.

    Args:
        text (str): Input text to process

    Returns:
        str: Processed text with sentences containing empty boxes and following sentences removed
    """
    # Split the text into paragraphs
    paragraphs = text.split('\n\n')
    processed_paragraphs = []

    for para_i, paragraph in enumerate(paragraphs):
        # Check if paragraph contains an empty box
        if r'$\boxed{}$' in paragraph or r'\boxed{}' in paragraph:
            # Split paragraph into sentences
            # This regex handles most common sentence endings (., !, ?)
            # and tries to handle the case where sentences end with quotation marks
            sentences = nltk.sent_tokenize(paragraph)

            # Find the index of the first sentence containing an empty box
            box_index = -1
            for i, sentence in enumerate(sentences):
                if r'$\boxed{}$' in sentence or r'\boxed{}' in sentence:
                    box_index = i
                    break

            # Keep only sentences before the box
            if box_index > 0:
                cleaned_paragraph = ' '.join(sentences[:box_index])
                processed_paragraphs.append(cleaned_paragraph)
            # If box is in the first sentence, skip the whole paragraph
            if para_i >= (len(paragraphs) - 2):
                break
        else:
            # If no box in paragraph, keep it as is
            processed_paragraphs.append(paragraph)

    # Join the processed paragraphs back together
    return '\n\n'.join(processed_paragraphs)

def remove_last_note(text):
    text = text.strip()
    sentences = nltk.sent_tokenize(text)
    if len(sentences) == 0:
        return text
    last_sentence = sentences[-1].strip()
    # result = text
    patterns = ["Note:", "(Note:", "*(Note:", "(*Note:"]

    for pattern in patterns:
        if last_sentence.startswith(pattern):
            result = text.rsplit(pattern, 1)[0].strip()
            return result

    for pattern in patterns:
        if pattern in last_sentence:
            position = text.find(last_sentence)
            if position != -1:
                result = text[:position].strip()
                return result
    return text

def process_jsonl_files(root_dir):
    # List to store all data
    all_data = []
    
    jsonl_files = []
    # Find all jsonl files in the root directory and its subdirectories
    new_files = glob(os.path.join(root_dir, "**", "*.jsonl"), recursive=True)
    jsonl_files.extend(new_files)
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
    
    wrong_question_number = 0
    # Filter data based on "Result: Yes"
    new_data = []
    for item in tqdm(all_data):
        # Extract the refineqa_output content
        refineqa_text = item["refineqa_output"]
        
        # Initialize empty strings for question and answer
        extracted_question = ""
        extracted_answer = ""
        
        if "**Refined Question:**" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("**Refined Question:**", 1)[1].strip("*# \n")
        elif "**Refined Question**:" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("**Refined Question**:", 1)[1].strip("*# \n")
        elif "Refined Question:" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("Refined Question:", 1)[1].strip("*# \n")
        elif "Refined Question" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("Refined Question", 1)[1].strip("*# \n")
        
        if "**Refined Question:**" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("**Refined Question:**", 1)[1].strip("*# \n")
        elif "**Refined Question**:" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("**Refined Question**:", 1)[1].strip("*# \n")
        elif "Refined Question:" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("Refined Question:", 1)[1].strip("*# \n")
        elif "Refined Question" in refineqa_text:
            refineqa_text = refineqa_text.rsplit("Refined Question", 1)[1].strip("*# \n")
        
        if "**Refined Answer:**" in refineqa_text:
            extracted_question, extracted_answer = refineqa_text.rsplit("**Refined Answer:**", 1)
        elif "**Refined Answer**:" in refineqa_text:
            extracted_question, extracted_answer = refineqa_text.rsplit("**Refined Answer**:", 1)
        elif "Refined Answer:" in refineqa_text:
            extracted_question, extracted_answer = refineqa_text.rsplit("Refined Answer:", 1)
        elif "Refined Answer" in refineqa_text:
            extracted_question, extracted_answer = refineqa_text.rsplit("Refined Answer", 1)
        
        extracted_answer = remove_empty_boxes(extracted_answer.strip("*# \n"))
        extracted_answer = remove_last_note(extracted_answer.strip("*# \n"))
        if not extracted_answer or not extracted_question:
            wrong_question_number += 1
            continue
        extracted_question = extracted_question.strip("*# \n")
        extracted_answer = extracted_answer.strip("*# \n")
            
        # # Extract the question
        # if "Question:" in refineqa_text:
        #     question_start = refineqa_text.find("Question:") + len("Question:")
        #     answer_start = refineqa_text.find("Answer:")
        #     if answer_start != -1:  # If "Answer:" is found
        #         extracted_question = refineqa_text[question_start:answer_start].strip()
        #     else:  # If "Answer:" is not found
        #         extracted_question = refineqa_text[question_start:].strip()
        
        # # Extract the answer
        # if "Answer:" in refineqa_text:
        #     answer_start = refineqa_text.find("Answer:") + len("Answer:")
        #     extracted_answer = refineqa_text[answer_start:].strip()
        
        # if "Question:" not in refineqa_text or "Answer:" not in refineqa_text:
        #     extracted_question = item["question"]
        #     extracted_answer = refineqa_text
        
        # Create a new dictionary with the original data plus the extracted question and answer
        suffixes_to_remove = ["refined answer:", "refined answer", "solution process", "refined solution"]
        if "refined answer" in extracted_question.lower() or "solution process" in extracted_question.lower() or "refined solution" in extracted_question.lower():
            suffix_flag = False
            for suffix in suffixes_to_remove:
                if extracted_question.lower().endswith(suffix.lower()):
                    extracted_question = extracted_question[:-len(suffix)].strip()
                    suffix_flag = True
            if not suffix_flag:
                wrong_question_number += 1
                continue
        item["refined_question"] = extracted_question
        item["refined_answer"] = extracted_answer
        
        # Add the new item to new_data
        new_data.append(item)
    print(f"wrong_question: {wrong_question_number}")
    print(f"new_data: {len(new_data)}")
    return new_data

def save_final_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(data)} items to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Directory containing the data to be processed"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        required=True,
        help="Output file path for the output data (JSONL format)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing files from: {args.input_dir}")
    print(f"Output will be saved to: {args.output_file}")
    
    # Process all the files
    new_data = process_jsonl_files(args.input_dir)
    
    # Save the final data
    save_final_data(new_data, args.output_file)
    
    print(f"Processing completed. Results saved to: {args.output_file}")
