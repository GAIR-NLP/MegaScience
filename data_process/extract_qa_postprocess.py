import json
import re
import argparse
import os
from tqdm import tqdm

def extract_qa(text):
    """
    Extract question-answer pairs from text using regex pattern.
    
    Args:
        text (str): Input text containing Q&A pairs
        
    Returns:
        list: List of dictionaries with 'question' and 'answer' keys
    """
    pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\s*Question:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    qa_pairs = [
        {
            "question": q.strip().strip('*').strip(), 
            "answer": a.strip().strip('*').strip()
        } 
        for q, a in matches
    ]
    return qa_pairs

def save_document(subject, text, document_save_path):
    """
    Save document to JSONL file.
    
    Args:
        subject (str): Subject of the document
        text (str): Document text content
        document_save_path (str): Path to document output file
    """
    document_data = {
        "subject": subject,
        "text": text
    }
    
    with open(document_save_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(document_data, ensure_ascii=False) + '\n')

def process_jsonl_file(file_path, output_path, document_save_path, document_idx_counter):
    """
    Process a single JSONL file and extract QA pairs.
    
    Args:
        file_path (str): Path to the JSONL file
        output_path (str): Path to output JSONL file for QA pairs
        document_save_path (str): Path to output JSONL file for documents
        document_idx_counter (dict): Counter for document index (passed by reference)
        
    Returns:
        tuple: (Number of QA pairs extracted, Number of documents saved)
    """
    qa_count = 0
    doc_count = 0
    
    print(f"Processing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                continue
            
            # Get subject and text from each item
            subject = item.get('subject')
            text = item.get('text')
            extract_output = item.get("extract_qa")
            
            if not text:
                continue
            
            # Extract QA pairs from text
            qa_pairs = extract_qa(extract_output)
            
            # Only save document and QA pairs if QA pairs were found
            if qa_pairs:
                # Save document with current document_idx
                save_document(subject, text, document_save_path)
                doc_count += 1
                
                # Save each QA pair with subject and document_idx
                for qa in qa_pairs:
                    qa_data = {
                        "question": qa["question"],
                        "answer": qa["answer"],
                        "subject": subject,
                        "document_idx": document_idx_counter['count']
                    }
                    
                    with open(output_path, 'a', encoding='utf-8') as out_f:
                        out_f.write(json.dumps(qa_data, ensure_ascii=False) + '\n')
                    
                    qa_count += 1
                
                # Increment document index after saving document and QA pairs
                document_idx_counter['count'] += 1
    
    print(f"Extracted {qa_count} QA pairs and {doc_count} documents from {file_path}")
    return qa_count, doc_count

def find_all_jsonl_files(directory):
    """
    Find all JSONL files in directory and subdirectories.
    
    Args:
        directory (str): Root directory to search
        
    Returns:
        list: List of paths to JSONL files
    """
    jsonl_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    
    return jsonl_files

def process_directory(input_directory, output_file_path, document_save_path):
    """
    Process all JSONL files in directory and subdirectories.
    
    Args:
        input_directory (str): Path to input directory
        output_file_path (str): Path to output JSONL file for QA pairs
        document_save_path (str): Path to output JSONL file for documents
    """
    # Clear output files if they exist
    open(output_file_path, 'w').close()
    open(document_save_path, 'w').close()
    
    # Find all JSONL files
    jsonl_files = find_all_jsonl_files(input_directory)
    
    if not jsonl_files:
        print(f"No JSONL files found in directory: {input_directory}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Initialize counters
    total_qa_count = 0
    total_doc_count = 0
    document_idx_counter = {'count': 0}  # Use dict to pass by reference
    
    # Process each file individually
    for i, file_path in enumerate(jsonl_files, 1):
        print(f"\n[{i}/{len(jsonl_files)}] Processing: {file_path}")
        
        try:
            qa_count, doc_count = process_jsonl_file(
                file_path, output_file_path, document_save_path, document_idx_counter
            )
            total_qa_count += qa_count
            total_doc_count += doc_count
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue
    
    print(f"\n" + "="*60)
    print(f"Processing completed!")
    print(f"Total files processed: {len(jsonl_files)}")
    print(f"Total QA pairs extracted: {total_qa_count}")
    print(f"Total documents saved: {total_doc_count}")
    print(f"QA pairs saved to: {output_file_path}")
    print(f"Documents saved to: {document_save_path}")
    print(f"Document index range: 0 to {document_idx_counter['count'] - 1}")

def main():
    """
    Main function to parse arguments and process data.
    """
    parser = argparse.ArgumentParser(description="Extract QA pairs from JSONL files in directory")
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to input directory containing JSONL files"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to output JSONL file for QA pairs"
    )
    parser.add_argument(
        "--document_save_path", 
        type=str, 
        required=True, 
        help="Path to output JSONL file for documents"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input):
        print(f"Error: Input path '{args.input}' is not a valid directory")
        return
    
    # Create output directories if they don't exist
    for output_path in [args.output, args.document_save_path]:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    process_directory(args.input, args.output, args.document_save_path)

if __name__ == "__main__":
    main()