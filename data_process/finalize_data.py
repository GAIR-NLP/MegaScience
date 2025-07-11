import json
import argparse
import os
from typing import Dict, Any

def transform_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform a single data entry to the desired format.
    
    Args:
        data: Original data entry from JSONL file
        
    Returns:
        Transformed data entry
    """
    # Use distill_cot if available, otherwise use refined_answer
    answer = data.get('distill_cot', data.get('refined_answer'))
    
    transformed = {
        "question": data.get('refined_question'),
        "answer": answer,
        "subject": data.get('subject'),
        "reference_answer": data.get('reference_answer')
    }
    
    return transformed

def process_jsonl_file(input_path: str, output_path: str) -> None:
    """
    Process JSONL file and transform data to desired format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as input_file, \
             open(output_path, 'w', encoding='utf-8') as output_file:
            
            processed_count = 0
            
            for line_num, line in enumerate(input_file, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                try:
                    # Parse JSON from line
                    data = json.loads(line)
                    
                    # Transform data
                    transformed_data = transform_data(data)
                    
                    # Write transformed data to output file
                    json.dump(transformed_data, output_file, ensure_ascii=False)
                    output_file.write('\n')
                    
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
            
            print(f"Successfully processed {processed_count} entries.")
            print(f"Output saved to: {output_path}")
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied when accessing files.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")

def main():
    """
    Main function to parse arguments and process the file.
    """
    parser = argparse.ArgumentParser(
        description="Transform JSONL data to specified format"
    )
    
    parser.add_argument(
        '--input_path',
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    
    parser.add_argument(
        '--output_path', 
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.input_path.endswith('.jsonl'):
        print("Warning: Input file doesn't have .jsonl extension")
    
    if not args.output_path.endswith('.jsonl'):
        print("Warning: Output file doesn't have .jsonl extension")
    
    # Create output directories if they don't exist
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # Process the file
    process_jsonl_file(args.input_path, args.output_path)

if __name__ == "__main__":
    main()