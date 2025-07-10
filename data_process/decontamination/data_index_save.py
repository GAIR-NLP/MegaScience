import argparse
from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

def read_data(path):
    """Read data from json or jsonl file"""
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for questions using SentenceTransformer")
    parser.add_argument("--model", type=str, required=True, help="Path to the SentenceTransformer model")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output data file")
    
    args = parser.parse_args()
    
    model = SentenceTransformer(args.model)
    batch_size = args.batch_size
    input_path = args.input_path
    output_data_path = args.output_path
   
    all_data = read_data(input_path)
    for i, d in enumerate(all_data):
        d["idx"] = i
   
    print(f"Total data: {len(all_data)}")
   
    total_batches = (len(all_data) + batch_size - 1) // batch_size
   
    for i in tqdm(range(0, len(all_data), batch_size),
                  desc="Processing batches",
                  total=total_batches):
        # Get current batch data
        batch_data = all_data[i:i + batch_size]
       
        # Extract all questions from current batch
        batch_questions = [d['refined_question'] for d in batch_data]
       
        # Batch encoding
        batch_embeddings = model.encode(batch_questions, normalize_embeddings=True)
       
        # Assign embeddings back to corresponding data items
        for j, embedding in enumerate(batch_embeddings):
            batch_data[j]["embedding"] = embedding.tolist()
   
    with open(output_data_path, 'w', encoding="utf-8") as fp:
        for d in all_data:
            fp.write(json.dumps(d) + '\n')

if __name__ == "__main__":
    main()