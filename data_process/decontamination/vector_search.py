import torch
import numpy as np
from typing import List, Dict, Any
import json
from tqdm import tqdm
import gc
import argparse

class GPUVectorSearch:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
    
    def batch_similarity_search(self, data1: List[Dict], data2: List[Dict], 
                               top_k: int = 5, batch_size: int = 1024):
        """
        Batch similarity calculation to avoid memory overflow
        """
        # Extract all data2 embeddings and convert to tensor
        embeddings2 = torch.tensor([item['embedding'] for item in data2], 
                                  dtype=torch.float32, device=self.device)
        
        print(f"Data2 embeddings shape: {embeddings2.shape}")
        
        final_data1 = []
        
        # Process data1 in batches
        for i in tqdm(range(0, len(data1), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(data1))
            batch_data1 = data1[i:batch_end]
            
            # Extract embeddings for current batch
            batch_embeddings1 = torch.tensor([item['embedding'] for item in batch_data1],
                                           dtype=torch.float32, device=self.device)
            
            # Calculate similarity matrix (batch_size, len(data2))
            similarity_matrix = torch.mm(batch_embeddings1, embeddings2.t())
            
            # Get top-k similar indices and scores
            top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
            
            # Store results back to data1
            for j, data_item in enumerate(batch_data1):
                retrieved_data = []
                for k in range(top_k):
                    idx = top_k_indices[j, k].item()
                    score = top_k_scores[j, k].item()
                    
                    # Copy data from data2 and add similarity score
                    retrieved_item = data2[idx].copy()
                    del retrieved_item['embedding']
                    retrieved_item['similarity_score'] = score
                    retrieved_data.append(retrieved_item)
                del data_item['embedding']
                
                data_item['retrieved_benchmark'] = retrieved_data
                final_data1.append(data_item)
            # Clean GPU memory
            del batch_embeddings1, similarity_matrix, top_k_scores, top_k_indices
            torch.cuda.empty_cache()
        return final_data1

def read_data(path):
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

def main(args):
    query_data = read_data(args.data_embedding_path)
    benchmark_data = read_data(args.benchmark_embedding_path)
    
    
    print("Using single GPU solution...")
    searcher = GPUVectorSearch()
    final_data = searcher.batch_similarity_search(query_data, benchmark_data, top_k=5, batch_size=1024)

    
    # Check results
    print("Retrieval results for the first query:")
    print(final_data[0]["refined_question"] + '\n')
    for i, item in enumerate(final_data[0]['retrieved_benchmark']):
        print(item['question'])
        print(f"  Top {i+1}: ID={item['idx']}, Benchmark={item['benchmark']} Similarity={item['similarity_score']:.4f}")
        print()
    
    with open(args.output_path, 'w', encoding="utf-8") as fp:
        for d in final_data:
            fp.write(json.dumps(d, ensure_ascii=False) + '\n')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Vector Search")
    parser.add_argument("--data_embedding_path", type=str, required=True,
                        help="Path to the data embedding file")
    parser.add_argument("--benchmark_embedding_path", type=str, required=True,
                        help="Path to the benchmark embedding file")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save the output results")
    
    args = parser.parse_args()
    main(args)