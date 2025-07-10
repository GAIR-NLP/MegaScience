import torch
import numpy as np
from typing import List, Dict, Any
import json
from tqdm import tqdm
import gc

class GPUVectorSearch:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
    
    def batch_similarity_search(self, data1: List[Dict], data2: List[Dict], 
                               top_k: int = 5, batch_size: int = 1024):
        """
        批量计算相似度，避免内存溢出
        """
        # 提取所有data2的embeddings并转换为tensor
        embeddings2 = torch.tensor([item['embedding'] for item in data2], 
                                  dtype=torch.float32, device=self.device)
        
        print(f"Data2 embeddings shape: {embeddings2.shape}")
        
        final_data1 = []
        
        # 分批处理data1
        for i in tqdm(range(0, len(data1), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(data1))
            batch_data1 = data1[i:batch_end]
            
            # 提取当前batch的embeddings
            batch_embeddings1 = torch.tensor([item['embedding'] for item in batch_data1],
                                           dtype=torch.float32, device=self.device)
            
            # 计算相似度矩阵 (batch_size, len(data2))
            similarity_matrix = torch.mm(batch_embeddings1, embeddings2.t())
            
            # 获取top-k相似的索引和分数
            top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
            
            # 将结果存储回data1
            for j, data_item in enumerate(batch_data1):
                retrieved_data = []
                for k in range(top_k):
                    idx = top_k_indices[j, k].item()
                    score = top_k_scores[j, k].item()
                    
                    # 复制data2中的数据并添加相似度分数
                    retrieved_item = data2[idx].copy()
                    del retrieved_item['embedding']
                    retrieved_item['similarity_score'] = score
                    retrieved_data.append(retrieved_item)
                
                data_item['retrieved_benchmark'] = retrieved_data
                final_data1.append(data_item)
            # 清理GPU内存
            del batch_embeddings1, similarity_matrix, top_k_scores, top_k_indices
            torch.cuda.empty_cache()
        return final_data1
    
    # def optimized_similarity_search(self, data1: List[Dict], data2: List[Dict], 
    #                                top_k: int = 5, batch_size: int = 500):
    #     """
    #     优化版本：使用更小的batch size和内存管理
    #     """
    #     # 预处理data2的embeddings
    #     embeddings2 = torch.tensor([item['embedding'] for item in data2], 
    #                               dtype=torch.float16, device=self.device)  # 使用float16节省内存
        
    #     # 归一化embeddings以使用余弦相似度
    #     embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
    #     print(f"Data2 embeddings shape: {embeddings2.shape}")
        
    #     for i in tqdm(range(0, len(data1), batch_size), desc="Processing queries"):
    #         batch_end = min(i + batch_size, len(data1))
    #         batch_data1 = data1[i:batch_end]
            
    #         # 当前batch的embeddings
    #         batch_embeddings1 = torch.tensor([item['embedding'] for item in batch_data1],
    #                                        dtype=torch.float16, device=self.device)
    #         batch_embeddings1 = torch.nn.functional.normalize(batch_embeddings1, p=2, dim=1)
            
    #         # 计算余弦相似度
    #         similarity_matrix = torch.mm(batch_embeddings1, embeddings2.t())
            
    #         # 获取top-k
    #         top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
            
    #         # 存储结果
    #         for j, data_item in enumerate(batch_data1):
    #             retrieved_data = []
    #             for k in range(top_k):
    #                 idx = top_k_indices[j, k].item()
    #                 score = top_k_scores[j, k].item()
                    
    #                 retrieved_item = data2[idx].copy()
    #                 retrieved_item['similarity_score'] = float(score)
    #                 retrieved_data.append(retrieved_item)
                
    #             data_item['retrieved_benchmark'] = retrieved_data
            
    #         # 清理内存
    #         del batch_embeddings1, similarity_matrix, top_k_scores, top_k_indices
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()

# def multi_gpu_search(data1: List[Dict], data2: List[Dict], top_k: int = 5):
#     """
#     多GPU并行处理
#     """
#     if not torch.cuda.is_available():
#         print("CUDA不可用，使用单GPU方案")
#         searcher = GPUVectorSearch()
#         searcher.batch_similarity_search(data1, data2, top_k)
#         return
    
#     num_gpus = torch.cuda.device_count()
#     print(f"检测到 {num_gpus} 个GPU")
    
#     if num_gpus == 1:
#         searcher = GPUVectorSearch()
#         searcher.optimized_similarity_search(data1, data2, top_k)
#         return
    
#     # 将data1分割到多个GPU
#     chunk_size = len(data1) // num_gpus
    
#     # 为每个GPU创建进程
#     import multiprocessing as mp
#     from concurrent.futures import ProcessPoolExecutor
    
#     def process_chunk(gpu_id, data1_chunk, data2, top_k):
#         torch.cuda.set_device(gpu_id)
#         searcher = GPUVectorSearch(device=f'cuda:{gpu_id}')
#         searcher.optimized_similarity_search(data1_chunk, data2, top_k)
#         return data1_chunk
    
#     # 分割数据
#     chunks = []
#     for i in range(num_gpus):
#         start_idx = i * chunk_size
#         end_idx = (i + 1) * chunk_size if i < num_gpus - 1 else len(data1)
#         chunks.append(data1[start_idx:end_idx])
    
#     # 并行处理
#     with ProcessPoolExecutor(max_workers=num_gpus) as executor:
#         futures = [executor.submit(process_chunk, i, chunks[i], data2, top_k) 
#                   for i in range(num_gpus)]
        
#         # 等待所有任务完成
#         for i, future in enumerate(futures):
#             processed_chunk = future.result()
#             # 结果已经存储在原始data1中
#             print(f"GPU {i} 处理完成")

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

# 使用示例
def example_usage():
    # 模拟数据
    data_path = "embedding/textbook_reasoning_distill_cot_filtering_qa_72w_embedding.jsonl"
    benchmark_path = "embedding/benchmark_embedding.jsonl"
    output_data_path = "search_results/textbook_reasoning_distill_cot_filtering_qa_72w_embedding_with_top5_similarity.jsonl"
    
    query_data = read_data(data_path)
    benchmark_data = read_data(benchmark_path)
    
    
    # 单GPU方案
    print("使用单GPU方案...")
    searcher = GPUVectorSearch()
    final_data = searcher.batch_similarity_search(query_data, benchmark_data, top_k=5, batch_size=1024)
    
    # # 多GPU方案
    # print("使用多GPU方案...")
    # multi_gpu_search(data1, data2, top_k=5)
    
    # 检查结果
    print("第一个查询的检索结果:")
    print(final_data[0]["refined_question"] + '\n')
    for i, item in enumerate(final_data[0]['retrieved_benchmark']):
        print(item['question'])
        print(f"  Top {i+1}: ID={item['idx']}, Benchmark={item['benchmark']} 相似度={item['similarity_score']:.4f}")
        print()
    
    with open(output_data_path, 'w', encoding="utf-8") as fp:
        for d in final_data:
            fp.write(json.dumps(d, ensure_ascii=False) + '\n')
    

# # 内存友好的大规模处理
# class LargeScaleVectorSearch:
#     def __init__(self, device='cuda'):
#         self.device = device
    
#     def process_large_dataset(self, data1_path: str, data2_path: str, 
#                              output_path: str, top_k: int = 5):
#         """
#         处理大规模数据集，支持文件流式读取
#         """
#         # 首先加载data2到GPU内存
#         print("加载data2到GPU...")
#         with open(data2_path, 'r') as f:
#             data2 = json.load(f)
        
#         embeddings2 = torch.tensor([item['embedding'] for item in data2], 
#                                   dtype=torch.float16, device=self.device)
#         embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
        
#         # 流式处理data1
#         print("开始流式处理data1...")
#         batch_size = 500
#         batch_data = []
        
#         with open(data1_path, 'r') as f_in, open(output_path, 'w') as f_out:
#             data1 = json.load(f_in)
            
#             for i, item in enumerate(tqdm(data1, desc="Processing data1")):
#                 batch_data.append(item)
                
#                 if len(batch_data) == batch_size or i == len(data1) - 1:
#                     # 处理当前batch
#                     self._process_batch(batch_data, data2, embeddings2, top_k)
                    
#                     # 写入结果
#                     for processed_item in batch_data:
#                         f_out.write(json.dumps(processed_item) + '\n')
                    
#                     batch_data = []
                    
#                     # 清理内存
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()
    
#     def _process_batch(self, batch_data: List[Dict], data2: List[Dict], 
#                       embeddings2: torch.Tensor, top_k: int):
#         """处理一个batch的数据"""
#         batch_embeddings = torch.tensor([item['embedding'] for item in batch_data],
#                                        dtype=torch.float16, device=self.device)
#         batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        
#         similarity_matrix = torch.mm(batch_embeddings, embeddings2.t())
#         top_k_scores, top_k_indices = torch.topk(similarity_matrix, k=top_k, dim=1)
        
#         for i, data_item in enumerate(batch_data):
#             retrieved_data = []
#             for j in range(top_k):
#                 idx = top_k_indices[i, j].item()
#                 score = top_k_scores[i, j].item()
                
#                 retrieved_item = data2[idx].copy()
#                 del retrieved_item['embedding']
#                 retrieved_item['similarity_score'] = float(score)
#                 retrieved_data.append(retrieved_item)
            
#             data_item['retrieved_benchmark'] = retrieved_data

if __name__ == "__main__":
    example_usage()