from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm

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


def main():
    model = SentenceTransformer('/inspire/hdd/global_user/liupengfei-24025/rzfan/models/bge-large-en-v1.5')
    batch_size = 1024
    
    
    input_path = "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/filtering_qa/final_output/v3_with_document_final_76w.jsonl"
    output_data_path = "embedding/textbook_reasoning_distill_cot_filtering_qa_72w_embedding.jsonl"
    
    all_data = read_data(input_path)
    for i, d in enumerate(all_data):
        d["idx"] = i
    
    print(f"Total data: {len(all_data)}")
    
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(all_data), batch_size), 
                  desc="Processing batches", 
                  total=total_batches):
        # 获取当前批次的数据
        batch_data = all_data[i:i + batch_size]
        
        # 提取当前批次的所有问题
        batch_questions = [d['refined_question'] for d in batch_data]
        
        # 批量编码
        batch_embeddings = model.encode(batch_questions, normalize_embeddings=True)
        
        # 将embedding分配回对应的数据项
        for j, embedding in enumerate(batch_embeddings):
            batch_data[j]["embedding"] = embedding.tolist()

    # for d in tqdm(all_data, desc="embedding"):
    #     embedding = model.encode(d['question'], normalize_embeddings=True)
    #     d["embedding"] = embedding.tolist()
    
    with open(output_data_path, 'w', encoding="utf-8") as fp:
        for d in all_data:
            fp.write(json.dumps(d) + '\n')
    
    # sentences_1 = ["样例数据-1", "样例数据-2"]
    # sentences_2 = ["样例数据-3", "样例数据-4"]
    
    # embeddings_1 = model.encode(sentences_1, normalize_embeddings=True)
    # embeddings_2 = model.encode(sentences_2, normalize_embeddings=True)
    # similarity = embeddings_1 @ embeddings_2.T
    # print(similarity)

if __name__ == "__main__":
    main()