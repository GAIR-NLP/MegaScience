import tiktoken
import json
from tqdm import tqdm

def split_text_by_tokens(text, max_tokens=4096, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        
        # 尝试在 max_tokens 附近找到最近的换行符对应的 token 位置
        while end > start and encoding.decode(tokens[start:end])[-1] != '\n':
            end -= 1
        
        # 如果没有找到合适的换行符，就按 max_tokens 硬切
        if end == start:
            end = min(start + max_tokens, len(tokens))
        
        chunks.append(encoding.decode(tokens[start:end]))
        start = end
    
    return chunks

# 示例用法
if __name__ == "__main__":
    file_name = "/inspire/hdd/global_user/liupengfei-24025/rzfan/MegaScience/data_process/data/documents/examples.jsonl"
    out_path = f"/inspire/hdd/global_user/liupengfei-24025/rzfan/MegaScience/data_process/data/documents/example_chunks.jsonl"
    
    number = 0
    with open(file_name, "r", encoding="utf-8") as file:
        for line in tqdm(file):
            data = json.loads(line)
            document = data["text"]
            subject = data["subject"]
            split_docs = split_text_by_tokens(document)
            out_data = []
            for doc in split_docs:
                out_data.append({
                    "subject": subject,
                    "text": doc
                })
            number += len(out_data)
            with open(out_path, "a+", encoding="utf-8") as w:
                for d in out_data:
                    w.write(json.dumps(d, ensure_ascii=False) + "\n")
            if number > 1000:
                break
            print(f"files: {len(out_data)}")