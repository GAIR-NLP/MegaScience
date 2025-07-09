import psutil
import os
import random
import json
import time
from typing import List, Dict, Any
import random
random.seed(42)
from math import ceil
import gzip
import pandas as pd
import json
from datetime import datetime


def json_serial(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()  # 转换为 ISO 8601 格式
    raise TypeError("Type not serializable")


def save_dataset_in_chunks(ds, output_dir, batch_size):
    """
    Save a dataset into multiple .jsonl.gz files based on the given batch size.

    Args:
        ds: The dataset to save (should be iterable).
        output_dir (str): Directory to save the output files.
        batch_size (int): Number of records per chunk file.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    total_records = len(ds)
    num_chunks = ceil(total_records / batch_size)


    df = ds.to_pandas()
    data = df.to_dict(orient="records")
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * batch_size
        end_idx = min((chunk_idx + 1) * batch_size, total_records)
        chunk_data = data[start_idx:end_idx]  
        output_path = os.path.join(output_dir, f"chunk_{chunk_idx + 1}.jsonl")

        with open(output_path, "wt", encoding="utf-8") as gz_file:
            for record in chunk_data:
                if "text" in record and "text" in record['metadata'] and record['text'] == record['metadata']['text']:
                    del record['metadata']['text']
                try:
                    gz_file.write(json.dumps(record, ensure_ascii=False, default=json_serial) + "\n")
                except Exception as e:
                    print(e)
                    print(record)
                    continue
            
        print(f"Chunk {chunk_idx + 1} saved to {output_path}")