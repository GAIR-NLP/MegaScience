import argparse
import json
import os
from typing import List, Dict, Optional
import torch
# import torch.distributed as dist
from vllm import LLM, SamplingParams
from config import GentaskConfig
from transformers import AutoTokenizer
from tqdm import tqdm

SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."

def split_into_batches(arguments, batch_size):
    """Split arguments into batches according to batch_size."""
    if batch_size is None or batch_size <= 0:
        return [arguments]
    else:
        return [
            arguments[i : i + batch_size] for i in range(0, len(arguments), batch_size)
        ]

def run_inference(args):
    print(args)
    config = GentaskConfig.from_yaml(args.config_path)
    config.__post_init__()
    
    print(f"Loading model: {config.model_path}")
    print(f"Tensor parallel size: {config.NODE_GPUS}")
    print(f"Pipeline parallel size: {config.N_NODES}")
    
    # Create LLM instance with distributed settings
    llm = LLM(
        model=config.model_path,
        tensor_parallel_size=config.tp_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,  # Adjust if needed
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p = config.top_p
    )
    
    print(f"Prepare data: {config.data_path}")
    arguments = []
    # prepare data
    if config.data_format == "jsonl":
        idx = 0
        with open(config.data_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    if idx % config.TOTAL_SPLIT == config.GLOBAL_RANK:
                        json_obj = json.loads(line)
                        arguments.append(json_obj)
                    idx += 1
    else:
        raise ValueError(f"Cannot support data format: {config.data_format}")
 
    dir_path = config.get_save_dir_path()
    base_name = config.save_name
    os.makedirs(dir_path, exist_ok=True)
    batches = split_into_batches(arguments, config.save_interval)
    
    score_prompt = open(config.prompt_path, "r").read()
    
    # Run batch inference
    for batch_idx, batch in enumerate(tqdm(batches, desc="Processing batches")):
        all_prompts = []      # Store all messages that need to be sent to the engine (including each chunk after splitting)
        mapping = []          # Used to record the sample index and chunk order in the original batch corresponding to each prompt
        # Iterate through each sample in the batch
        for sample_idx, sample in enumerate(tqdm(batch, total=len(batch), unit="sample", desc="Tokenizing samples")):
            question = sample["refined_question"]
            answer = sample["refined_answer"]
            total_msg = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": score_prompt.replace("<PROBLEM>", question).replace("<ANSWER>", answer)},
            ]
            # If the number of samples is small, print partial content for debugging
            if sample_idx <= 1:
                print("Debug - Prompt message:")
                # print(total_msg)
                print(tokenizer.apply_chat_template(total_msg, tokenize=False))
            all_prompts.append(tokenizer.apply_chat_template(total_msg, tokenize=True, add_generation_prompt=True))
            # Record which document this prompt comes from and which chunk it is in the document
            mapping.append(sample_idx)
        
        # Call the generation engine to process all prompts
        outputs = llm.generate(
            sampling_params=sampling_params, prompt_token_ids=all_prompts
        )
        # Remove leading and trailing whitespace from outputs
        outputs = [item.outputs[0].text.strip() for item in outputs]
        
        # Debug print partial generation results
        for idx_out, item in enumerate(outputs):
            if idx_out > 2:
                break
            print(f"Output {idx_out}:")
            print(item)
            print("-" * 100)
        
        
        # Generate a new results list, where each element corresponds to an original document and contains generation results from all chunks
        new_results = []
        for idx, sample in enumerate(batch):
            sample["judge_cot_output"] = outputs[idx]
            new_results.append(sample)
        
        # Save the results of this batch as a jsonl file
        out_path = os.path.join(
            dir_path,
            f"{base_name}_{batch_idx + 1}_{(len(arguments) - 1) // config.save_interval + 1}.jsonl",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for res in new_results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        # write into logging file
        os.makedirs(os.path.join(config.logging_path, config.save_name), exist_ok=True)
        with open(
            os.path.join(config.logging_path, config.save_name, "finished.log"), "a"
        ) as f:
            f.write(f"Rank [{config.GLOBAL_RANK}] Finished\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    run_inference(args)