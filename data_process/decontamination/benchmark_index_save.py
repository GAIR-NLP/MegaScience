from sentence_transformers import SentenceTransformer
import json
from tqdm import tqdm
import argparse

benchmark_list = {
    "mmlu":{
        "path": "decontamination/benchmarks/mmlu.json",
        "key": "question"
    },
    "mmlu_pro":{
        "path": "decontamination/benchmarks/mmlu_pro.json",
        "key": "question"
    },
    "gpqa_diamond":{
        "path": "decontamination/benchmarks/gpqa_diamond.json",
        "key": "Question"
    },
    "gpqa_main":{
        "path": "decontamination/benchmarks/gpqa_main.json",
        "key": "Question"
    },
    "super_gpqa":{
        "path": "decontamination/benchmarks/super_gpqa.json",
        "key": "question"
    },
    "gsm8k":{
        "path": "decontamination/benchmarks/gsm8k.jsonl",
        "key": "question"
    },
    "math":{
        "path": "decontamination/benchmarks/math.jsonl",
        "key": "problem"
    },
    "math-500":{
        "path": "decontamination/benchmarks/math500.jsonl",
        "key": "problem"
    },
    "ChemBench-multi-choise":{
        "path": "decontamination/benchmarks/chembench_multi_choise.json",
        "key": None
    },
    "ChemBench-str-match":{
        "path": "decontamination/benchmarks/chembench_str_match.json",
        "key": None
    },
    "cs-bench-assertion-questions":{
        "path": "decontamination/benchmarks/cs_bench_assertion.json",
        "key": "Question"
    },
    "cs-bench-multiple-choice-questions":{
        "path": "decontamination/benchmarks/cs_bench_multiple_choice.json",
        "key": "Question"
    },
    "medqa-us":{
        "path": "decontamination/benchmarks/medqa_US.jsonl",
        "key": "question"
    },
    "medmcqa":{
        "path": "decontamination/benchmarks/medmcqa.json",
        "key": "question"
    },
    "PubMedQA":{
        "path": "decontamination/benchmarks/pubmedqa.json",
        "key": "question"
    },
    "NEWTON-confident-questions":{
        "path": "decontamination/benchmarks/newton_confident_questions.json",
        "key": "question"
    },
    "NEWTON-explicit-questions-bool":{
        "path": "decontamination/benchmarks/newton_explicit_questions_bool.json",
        "key": "question"
    },
    "NEWTON-explicit-questions-mc":{
        "path": "decontamination/benchmarks/newton_explicit_questions_mc.json",
        "key": "question"
    },
    "NEWTON-implicit-questions":{
        "path": "decontamination/benchmarks/newton_implicit_questions.json",
        "key": "question"
    },
    "piqa":{
        "path": "decontamination/benchmarks/piqa_valid.json",
        "key": "goal"
    },
    "scibench-physics":{
        "path": "decontamination/benchmarks/scibench_physics.json",
        "key": "problem_text"
    },
    "scibench-chemistry":{
        "path": "decontamination/benchmarks/scibench_chemistry.json",
        "key": "problem_text"
    },
    "scibench-math":{
        "path": "decontamination/benchmarks/scibench_math.json",
        "key": "problem_text"
    }
}

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
    model = SentenceTransformer(args.model)
    batch_size = args.batch_size
    all_data = []
    idx = 0
    for name, info in tqdm(benchmark_list.items()):
        data = read_data(info["path"])
        for d in data:
            if info["key"] == None:
                question = d["examples"][0]["input"]
            else:
                question = d[info["key"]]
            all_data.append({
                "idx": idx,
                "question": question,
                "benchmark": name
            })
            idx += 1
    print(f"Total benchmark data: {len(all_data)}")
    
    total_batches = (len(all_data) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(all_data), batch_size), 
                  desc="Processing batches", 
                  total=total_batches):
        # Get current batch data
        batch_data = all_data[i:i + batch_size]
        
        # Extract all questions from current batch
        batch_questions = [d['question'] for d in batch_data]
        
        # Batch encoding
        batch_embeddings = model.encode(batch_questions, normalize_embeddings=True)
        
        # Assign embeddings back to corresponding data items
        for j, embedding in enumerate(batch_embeddings):
            batch_data[j]["embedding"] = embedding.tolist()
    
    with open(args.output_path, 'w', encoding="utf-8") as fp:
        for d in all_data:
            fp.write(json.dumps(d) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for benchmark datasets")
    parser.add_argument("--model", type=str, required=True, help="Path to the embedding model")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--output_path", type=str, required=True, help="Output file path for embeddings")
    
    args = parser.parse_args()
    main(args)