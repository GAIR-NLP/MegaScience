# MegaScience
MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning


## 🔥 News

## 💎 Resources

## 🚀 Introduction

## ⚙️ Data Process Pipeline

### Step 0. Install Environment

```
cd data_process
conda create --name megascience python=3.10
conda activate megascience
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -U pynvml
```

Launch the Python interpreter and download the necessary NLTK tokenizer data:

```python
python -c "import nltk; nltk.download('punkt_tab')"
```

Alternatively, you can run this interactively:
```python
python
>>> import nltk
>>> nltk.download('punkt_tab')
>>> exit()
```

### Step 1. PDF Digitalization

We utilize [olmOCR](https://github.com/allenai/olmocr) to convert PDF documents into text format. Please follow the [olmOCR documentation](https://github.com/allenai/olmocr) to process your PDFs, then segment the resulting documents into 4096-token chunks and store them in the `text` field.

### Step 2: QA Extraction

Configure the QA extraction process by modifying `data_process/vllm_inference/task_config/extract_qa.yaml`. Set the following parameters:
- Model path
- Number of GPUs
- Input data path
- Prompt path
- Save path

Execute the extraction script:
```bash
bash script/extract_qa.sh
```

After extraction completes, run the post-processing step to finalize the QA pairs:
```bash
python vllm_inference/extract_qa_postprocess.py 
    --input data/extract_qa/original_qa \
    --output data/extract_qa/final_qa/extract_qa.jsonl \
    --document_save_path data/extract_qa/final_qa/documents.jsonl
```
### Step 3. Question Deduplication

We employ [text-dedup](https://github.com/ChenghaoMou/text-dedup) to remove duplicate questions from the extracted dataset. This tool provides efficient text deduplication capabilities using various algorithms including MinHash, SimHash, and exact hash matching.

#### Environment Setup

First, create and activate a dedicated conda environment:
```bash
conda create --name textdedup python=3.10
conda activate textdedup
pip install text-dedup
```

#### Configuration

Configure the deduplication parameters in the script:
```bash
script/question_dedup.sh
```

#### Execution

Run the deduplication script:
```bash
bash script/question_dedup.sh
```

#### Output

After processing, merge all deduplicated chunks into a single JSONL file for downstream use.

### Step 4. QA Refinement

Configure the QA refinement process by modifying `data_process/vllm_inference/task_config/refine_qa.yaml`.

Execute the extraction script:

```
bash script/refine_qa.sh
```

After refinement completes, run the post-processing step to finalize the refined QA pairs:

```
python vllm_inference/refine_qa_postprocess.py \
    --input_dir data/refine_qa/original_data \
    --output_file data/refine_qa/final_data/refined_qa.jsonl
```

### Step 5. CoT Augmentation

#### Find no CoT data (judge CoT)

Configure the CoT judgement process by modifying `data_process/vllm_inference/task_config/judge_cot.yaml`.

Execute the extraction script:

```
bash script/judge_cot.sh
```

After refinement completes, run the post-processing step to finalize the QA pairs with CoT and no CoT:

```
python vllm_inference/judge_cot_postprocess.py \
    --input_file data/augment_cot/judge_cot/original_data \
    --output_cot data/augment_cot/judge_cot/final_data/cot_data.jsonl \
    --output_no_cot data/augment_cot/judge_cot/final_data/no_cot_data.jsonl
```

#### Distill CoT for no CoT data

Configure the CoT distillation process by modifying `data_process/vllm_inference/task_config/distill_cot.yaml`.

Execute the extraction script:

```
bash script/distill_cot.sh
```

After distillation completes, run the post-processing step to finalize the QA pairs:

```
python vllm_inference/distill_cot_postprocess.py \
    --input_no_cot_dir data/augment_cot/distill_cot/original_data \
    --input_cot_file data/augment_cot/judge_cot/final_data/cot_data.jsonl \
    --output data/augment_cot/distill_cot/final_data/refined_augmented_cot_qa.jsonl
```

### Step 6. QA Filtering

Configure the QA filtering process by modifying `data_process/vllm_inference/task_config/filter_qa.yaml`.

Execute the extraction script:

```
bash script/filter_qa.sh
```

After filtering completes, run the post-processing step to finalize the QA pairs:

```
python vllm_inference/filter_qa_postprocess.py \
    --input_dir data/filter_qa/original_data \
    --output_file data/filter_qa/final_data/refined_augmented_cot_filtering_qa.jsonl
```

### Step 7. LLM-based Question Decontamination

To ensure data quality and prevent benchmark contamination, we implement a comprehensive decontamination pipeline using embedding-based similarity search followed by LLM-based semantic judgment.

#### Generate Benchmark Embeddings

First, generate embeddings for benchmark questions to create a searchable index:

```
python decontamination/benchmark_index_save.py \
    --model BAAI/bge-large-en-v1.5 \
    --batch_size 1024 \
    --output_path decontamination/index/benchmark_embedding.jsonl
```

#### Generate Dataset Embeddings

Generate embeddings for your dataset questions:

```
python decontamination/data_index_save.py \
    --model BAAI/bge-large-en-v1.5 \
    --batch_size 1024 \
    --input_path data/filter_qa/final_data/refined_augmented_cot_filtering_qa.jsonl \
    --output_path decontamination/index/data_embedding.jsonl
```

#### Similarity Search

Perform vector similarity search to identify potentially similar questions between your dataset and benchmark:

```
python decontamination/vector_search.py \
    --data_embedding_path decontamination/index/data_embedding.jsonl \
    --benchmark_embedding_path decontamination/index/benchmark_embedding.jsonl \
    --output_path decontamination/results/refined_augmented_cot_filtering_qa_with_top5_similarity_benchmark.jsonl
```

#### LLM-based Similarity Judge

Configure the decontamination process by modifying the task configuration:g `data_process/vllm_inference/task_config/llm_based_decontamination.yaml`.

Execute the LLM-based similarity judgment:

```
bash script/llm_based_decontamination.sh
```

After the LLM judgment completes, run the post-processing step to generate the final decontaminated dataset:

```
python vllm_inference/llm_based_decontamination_postprocess.py \
    --input_data_dir data/llm_based_decontamination/original_data \
    --output_path data/llm_based_decontamination/final_data/refined_augmented_cot_filtering_qa_decontamination.jsonl
```

> **Note**: Our open-source code does not include OlympicArena benchmarks since their test sets are not publicly available. However, we have internally performed decontamination against [OlympicArena](https://github.com/GAIR-NLP/OlympicArena) to ensure there is no data overlap between our dataset and this benchmark.

### Step 8. Reference Answer Extraction

Configure the reference answer extraction process by modifying `data_process/vllm_inference/task_config/extract_reference_answer.yaml`.

Execute the extraction script:

```
bash script/extract_reference_answer.sh
```

After extraction completes, run the post-processing step to finalize the QA pairs:

```
python vllm_inference/extract_reference_answer_postprocess.py \
    --input_data_dir data/extract_reference_answer/original_data \
    --output_path data/extract_reference_answer/final_data/refined_augmented_cot_filtering_qa_decontamination_reference_answer.jsonl
```

### Step 9. Data Finalization

Transform the data format into:

```
{
    "question": ,
    "answer": ,
    "subject": ,
    "reference_answer": 
}
```

Run the following process:
```
python finalize_data.py \
    --input_path data/extract_reference_answer/final_data/refined_augmented_cot_filtering_qa_decontamination_reference_answer.jsonl \
    --output_path data/final_data.jsonl
```

## 🏋️ Supervised Finetuning

## 🎯 Evaluation

## 🥳 Citation