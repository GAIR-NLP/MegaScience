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
python extract_qa_postprocess.py --input data/extract_qa/original_qa --output data/extract_qa/final_qa/extract_qa.jsonl --document_save_path data/extract_qa/final_qa/documents.jsonl
```

### Step 3. Question Deduplication

### Step 4. QA Refinement

### Step 5. CoT Augmentation

### Step 6. QA Filtering

### Step 7. LLM-based Question Decontamination

### Step 8. Reference Answer Extraction

## 🏋️ Supervised Finetuning

## 🎯 Evaluation

## 🥳 Citation