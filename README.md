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

We adopt [olmOCR](https://github.com/allenai/olmocr) to convert PDFs into text documents.

### Step 2. QA Extraction



### Step 3. Question Deduplication

### Step 4. QA Refinement

### Step 5. CoT Augmentation

### Step 6. QA Filtering

### Step 7. LLM-based Question Decontamination

### Step 8. Reference Answer Extraction

## 🏋️ Supervised Finetuning

## 🎯 Evaluation

## 🥳 Citation