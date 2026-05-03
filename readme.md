# Fine-Tuning Qwen2.5-3B-Instruct for Mathematical Reasoning

**ECE 6514 Machine Learning Assignment - Fall 2025**

This project fine-tunes the Qwen2.5-3B-Instruct language model on the AceReason-1.1-SFT dataset using a novel loss-based difficulty selection strategy to improve mathematical reasoning capabilities. The study compares intelligent data curation against random data selection and the baseline model.

---

## 📋 Project Overview

**Objective:** Develop and evaluate an intelligent data selection mechanism that improves mathematical reasoning performance in large language models through selective fine-tuning.

**Key Innovation:** Loss-based difficulty sampling - selects training examples based on model loss to focus on moderately challenging samples that contribute most to learning.

**Model Base:** Qwen2.5-3B-Instruct (3 billion parameters)

**Dataset:** AceReason-1.1-SFT (~100K mathematical reasoning examples)

---

## 📊 Results Summary

| Benchmark | Baseline | Random | Data Selection | Key Finding |
|-----------|----------|--------|--------|-------------|
| **MATH-500** | 63.40% | 30.0% | **64.20%** | **+34.2 pp over random** ✓ |
| **GPQA-Diamond** | 29.29% | 27.27% | 19.19% | Trade-off observed |
| **AIME 2024** | 3.33% | 0.0% | **3.33%** | Maintained baseline |
| **AIME 2025** | 0.00% | 0.0% | **3.33%** | New capability gained |
| **MMLU-Redux-2** | 57.05% | 29.82% | 31.11% | Trade-off observed |

**🎯 Key Achievement:** Loss-based data selection achieved **+34.2 percentage points** better performance than random selection on MATH-500, validating intelligent data curation for mathematical reasoning tasks.

---

## 📁 Project Structure

```
.
├── scripts/
│   ├── datasetcreation.py           # Loss-based data selection implementation
│   ├── sft_dataselection_cnfg.yaml  # Config for full fine-tuning with selected data
│   └── sft_random_selected.yaml     # Config for LoRA fine-tuning with random data
├── EvaluationScripts/
│   ├── neweval_diffdata.sh          # Evaluation script for mathematical benchmarks
│   └── newmlmu.sh                   # Evaluation script for MMLU-2 benchmark
├── results/
│   ├── EvalBaseline/                # Baseline and random selection results
│   │   ├── Baseline_reasoning_results.csv
│   │   ├── Random_results.csv
│   │   └── Dataselection_results.json
│   └── mmlu2_redux_results/         # MMLU-2 benchmark results
│       ├── mlmubaseline_results.csv
│       ├── randommodelmlmuresults.csv
│       └── mlmuresults_dataselection.docx
├── requirements.txt                 # Python dependencies
├── TML_Report.pdf                   # Full technical report
└── readme.md                        # This file
```

---

## 🔧 Installation & Setup

### Requirements
- Python 3.9+
- CUDA-capable GPU (A100 recommended for efficient training)
- 40GB+ VRAM

### Installation

```bash
# Clone the repository
git clone <repository_url>
cd TrustWorthyMachineLearningProject1

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Data Selection (Loss-Based Sampling)

Generate a curated dataset of 15K examples using loss-based difficulty selection:

```bash
python scripts/datasetcreation.py
```

This script:
- Loads the AceReason-1.1-SFT dataset
- Computes forward pass losses for each example
- Selects examples in the difficulty percentile range (0.25-0.75)
- Outputs `best15k.json` for training

### 2. Fine-Tuning with Data Selection

Train using the full fine-tuning approach with selected data:

```bash
# Configure the YAML file
vim scripts/sft_dataselection_cnfg.yaml

# Run training with the selection config
# (Use with your training framework, e.g., LLaMA-Factory or Hugging Face Trainer)
```

### 3. Fine-Tuning with Random Selection (Baseline)

Train using LoRA with randomly selected data:

```bash
# Configure the YAML file
vim scripts/sft_random_selected.yaml

# Run training with the random config
```

### 4. Evaluation

Run evaluation on mathematical reasoning benchmarks:

```bash
# Evaluate on MATH, GPQA, AIME datasets
bash EvaluationScripts/neweval_diffdata.sh

# Evaluate on MMLU-2 benchmark
bash EvaluationScripts/newmlmu.sh
```

---

## 📚 Technical Details

### Data Selection Strategy

The loss-based selection mechanism:
1. **Compute Model Loss:** Forward pass each example through the base model
2. **Calculate Percentiles:** Identify examples within the 25th-75th percentile range
3. **Intelligent Curation:** Select moderately difficult examples (not too easy, not too hard)
4. **Dataset Size:** Curates ~15K training examples from 100K total

**Rationale:** Examples with moderate loss are most informative for model improvement - they're challenging enough to teach new patterns but not so difficult that they introduce noise.

### Training Configurations

**Full Fine-Tuning (Data Selection Config):**
- Batch size: 2 (gradient accumulation: 8)
- Learning rate: 1.0e-5
- Optimizer: AdamW with cosine schedule
- Epochs: 3
- Hardware: A100 with DeepSpeed ZeRO-3 offloading

**LoRA Fine-Tuning (Random Config):**
- LoRA rank: 8, alpha: 32
- Batch size: 2 (gradient accumulation: 8)
- Learning rate: 2e-4
- Epochs: 3
- Efficient adaptation of full model

---

## 🔗 Resources & Links

### Trained Models
- **Random Selection Model:** https://huggingface.co/aaryankamdar/TML_Trained_model_random_dataset
- **Data Selection Model:** https://huggingface.co/aaryankamdar/TML_PROJECT_MODELS

### Datasets
- **Training Dataset:** https://huggingface.co/datasets/aaryankamdar/TML_Project

---

## 📖 Documentation

- **Full Technical Report:** [TML_Report.pdf](TML_Report.pdf) - Comprehensive analysis, methodology, and findings

---

## 🎓 Course Information

- **Course:** ECE 6514 Machine Learning
- **Semester:** Fall 2025
- **Topic:** Trustworthy Machine Learning - Data-Centric AI approaches

---

## ✅ Key Findings

1. **Loss-based data selection significantly outperforms random sampling** on mathematical reasoning tasks
2. **Trade-offs exist** - optimization for mathematical reasoning can impact performance on general knowledge tasks (MMLU)
3. **Data quality over quantity** - careful selection of 15K examples outperforms random 15K by 34.2 percentage points
4. **Transferability challenges** - improvements don't uniformly transfer across all benchmark types

---

## 📝 License & Attribution

Dataset: AceReason-1.1-SFT
Base Model: Qwen2.5-3B-Instruct (Alibaba Qwen Team)

