# Fine-Tuning Qwen2.5-3B-Instruct for Mathematical Reasoning

**ECE 6514 Machine Learning Assignment - Fall 2025**

Fine-tuning Qwen2.5-3B-Instruct on AceReason-1.1-SFT using loss-based difficulty selection to improve mathematical reasoning capabilities.

---

## 🔗 Links

### Models 

1. Randomly selected model: https://huggingface.co/aaryankamdar/TML_Trained_model_random_dataset 

2. Data selection: https://huggingface.co/aaryankamdar/TML_PROJECT_MODELS 

### DatasetsDataset Links: https://huggingface.co/datasets/aaryankamdar/TML_Project 


##  Results Summary

| Benchmark | Baseline | Random | Advanced | Improvement |
|-----------|----------|--------|----------|-------------|
| **MATH-500** | 63.40% | 30.0% | **64.20%** | **+34.2%** over random  |
| **GPQA-Diamond** | 29.29% | 27.27% | 19.19% | Trade-off  |
| **AIME 2024** | 3.33% | 0.0% | **3.33%** | Maintained  |
| **AIME 2025** | 0.00% | 0.0% | **3.33%** | New capability |
| **MMLU-Redux-2** | 57.05% | 29.82% | 31.11% | Trade-off  |

**Key Achievement:** Loss-based data selection achieved **+34.2 percentage points** better performance than random selection on MATH-500, validating intelligent data curation for reasoning tasks.

---
### Documentation
- **Full Technical Report:** TML_PROJECT.pdf

