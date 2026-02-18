# XREAD: Radiology Report Pathology Classification with LLM-Based Uncertainty Resolution

## Overview

XREAD is a novel approach to handling uncertain labels in chest X-ray pathology classification. Instead of using blanket strategies (U-Zeros, U-Ones, U-Ignore) to handle the uncertain labels (-1) extracted by NLP systems from hedging language in radiology reports, we use a local LLM to **read the actual report text and classify the type of uncertainty**, then resolve labels intelligently.

We fine-tune [RadBERT](https://huggingface.co/StanfordAIMI/RadBERT) on the resolved labels and evaluate across a 4x3 experiment matrix (4 uncertainty strategies x 3 model architectures).

## Project Structure

```
XREAD/
├── main.py                       # CLI entrypoint
├── scripts/
│   ├── make_dataset.py           # Data loading, merging, splitting
│   ├── resolve_uncertainty.py    # Ollama LLM uncertainty resolution
│   ├── build_features.py         # TF-IDF + tokenization
│   ├── model.py                  # Model architectures (Naive, Classical, RadBERT)
│   ├── train.py                  # Training loop orchestration
│   ├── evaluate.py               # Metrics + experiment comparison
│   └── app.py                    # Gradio deployment app
├── configs/
│   └── experiment.yaml           # Hyperparameters
├── models/                       # Saved model artifacts
├── data/
│   ├── raw/                      # Downloaded datasets
│   ├── processed/                # Cleaned splits
│   └── outputs/                  # Predictions, figures
└── notebooks/                    # Exploration
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (tested on RTX 4070 Ti SUPER)
- [Ollama](https://ollama.ai/) with `qwen3-coder:30b` model (for uncertainty resolution)

### Installation

```bash
pip install -r requirements.txt
```

### Data Download

1. **CheXpert Plus CSV** (386 MB): Download `df_chexpert_plus_240401.csv` from [AIMI](https://aimi.stanford.edu/chexpert-plus) to `data/raw/`
2. **CheXbert Labels** (6 MB): Download and extract `chexbert_labels.zip` to `data/raw/chexbert_labels/`
3. **CheXpert-small images** (~11.5 GB): `kaggle datasets download ashery/chexpert -p data/raw/`
4. **RadBERT**: Auto-downloaded on first run via HuggingFace

## Usage

### Full Pipeline

```bash
# Phase 1: Process data
python main.py data

# Phase 2: Resolve uncertain labels
python main.py resolve                    # Full Ollama run (~17 hrs)
python main.py resolve --skip-ollama      # Regex-only (fast, ~5 min)
python main.py resolve --sample-size 2000 # Sample for Ollama

# Phase 3: Build features
python main.py features

# Phase 4: Train models
python main.py train                                    # All 12 models
python main.py train --models naive classical           # Skip RadBERT
python main.py train --strategies u_zeros u_llm         # Specific strategies

# Phase 5: Evaluate
python main.py evaluate

# Phase 6: Launch app
python main.py app
python main.py app --share  # Public link
```

## Models

| Model | Description | Input |
|-------|-------------|-------|
| **Naive Baseline** | Predicts training prevalence | None |
| **Classical ML** | 14 independent LogisticRegression | TF-IDF features |
| **RadBERT** | Fine-tuned radiology BERT | Raw report text |

## Uncertainty Strategies

| Strategy | Uncertain (-1) Treatment |
|----------|-------------------------|
| **U-Zeros** | Map to negative (0) |
| **U-Ones** | Map to positive (1) |
| **U-Ignore** | Mask from loss (NaN) |
| **U-LLM** | Classify type, resolve intelligently |

## 14 CheXpert Pathologies

No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices

## Hardware

Developed and tested on:
- GPU: NVIDIA RTX 4070 Ti SUPER (16 GB VRAM)
- Ollama: qwen3-coder:30b

## References

- Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison" (2019)
- Yan et al., "RadBERT: Adapting Transformer-based Language Models to Radiology" (2022)
- Chambon et al., "CheXpert Plus: Augmenting a Large Chest X-ray Dataset with Text Radiology Reports, Patient Demographics and Additional Image Formats" (2024)
