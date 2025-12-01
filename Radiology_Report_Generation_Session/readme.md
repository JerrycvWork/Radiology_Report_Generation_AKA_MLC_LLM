# Radiology Report Generation Using Automatic Keyword Adaptation & LLMs

This repository contains the radiology report generation part of official implementation for the paper:  
**"Radiology report generation using automatic keyword adaptation, frequency-based multi-label classification and text-to-text large language models"** *Published in Computers in Biology and Medicine, September 2025.*

[**[Read the Paper]**](https://www.sciencedirect.com/science/article/pii/S001048252500976X)

This project provides a unified framework for generating radiology reports from chest X-ray images. It replaces traditional black-box visual features with transparent keyword lists, utilizing a frequency-based multi-label classification strategy and a pre-trained text-to-text Large Language Model (LLM) to generate clinically coherent reports.

## 📂 Project Structure

```

.
├── run\_report\_gen.py    \# Main entry point for training and testing
├── dataset.py           \# Dataset loading and processing logic (MIMIC-CXR, IU-Xray, Custom)
├── evaluate.py          \# Evaluation metrics (NLP & Clinical: RadGraph, CheXbert)
├── requirements.txt     \# Python dependencies
└── README.md            \# This file

```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/JerrycvWork/Radiology_Report_Generation_AKA_MLC_LLM](https://github.com/JerrycvWork/Radiology_Report_Generation_AKA_MLC_LLM)
   cd your-repo
   ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: For clinical metrics (BLEU, METEOR, etc.), ensure you have a Java runtime installed (required for `pycocoevalcap`).*

3.  **External Resources (Optional for Evaluation):**
    To calculate **RadGraph** and **CheXbert** scores, download the model weights:

      * **CheXbert:** [Download `chexbert.pth`](https://github.com/stanfordmlgroup/CheXbert)
      * **RadGraph:** [Download `model.tar.gz`](https://physionet.org/content/radgraph/1.0.0/)

## 📊 Data Preparation

The code expects CSV files with at least two columns:

  * **Source Column:** Keywords or findings (Default: `Level1_keywords`)
  * **Target Column:** Ground truth report (Default: `Ground-Truth`)

### Supported Datasets

  * **MIMIC-CXR**
  * **IU-Xray**
  * **Custom:** Place `train.csv`, `val.csv`, and `test.csv` in a folder.

## 🚀 Training

To train a model, use `run_report_gen.py` with `--mode train`.

### Example: Train on MIMIC-CXR with ClinicalT5

```bash
python run_report_gen.py \
  --mode train \
  --dataset_name mimic \
  --data_dir "path/to/mimic_dataset" \
  --model_type clinicalt5 \
  --model_name "luqh/ClinicalT5-base" \
  --epochs 10 \
  --batch_size 4
```

### Example: Train on Custom Data

```bash
python run_report_gen.py \
  --mode train \
  --dataset_name custom \
  --data_dir "data/my_custom_dataset" \
  --source_col "Keywords" \
  --target_col "Report" \
  --model_type t5 \
  --model_name "t5-base"
```

**Key Arguments:**

  * `--dataset_name`: `mimic`, `iuxray`, or `custom`
  * `--model_type`: The architecture type (`t5`, `flan_t5`, `clinicalt5`)
  * `--model_name`: The HuggingFace model path (e.g., `google/flan-t5-base`)

## 📉 Evaluation

Evaluation automatically calculates **NLP Metrics** (BLEU, ROUGE, METEOR, CIDEr). If paths are provided, it also calculates **Clinical Metrics** (RadGraph F1, CheXbert Similarity).

### Run Inference & Evaluation

```bash
python run_report_gen.py \
  --mode test \
  --dataset_name mimic \
  --data_dir "path/to/mimic_dataset" \
  --checkpoint_path "outputs/mimic_clinicalt5_run/simplet5-epoch-9..." \
  --chexbert_path "checkpoints/chexbert.pth" \
  --radgraph_path "checkpoints/radgraph_model.tar.gz"
```

**Outputs:**

  * `results.csv`: Contains generated reports vs. ground truth.
  * `nlp_metrics.txt`: BLEU, ROUGE, etc.
  * `clinical_metrics.csv`: Detailed RadGraph and CheXbert scores.

## 📜 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{He2025Radiology,
  title = {Radiology report generation using automatic keyword adaptation, frequency-based multi-label classification and text-to-text large language models},
  author = {He, Zebang and Wong, Alex Ngai Nick and Yoo, Jung Sun},
  journal = {Computers in Biology and Medicine},
  volume = {196},
  pages = {110625},
  year = {2025},
  month = {September},
  issn = {0010-4825},
  doi = {10.1016/j.compbiomed.2025.110625},
  url = {[https://www.sciencedirect.com/science/article/pii/S001048252500976X](https://www.sciencedirect.com/science/article/pii/S001048252500976X)}
}
```

## 🤝 Acknowledgements

This codebase leverages [SimpleT5](https://github.com/Shivanandroy/simpleT5) for model training and [CXRMetric](https://www.google.com/search?q=https://github.com/aehrc/CXRMetric) for clinical evaluation.
