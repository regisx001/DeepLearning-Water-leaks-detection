# DeepLearning-Water-leaks-detection

End-to-end project for water leak detection in urban distribution networks using:
- a hybrid deep learning model (CNN-BiLSTM-Attention)
- classical ML baselines (Logistic Regression, Random Forest)
- benchmark comparison and paper-ready outputs

## Project Goals

- Detect leak events from multivariate sensor time-series data.
- Build a robust deep learning pipeline with threshold calibration.
- Compare deep learning against classical ML on the same split strategy.
- Produce reproducible results and figures for paper writing (English and French versions).

## Repository Structure

```text
.
|- data/
|  |- X_train.npy, y_train.npy, X_test.npy, y_test.npy
|  |- benchmark_metrics_all_models.csv
|  |- benchmark_metrics_classical_models.csv
|  |- final_metrics.csv
|- models/
|  |- best_model.keras
|  |- best_threshold.npy
|- notebooks/
|  |- 01-Exploratory-Data-Analysis.ipynb
|  |- 02-Feature-Engineering.ipynb
|  |- 03-Windowing.ipynb
|  |- 04-Model-Training-Evaluation.ipynb
|  |- 05-Train-benchmark-models.ipynb
|- paper/
|  |- main.tex
|  |- sections/
|- paper_fr/
|  |- main.tex
|  |- sections/
|- MODEL_COMPARISON.md
|- MODEL_COMPARISON_FR.md
|- requirements.txt
```

## Environment Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main libraries used:
- tensorflow
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Data and Pipeline

The workflow follows these steps:

1. Exploratory analysis and preprocessing.
2. Feature engineering on sensor variables.
3. Sliding-window sequence construction.
4. Deep model training and threshold calibration.
5. Classical ML benchmark training and comparison.

Recommended notebook execution order:

1. notebooks/01-Exploratory-Data-Analysis.ipynb
2. notebooks/02-Feature-Engineering.ipynb
3. notebooks/03-Windowing.ipynb
4. notebooks/04-Model-Training-Evaluation.ipynb
5. notebooks/05-Train-benchmark-models.ipynb

## Models

### Deep Learning
- CNN-BiLSTM-Attention architecture
- Saved model: models/best_model.keras
- Calibrated threshold: models/best_threshold.npy

### Classical ML Baselines
- Logistic Regression
- Random Forest

Classical baselines are trained on tabular features extracted from windows:
- mean, std, min, max, and trend descriptors per feature channel.

## Benchmark Results

Latest benchmark file: data/benchmark_metrics_all_models.csv

| Model | AUC-ROC | AUC-PR | Accuracy | Precision (Leak) | Recall (Leak) | F1-score (Leak) |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9710 | 0.9652 | 0.9050 | 0.9487 | 0.8132 | 0.8757 |
| Logistic Regression | 0.9524 | 0.9595 | 0.9231 | 0.9405 | 0.8681 | 0.9029 |
| Deep Learning (CNN-BiLSTM-Attention) | 0.9412 | 0.9583 | 0.9457 | 0.9647 | 0.9011 | 0.9318 |

Summary:
- Random Forest provides the strongest ranking metrics (AUC-based).
- Deep Learning provides the strongest thresholded operational performance (best Accuracy/F1 and lowest false alarms/misses).

Detailed reports:
- English: MODEL_COMPARISON.md
- French: MODEL_COMPARISON_FR.md

## Generated Outputs

- Metrics CSV files in data/
- Benchmark figures in paper/images/
- Paper sources in paper/ and paper_fr/

Key benchmark figures:
- paper/images/fig_benchmark_classical_vs_deep.png
- paper/images/fig_benchmark_confusion_matrices_classical.png

## Papers

English paper:
- paper/main.tex

French paper:
- paper_fr/main.tex

Both versions include full sections, figures, and bibliography sources.

## Reproducibility Notes

- Random seed is fixed in training/benchmark notebooks (RANDOM_STATE = 42).
- Data split strategy uses temporal separation and calibration for threshold tuning.
- To regenerate benchmark artifacts, run notebooks/05-Train-benchmark-models.ipynb from top to bottom.

## Authors

- ZARQI Ezzoubair
- El Gharib Mahmoud

Supervised by:
- Pr. Mostafa EZZIYANI