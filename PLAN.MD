Here is the complete 10-phase plan for your water leak detection deep learning project and scientific paper.
---
## Dataset Reality Check (Binary Focus)
From the analysis of your CSV, here are the exact numbers you're working with for **leak detection only**: [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/84862668/69c6f574-8da3-441e-b1a9-270ddfb18a4b/water_leak_detection_1000_rows.csv)

- **Normal**: 981 rows (98.1%)
- **Leak**: 19 rows (1.9%)
- **Data span**: 3.5 days, 5-min intervals, 10 sensors (S001–S010)
- **~100 readings per sensor** over the full period

This is an extreme imbalance problem. Your entire project strategy must be built around it.

***
## Phase 1 — EDA & Data Analysis (`01-Exploratory-Data-Analysis.ipynb`)
Your existing EDA notebook already covers this. The key findings to **document in the paper**:
- Pressure mean drops from 3.26 bar (Normal) → 2.19 bar (Leak) — primary signal
- Flow Rate increases slightly (124 → 149 L/s) — secondary signal
- Temperature is **non-discriminative** (17.4 vs 17.9°C) — mention this explicitly
- Sensor S010 has the most readings; leaks appear distributed across sensors

***
## Phase 2 — Feature Engineering (`02_feature_eng.ipynb`)
Add 3 engineered features per timestep before windowing:

| Feature | Formula | Why It Matters |
|---|---|---|
| `pressure_delta` | \(P_t - P_{t-1}\) per sensor | Detects sudden pressure drops |
| `flow_zscore` | \((F_t - \mu_s) / \sigma_s\) per sensor | Flags abnormal flow surges |
| `pf_ratio` | \(P_t / F_t\) | Drops sharply as leaks develop |

This gives you **6 features per timestep** (3 raw + 3 engineered), making your input shape `(window_size, 6)`.

***
## Phase 3 — Per-Sensor Windowing (`03_windowing.ipynb`)
This is the most critical fix over your current project:

```python
# CORRECT approach
df_sorted = df.sort_values(['Sensor_ID', 'Timestamp'])  # group by sensor first
# Window size: 20 timesteps = 100 minutes of sensor history
# Step size: 5 (reduce overlap, reduce data leakage)
# Label: LAST timestep in window (not "any leak in window")
WINDOW_SIZE = 20
STEP_SIZE   = 5
```

**Why window=20, step=5?** With ~100 readings per sensor and only 19 leaks, window=50/step=1 inflates the dataset artificially. Window=20/step=5 keeps window overlap at 75% (acceptable) while producing ~200 real windows instead of 510 fake-inflated ones.

***
## Phase 4 — Train / Val / Test Split (`03_windowing.ipynb`)
```python
# Temporal split per sensor — NO random shuffling across time
# 70% train | 15% val | 15% test
from sklearn.model_selection import StratifiedShuffleSplit
# Use stratify=y to preserve the 1.9% leak ratio in all 3 sets
```

You need **3 separate sets** for a scientific paper: train for learning, val for hyperparameter tuning, test for final reporting. Never report val results as final results.

***
## Phase 5 — Architecture (`04_model.ipynb`) ⭐ Core Contribution
The **CNN-BiLSTM with Self-Attention** is your paper's headline contribution:

```
Input (batch, 20, 6)
  │
  ├─ Conv1D(32, kernel=3, ReLU) + BatchNorm    # local patterns (15 min)
  ├─ Conv1D(64, kernel=5, ReLU) + BatchNorm    # mid-scale patterns (25 min)
  │
  ├─ Bidirectional LSTM(64 units)              # temporal context both directions
  │
  ├─ Self-Attention Layer                      # learn which timesteps matter
  │
  ├─ GlobalAveragePooling1D
  ├─ Dropout(0.4)
  ├─ Dense(32, ReLU)
  │
  └─ Dense(1, sigmoid)  → Leak probability
```

**Total parameters: ~46,000** — appropriately sized for your small dataset. Bigger models will overfit.

***
## Phase 6 — Training Strategy (`04_model.ipynb`)
Replace `binary_crossentropy` with **Focal Loss** — this is a key paper contribution:

```python
# Focal Loss: penalizes easy negatives less, focuses on hard positives
def focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        return -alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t + 1e-8)
    return loss

model.compile(
    optimizer=Adam(lr=0.001),
    loss=focal_loss(gamma=2.0, alpha=0.75),
    metrics=['accuracy', AUC(name='auc')]
)
```

Training callbacks:
- `EarlyStopping(patience=15, monitor='val_auc', mode='max')`
- `ReduceLROnPlateau(factor=0.5, patience=7)`
- `ModelCheckpoint` saving best weights

***
## Phase 7 — Threshold Tuning (`04_model.ipynb`)
Sigmoid outputs a probability. The **default 0.5 threshold is wrong** for imbalanced data:

```python
# Sweep thresholds on validation set
from sklearn.metrics import f1_score, precision_recall_curve
prec, rec, thresh = precision_recall_curve(y_val, y_proba_val)
f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
best_threshold = thresh[np.argmax(f1_scores)]
# Typically lands around 0.2–0.35 for this dataset
```

For leak detection specifically, you want **high recall** (missing a leak is worse than a false alarm). Pick the threshold where recall ≥ 0.90.

***
## Phase 8 — Evaluation (`05_evaluation.ipynb`)
Report **all 5 of these** — using only accuracy is a red flag for reviewers:

- **F1-Score** (macro and for Leak class specifically)
- **AUC-ROC** curve
- **Precision-Recall curve** (most informative for imbalanced tasks)
- **Confusion matrix** (normalized)
- **Ablation table**: CNN only → +BiLSTM → +Attention → +Focal Loss

The ablation study is the most important part of a hackathon paper — it proves each component contributes.

***
## Phase 9 — Interpretability (`05_evaluation.ipynb`)
Visualize the attention weights on a window containing a leak event:

```python
# Extract attention weights from the attention layer
attention_model = Model(inputs=model.input,
                        outputs=model.get_layer('attention').output)
weights = attention_model.predict(X_leak_window)  # shape (1, 20, 1)
# Plot as heatmap over time → shows model "looks at" timestep before leak
```

This produces a **heatmap figure** showing which 5-minute intervals triggered the alert — crucial for the paper's discussion section and very impressive for hackathon judges.

***
## Phase 10 — Scientific Paper Structure
```
Title: "CNN-BiLSTM with Self-Attention for Water Leak Detection
        in IoT Sensor Networks"

1. Abstract (150 words)
2. Introduction — Water loss costs, ML gap, contribution
3. Related Work — Threshold methods, SVM, vanilla CNN baselines
4. Dataset & Preprocessing — EDA findings, feature engineering
5. Methodology — Architecture diagram, Focal Loss justification
6. Experiments — Ablation table, comparison with baselines
7. Results — F1, AUC-ROC, PR curve, confusion matrix
8. Interpretability — Attention heatmap on leak events
9. Conclusion + Future Work — Real-time IoT inference
```

The **ablation study in Section 6** is what makes this publishable: show CNN → CNN+BiLSTM → CNN+BiLSTM+Attention → +FocalLoss each add measurable F1 improvement.

***
## Notebook Structure
```
water-leak-detection/
├── 01-Exploratory-Data-Analysis.ipynb                ← already done ✓
├── 02_feature_engineering.ipynb  ← NEW
├── 03_windowing.ipynb          ← refactor (window=20, step=5, last-label)
├── 04_model_training.ipynb     ← NEW (CNN-BiLSTM+Attention + Focal Loss)
├── 05_evaluation.ipynb         ← NEW (ablation + all metrics + heatmaps)
├── data/
│   ├── water_leak_detection_1000_rows.csv
│   └── processed/  ← X_windows.npy, y_windows.npy (new)
└── paper/
    └── paper.tex
```

Would you like me to write the complete implementation code for **Phase 3 (windowing) + Phase 4 (model)** as ready-to-run notebooks?