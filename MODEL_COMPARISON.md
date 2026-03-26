# Model Comparison: Deep Learning vs Classical ML

This report compares the three trained models on the same test set:
- Logistic Regression
- Random Forest
- Deep Learning (CNN-BiLSTM-Attention)

## Source Data
- data/benchmark_metrics_all_models.csv
- data/final_metrics.csv

## Benchmark Table

| Model | Threshold | AUC-ROC | AUC-PR | Accuracy | Balanced Accuracy | Precision (Leak) | Recall (Leak) | F1-score (Leak) | False Alarm Rate | Miss Rate | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.3752 | 0.9524 | 0.9595 | 0.9231 | 0.9148 | 0.9405 | 0.8681 | 0.9029 | 0.0385 | 0.1319 | 79 | 5 | 125 | 12 |
| Random Forest | 0.3695 | 0.9710 | 0.9652 | 0.9050 | 0.8912 | 0.9487 | 0.8132 | 0.8757 | 0.0308 | 0.1868 | 74 | 4 | 126 | 17 |
| Deep Learning (CNN-BiLSTM-Attention) | 0.0333 | 0.9412 | 0.9583 | 0.9457 | 0.9390 | 0.9647 | 0.9011 | 0.9318 | 0.0231 | 0.0989 | 82 | 3 | 127 | 9 |

## Key Findings

1. Best ranking metrics:
- Random Forest has the best AUC-ROC (0.9710) and AUC-PR (0.9652).

2. Best operational performance at selected threshold:
- Deep Learning achieves the best Accuracy (0.9457), Balanced Accuracy (0.9390), Precision (0.9647), Recall (0.9011), and F1-score (0.9318).

3. Error trade-off:
- Deep Learning has the lowest False Alarm Rate (0.0231) and lowest Miss Rate (0.0989), with the fewest false positives (3) and false negatives (9).

4. Classical model behavior:
- Logistic Regression is more balanced than Random Forest at the chosen threshold and is closer to Deep Learning than Random Forest on Recall and F1.
- Random Forest produces stronger probability ranking but weaker thresholded classification outcomes under the current thresholding strategy.

## Overall Conclusion

For deployment-focused leak detection (maximize leak capture while controlling false alarms), the Deep Learning model is the strongest overall option in this benchmark.

If your objective shifts toward probability ranking quality only, Random Forest is highly competitive and can be reconsidered with alternative threshold tuning or calibration.

## Recommendation

- Keep Deep Learning as the primary model for production decisions.
- Keep Logistic Regression as a lightweight baseline.
- Revisit Random Forest threshold calibration if you want to exploit its strong AUC metrics in a different operating regime.
