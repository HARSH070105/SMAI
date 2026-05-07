# Overall Verification Metrics

### Equal Error Rate (EER) Matrix [Lower is Better]

| Model \ Prompt         | baseline               | ensemble               | custom_ensemble        |
|------------------------|------------------------|------------------------|------------------------|
| base                   | 7.88%                  | 7.50%                  | 6.37%                  |
| baseline_lora          | 1.74%                  | 1.88%                  | 2.11%                  |
| ensemble_lora          | 1.67%                  | 1.68%                  | 2.03%                  |
| custom_ensemble_lora   | 2.76%                  | 2.46%                  | 1.79%                  |

### ROC AUC Score Matrix [Higher is Better]

| Model \ Prompt         | baseline               | ensemble               | custom_ensemble        |
|------------------------|------------------------|------------------------|------------------------|
| base                   | 0.9774                 | 0.9803                 | 0.9845                 |
| baseline_lora          | 0.9987                 | 0.9987                 | 0.9981                 |
| ensemble_lora          | 0.9986                 | 0.9988                 | 0.9983                 |
| custom_ensemble_lora   | 0.9970                 | 0.9975                 | 0.9990                 |

### True Accept Rate (TAR) @ 1.0% FAR Matrix [Higher is Better]

| Model \ Prompt         | baseline               | ensemble               | custom_ensemble        |
|------------------------|------------------------|------------------------|------------------------|
| base                   | 69.01%                 | 69.91%                 | 72.67%                 |
| baseline_lora          | 97.31%                 | 97.10%                 | 95.51%                 |
| ensemble_lora          | 96.62%                 | 96.83%                 | 95.86%                 |
| custom_ensemble_lora   | 93.17%                 | 94.62%                 | 97.38%                 |

