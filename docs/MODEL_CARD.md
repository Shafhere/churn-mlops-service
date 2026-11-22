# ChurnGuard — Model Card

## 1. Purpose
Predict the probability of customer churn for a
telecommunications customer.

The output supports customer-retention decisions.
It is not an automatic decision to deny service.

## 2. Dataset
Source: IBM Telco Customer Churn sample.

Records: 7043
Training: 4930
Validation: 1056
Test: 1057

Raw dataset SHA-256:
`16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

Training split SHA-256:
`365bbc61d0d52c9a80d579e6fe27c09395e27a9cd14a61e79200feb9c6db34ac`

## 3. Preprocessing
- Validate required columns and value ranges.
- Exclude customerID and the Churn target.
- Convert TotalCharges to numeric.
- Impute missing values using training statistics.
- Scale numeric features and one-hot encode categories.
- Fit preprocessing only on the training split.

## 4. Model Selection
Selected: **logistic_balanced**

| Candidate | PR-AUC | Recall | Cost | Threshold |
|---|---:|---:|---:|---:|
| dummy_baseline | 0.2652 | 1.0000 | 776 | 0.26 |
| logistic_regression | 0.6310 | 0.9143 | 433 | 0.17 |
| logistic_balanced | 0.6308 | 0.9107 | 431 | 0.37 |
| random_forest_balanced | 0.6219 | 0.8679 | 443 | 0.36 |

Selection: minimum validation business cost,
with PR-AUC and recall as tie-breakers.

## 5. Decision Threshold
Threshold: **0.37**

Assumed false-positive cost: 1
Assumed false-negative cost: 5

These costs are illustrative, not verified
financial costs from a telecom company.

## 6. Final Holdout Test Results
- PR-AUC (Average Precision): 0.6586
- ROC-AUC: 0.8446
- Precision: 0.4540
- Recall: 0.8790
- F1-score: 0.5988
- Brier score: 0.1630
- Assumed business cost: 467

The test split was excluded from model selection
and threshold tuning.

## 7. Global Feature Importance
Measured by permutation importance on validation data.

| Feature | Mean PR-score decrease |
|---|---:|
| tenure | 0.241242 |
| InternetService | 0.079760 |
| MonthlyCharges | 0.047481 |
| Contract | 0.044607 |
| TotalCharges | 0.037576 |
| StreamingMovies | 0.015776 |
| PaperlessBilling | 0.003217 |
| TechSupport | 0.002625 |
| SeniorCitizen | 0.001888 |
| gender | 0.001299 |

Feature importance is not causal evidence
or a per-customer explanation.

## 8. Version and Traceability
Registered model: churnguard-churn-model
Registry version: 1
Model URI: `models:/churnguard-churn-model/1`
MLflow run ID: `b62a637d0f574ec4be17b61db225d0c9`
Random seed: 42

## 9. Limitations
- This is a public sample dataset.
- Results are not validated against live telecom traffic.
- The cost assumptions need business validation.
- Correlated features can distort importance rankings.
- Customer subgroups have not been fairness-audited.
- A random split may not represent future customers.
- Model calibration and stability need further review.

## 10. Security and Privacy
- Customer IDs are excluded from model features.
- Raw data is excluded from version control.
- Experiment logs contain metrics and aggregate data.
- No individual customer records are logged as examples.
- Serialized model files must come from trusted sources.

## 11. Monitoring and Maintenance
Monitor:
- Prediction latency and error rate.
- Prediction probability distribution.
- Feature drift compared with training.
- Model quality when labeled outcomes become available.

Investigate significant drift before considering retraining.
Any replacement model must pass evaluation and approval.

## 12. Deployment Status
Candidate model: requires application testing,
security review and explicit deployment approval.
