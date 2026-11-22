
def build_model_card(report):
    """Create a model card from measured results."""

    test = report["test_metrics"]
    registry = report["registry"]
    dataset = report["dataset"]
    costs = report["business_assumptions"]

    comparison_rows = "\n".join(
        f"| {r['model']} | {r['pr_auc']:.4f} | "
        f"{r['recall']:.4f} | "
        f"{r['business_cost']} | {r['threshold']:.2f} |"
        for r in report["validation_comparison"]
    )

    feature_rows = "\n".join(
        f"| {r['feature']} | {r['importance']:.6f} |"
        for r in report["feature_importance"]
    )

    return f"""# ChurnGuard — Model Card

## 1. Purpose
Predict the probability of customer churn for a
telecommunications customer.

The output supports customer-retention decisions.
It is not an automatic decision to deny service.

## 2. Dataset
Source: IBM Telco Customer Churn sample.

Records: {dataset['records']}
Training: {dataset['train_rows']}
Validation: {dataset['val_rows']}
Test: {dataset['test_rows']}

Raw dataset SHA-256:
`{dataset['source_sha256']}`

Training split SHA-256:
`{dataset['train_split_sha256']}`

## 3. Preprocessing
- Validate required columns and value ranges.
- Exclude customerID and the Churn target.
- Convert TotalCharges to numeric.
- Impute missing values using training statistics.
- Scale numeric features and one-hot encode categories.
- Fit preprocessing only on the training split.

## 4. Model Selection
Selected: **{report['selected_model']}**

| Candidate | PR-AUC | Recall | Cost | Threshold |
|---|---:|---:|---:|---:|
{comparison_rows}

Selection: minimum validation business cost,
with PR-AUC and recall as tie-breakers.

## 5. Decision Threshold
Threshold: **{report['threshold']:.2f}**

Assumed false-positive cost: {costs['false_positive_cost']}
Assumed false-negative cost: {costs['false_negative_cost']}

These costs are illustrative, not verified
financial costs from a telecom company.

## 6. Final Holdout Test Results
- PR-AUC (Average Precision): {test['pr_auc']:.4f}
- ROC-AUC: {test['roc_auc']:.4f}
- Precision: {test['precision']:.4f}
- Recall: {test['recall']:.4f}
- F1-score: {test['f1']:.4f}
- Brier score: {test['brier_score']:.4f}
- Assumed business cost: {test['business_cost']}

The test split was excluded from model selection
and threshold tuning.

## 7. Global Feature Importance
Measured by permutation importance on validation data.

| Feature | Mean PR-score decrease |
|---|---:|
{feature_rows}

Feature importance is not causal evidence
or a per-customer explanation.

## 8. Version and Traceability
Registered model: {registry['name']}
Registry version: {registry['version']}
Model URI: `{registry['uri']}`
MLflow run ID: `{registry['run_id']}`
Random seed: {report['random_state']}

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
"""
