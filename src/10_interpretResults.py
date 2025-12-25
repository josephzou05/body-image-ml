import pandas as pd
import matplotlib.pyplot as plt

# load labeled and clustered data
labeledDf = pd.read_csv("outputs/labeledData.csv")
clusteredDf = pd.read_csv("outputs/clusteredData.csv")

# model performance summary

# store model metrics
modelResults = pd.DataFrame({
    "model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "rocAuc": [
        0.892,
        0.770,
        0.971
    ],
    "recallAtRisk": [
        0.33,
        0.50,
        0.50
    ]
})

# plot ROC-AUC comparison
plt.figure(figsize=(7, 4))
plt.bar(modelResults["model"], modelResults["rocAuc"])
plt.ylim(0, 1)
plt.title("ROC-AUC Comparison Across Classification Models")
plt.ylabel("ROC-AUC")
plt.tight_layout()
plt.savefig("outputs/figures/roc_auc_comparison.png")
plt.show()

# plot recall comparison for at-risk class
plt.figure(figsize=(7, 4))
plt.bar(modelResults["model"], modelResults["recallAtRisk"])
plt.ylim(0, 1)
plt.title("Recall for At-Risk Class Across Models")
plt.ylabel("Recall (Class = At Risk)")
plt.tight_layout()
plt.savefig("outputs/figures/recall_comparison.png")
plt.show()

# cluster-level risk analysis

# merge cluster labels with risk label
mergedDf = clusteredDf.merge(
    labeledDf[["atRisk"]],
    left_index=True,
    right_index=True
)

# compute at-risk rate per cluster
clusterRisk = (
    mergedDf
    .groupby("cluster")["atRisk"]
    .mean()
    .reset_index()
)

# plot at-risk rate by cluster
plt.figure(figsize=(7, 4))
plt.bar(clusterRisk["cluster"], clusterRisk["atRisk"])
plt.ylim(0, 1)
plt.title("At-Risk Rate by Gym-Goer Cluster")
plt.xlabel("Cluster")
plt.ylabel("Proportion At Risk")
plt.tight_layout()
plt.savefig("outputs/figures/cluster_risk_rates.png")
plt.show()

# print summary tables
print("Model Performance Summary")
print(modelResults)

print("\nAt-Risk Rate by Cluster")
print(clusterRisk)
