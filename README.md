# Body Image & Gym Habits: Machine Learning Analysis

This project applies supervised and unsupervised machine learning techniques to survey data exploring body image, gym behavior, and body dissatisfaction risk among gym-goers.

The project has two primary goals:
1. Predict individuals at risk for low body satisfaction based on gym-related behaviors
2. Identify latent gym-goer archetypes and analyze how body dissatisfaction risk varies across these groups

---

## Dataset

- Source: Original survey on body image and gym habits
- Sample size: 92 respondents
- Data type: Mixed numeric and categorical survey responses

The raw dataset is stored in the `data/` directory and is never modified. All preprocessing steps output new datasets to ensure reproducibility.

---

## Methods

### Classification

A binary classification label (`atRisk`) was created using the bottom 25% of the body satisfaction score distribution.

Models trained:
- Logistic Regression (baseline)
- Decision Tree
- Random Forest

Evaluation metrics:
- ROC–AUC
- Recall for the at-risk class

---

### Clustering

Unsupervised learning was used to discover gym-goer archetypes:
- K-means clustering (k = 4)
- Hierarchical clustering using Ward linkage for validation

Clusters were interpreted by analyzing the proportion of at-risk individuals within each cluster.

---

## Key Results

### Model Performance

| Model               | ROC–AUC | Recall (At Risk) |
|--------------------|---------|------------------|
| Logistic Regression | 0.89    | 0.33             |
| Decision Tree       | 0.77    | 0.50             |
| Random Forest       | 0.97    | 0.50             |

- Logistic regression showed strong ranking ability but low recall for at-risk individuals
- Tree-based models captured nonlinear relationships
- Random forest achieved the best overall performance

---

### Cluster-Level Risk

| Cluster | At-Risk Rate |
|--------|--------------|
| 0      | ~5%          |
| 1      | ~68%         |
| 2      | ~14%         |
| 3      | ~19%         |

One cluster exhibited substantially higher risk of low body satisfaction, indicating meaningful behavioral differences among gym-goers.

---

## Visualizations

All figures are saved in `outputs/figures/`, including:
- ROC–AUC comparison across classification models
- Recall comparison for at-risk classification
- At-risk rate by gym-goer cluster
- Hierarchical clustering dendrogram

---

## Project Structure

```
body-image-ml/
├── data/
│   └── body_image_gym_survey.csv
├── src/
│   ├── 01_explore.py
│   ├── 02_cleanColumns.py
│   ├── 03_selectFeatures.py
│   ├── 04_createLabels.py
│   ├── 05_encodeAndSplit.py
│   ├── 06_logisticRegression.py
│   ├── 07_treeModels.py
│   ├── 08_kMeansClustering.py
│   ├── 09_hierarchicalClustering.py
│   └── 10_interpretResults.py
├── outputs/
│   ├── cleanData.csv
│   ├── modelData.csv
│   ├── labeledData.csv
│   ├── clusteredData.csv
│   ├── figures/
│   │   ├── roc_auc_comparison.png
│   │   ├── recall_comparison.png
│   │   ├── cluster_risk_rates.png
│   │   └── hierarchical_dendrogram.png
│   └── tables/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Technologies Used

- Python
- pandas, numpy
- scikit-learn
- matplotlib
- scipy

---

## Takeaway

This project demonstrates how combining supervised and unsupervised learning can uncover both predictive signals and latent behavioral structure in social science survey data. Tree-based models were especially effective at identifying individuals at risk for low body satisfaction, while clustering revealed distinct gym-goer archetypes with significantly different risk profiles.
