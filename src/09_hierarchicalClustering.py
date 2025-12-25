import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.cluster.hierarchy import linkage, dendrogram

# load model-ready data
df = pd.read_csv("outputs/modelData.csv")

# separate features
X = df.drop(columns=["bodySatisfactionScore"])

# identify numeric and categorical columns
numericCols = ["age"]
categoricalCols = [col for col in X.columns if col not in numericCols]

# preprocess features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numericCols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categoricalCols)
    ]
)

X_processed = preprocessor.fit_transform(X)

# perform hierarchical clustering
Z = linkage(X_processed.toarray(), method="ward")

# plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Individuals")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()
