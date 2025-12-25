import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

# load model-ready data
df = pd.read_csv("outputs/modelData.csv")

# separate features and outcome
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

# fit k-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_processed)

# attach cluster labels
df["cluster"] = clusters

# save clustered data
df.to_csv("outputs/clusteredData.csv", index=False)

print("K-means clustering complete")
print(df["cluster"].value_counts())
