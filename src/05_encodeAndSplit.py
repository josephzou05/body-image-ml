import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# load labeled data
df = pd.read_csv("outputs/labeledData.csv")

# separate features and label
X = df.drop(columns=["bodySatisfactionScore", "atRisk"])
y = df["atRisk"]

# identify numeric and categorical columns
numericCols = ["age"]
categoricalCols = [col for col in X.columns if col not in numericCols]

# define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numericCols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categoricalCols)
    ]
)

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# fit preprocessor on training data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# save processed arrays
pd.DataFrame(X_train_processed.toarray()).to_csv("outputs/X_train.csv", index=False)
pd.DataFrame(X_test_processed.toarray()).to_csv("outputs/X_test.csv", index=False)
y_train.to_csv("outputs/y_train.csv", index=False)
y_test.to_csv("outputs/y_test.csv", index=False)

print("Encoded and split data saved to outputs/")
