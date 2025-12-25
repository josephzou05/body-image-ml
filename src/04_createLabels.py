import pandas as pd

# load model-ready data
df = pd.read_csv("outputs/modelData.csv")

# create classification label
riskThreshold = df["bodySatisfactionScore"].quantile(0.25)
df["atRisk"] = (df["bodySatisfactionScore"] <= riskThreshold).astype(int)

# sanity checks
print("At-risk threshold:", riskThreshold)
print(df["atRisk"].value_counts())
print(df.groupby("atRisk")["bodySatisfactionScore"].mean())

# save labeled data
df.to_csv("outputs/labeledData.csv", index=False)
