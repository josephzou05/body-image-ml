import pandas as pd

# Load cleaned data
df = pd.read_csv("outputs/cleanData.csv")

# Columns to keep for ML
keepColumns = [
    "age",
    "gender",
    "gymFrequency",
    "fitnessGoals",
    "trackProgress",
    "followInfluencers",
    "comparePhysique",
    "needMoreMuscle",
    "feelTooSmall",
    "gymPressure",
    "bodyChecking",
    "avoidPostingPics",
    "bodySatisfactionScore"
]

df = df[keepColumns]

# Save feature-selected data
df.to_csv("outputs/modelData.csv", index=False)

print("Model-ready data saved to outputs/modelData.csv")
print("Columns used:")
print(df.columns.tolist())
