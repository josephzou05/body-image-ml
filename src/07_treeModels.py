import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# load encoded train and test data
X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")
y_train = pd.read_csv("outputs/y_train.csv").values.ravel()
y_test = pd.read_csv("outputs/y_test.csv").values.ravel()

# train decision tree model
treeModel = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
treeModel.fit(X_train, y_train)

# evaluate decision tree
treePred = treeModel.predict(X_test)
treeProb = treeModel.predict_proba(X_test)[:, 1]

print("Decision Tree Results")
print(classification_report(y_test, treePred))
print("ROC-AUC:", roc_auc_score(y_test, treeProb))

# train random forest model
forestModel = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42
)
forestModel.fit(X_train, y_train)

# evaluate random forest
forestPred = forestModel.predict(X_test)
forestProb = forestModel.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results")
print(classification_report(y_test, forestPred))
print("ROC-AUC:", roc_auc_score(y_test, forestProb))
