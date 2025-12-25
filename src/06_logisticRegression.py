import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# load encoded train and test data
X_train = pd.read_csv("outputs/X_train.csv")
X_test = pd.read_csv("outputs/X_test.csv")
y_train = pd.read_csv("outputs/y_train.csv").values.ravel()
y_test = pd.read_csv("outputs/y_test.csv").values.ravel()

# train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# generate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# evaluate model performance
print("Classification report:")
print(classification_report(y_test, y_pred))

print("ROC-AUC score:")
print(roc_auc_score(y_test, y_prob))
