import json
from pathlib import Path
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from model_utils import load_data, split_data

X, y, cleaned = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=1000, random_state=42))]),
    "KNN (k=5)": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "Linear SVM": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="linear", random_state=42))]),
    "Kernel SVM (RBF)": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", random_state=42))]),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest (10 estimators)": RandomForestClassifier(n_estimators=10, random_state=42),
}
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[2,4])
    results.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                    "TN": int(cm[0,0]), "FP": int(cm[0,1]),
                    "FN": int(cm[1,0]), "TP": int(cm[1,1]),
                    "Confusion Matrix": cm.tolist()})

yb = y.map({2:0,4:1})
Xtr, Xte, ytr, yte = split_data(X, yb)
xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                    random_state=42, eval_metric="logloss")
xgb.fit(Xtr, ytr)
pred = xgb.predict(Xte)
cm = confusion_matrix(yte, pred, labels=[0,1])
results.append({"Model": "XGBoost", "Accuracy": accuracy_score(yte, pred),
                "TN": int(cm[0,0]), "FP": int(cm[0,1]),
                "FN": int(cm[1,0]), "TP": int(cm[1,1]),
                "Confusion Matrix": cm.tolist()})

out = Path(__file__).resolve().parent
pd.DataFrame(results).to_csv(out / "model_results.csv", index=False)
(out / "model_results.json").write_text(json.dumps({
    "original_rows": 699, "usable_rows": len(cleaned),
    "training_rows": len(X_train), "testing_rows": len(X_test),
    "random_state": 42, "stratified": True, "results": results
}, indent=2), encoding="utf-8")
print(pd.DataFrame(results).to_string(index=False))
