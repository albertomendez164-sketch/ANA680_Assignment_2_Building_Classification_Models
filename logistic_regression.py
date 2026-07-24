from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from model_utils import load_data, split_data, print_results

X, y, _ = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = Pipeline([("scaler", StandardScaler()),
                  ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
model.fit(X_train, y_train)
print_results("Logistic Regression", y_test, model.predict(X_test))
