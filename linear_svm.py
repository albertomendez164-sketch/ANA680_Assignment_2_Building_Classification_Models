from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from model_utils import load_data, split_data, print_results

X, y, _ = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = Pipeline([("scaler", StandardScaler()),
                  ("classifier", SVC(kernel="linear", random_state=42))])
model.fit(X_train, y_train)
print_results("Linear SVM", y_test, model.predict(X_test))
