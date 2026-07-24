from sklearn.ensemble import RandomForestClassifier
from model_utils import load_data, split_data, print_results

X, y, _ = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
print_results("Random Forest (10 estimators)", y_test, model.predict(X_test))
