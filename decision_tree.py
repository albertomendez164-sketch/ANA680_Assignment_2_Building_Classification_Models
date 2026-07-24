from sklearn.tree import DecisionTreeClassifier
from model_utils import load_data, split_data, print_results

X, y, _ = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
print_results("Decision Tree", y_test, model.predict(X_test))
