from sklearn.naive_bayes import GaussianNB
from model_utils import load_data, split_data, print_results

X, y, _ = load_data()
X_train, X_test, y_train, y_test = split_data(X, y)
model = GaussianNB()
model.fit(X_train, y_train)
print_results("Naive Bayes", y_test, model.predict(X_test))
