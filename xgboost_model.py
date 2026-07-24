from xgboost import XGBClassifier
from model_utils import load_data, split_data, print_results

X, y, _ = load_data()
y_binary = y.map({2: 0, 4: 1})
X_train, X_test, y_train, y_test = split_data(X, y_binary)
model = XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    random_state=42, eval_metric="logloss"
)
model.fit(X_train, y_train)
pred_binary = model.predict(X_test)
y_test_original = y_test.map({0: 2, 1: 4})
pred_original = [2 if value == 0 else 4 for value in pred_binary]
print_results("XGBoost", y_test_original, pred_original)
