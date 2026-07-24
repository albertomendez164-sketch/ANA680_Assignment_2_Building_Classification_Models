"""XGBoost model for ANA680 Assignment 1."""
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from data_preparation import load_and_split_data


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()

    # XGBoost requires zero-based class labels.
    y_train_binary = y_train.map({2: 0, 4: 1})
    y_test_binary = y_test.map({2: 0, 4: 1})

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train_binary)
    predictions = model.predict(X_test)

    print("=" * 55)
    print("XGBoost")
    print(f"Accuracy: {accuracy_score(y_test_binary, predictions):.4f}")
    print("Confusion Matrix (labels ordered as 0=benign, 1=malignant):")
    print(confusion_matrix(y_test_binary, predictions, labels=[0, 1]))


if __name__ == "__main__":
    main()
