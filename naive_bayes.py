"""Naive Bayes model for ANA680 Assignment 1."""
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from data_preparation import load_and_split_data


def main():
    X_train, X_test, y_train, y_test = load_and_split_data()
    model = GaussianNB()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("=" * 55)
    print("Naive Bayes")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("Confusion Matrix (labels ordered as 2=benign, 4=malignant):")
    print(confusion_matrix(y_test, predictions, labels=[2, 4]))


if __name__ == "__main__":
    main()
