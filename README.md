# ANA680 – Assignment Week one
## Breast Cancer Classification Using Machine Learning

### Overview
This project implements eight supervised machine learning classification algorithms using the Breast Cancer Wisconsin dataset from the UCI Machine Learning Repository.

The objective is to compare the performance of several classification models using a 75% training and 25% testing split.

Dataset:
- Breast_Cancer_Data.csv
- Approximately 683 patient records
- 10 predictor features
- Target Class:
  - 2 = Benign
  - 4 = Malignant

---

## Models Implemented

1. Logistic Regression
2. K-Nearest Neighbors (k = 5)
3. Linear Support Vector Machine
4. Kernel Support Vector Machine (RBF)
5. Gaussian Naïve Bayes
6. Decision Tree
7. Random Forest (10 estimators)
8. XGBoost

---

## Software Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn xgboost
```

or

```bash
pip install -r requirements.txt
```

---

## Files Included

```
Breast_Cancer_Data.csv
logistic_regression.py
knn.py
linear_svm.py
kernel_svm.py
naive_bayes.py
decision_tree.py
random_forest.py
xgboost_model.py
requirements.txt
README.md
```

---

## Running the Programs

Run each classifier individually:

```bash
python logistic_regression.py
python knn.py
python linear_svm.py
python kernel_svm.py
python naive_bayes.py
python decision_tree.py
python random_forest.py
python xgboost_model.py
```

Each program prints:
- Classification Accuracy
- Confusion Matrix

---

## Assignment Requirements

- Python
- Scikit-learn
- Separate Python file for each classifier
- 75/25 train-test split
- Accuracy reported
- Confusion matrix reported
- Results summarized in a Word document

---

## Author

Alberto Mendez

National University

ANA680 – Machine Learning
