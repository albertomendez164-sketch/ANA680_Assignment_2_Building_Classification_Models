# ANA680 Assignment Week One: Breast Cancer Classification

## Overview
This repository contains the code and results for ANA680 Assignment 1. Eight classification models were trained and evaluated on the Wisconsin Breast Cancer (Original) dataset. Each required classifier is implemented in a separate Python file.

## Dataset
The dataset was originally obtained from the UCI Machine Learning Repository and donated by Dr. William H. Wolberg of the University of Wisconsin Hospitals.

- Original records: **699**
- Records with complete data used for modeling: **683**
- Predictor variables: **9**
- Target labels: **2 = benign**, **4 = malignant**
- Missing values: **16 rows containing `?` were removed**
- The sample code number was excluded because it is an identifier rather than a predictive feature.

## Experimental Setup
- Training set: **75% (512 records)**
- Test set: **25% (171 records)**
- Split: stratified by class
- Random seed: `42`
- Scaling: StandardScaler for Logistic Regression, KNN, Linear SVM, and RBF SVM

## Models
1. Logistic Regression
2. K-Nearest Neighbors (`k = 5`)
3. Linear SVM (`kernel = linear`)
4. Kernel SVM (`kernel = rbf`)
5. Gaussian Naive Bayes
6. Decision Tree
7. Random Forest (`n_estimators = 10`)
8. XGBoost

## Results
| Model | Accuracy | Confusion Matrix |
|---|---:|---|
| Logistic Regression | 95.91% | `[[106, 5], [2, 58]]` |
| KNN (k=5) | 95.32% | `[[106, 5], [3, 57]]` |
| Linear SVM | 95.91% | `[[106, 5], [2, 58]]` |
| Kernel SVM (RBF) | 96.49% | `[[106, 5], [1, 59]]` |
| Naive Bayes | 95.91% | `[[106, 5], [2, 58]]` |
| Decision Tree | 95.91% | `[[105, 6], [1, 59]]` |
| Random Forest (10 estimators) | 95.91% | `[[106, 5], [2, 58]]` |
| XGBoost | 97.08% | `[[106, 5], [0, 60]]` |

The confusion matrices use rows as actual classes and columns as predicted classes. The order is benign first and malignant second.

## Repository Files
- Individual model scripts: one file per required classifier
- `model_utils.py`: common data loading, cleaning, splitting, and reporting functions
- `run_all_models.py`: runs all eight models and exports the combined results
- `model_results.csv` and `model_results.json`: machine-readable results
- `ANA680_Assignment1_Final_Report.docx`: formatted assignment report
- `requirements.txt`: required Python packages

## Installation
```bash
pip install -r requirements.txt
```

## Running the Models
Run any classifier individually, for example:
```bash
python logistic_regression.py
```

Run every classifier and regenerate the result files:
```bash
python run_all_models.py
```

## Dataset Citation
Wolberg, W. H., and Mangasarian, O. L. (1990). Multisurface method of pattern separation for medical diagnosis applied to breast cytology. *Proceedings of the National Academy of Sciences, 87*, 9193-9196.

UCI Machine Learning Repository. Wisconsin Breast Cancer (Original) dataset.

## Author
Alberto Mendez  
National University  
ANA680

