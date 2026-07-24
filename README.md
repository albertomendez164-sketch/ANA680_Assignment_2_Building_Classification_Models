ANA680 Assignment: Breast Cancer Classification
This repository contains eight classification models for the Wisconsin Breast Cancer dataset. Each required model is implemented in a separate Python file.
Models
Logistic Regression
K-Nearest Neighbors (`k = 5`)
Linear Support Vector Machine
RBF Kernel Support Vector Machine
Gaussian Naive Bayes
Decision Tree
Random Forest (`n_estimators = 10`)
XGBoost
Assignment Settings
Test set: 25% of the cleaned dataset
Training set: 75%
Split seed: `random_state=42`
Stratified split: Yes
Original labels: `2 = benign`, `4 = malignant`
Missing UCI values represented by `?` are removed during preprocessing.
Common sample/patient ID columns are excluded from the predictors.
Setup
Replace the placeholder `Breast_Cancer_Data.csv` with the CSV supplied by the instructor.
Open a terminal in this project folder.
Create and activate a virtual environment if desired.
Install the packages:
```bash
python -m pip install -r requirements.txt
```
Run Each Required File
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
A convenience script is also provided:
```bash
python run_all_models.py
```
Record the printed accuracy and 2-by-2 confusion matrix in `ANA680_Assignment1_Results_Template.docx`.
Confusion Matrix Layout
For the scikit-learn models, labels are ordered as `[2, 4]`:
```text
[[True Benign,  Benign predicted as Malignant],
 [Malignant predicted as Benign, True Malignant]]
```
For XGBoost, the labels are converted to `0 = benign` and `1 = malignant`, but the matrix positions have the same interpretation.
GitHub Submission
Create a new GitHub repository and upload all project files. Do not forget to replace the placeholder dataset before running the models. Include the completed Word results document in the repository if permitted by the instructor.
