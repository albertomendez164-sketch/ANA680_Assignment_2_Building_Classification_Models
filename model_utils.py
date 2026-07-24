"""Shared data-loading and evaluation utilities for ANA680 Assignment 1."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

COLUMN_NAMES = [
    "sample_code_number", "clump_thickness", "uniformity_cell_size",
    "uniformity_cell_shape", "marginal_adhesion",
    "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
    "normal_nucleoli", "mitoses", "class"
]

def load_data(filename="breast-cancer-wisconsin.data"):
    path = Path(__file__).resolve().parent / filename
    data = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    data = data.dropna().copy()
    data["bare_nuclei"] = data["bare_nuclei"].astype(int)
    X = data.drop(columns=["sample_code_number", "class"])
    y = data["class"].astype(int)
    return X, y, data

def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

def print_results(model_name, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=[2, 4])
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")
    print("Confusion Matrix (rows=true, columns=predicted; labels=[2, 4]):")
    print(matrix)
    return accuracy, matrix
