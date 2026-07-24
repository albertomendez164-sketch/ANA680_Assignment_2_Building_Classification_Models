"""Shared data-loading utilities for ANA680 Assignment 1."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).with_name("Breast_Cancer_Data.csv")


def load_and_split_data():
    """Load, clean, and split the breast cancer dataset.

    The function supports the common UCI Wisconsin format, including an ID
    column and missing values represented by question marks.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE.name}. Place it in the project folder."
        )

    data = pd.read_csv(DATA_FILE, na_values=["?", "NA", "N/A", ""])
    data.columns = [str(column).strip() for column in data.columns]

    # Remove fully empty rows and columns.
    data = data.dropna(axis=0, how="all").dropna(axis=1, how="all")

    # Locate the target column. If no common name is found, use the last column.
    normalized = {column.lower().replace(" ", "_"): column for column in data.columns}
    target_column = None
    for candidate in ("class", "diagnosis", "target", "label"):
        if candidate in normalized:
            target_column = normalized[candidate]
            break
    if target_column is None:
        target_column = data.columns[-1]

    # Remove common patient/sample identifier columns from the predictors.
    id_columns = [
        column for column in data.columns
        if column != target_column
        and column.lower().replace(" ", "_")
        in {"id", "sample_code_number", "sample_id", "patient_id"}
    ]

    data = data.drop(columns=id_columns)
    data = data.apply(pd.to_numeric, errors="coerce").dropna()

    X = data.drop(columns=[target_column])
    y = data[target_column].astype(int)

    # Keep original labels 2 (benign) and 4 (malignant) for scikit-learn models.
    return train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
