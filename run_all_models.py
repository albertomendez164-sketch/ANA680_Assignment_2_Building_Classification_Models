"""Convenience script that runs all eight model files sequentially."""
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "logistic_regression.py",
    "knn.py",
    "linear_svm.py",
    "kernel_svm.py",
    "naive_bayes.py",
    "decision_tree.py",
    "random_forest.py",
    "xgboost_model.py",
]


def main():
    project_dir = Path(__file__).resolve().parent
    for script in SCRIPTS:
        print(f"\nRunning {script}...")
        subprocess.run(
            [sys.executable, str(project_dir / script)],
            cwd=project_dir,
            check=True,
        )


if __name__ == "__main__":
    main()
