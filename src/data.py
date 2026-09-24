"""
data.py — Data Loading, Splitting, and Preprocessing for UCI HAR Dataset

Frozen data protocol (R0):
  - Dataset: UCI HAR Using Smartphones v1.0
  - Split: Subject-aware (original 21/9 subject split)
  - Preprocessing: fit on train only, transform val/test
  - Random seed: 42
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

# === Configuration (Frozen at R0) ===
RANDOM_SEED = 42
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Activity labels
ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def download_dataset():
    """Download UCI HAR dataset if not already present."""
    import urllib.request
    import zipfile

    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    zip_path = DATA_DIR / "har.zip"

    if (RAW_DIR / "UCI HAR Dataset").exists():
        print("Dataset already downloaded.")
        return

    print(f"Downloading UCI HAR dataset from {url}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    zip_path.unlink()
    print("Done.")


def load_data(subset="train"):
    """
    Load UCI HAR data for a given subset ('train' or 'test').

    Returns:
        X: np.ndarray of shape (n_samples, 561) — feature vectors
        y: np.ndarray of shape (n_samples,) — activity labels (1-6)
        subjects: np.ndarray of shape (n_samples,) — subject IDs
    """
    base = RAW_DIR / "UCI HAR Dataset" / subset

    X = pd.read_csv(base / "X_{}.txt".format(subset), sep=r"\s+", header=None).values
    y = pd.read_csv(base / "y_{}.txt".format(subset), header=None).values.ravel()
    subjects = pd.read_csv(
        base / "subject_{}.txt".format(subset), header=None
    ).values.ravel()

    return X, y, subjects


def get_train_test_data():
    """
    Load and return train/test data with subject-aware split.

    Returns:
        X_train, y_train, X_test, y_test
    """
    X_train, y_train, _ = load_data("train")
    X_test, y_test, _ = load_data("test")

    return X_train, y_train, X_test, y_test


def get_feature_names():
    """Load feature names from the dataset."""
    features_path = RAW_DIR / "UCI HAR Dataset" / "features.txt"
    features = pd.read_csv(features_path, sep=r"\s+", header=None, names=["idx", "name"])
    return features["name"].tolist()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UCI HAR Dataset Manager")
    parser.add_argument("--download", action="store_true", help="Download the dataset")
    parser.add_argument("--info", action="store_true", help="Print dataset info")
    args = parser.parse_args()

    if args.download:
        download_dataset()

    if args.info or not args.download:
        try:
            X_train, y_train, X_test, y_test = get_train_test_data()
            print(f"Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            print(f"Test:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
            print(f"Classes: {np.unique(y_train)}")
            print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
            print(f"Test class distribution:  {dict(zip(*np.unique(y_test, return_counts=True)))}")
        except FileNotFoundError:
            print("Dataset not found. Run: python src/data.py --download")
