"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
File: DATA_SPLIT.py
Description:
    Performs stratified two-stage train-validation-test partitioning of the
    extracted feature dataset (70% Training, 10% Validation, 20% Testing).
    Stratification preserves identical class ratios across all partitions to
    prevent class distribution skewing during model training and evaluation.
================================================================================
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def split_dataset():
    print("================================================================")
    print("  STRATIFIED DATASET PARTITIONING")
    print("  Project ID: 002/2025 | Author: M NANDISH")
    print("================================================================")

    feat_path = os.path.join(BASE_DIR, "features.npy")
    lbl_path = os.path.join(BASE_DIR, "labels.npy")

    if not os.path.exists(feat_path) or not os.path.exists(lbl_path):
        print("Error: features.npy or labels.npy not found. Run DATA_PREPARATION.py first.")
        return

    X = np.load(feat_path)
    y = np.load(lbl_path)

    print(f"Total dataset size: {len(X)} samples, {X.shape[1]} features")

    # STAGE 1: Split into Train (70%) and Temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # STAGE 2: Split Temp (30%) into Validation (10% of total) and Test (20% of total)
    # test_size = 20 / 30 = 0.6666667
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.66, random_state=42, stratify=y_temp
    )

    # Save splits to disk
    np.save(os.path.join(BASE_DIR, "train_X.npy"), X_train)
    np.save(os.path.join(BASE_DIR, "train_y.npy"), y_train)
    np.save(os.path.join(BASE_DIR, "val_X.npy"), X_val)
    np.save(os.path.join(BASE_DIR, "val_y.npy"), y_val)
    np.save(os.path.join(BASE_DIR, "test_X.npy"), X_test)
    np.save(os.path.join(BASE_DIR, "test_y.npy"), y_test)

    print("\nPartitioning Complete:")
    print(f"  Training Split   (70%) : {len(X_train)} samples")
    print(f"  Validation Split (10%) : {len(X_val)} samples")
    print(f"  Test Split       (20%) : {len(X_test)} samples")

    print("\nClass Distribution per Partition:")
    for split_name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(labels, return_counts=True)
        dist_str = ", ".join([f"{u}: {c} ({c/len(labels)*100:.1f}%)" for u, c in zip(unique, counts)])
        print(f"  {split_name:6s} -> {dist_str}")
    print("----------------------------------------------------------------\n")


if __name__ == "__main__":
    split_dataset()
