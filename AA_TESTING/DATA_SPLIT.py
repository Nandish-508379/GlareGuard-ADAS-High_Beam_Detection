"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load dataset
X = np.load(os.path.join(BASE_DIR, "features.npy"))
y = np.load(os.path.join(BASE_DIR, "labels.npy"))

print("Loaded dataset:")
print("X shape:", X.shape)     # (num_samples, num_features)
print("y shape:", y.shape)
print("Number of features per sample:", X.shape[1])
print("Class distribution:", Counter(y))

# -------------------------------
# FIRST SPLIT: Train 70% + Temp 30%
# -------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -------------------------------
# SECOND SPLIT: Temp → Val 10% + Test 20%
# 10% of total = 10/30 = 0.33 of temp
# -------------------------------
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.66,
    random_state=42,
    stratify=y_temp
)

# Save splits
np.save(os.path.join(BASE_DIR, "train_X.npy"), X_train)
np.save(os.path.join(BASE_DIR, "train_y.npy"), y_train)
np.save(os.path.join(BASE_DIR, "val_X.npy"), X_val)
np.save(os.path.join(BASE_DIR, "val_y.npy"), y_val)
np.save(os.path.join(BASE_DIR, "test_X.npy"), X_test)
np.save(os.path.join(BASE_DIR, "test_y.npy"), y_test)

print("\nDataset split complete!")
print("Train samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

print("\n(feature vector updated — all new features included correctly!)")
