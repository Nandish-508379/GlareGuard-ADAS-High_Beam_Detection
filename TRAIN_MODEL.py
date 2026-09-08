"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
File: TRAIN_MODEL.py
Description:
    Trains an ensemble Random Forest Classifier on the 275-dimensional
    photometric and LBP texture features extracted from training samples.
    Evaluates generalization on the held-out validation set and serializes
    the trained model to highbeam_rf_model.pkl.
================================================================================
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def train():
    print("================================================================")
    print("  RANDOM FOREST MODEL TRAINING & VALIDATION")
    print("  Project ID: 002/2025 | Author: M NANDISH")
    print("================================================================")

    train_X_path = os.path.join(BASE_DIR, "train_X.npy")
    train_y_path = os.path.join(BASE_DIR, "train_y.npy")
    val_X_path = os.path.join(BASE_DIR, "val_X.npy")
    val_y_path = os.path.join(BASE_DIR, "val_y.npy")

    if not os.path.exists(train_X_path):
        print("Error: Training data not found. Run DATA_SPLIT.py first.")
        return

    # Load splits
    X_train = np.load(train_X_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_X_path)
    y_val = np.load(val_y_path)

    print(f"Training samples   : {X_train.shape[0]} (Feature dimension: {X_train.shape[1]})")
    print(f"Validation samples : {X_val.shape[0]}")

    # Build the Random Forest Classifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        oob_score=True,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    print("\nFitting Random Forest ensemble (300 estimators, balanced weights)...")
    model.fit(X_train, y_train)

    # Validation evaluation
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)

    print("\n----------------------------------------------------------------")
    print(f"VALIDATION PERFORMANCE (Accuracy: {val_acc*100:.2f}%):")
    print(f"Out-of-Bag (OOB) Score: {model.oob_score_*100:.2f}%")
    print("----------------------------------------------------------------")
    print(classification_report(y_val, val_preds, digits=4))

    # Save trained model
    model_path = os.path.join(BASE_DIR, "highbeam_rf_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model successfully saved to: {model_path}\n")


if __name__ == "__main__":
    train()
