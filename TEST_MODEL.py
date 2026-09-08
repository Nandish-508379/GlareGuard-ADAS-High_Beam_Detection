"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
File: TEST_MODEL.py
Description:
    Evaluates the finalized Random Forest model on the untouched test partition
    (99 unseen ROI samples). Computes strict multi-class classification
    metrics (Precision, Recall, F1-Score, Overall Accuracy) and outputs the
    Confusion Matrix.
================================================================================
"""

import os
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def test():
    print("================================================================")
    print("  FINAL MODEL EVALUATION ON UNTOUCHED TEST SET")
    print("  Project ID: 002/2025 | Author: M NANDISH")
    print("================================================================")

    model_path = os.path.join(BASE_DIR, "highbeam_rf_model.pkl")
    test_x_path = os.path.join(BASE_DIR, "test_X.npy")
    test_y_path = os.path.join(BASE_DIR, "test_y.npy")

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run TRAIN_MODEL.py first.")
        return

    # Load test data and trained model
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    model = joblib.load(model_path)

    classes = list(getattr(model, "classes_", np.unique(y_test)))

    # Run predictions
    test_preds = model.predict(X_test)
    acc = accuracy_score(y_test, test_preds)

    print(f"Test Samples Loaded : {len(y_test)} (Dimensions: {X_test.shape[1]})")
    print(f"Classes Evaluated   : {classes}")
    print("\n----------------------------------------------------------------")
    print(f"OVERALL TEST ACCURACY: {acc * 100:.2f}%")
    print("----------------------------------------------------------------")
    print(classification_report(y_test, test_preds, digits=4))

    cm = confusion_matrix(y_test, test_preds, labels=classes)
    print("CONFUSION MATRIX:")
    header = f"{'True \\ Pred':<14} | " + " | ".join([f"{c:<10}" for c in classes])
    print(header)
    print("-" * len(header))
    for i, row_cls in enumerate(classes):
        row_vals = " | ".join([f"{cm[i, j]:<10d}" for j in range(len(classes))])
        print(f"{row_cls:<14} | {row_vals}")
    print("----------------------------------------------------------------\n")


if __name__ == "__main__":
    test()
