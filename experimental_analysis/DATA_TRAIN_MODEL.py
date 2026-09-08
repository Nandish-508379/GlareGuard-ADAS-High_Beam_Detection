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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load splits
X_train = np.load(os.path.join(BASE_DIR, "train_X.npy"))
y_train = np.load(os.path.join(BASE_DIR, "train_y.npy"))
X_val = np.load(os.path.join(BASE_DIR, "val_X.npy"))
y_val = np.load(os.path.join(BASE_DIR, "val_y.npy"))

print("Training samples:", X_train.shape)
print("Validation samples:", X_val.shape)
print("Feature count per sample:", X_train.shape[1])

# -------------------------------------------------
# BUILD THE RANDOM FOREST MODEL (IMPROVED)
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,                  # limit overfitting
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",       # fixes imbalance
    oob_score=True,                # free validation check
    bootstrap=True,                # required for OOB
    random_state=42,
    n_jobs=-1                      # use all CPU cores
)

print("\nTraining model...")
model.fit(X_train, y_train)

# -------------------------------------------------
# VALIDATION RESULTS
# -------------------------------------------------
val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)

print("\nVALIDATION ACCURACY:", val_acc)
print("\nDETAILED REPORT:")
print(classification_report(y_val, val_preds))

# -------------------------------------------------
# OUT-OF-BAG SCORE
# -------------------------------------------------
if hasattr(model, "oob_score_"):
    print("\nOOB SCORE (approx validation):", model.oob_score_)

# -------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------
importances = model.feature_importances_
print("\nTop 20 important features:")
top_idx = np.argsort(importances)[::-1][:20]
for i, idx in enumerate(top_idx):
    print(f"{i+1}. Feature[{idx}] = {importances[idx]:.4f}")

# -------------------------------------------------
# SAVE MODEL + LABELS
# -------------------------------------------------
joblib.dump({
    "model": model,
    "classes": np.unique(y_train)
}, os.path.join(BASE_DIR, "highbeam_rf_model.pkl"))

print("\nModel saved as highbeam_rf_model.pkl")
