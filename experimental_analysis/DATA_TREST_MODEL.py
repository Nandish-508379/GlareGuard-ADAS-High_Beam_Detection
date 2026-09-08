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
import matplotlib
matplotlib.use("Agg")     # Headless-safe backend

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# LOAD MODEL + TEST DATA
# ---------------------------------------------------------
data = joblib.load(os.path.join(BASE_DIR, "highbeam_rf_model.pkl"))
model = data["model"]
classes = list(data["classes"])

X_test = np.load(os.path.join(BASE_DIR, "test_X.npy"))
y_test = np.load(os.path.join(BASE_DIR, "test_y.npy"))

print("Loaded test set:", X_test.shape)
print("Loaded classes:", classes)

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
preds = model.predict(X_test)

# ---------------------------------------------------------
# PRINT METRICS
# ---------------------------------------------------------
acc = accuracy_score(y_test, preds)
print("\n============================")
print("FINAL TEST ACCURACY:", acc)
print("============================\n")

print("CLASSIFICATION REPORT:")
print(classification_report(y_test, preds, target_names=classes))

# ---------------------------------------------------------
# CONFUSION MATRIX (ONLY GRAPH SHOWN)
# ---------------------------------------------------------
cm = confusion_matrix(y_test, preds, labels=classes)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

plt.show()   # <-- Shows the graph (does NOT save)
