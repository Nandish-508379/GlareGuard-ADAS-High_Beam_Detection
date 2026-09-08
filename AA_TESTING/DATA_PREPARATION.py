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

import cv2
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.stats import entropy as scipy_entropy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
import time

# ---------------------------------------------------------
# PATHS (Dynamic & Portable)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "MARKERS")
FIG_OUT = os.path.join(BASE_DIR, "FIGURES")
PCA_OUT = os.path.join(BASE_DIR, "PCA_REPORT")

os.makedirs(FIG_OUT, exist_ok=True)
os.makedirs(PCA_OUT, exist_ok=True)

# ---------------------------------------------------------
# PRETTY PROGRESS BAR
# ---------------------------------------------------------
def progress(i, total, msg=""):
    p = (i / total)
    bar_len = 40
    filled = int(p * bar_len)
    bar = "█" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r[{bar}] {p*100:5.1f}% | {msg}")
    sys.stdout.flush()

# ---------------------------------------------------------
# FEATURE FUNCTIONS
# ---------------------------------------------------------
def extract_lbp(gray):
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp, bins=59, range=(0, 60))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return lbp, hist

def compute_entropy(gray):
    h, _ = np.histogram(gray, bins=256, range=[0, 256])
    h = h / (h.sum() + 1e-6)
    return scipy_entropy(h)

def compute_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_gradient_strength(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return np.mean(cv2.magnitude(gx, gy))

def compute_bright_ratio(gray):
    return np.count_nonzero(gray > 220) / (gray.size + 1e-6)

def extract_features(roi):
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

    mean_int = roi_gray.mean()
    std_int  = roi_gray.std()
    max_int  = roi_gray.max()

    hist16 = cv2.calcHist([roi_gray], [0], None, [16], [0, 256]).flatten()
    hist16 /= (hist16.sum() + 1e-6)

    lbp_img, lbp_hist = extract_lbp(roi_gray)

    feat = np.hstack([
        mean_int, std_int, max_int,
        compute_entropy(roi_gray),
        compute_sharpness(roi_gray),
        compute_gradient_strength(roi_gray),
        compute_bright_ratio(roi_gray),
        hist16,
        lbp_hist
    ])

    return feat, roi_gray, lbp_img, hist16, lbp_hist

# ======================================================
# PHASE 1 — PROCESS IMAGES + MAKE FIGURES
# ======================================================
X, y, meta = [], [], []
clahe = cv2.createCLAHE(3.0, (8, 8))

files = [f for f in os.listdir(DATASET) if f.endswith(".json")]
total_files = len(files)

print("\nStarting processing...\n")

for idx, file in enumerate(files):
    progress(idx, total_files, msg=f"Loading {file}")
    time.sleep(0.02)

    img_name = file.replace(".json", ".jpg")
    img_path = os.path.join(DATASET, img_name)
    json_path = os.path.join(DATASET, file)

    progress(idx, total_files, msg="Reading Image")
    image = cv2.imread(img_path)
    if image is None:
        print(f"\n[WARNING] Missing image file {img_name}\n")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)

    progress(idx, total_files, msg="Reading JSON")
    with open(json_path, "r") as f:
        ann = json.load(f)

    n_roi = len(ann)
    total_rows = 2 + n_roi

    fig, axs = plt.subplots(total_rows, 4, figsize=(18, 3 + 2.7*n_roi))
    axs = axs.reshape(total_rows, 4)

    axs[0, 0].set_title("Original")
    axs[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0, 0].axis("off")

    axs[0, 1].set_title("CLAHE")
    axs[0, 1].imshow(clahe_img, cmap="gray")
    axs[0, 1].axis("off")

    overlay = image.copy()
    for r in ann:
        cv2.rectangle(overlay, (r["x_min"], r["y_min"]), (r["x_max"], r["y_max"]), (0,255,0), 2)

    axs[0, 2].set_title("ROI Map")
    axs[0, 2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[0, 2].axis("off")

    axs[0, 3].axis("off")

    row = 1
    for r in ann:
        progress(idx, total_files, msg="Extracting ROI + Features")

        x1, y1, x2, y2 = r["x_min"], r["y_min"], r["x_max"], r["y_max"]
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        feats, roi_gray, lbp_img, hist16, lbp_hist = extract_features(roi)

        axs[row, 0].set_title(f"ROI — {r['class']}")
        axs[row, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        axs[row, 0].axis("off")

        axs[row, 1].set_title("LBP")
        axs[row, 1].imshow(lbp_img, cmap="gray")
        axs[row, 1].axis("off")

        axs[row, 2].set_title("Feature Table")
        axs[row, 2].axis("off")

        text = (
            f"Mean: {feats[0]:.2f}\n"
            f"Std: {feats[1]:.2f}\n"
            f"Max: {feats[2]:.2f}\n"
            f"Entropy: {feats[3]:.4f}\n"
            f"Sharpness: {feats[4]:.2f}\n"
            f"Gradient: {feats[5]:.2f}\n"
            f"Bright Ratio: {feats[6]:.4f}"
        )

        axs[row, 2].text(0.05, 0.95, text, fontsize=9, family="monospace", va="top")

        axs[row, 3].set_title("Histograms")
        axs[row, 3].plot(hist16, label="16-bin Intensity")
        axs[row, 3].plot(lbp_hist, label="LBP 59-bin")
        axs[row, 3].legend(fontsize=7)
        axs[row, 3].grid(False)

        X.append(feats)
        y.append(r["class"])
        meta.append(img_name)

        row += 1

    progress(idx, total_files, msg="Saving Figure")
    save_path = os.path.join(FIG_OUT, img_name.replace(".jpg", "_FIG.png"))
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ======================================================
# SAVE RAW FEATURES (.npy)
# ======================================================
progress(total_files, total_files, msg="Saving Feature Arrays")

X = np.array(X)
y = np.array(y)
meta = np.array(meta)

np.save(os.path.join(PCA_OUT, "features.npy"), X)
np.save(os.path.join(PCA_OUT, "labels.npy"), y)
np.save(os.path.join(PCA_OUT, "filenames.npy"), meta)

time.sleep(0.5)

# ======================================================
# FULL NORMALIZED FEATURE CSV (81 features, all images)
# ======================================================
print("\nCreating FULL_NORMALIZED_FEATURES.csv ...")

scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

num_features = X_norm.shape[1]
feature_cols = [f"F{i+1}" for i in range(num_features)]

df_full = pd.DataFrame(X_norm, columns=feature_cols)
df_full["label"] = y
df_full["filename"] = meta

csv_path = os.path.join(PCA_OUT, "FULL_NORMALIZED_FEATURES.csv")
df_full.to_csv(csv_path, index=False)

print("Saved:", csv_path)

# ======================================================
# PCA REPORT
# ======================================================
progress(total_files, total_files, msg="Running PCA...")
time.sleep(0.5)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

df_report = pd.DataFrame({
    "filename": meta,
    "label": y,
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1]
})
df_report.to_csv(os.path.join(PCA_OUT, "PCA_REPORT.csv"), index=False)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1_loading", "PC2_loading"]
)
loadings.to_csv(os.path.join(PCA_OUT, "PCA_LOADINGS.csv"), index=False)

var_df = pd.DataFrame({
    "PC": ["PC1", "PC2"],
    "Explained Variance": pca.explained_variance_ratio_
})
var_df.to_csv(os.path.join(PCA_OUT, "PCA_VARIANCE.csv"), index=False)

progress(total_files, total_files, msg="DONE ✔")
print("\nProcessing complete.\n")
