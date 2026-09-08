"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
File: DATA_PREPARATION.py
Description:
    Processes annotated regions of interest (ROIs) from MARKERS/ directory,
    extracts photometric intensity descriptors (mean, std, max), a 16-bin
    normalized grayscale histogram, and a 256-bin Local Binary Pattern (LBP)
    micro-texture representation.
    Outputs extracted feature matrix (X), class labels (y), and source metadata.
================================================================================
"""

import cv2
import os
import json
import numpy as np

# -------------------------
# PATH CONFIGURATION (Dynamic & Portable)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "MARKERS")


def extract_lbp_features(gray):
    """
    Computes Local Binary Pattern (LBP) texture representation across an 8-neighborhood.
    Formula:
        LBP(x, y) = SUM_{p=0..7} s(i_p - i_c) * 2^p
        where s(x) = 1 if x >= 0 else 0.
    Returns:
        Normalized 256-bin probability histogram.
    """
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            code = 0
            code |= (gray[i-1, j-1] > center) << 7
            code |= (gray[i-1, j]   > center) << 6
            code |= (gray[i-1, j+1] > center) << 5
            code |= (gray[i,   j+1] > center) << 4
            code |= (gray[i+1, j+1] > center) << 3
            code |= (gray[i+1, j]   > center) << 2
            code |= (gray[i+1, j-1] > center) << 1
            code |= (gray[i,   j-1] > center) << 0
            lbp[i, j] = code

    # 256-bin LBP histogram
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_features(roi):
    """
    Extracts a fused 275-dimensional feature vector from a candidate optical ROI:
      - 3 Photometric Intensity Statistics (mean, std, max)
      - 16 Normalized Grayscale Intensity Histogram Bins
      - 256 Normalized Local Binary Pattern (LBP) Texture Histogram Bins
    """
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (64, 64))

    # 1. Intensity statistics
    mean_intensity = np.mean(roi_gray)
    std_intensity = np.std(roi_gray)
    max_intensity = np.max(roi_gray)

    # 2. 16-bin intensity histogram
    hist = cv2.calcHist([roi_gray], [0], None, [16], [0, 256])
    hist = (hist.flatten() / (np.sum(hist) + 1e-6)).astype("float32")

    # 3. 256-bin LBP texture descriptor
    lbp = extract_lbp_features(roi_gray)

    return np.hstack([mean_intensity, std_intensity, max_intensity, hist, lbp])


def prepare_dataset():
    """Iterate through all annotations in MARKERS, extract features, and save .npy arrays."""
    print("================================================================")
    print("  FEATURE EXTRACTION & DATA PREPARATION PIPELINE")
    print("  Project ID: 002/2025 | Author: M NANDISH")
    print("================================================================")

    if not os.path.exists(DATASET):
        print(f"Error: Dataset directory does not exist: {DATASET}")
        return

    X = []
    y = []
    meta = []

    json_files = [f for f in sorted(os.listdir(DATASET)) if f.endswith(".json")]
    total_files = len(json_files)
    print(f"Found {total_files} annotation records in {DATASET}\n")

    for idx, file in enumerate(json_files, 1):
        img_name = file.replace(".json", ".jpg")
        img_path = os.path.join(DATASET, img_name)
        json_path = os.path.join(DATASET, file)

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARNING] Cannot read image {img_name}")
            continue

        h, w, _ = image.shape

        with open(json_path, "r") as f:
            ann_list = json.load(f)

        for region in ann_list:
            x1, y1 = region["x_min"], region["y_min"]
            x2, y2 = region["x_max"], region["y_max"]

            # Fix inverted coordinates if any
            if x2 < x1: x1, x2 = x2, x1
            if y2 < y1: y1, y2 = y2, y1

            # Clamp boundaries to valid image dimensions
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            # Skip invalid or tiny ROIs
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                continue

            roi = image[y1:y2, x1:x2]
            if roi is None or roi.size == 0:
                continue

            feats = extract_features(roi)
            X.append(feats)
            y.append(region["class"])
            meta.append(img_name)

        if idx % 10 == 0 or idx == total_files:
            print(f"Processed [{idx:3d}/{total_files:3d}] images | Total ROIs extracted: {len(X)}")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    meta = np.array(meta)

    # Save to disk
    np.save(os.path.join(BASE_DIR, "features.npy"), X)
    np.save(os.path.join(BASE_DIR, "labels.npy"), y)
    np.save(os.path.join(BASE_DIR, "filenames.npy"), meta)

    print("\n----------------------------------------------------------------")
    print("Dataset Preparation Summary:")
    print(f"  Total ROI Samples Extracted : {len(X)}")
    print(f"  Feature Dimensions per ROI  : {X.shape[1]}")
    unique_cls, counts = np.unique(y, return_counts=True)
    for c, cnt in zip(unique_cls, counts):
        print(f"    - {c:12s}: {cnt} samples ({cnt/len(y)*100:.1f}%)")
    print(f"  Saved files: features.npy, labels.npy, filenames.npy")
    print("----------------------------------------------------------------\n")


if __name__ == "__main__":
    prepare_dataset()
