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
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern
from scipy.stats import entropy as scipy_entropy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------------------
data = joblib.load(os.path.join(BASE_DIR, "highbeam_rf_model.pkl"))
model = data["model"]
classes = list(data["classes"])

# ------------------------------------------------------------
# FEATURE FUNCTIONS (MUST MATCH TRAINING EXACTLY)
# ------------------------------------------------------------
def extract_lbp(gray):
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 59))
    hist = hist.astype("float32"); hist /= (hist.sum() + 1e-6)
    return lbp, hist

def compute_entropy(gray):
    h, _ = np.histogram(gray.flatten(), bins=256, range=[0, 256])
    h = h / (h.sum() + 1e-6)
    return scipy_entropy(h)

def compute_sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def compute_gradient(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.mean(cv2.magnitude(gx, gy))

def compute_bright_ratio(gray):
    return np.sum(gray > 220) / (gray.size + 1e-6)

# ------------------------------------------------------------
# MAIN FEATURE EXTRACTOR (MATCHING TRAINING)
# ------------------------------------------------------------
def extract_features(roi):

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (64, 64))

    mean_int = np.mean(roi_gray)
    std_int  = np.std(roi_gray)
    max_int  = np.max(roi_gray)

    hist16 = cv2.calcHist([roi_gray], [0], None, [16], [0, 256])
    hist16 = hist16.flatten() / (hist16.sum() + 1e-6)

    lbp_img, lbp_hist = extract_lbp(roi_gray)

    ent = compute_entropy(roi_gray)
    sharp = compute_sharpness(roi_gray)
    grad = compute_gradient(roi_gray)
    bright_ratio = compute_bright_ratio(roi_gray)

    return np.hstack([
        mean_int, std_int, max_int,
        ent, sharp, grad, bright_ratio,
        hist16,
        lbp_hist
    ])

# ------------------------------------------------------------
# DETECT BRIGHT REGIONS (SAME AS BEFORE)
# ------------------------------------------------------------
def detect_bright_regions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:
            continue
        boxes.append((x, y, x+w, y+h))
    return boxes

# ------------------------------------------------------------
# FULL INFERENCE PIPELINE (CORRECT)
# ------------------------------------------------------------
def predict_on_image(img_path):

    image = cv2.imread(img_path)
    boxes = detect_bright_regions(image)

    for (x1, y1, x2, y2) in boxes:
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        feats = extract_features(roi).reshape(1, -1)
        pred = model.predict(feats)[0]

        label = "HIGH BEAM" if pred == "high_beam" else "GLARE"
        color = (0, 255, 0) if pred == "high_beam" else (0, 128, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("High-Beam Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------------------------------------
# RUN PREDICTION
# ------------------------------------------------------------
predict_on_image("unseen_image.jpg")
