"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
File: DEPLOYMENT_TESTING.py
Description:
    Real-time inference & deployment pipeline for full night-driving camera
    frames. Implements automated optical hotspot proposal via adaptive
    thresholding, morphological filtering, and contour extraction.
    Extracts photometric & LBP texture features per candidate ROI and uses
    the pre-trained Random Forest model to classify each optical source into
    HIGH BEAM (green box) or GLARE (orange box).
    Supports headless execution and automatic output image exporting.
================================================================================
"""

import cv2
import numpy as np
import joblib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "highbeam_rf_model.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# LBP FEATURE FUNCTION (identical to training)
# ----------------------
def extract_lbp_features(gray):
    """Computes normalized 256-bin LBP histogram for 8-neighborhood."""
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

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float32")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_features(roi):
    """Extracts 275-dimensional feature vector (3 intensity stats + 16-bin hist + 256-bin LBP)."""
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (64, 64))

    mean_intensity = np.mean(roi_gray)
    std_intensity = np.std(roi_gray)
    max_intensity = np.max(roi_gray)

    hist = cv2.calcHist([roi_gray], [0], None, [16], [0, 256])
    hist = (hist.flatten() / (np.sum(hist) + 1e-6)).astype("float32")

    lbp = extract_lbp_features(roi_gray)

    return np.hstack([mean_intensity, std_intensity, max_intensity, hist, lbp])


def detect_bright_regions(image):
    """
    Candidate ROI Proposal:
    Applies Gaussian smoothing followed by high-intensity thresholding and
    morphological opening/closing to isolate headlight optical cores and glare blooms.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # Threshold bright optical centers
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

    # Morphological clean-up
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 8 or h < 8:  # filter negligible noise artifacts
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def predict_on_image(img_path, model=None, show_gui=False, save_dir=OUTPUT_DIR):
    """Runs complete end-to-end detection and classification on an input image."""
    if not os.path.exists(img_path):
        print(f"[ERROR] Target image not found: {img_path}")
        return None

    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Run TRAIN_MODEL.py first.")
        model = joblib.load(MODEL_PATH)

    image = cv2.imread(img_path)
    if image is None:
        print(f"[ERROR] Could not decode image: {img_path}")
        return None

    vis_img = image.copy()
    boxes = detect_bright_regions(image)
    print(f"\nProcessing '{os.path.basename(img_path)}': Detected {len(boxes)} candidate optical regions.")

    detections = []
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        features = extract_features(roi).reshape(1, -1)
        pred = model.predict(features)[0]

        # Calculate prediction probabilities if supported
        prob_str = ""
        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            idx = classes.index(pred)
            prob = model.predict_proba(features)[0][idx]
            prob_str = f" ({prob*100:.1f}%)"

        label = "HIGH BEAM" if pred == "high_beam" else "GLARE"
        color = (0, 255, 0) if pred == "high_beam" else (0, 140, 255)  # Green or Vibrant Orange

        # Draw bounding box and stylized text label
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        tag = f"{label}{prob_str}"
        cv2.putText(vis_img, tag, (x1, max(y1 - 10, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        detections.append({
            "box": (x1, y1, x2, y2),
            "label": label,
            "prediction": pred
        })
        print(f"  Region {i}: Box=({x1},{y1})-({x2},{y2}) -> Prediction: {label}{prob_str}")

    # Save output visualization
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(save_dir, f"{base_name}_classified.png")
        cv2.imwrite(out_path, vis_img)
        print(f"  Saved classified result to: {out_path}")

    if show_gui:
        window_name = f"Inference Result: {os.path.basename(img_path)}"
        cv2.imshow(window_name, vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return vis_img, detections


def main():
    print("================================================================")
    print("  REAL-TIME DEPLOYMENT & INFERENCE PIPELINE")
    print("  Project ID: 002/2025 | Author: M NANDISH")
    print("================================================================")

    # Process all available unseen test images
    test_images = [
        os.path.join(BASE_DIR, "unseen_image.jpg"),
        os.path.join(BASE_DIR, "unseen_image_1.jpg"),
        os.path.join(BASE_DIR, "unseen_image_2.jpg")
    ]

    # Check CLI arguments for custom image
    if len(sys.argv) > 1:
        custom_img = sys.argv[1]
        test_images = [custom_img]

    show_gui = "--gui" in sys.argv

    model = joblib.load(MODEL_PATH)
    for img_path in test_images:
        if os.path.exists(img_path):
            predict_on_image(img_path, model=model, show_gui=show_gui, save_dir=OUTPUT_DIR)

    print("\nDeployment testing completed for all test scenes.")
    print(f"All classified visualizations saved in: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
