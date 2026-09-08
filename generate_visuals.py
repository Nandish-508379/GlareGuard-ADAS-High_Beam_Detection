"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
File: generate_visuals.py
Description:
    Generates high-resolution publication-grade visual artifacts:
      1. Flowchart & System Architecture diagram
      2. Confusion Matrix Heatmap (Test set)
      3. ROC Curve & Precision-Recall Curve
      4. Feature Importance Ranking (Random Forest)
      5. Dataset Distribution & Stratified Partitioning Plot
      6. Visual Detections on Unseen Images (unseen_image, unseen_image_1, unseen_image_2)
      7. Unified Evaluation Summary Dashboard
    All figures are saved to assets/ and outputs/ for documentation & GitHub.
================================================================================
"""

import os
import cv2
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for robust script rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# Set global aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. GENERATE SYSTEM FLOWCHART
# ----------------------------------------------------------------------
def generate_system_flowchart():
    """Renders a detailed, high-resolution visual flowchart of the entire pipeline."""
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_facecolor("#0F172A")  # Deep Navy / Slate background
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    # Define stage nodes
    stages = [
        {"id": 0, "x": 0.08, "y": 0.72, "w": 0.16, "h": 0.18, "title": "1. Optical Acquisition\n& Video Frames", "desc": "Night road scenes\n(DATASET/)\nRaw Camera Stream", "color": "#3B82F6"},
        {"id": 1, "x": 0.28, "y": 0.72, "w": 0.16, "h": 0.18, "title": "2. Bright Spot\nProposal", "desc": "Gaussian Blur (11x11)\nBinary Thresh (>200)\nMorph. Erode & Dilate", "color": "#06B6D4"},
        {"id": 2, "x": 0.48, "y": 0.72, "w": 0.18, "h": 0.18, "title": "3. Candidate ROI\nStandardization", "desc": "Bounding Box Crop\n64x64 Resize\nGray Conversion", "color": "#10B981"},
        {"id": 3, "x": 0.70, "y": 0.72, "w": 0.22, "h": 0.18, "title": "4. Feature Extraction\n(275-D Fused Vector)", "desc": "• Photometric: Mean, Std, Max (3)\n• Intensity Hist: 16 bins\n• Texture: 256-bin LBP Operator", "color": "#8B5CF6"},
        
        {"id": 4, "x": 0.70, "y": 0.22, "w": 0.22, "h": 0.18, "title": "5. Random Forest\nEnsemble (300 Trees)", "desc": "Gini Impurity Splits\nBalanced Class Weights\nOut-Of-Bag (OOB) Check", "color": "#EC4899"},
        {"id": 5, "x": 0.44, "y": 0.22, "w": 0.20, "h": 0.18, "title": "6. Optical Source\nClassification", "desc": "Probabilistic Output\n• HIGH BEAM (Focused)\n• GLARE (Diffuse scatter)", "color": "#F59E0B"},
        {"id": 6, "x": 0.14, "y": 0.22, "w": 0.24, "h": 0.18, "title": "7. ADAS Actuation\n& Glare-Guard Control", "desc": "Adaptive Matrix Dimming\nNight Driving Hazard Alert\nAutomatic Headlight Dipper", "color": "#EF4444"}
    ]

    for s in stages:
        rect = patches.FancyBboxPatch(
            (s["x"], s["y"]), s["w"], s["h"],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#1E293B",
            edgecolor=s["color"],
            linewidth=2.5,
            zorder=2
        )
        ax.add_patch(rect)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.70, s["title"],
                color="#FFFFFF", fontsize=11, fontweight="bold", ha="center", va="center", zorder=3)
        ax.text(s["x"] + s["w"]/2, s["y"] + s["h"]*0.30, s["desc"],
                color="#94A3B8", fontsize=9, ha="center", va="center", zorder=3)

    # Connections
    arrows = [
        ((0.24, 0.81), (0.28, 0.81)),
        ((0.44, 0.81), (0.48, 0.81)),
        ((0.66, 0.81), (0.70, 0.81)),
        ((0.81, 0.72), (0.81, 0.40)),  # Downwards from Stage 4 to Stage 5
        ((0.70, 0.31), (0.64, 0.31)),  # Leftwards to Stage 6
        ((0.44, 0.31), (0.38, 0.31))   # Leftwards to Stage 7
    ]

    for (p1, p2) in arrows:
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(facecolor="#38BDF8", edgecolor="#38BDF8",
                                    arrowstyle="-|>", lw=2.2, shrinkA=4, shrinkB=4),
                    zorder=4)

    # Header & Metadata banner
    ax.text(0.5, 0.96, "INTELLIGENT HIGH BEAM VS GLARE CLASSIFICATION PIPELINE",
            color="#F8FAFC", fontsize=16, fontweight="bold", ha="center", va="center")
    ax.text(0.5, 0.92, "Project ID: 002/2025  |  Year: 2025  |  Author: M NANDISH  |  Automotive ADAS Vision",
            color="#38BDF8", fontsize=11, ha="center", va="center")

    out_path = os.path.join(ASSETS_DIR, "system_flowchart.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[GENERATED] Flowchart saved -> {out_path}")


# ----------------------------------------------------------------------
# 2. CONFUSION MATRIX HEATMAP
# ----------------------------------------------------------------------
def generate_confusion_matrix_plot(y_test, preds, classes):
    """Generates an annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_test, preds, labels=classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    # Custom annotations with count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    class_names_disp = ["Glare\n(Low Beam/Diffuse)", "High Beam\n(Focused Direct)"]
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=class_names_disp, yticklabels=class_names_disp,
                annot_kws={"fontsize": 13, "fontweight": "bold"}, ax=ax,
                linewidths=1.5, linecolor="#CBD5E1")

    ax.set_title("Test Confusion Matrix (Project 002/2025)\nAuthor: M NANDISH",
                 fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold", labelpad=10)

    acc = accuracy_score(y_test, preds)
    fig.text(0.5, 0.01, f"Overall Accuracy: {acc*100:.2f}% | Total Test Samples: {len(y_test)}",
             ha="center", fontsize=10, style="italic", color="#334155")

    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, "confusion_matrix.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[GENERATED] Confusion Matrix saved -> {out_path}")


# ----------------------------------------------------------------------
# 3. ROC AND PRECISION-RECALL CURVES
# ----------------------------------------------------------------------
def generate_roc_and_pr_curves(model, X_test, y_test, classes):
    """Computes and plots ROC-AUC and Precision-Recall curves."""
    # Convert labels to binary (high_beam = 1, glare = 0)
    pos_idx = list(model.classes_).index("high_beam")
    y_true_binary = (y_test == "high_beam").astype(int)
    y_prob = model.predict_proba(X_test)[:, pos_idx]

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true_binary, y_prob)
    pr_auc = average_precision_score(y_true_binary, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # ROC Subplot
    ax1.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], color="#94A3B8", lw=1.5, linestyle="--", label="Random Guess (AUC = 0.500)")
    ax1.fill_between(fpr, tpr, color="#3B82F6", alpha=0.15)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontweight="bold")
    ax1.set_title("Receiver Operating Characteristic (ROC)", fontweight="bold", pad=10)
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PR Subplot
    ax2.plot(recall, precision, color="#10B981", lw=2.5, label=f"PR Curve (AP = {pr_auc:.3f})")
    ax2.fill_between(recall, precision, color="#10B981", alpha=0.15)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall (High Beam Sensitivity)", fontweight="bold")
    ax2.set_ylabel("Precision (Positive Predictive Value)", fontweight="bold")
    ax2.set_title("Precision-Recall Curve", fontweight="bold", pad=10)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle("Performance Evaluation Curves | Project ID: 002/2025 | Author: M NANDISH",
                 fontsize=13, fontweight="bold", y=1.00)
    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, "roc_pr_curves.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[GENERATED] ROC & PR Curves saved -> {out_path}")


# ----------------------------------------------------------------------
# 4. FEATURE IMPORTANCES
# ----------------------------------------------------------------------
def generate_feature_importance_plot(model):
    """Plots the top discriminatory features identified by Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]

    # Map indices to meaningful semantic feature names
    feature_names = []
    for idx in indices:
        if idx == 0:
            name = "Intensity Mean"
        elif idx == 1:
            name = "Intensity Std Dev (Contrast)"
        elif idx == 2:
            name = "Peak Intensity Max"
        elif 3 <= idx < 19:
            name = f"Intensity Hist Bin {idx - 3}"
        else:
            lbp_bin = idx - 19
            name = f"LBP Micro-texture Bin {lbp_bin}"
        feature_names.append(name)

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    y_pos = np.arange(len(indices))
    colors = ["#2563EB" if "Intensity" in fn else "#8B5CF6" for fn in feature_names]

    bars = ax.barh(y_pos, importances[indices], align="center", color=colors, edgecolor="#1E293B", lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names, fontsize=9.5)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Relative Gini Feature Importance", fontweight="bold", labelpad=10)
    ax.set_title("Top 20 Discriminative Features (Random Forest Ensemble)\nProject ID: 002/2025 | Author: M NANDISH",
                 fontsize=12, fontweight="bold", pad=12)

    # Annotate values on bars
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height()/2, f"{w:.4f}",
                va="center", fontsize=8.5, color="#1E293B")

    # Legend for feature domains
    custom_lines = [
        patches.Patch(facecolor="#2563EB", label="Photometric Intensity Descriptors"),
        patches.Patch(facecolor="#8B5CF6", label="LBP Micro-texture Descriptors")
    ]
    ax.legend(handles=custom_lines, loc="lower right", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6, axis="x")

    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, "feature_importance.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[GENERATED] Feature Importance plot saved -> {out_path}")


# ----------------------------------------------------------------------
# 5. DATASET DISTRIBUTION PLOT
# ----------------------------------------------------------------------
def generate_dataset_distribution_plot():
    """Plots class balance across Train, Validation, and Test splits."""
    train_y = np.load(os.path.join(BASE_DIR, "train_y.npy"))
    val_y = np.load(os.path.join(BASE_DIR, "val_y.npy"))
    test_y = np.load(os.path.join(BASE_DIR, "test_y.npy"))

    splits = ["Train (70%)", "Validation (10%)", "Test (20%)"]
    glare_counts = [np.sum(train_y == "glare"), np.sum(val_y == "glare"), np.sum(test_y == "glare")]
    hb_counts = [np.sum(train_y == "high_beam"), np.sum(val_y == "high_beam"), np.sum(test_y == "high_beam")]

    x = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    b1 = ax.bar(x - width/2, glare_counts, width, label="Glare (Low Beam / Reflections)",
                color="#F59E0B", edgecolor="#B45309", lw=1.2)
    b2 = ax.bar(x + width/2, hb_counts, width, label="High Beam (Direct High Intensity)",
                color="#10B981", edgecolor="#047857", lw=1.2)

    ax.set_ylabel("Number of Annotated ROIs", fontweight="bold")
    ax.set_title("Dataset Stratification Breakdown across Partitions\nProject ID: 002/2025 | Author: M NANDISH",
                 fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(splits, fontweight="bold")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")

    # Add count labels above bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 3, f"{h}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    total_samples = len(train_y) + len(val_y) + len(test_y)
    fig.text(0.5, 0.01, f"Total Annotated Samples: {total_samples} ROIs across 100 Camera Scenes",
             ha="center", fontsize=9.5, style="italic", color="#475569")

    plt.tight_layout()
    out_path = os.path.join(ASSETS_DIR, "dataset_distribution.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[GENERATED] Dataset Distribution plot saved -> {out_path}")


# ----------------------------------------------------------------------
# 6. UNSEEN IMAGE DETECTIONS & COMPOSITE VISUALS
# ----------------------------------------------------------------------
def generate_unseen_detections_and_dashboard(model):
    """
    Runs end-to-end inference on unseen_image.jpg, unseen_image_1.jpg,
    unseen_image_2.jpg, saves standalone screenshots to assets/ & outputs/,
    and creates a consolidated visual evaluation dashboard.
    """
    from DEPLOYMENT_TESTING import predict_on_image

    unseen_files = ["unseen_image.jpg", "unseen_image_1.jpg", "unseen_image_2.jpg"]
    annotated_results = []

    for idx, fname in enumerate(unseen_files, 1):
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            continue

        vis_img, dets = predict_on_image(fpath, model=model, show_gui=False, save_dir=OUTPUTS_DIR)
        
        # Save high-res PNG into assets/
        asset_out = os.path.join(ASSETS_DIR, f"unseen_prediction_{idx}.png")
        cv2.imwrite(asset_out, vis_img)
        print(f"[GENERATED] Detection result saved -> {asset_out}")
        annotated_results.append((fname, vis_img, dets))

    # Create composite multi-panel dashboard
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    for ax, (fname, vis_img, dets) in zip(axs, annotated_results):
        rgb_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_img)
        ax.axis("off")
        title = f"Test Scene: {fname}\n" + ", ".join([f"{d['label']}" for d in dets])
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    fig.suptitle("Real-Time Night Driving Inference on Unseen Test Frames\nProject ID: 002/2025 | Author: M NANDISH",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    dash_path = os.path.join(ASSETS_DIR, "unseen_detections_composite.png")
    fig.savefig(dash_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[GENERATED] Composite Detections saved -> {dash_path}")


# ----------------------------------------------------------------------
# 7. UNIFIED EVALUATION DASHBOARD (Multi-panel Summary)
# ----------------------------------------------------------------------
def generate_evaluation_dashboard(model, X_test, y_test, classes):
    """Creates a unified 4-quadrant executive dashboard figure."""
    fig = plt.figure(figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")

    # Grid specification
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # 1. Confusion Matrix (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds, labels=classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)"

    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False,
                xticklabels=["Glare", "High Beam"], yticklabels=["Glare", "High Beam"],
                annot_kws={"fontsize": 11, "fontweight": "bold"}, ax=ax1)
    ax1.set_title("A. Confusion Matrix on Test Split", fontweight="bold", fontsize=11, pad=10)
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    # 2. ROC & PR Curves (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    pos_idx = list(model.classes_).index("high_beam")
    y_true_binary = (y_test == "high_beam").astype(int)
    y_prob = model.predict_proba(X_test)[:, pos_idx]
    fpr, tpr, _ = roc_curve(y_true_binary, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_prob)
    pr_auc = average_precision_score(y_true_binary, y_prob)

    ax2.plot(fpr, tpr, color="#2563EB", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax2.plot(recall, precision, color="#10B981", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    ax2.plot([0, 1], [0, 1], color="#94A3B8", lw=1, linestyle="--")
    ax2.set_title("B. Diagnostic ROC & Precision-Recall", fontweight="bold", fontsize=11, pad=10)
    ax2.set_xlabel("Rate / Recall")
    ax2.set_ylabel("Value / Precision")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Top Features (Bottom Left)
    ax3 = fig.add_subplot(gs[1, 0])
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    names = []
    for idx in indices:
        if idx == 0: names.append("Mean Intensity")
        elif idx == 1: names.append("Std Dev (Contrast)")
        elif idx == 2: names.append("Max Intensity")
        elif 3 <= idx < 19: names.append(f"Hist Bin {idx-3}")
        else: names.append(f"LBP Bin {idx-19}")
    
    y_pos = np.arange(len(indices))
    ax3.barh(y_pos, importances[indices], color="#6366F1", edgecolor="#1E293B", lw=0.6)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(names, fontsize=8.5)
    ax3.invert_yaxis()
    ax3.set_title("C. Top 10 Random Forest Features", fontweight="bold", fontsize=11, pad=10)
    ax3.set_xlabel("Gini Importance")
    ax3.grid(True, linestyle=":", alpha=0.6, axis="x")

    # 4. Sample Classified Scene (Bottom Right)
    ax4 = fig.add_subplot(gs[1, 1])
    sample_img_path = os.path.join(OUTPUTS_DIR, "unseen_image_2_classified.png")
    if os.path.exists(sample_img_path):
        sample_bgr = cv2.imread(sample_img_path)
        sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
        ax4.imshow(sample_rgb)
    ax4.axis("off")
    ax4.set_title("D. Live Classified Road Frame (unseen_image_2)", fontweight="bold", fontsize=11, pad=10)

    # Master title
    fig.suptitle("INTELLIGENT HIGH BEAM VS GLARE CLASSIFICATION DASHBOARD\nProject ID: 002/2025 | Year: 2025 | Author: M NANDISH | Department of ETE, MSRIT",
                 fontsize=13, fontweight="bold", y=0.98, color="#0F172A")

    out_path = os.path.join(ASSETS_DIR, "evaluation_dashboard.png")
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[GENERATED] Master Evaluation Dashboard saved -> {out_path}")


# ----------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------
def main():
    print("================================================================")
    print("  AUTOMATED ARTIFACT & VISUAL GENERATION SUITE")
    print("  Project ID: 002/2025 | Author: M NANDISH")
    print("================================================================")

    # 1. Flowchart
    generate_system_flowchart()

    # Load model and test partition
    model_path = os.path.join(BASE_DIR, "highbeam_rf_model.pkl")
    test_x_path = os.path.join(BASE_DIR, "test_X.npy")
    test_y_path = os.path.join(BASE_DIR, "test_y.npy")

    model = joblib.load(model_path)
    X_test = np.load(test_x_path)
    y_test = np.load(test_y_path)
    classes = list(getattr(model, "classes_", np.unique(y_test)))
    preds = model.predict(X_test)

    # 2. Confusion Matrix
    generate_confusion_matrix_plot(y_test, preds, classes)

    # 3. ROC & PR curves
    generate_roc_and_pr_curves(model, X_test, y_test, classes)

    # 4. Feature Importance
    generate_feature_importance_plot(model)

    # 5. Dataset distribution
    generate_dataset_distribution_plot()

    # 6. Unseen detections
    generate_unseen_detections_and_dashboard(model)

    # 7. Unified dashboard
    generate_evaluation_dashboard(model, X_test, y_test, classes)

    print("\n================================================================")
    print(f"All figures successfully generated in: {ASSETS_DIR}")
    print("================================================================\n")


if __name__ == "__main__":
    main()
