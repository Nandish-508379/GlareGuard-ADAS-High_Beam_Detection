"""
================================================================================
Intelligent High Beam vs Glare Classification System
Project ID   : 002/2025
Project Year : 2025
Author       : M NANDISH
Department   : Department of Electronics & Telecommunication Engineering
Institution  : Ramaiah Institute of Technology (MSRIT), Bengaluru
================================================================================
File: ANNOTATION.py
Description:
    Interactive ROI annotation tool built with OpenCV. Allows the user to load
    raw road camera frames, manually draw bounding boxes over illuminated
    optical regions, and classify them as High Beam (HB) or Glare (GL).
    Saves image duplicates and structured JSON bounding boxes for dataset building.

Academic & Fair Use Disclaimer:
    Images in DATASET/ were retrieved from public internet resources strictly
    for non-commercial, academic study and educational benchmarking under Fair Use.
================================================================================
"""

import cv2
import os
import json

# -------------------------
# PATH CONFIGURATION (Dynamic & Portable)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "DATASET")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "MARKERS")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# GLOBALS FOR INTERACTIVE DRAWING
# -------------------------
drawing = False
start_pt = (-1, -1)
current_rect = None
rectangles = []        # list of tuples: (pt1, pt2, class)
display = None
original = None


def draw_labeled_box(img, pt1, pt2, cls):
    """Draw a bounding box with class label badge."""
    color = (0, 255, 0) if cls == "high_beam" else (0, 128, 255)
    label = "HB" if cls == "high_beam" else "GL"

    cv2.rectangle(img, pt1, pt2, color, 2)
    x, y = pt1
    cv2.putText(img, label, (x, max(y - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def redraw_display():
    """Redraw all confirmed bounding boxes on top of a fresh copy of the image."""
    global display
    display = original.copy()
    for (pt1, pt2, cls) in rectangles:
        draw_labeled_box(display, pt1, pt2, cls)


def mouse_callback(event, x, y, flags, param):
    """OpenCV mouse callback for interactive rectangle drawing."""
    global drawing, start_pt, current_rect, display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = display.copy()
            cv2.rectangle(temp, start_pt, (x, y), (0, 255, 0), 2)
            cv2.imshow("Annotator - Project 002/2025 (M NANDISH)", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect = (start_pt, (x, y))
        cv2.rectangle(display, start_pt, (x, y), (0, 255, 0), 2)
        cv2.imshow("Annotator - Project 002/2025 (M NANDISH)", display)


def save_annotation(img_name, rectangles):
    """Save annotated image copy and corresponding JSON bounding box coordinates."""
    out_img_path = os.path.join(OUTPUT_FOLDER, img_name)
    cv2.imwrite(out_img_path, original)

    ann = []
    for (pt1, pt2, cls) in rectangles:
        x1, y1 = pt1
        x2, y2 = pt2
        xmin = min(x1, x2)
        ymin = min(y1, y2)
        xmax = max(x1, x2)
        ymax = max(y1, y2)

        ann.append({
            "class": cls,
            "x_min": xmin,
            "y_min": ymin,
            "x_max": xmax,
            "y_max": ymax
        })

    json_name = os.path.splitext(img_name)[0] + ".json"
    json_path = os.path.join(OUTPUT_FOLDER, json_name)
    with open(json_path, "w") as f:
        json.dump(ann, f, indent=4)

    print(f"[SAVED] {img_name} with {len(rectangles)} annotated regions.")


def annotate():
    """Main loop for iterating through input images and collecting annotations."""
    global display, original, rectangles, current_rect

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder does not exist: {INPUT_FOLDER}")
        return

    images = [f for f in sorted(os.listdir(INPUT_FOLDER))
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    print("================================================================")
    print("  ROI ANNOTATION TOOL | Project ID: 002/2025 | Author: M NANDISH")
    print("================================================================")
    print(f"Loaded images: {len(images)}")
    print("""
Controls:
  Left-drag : Draw bounding box
  H         : Classify selection as HIGH BEAM (HB)
  G         : Classify selection as GLARE (GL)
  U         : Undo last drawn box
  S         : Save annotations and advance to next image
  N         : Skip current image and advance
  Q         : Quit tool
    """)

    window_name = "Annotator - Project 002/2025 (M NANDISH)"

    for img_name in images:
        img_path = os.path.join(INPUT_FOLDER, img_name)
        original = cv2.imread(img_path)
        if original is None:
            continue

        display = original.copy()
        rectangles = []
        current_rect = None

        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)

        while True:
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('h') and current_rect is not None:
                rectangles.append((current_rect[0], current_rect[1], "high_beam"))
                current_rect = None
                redraw_display()

            elif key == ord('g') and current_rect is not None:
                rectangles.append((current_rect[0], current_rect[1], "glare"))
                current_rect = None
                redraw_display()

            elif key == ord('u'):
                if rectangles:
                    removed = rectangles.pop()
                    print(f"Removed last annotation ({removed[2]})")
                    redraw_display()
                else:
                    print("Undo: no boxes to remove.")

            elif key == ord('s'):
                save_annotation(img_name, rectangles)
                break

            elif key == ord('n'):
                print("Skipped:", img_name)
                break

            elif key == ord('q'):
                print("Exiting annotation tool.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()


if __name__ == "__main__":
    annotate()
