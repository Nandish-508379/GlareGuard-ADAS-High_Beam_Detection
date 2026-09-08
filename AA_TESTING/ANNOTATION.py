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

# -------------------------
# CONFIG (Dynamic & Portable)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "DATASET")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "MARKERS")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------
# GLOBALS
# -------------------------
drawing = False
start_pt = (-1, -1)
current_rect = None
rectangles = []        # (pt1, pt2, class)
display = None
original = None


def draw_labeled_box(img, pt1, pt2, cls):
    color = (0,255,0) if cls == "high_beam" else (0,128,255)
    label = "HB" if cls == "high_beam" else "GL"

    cv2.rectangle(img, pt1, pt2, color, 2)

    x, y = pt1
    cv2.putText(img, label, (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)


def redraw_display():
    global display
    display = original.copy()
    for (pt1, pt2, cls) in rectangles:
        draw_labeled_box(display, pt1, pt2, cls)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_pt, current_rect, display

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_pt = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp = display.copy()
            cv2.rectangle(temp, start_pt, (x, y), (0,255,0), 2)
            cv2.imshow("Annotator", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_rect = (start_pt, (x, y))
        cv2.rectangle(display, start_pt, (x, y), (0,255,0), 2)
        cv2.imshow("Annotator", display)


def save_annotation(img_name, rectangles):
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

    json_name = img_name.replace(".jpg", ".json")
    json_path = os.path.join(OUTPUT_FOLDER, json_name)
    with open(json_path, "w") as f:
        json.dump(ann, f, indent=4)

    print(f"[SAVED] {img_name} with {len(rectangles)} regions.")


def annotate():
    global display, original, rectangles, current_rect

    images = [f for f in os.listdir(INPUT_FOLDER)
              if f.lower().endswith(".jpg")]

    print("Loaded images:", len(images))
    print("""
Controls:
Left-drag        = Draw box
H               = Label HIGH BEAM (HB)
G               = Label GLARE (GL)
U               = Undo last box
S               = Save & next
N               = Skip & next
Q               = Quit tool
    """)

    for img_name in images:
        img_path = os.path.join(INPUT_FOLDER, img_name)
        original = cv2.imread(img_path)
        display = original.copy()
        rectangles = []
        current_rect = None

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", mouse_callback)

        while True:
            cv2.imshow("Annotator", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('h') and current_rect is not None:
                rectangles.append((current_rect[0], current_rect[1], "high_beam"))
                current_rect = None
                redraw_display()

            if key == ord('g') and current_rect is not None:
                rectangles.append((current_rect[0], current_rect[1], "glare"))
                current_rect = None
                redraw_display()

            if key == ord('u'):
                if rectangles:
                    rectangles.pop()
                    redraw_display()
                else:
                    print("Undo: nothing to remove.")

            if key == ord('s'):
                save_annotation(img_name, rectangles)
                break

            if key == ord('n'):
                print("Skipped:", img_name)
                break

            if key == ord('q'):
                print("Exiting.")
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()


annotate()
