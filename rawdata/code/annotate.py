# ChatGPT generated annotation tool
# usage: python annotate.py *.jpg
# loads image files and based on file stem (candy, cards, cheetos),
# assigns a class. allows user to set annotation of object
# click upper left corner (defines positional anchor)
# click lower left corner (defines height)
# click to lower right corner (defines width and rotation)
# d to delete and re-annotate image
# n to go to next image
# q to quit
# annotations are saved as .txt files with same file stem

import cv2
import numpy as np
import glob
import os
import sys

# ---------------------------
# Global state
# ---------------------------
state = {
    "phase": 0,  # 0=waiting first click, 1=waiting second click, 2=adjusting rectangle
    "anchor": None,
    "height_pt": None,
    "corners": None,
    "img_copy": None,
    "filename": None,
}

# ---------------------------
# Class mapping
# ---------------------------
class_map = {
    "candy": 0,
    "cards": 1,
    "cheetos": 2,
}

def get_class_from_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    for key in class_map:
        if stem.lower().startswith(key):
            return key, class_map[key]
    return "unknown", -1


# ---------------------------
# Geometry
# ---------------------------
def compute_rectangle(anchor, height_pt, mouse_pt):
    """
    Compute rotated rectangle anchored at 'anchor' (first click):
    - Height magnitude = distance(anchor -> height_pt)
    - Lower-right corner = mouse_pt
    - Width computed to hit mouse_pt
    - Corners ordered UL -> LL -> LR -> UR
    """

    x0, y0 = anchor
    x1_init, y1_init = height_pt
    x2, y2 = mouse_pt

    # ---------------------------------
    # Fixed height from first two clicks
    # ---------------------------------
    dx_10 = x1_init - x0
    dy_10 = y1_init - y0
    H = np.sqrt(dx_10**2 + dy_10**2)

    if H < 1e-6:
        return None

    # ---------------------------------
    # Diagonal from anchor to mouse
    # ---------------------------------
    dx_20 = x2 - x0
    dy_20 = y2 - y0
    D = np.sqrt(dx_20**2 + dy_20**2)

    if D <= H:
        return None  # invalid geometry (mouse too close)

    # ---------------------------------
    # Width from Pythagorean theorem
    # ---------------------------------
    W = np.sqrt(D**2 - H**2)

    # ---------------------------------
    # Angles (your formulation)
    # ---------------------------------
    alpha = np.arctan2(dx_20, dy_20)  # angle from vertical
    beta = np.arctan2(H, W)

    theta = alpha + beta

    # ---------------------------------
    # Recompute lower-left corner
    # ---------------------------------
    x1 = x2 - W * np.sin(theta)
    y1 = y2 - W * np.cos(theta)

    # ---------------------------------
    # Construct rectangle corners
    # ---------------------------------
    A = np.array([x0, y0])
    B = np.array([x1, y1])
    C = np.array([x2, y2])
    D_pt = A + (C - B)

    corners = np.array([A, B, C, D_pt], dtype=np.int32)

    center = corners.mean(axis=0)
    angle = np.degrees(theta)

    return corners, center, H, W, angle

# ---------------------------
# Overlay
# ---------------------------
def draw_overlay(img):
    overlay = img.copy()

    fname = state["filename"]
    class_name, class_id = get_class_from_filename(fname)

    phase_text = {
        0: "Click upper-left corner",
        1: "Click lower-left corner (height)",
        2: "Click lower-right corner (width & rotation)",
    }

    lines = [
        f"Filename: {fname}",
        f"Class: {class_name} ({class_id})",
        f"State: {phase_text.get(state['phase'], '')}",
        "",
        "n: next image",
        "d: delete annotation",
        "q: quit",
    ]

    y0 = 25
    for i, line in enumerate(lines):
        y = y0 + i * 10
        cv2.putText(
            overlay,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


# ---------------------------
# Drawing
# ---------------------------
def draw_rectangle(img, corners):
    if corners is None:
        return img
    preview = img.copy()
    cv2.polylines(preview, [corners], isClosed=True, color=(0, 255, 0), thickness=3)
    # Orientation line from top edge to center
    A, B, C, D = corners
    center = corners.mean(axis=0).astype(int)
    top_center = ((A + D)/2).astype(int)
    cv2.line(preview, tuple(top_center), tuple(center), (0, 255, 0), 3)
    return preview

# ---------------------------
# Mouse callback
# ---------------------------
def mouse_callback(event, x, y, flags, param):
    img = state["img_copy"].copy()

    # Phase 0: waiting first click
    if state["phase"] == 0:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["anchor"] = (x, y)
            state["phase"] = 1
        img = draw_overlay(img)
        cv2.imshow("Annotate", img)
        return

    # Phase 1: moving toward second click
    elif state["phase"] == 1:
        anchor = state["anchor"]
        cv2.line(img, anchor, (x, y), (0, 255, 0), 3)  # green line width=3
        img = draw_overlay(img)
        cv2.imshow("Annotate", img)
        if event == cv2.EVENT_LBUTTONDOWN:
            state["height_pt"] = (x, y)
            state["phase"] = 2

    # Phase 2: rectangle rubber-band toward lower-right
    elif state["phase"] == 2:
        result = compute_rectangle(state["anchor"], state["height_pt"], (x, y))
        if result is not None:
            corners, center, h, w, angle = result
            state["corners"] = corners
            img = draw_rectangle(img, corners)
        img = draw_overlay(img)
        cv2.imshow("Annotate", img)
        if event == cv2.EVENT_LBUTTONDOWN:
            if state["corners"] is not None:
                save_annotation(state["filename"], state["corners"])
                # reset state
                state["phase"] = 0
                state["anchor"] = None
                state["height_pt"] = None
                state["corners"] = None

# ---------------------------
# Save annotation
# ---------------------------
def save_annotation(filename, corners):
    stem = os.path.splitext(os.path.basename(filename))[0]
    outname = f"{stem}.txt"

    class_id = None
    for key in class_map:
        if stem.lower().startswith(key):
            class_id = class_map[key]
            break

    if class_id is None:
        print(f"Unknown class for {stem}")
        return

    # ----------------------------
    # Image size
    # ----------------------------
    img = state["img_copy"]
    h_img, w_img = img.shape[:2]

    A, B, C, D = corners.astype(np.float32)

    # ----------------------------
    # Center
    # ----------------------------
    center = (A + C) / 2.0
    x_center = center[0] / w_img
    y_center = center[1] / h_img

    # ----------------------------
    # Width and Height
    # ----------------------------
    height = np.linalg.norm(B - A)
    width = np.linalg.norm(C - B)

    width_norm = width / w_img
    height_norm = height / h_img

    # ----------------------------
    # Angle from vertical
    # Vector: center -> top edge midpoint
    # ----------------------------
    top_mid = (A + D) / 2.0
    v = top_mid - center

    angle = np.degrees(np.arctan2(v[0], -v[1])) # dx, dy (angle from vertical)


    # ----------------------------
    # Save YOLO OBB format
    # class x_center y_center width height angle
    # ----------------------------
    with open(outname, "w") as f:
        f.write(
            f"{class_id} "
            f"{x_center:.6f} {y_center:.6f} "
            f"{width_norm:.6f} {height_norm:.6f} "
            f"{angle:.6f}\n"
        )

    print(f"Saved {outname}")


# ---------------------------
# Main loop
# ---------------------------
def annotate_files(filelist):
    for fname in filelist:
        state["filename"] = fname
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue
        state["img_copy"] = img.copy()
        state["phase"] = 0
        img_display = draw_overlay(img)
        cv2.imshow("Annotate", img_display)
        cv2.setMouseCallback("Annotate", mouse_callback)

        while True:
            key = cv2.waitKey(10) & 0xFF
            if key == ord('n'):
                break
            elif key == ord('q'):
                return
            elif key == ord('d'):
                # reset current image
                state["phase"] = 0
                state["anchor"] = None
                state["height_pt"] = None
                state["corners"] = None
                img_display = draw_overlay(img)
                cv2.imshow("Annotate", state["img_copy"])

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    files = []
    for arg in sys.argv[1:]:
        files.extend(glob.glob(arg))
    if not files:
        files = glob.glob("*.png")
    print(f"Found {len(files)} images to annotate")
    annotate_files(files)
    cv2.destroyAllWindows()
