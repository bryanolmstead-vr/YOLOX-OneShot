# show_annotation_aa.py
# usage: python show_annotation_aa.py *.jpg
# Loads images and corresponding YOLO OBB .txt metadata
# Draws green oriented bounding boxes ignoring rotation
# Press any key for next image
# Press q to quit

import cv2
import numpy as np
import glob
import os
import sys


# ---------------------------------
# Convert YOLO Axis Aligned → 4 corner points
# ---------------------------------
def yolo_obb_to_corners_aa(img, x_center, y_center, width, height):
    h_img, w_img = img.shape[:2]

    # Denormalize
    cx = x_center * w_img
    cy = y_center * h_img
    w = width * w_img
    h = height * h_img

    # Half sizes
    hw = w / 2.0
    hh = h / 2.0

    # Corners (UL, LL, LR, UR)
    A = np.array([cx - hw, cy - hh])
    B = np.array([cx - hw, cy + hh])
    C = np.array([cx + hw, cy + hh])
    D = np.array([cx + hw, cy - hh])

    return np.array([A, B, C, D], dtype=np.int32)


# ---------------------------------
# Draw axis aligned rectangle
# ---------------------------------
def draw_aa(img, corners):
    overlay = img.copy()

    cv2.polylines(
        overlay,
        [corners],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    return overlay


# ---------------------------------
# Main loop
# ---------------------------------
def visualize(filelist):
    for fname in filelist:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue

        stem = os.path.splitext(fname)[0]
        txtfile = stem + "_aa.txt"

        display = img.copy()

        if os.path.exists(txtfile):
            with open(txtfile, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    _, xc, yc, w, h = parts
                    corners = yolo_obb_to_corners_aa(
                        img,
                        float(xc),
                        float(yc),
                        float(w),
                        float(h),
                    )

                    display = draw_aa(display, corners)

        cv2.imshow("AA Bounding Box Viewer", display)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


# ---------------------------------
# Entry
# ---------------------------------
if __name__ == "__main__":
    files = []
    for arg in sys.argv[1:]:
        files.extend(glob.glob(arg))

    if not files:
        files = glob.glob("*.png")

    print(f"Found {len(files)} images")
    visualize(files)
