# visualize_obb.py
# usage: python visualize_obb.py *.jpg
# Loads images and corresponding YOLO OBB .txt metadata
# Draws green oriented bounding boxes with 12:00 marker
# Press any key for next image
# Press q to quit

import cv2
import numpy as np
import glob
import os
import sys


# ---------------------------------
# Convert YOLO OBB → 4 corner points
# ---------------------------------
def yolo_obb_to_corners(img, x_center, y_center, width, height, angle_deg):
    h_img, w_img = img.shape[:2]

    # Denormalize
    cx = x_center * w_img
    cy = y_center * h_img
    w = width * w_img
    h = height * h_img

    theta = np.radians(angle_deg)

    # Direction vectors

    # Height direction (12:00 direction)
    dx_h = -np.sin(theta)
    dy_h =  np.cos(theta)

    # Width direction (90° rotated from height)
    dx_w =  np.cos(theta)
    dy_w =  np.sin(theta)

    # Half sizes
    hw = w / 2.0
    hh = h / 2.0

    # Corners (UL, LL, LR, UR)
    A = np.array([cx - hw*dx_w - hh*dx_h, cy - hw*dy_w - hh*dy_h])
    B = np.array([cx - hw*dx_w + hh*dx_h, cy - hw*dy_w + hh*dy_h])
    C = np.array([cx + hw*dx_w + hh*dx_h, cy + hw*dy_w + hh*dy_h])
    D = np.array([cx + hw*dx_w - hh*dx_h, cy + hw*dy_w - hh*dy_h])

    return np.array([A, B, C, D], dtype=np.int32)


# ---------------------------------
# Draw rectangle + 12:00 marker
# ---------------------------------
def draw_obb(img, corners):
    overlay = img.copy()

    cv2.polylines(
        overlay,
        [corners],
        isClosed=True,
        color=(0, 255, 0),
        thickness=3,
    )

    # 12:00 orientation marker
    A, B, C, D = corners
    center = corners.mean(axis=0).astype(int)
    top_center = ((A + D) / 2).astype(int)

    cv2.line(
        overlay,
        tuple(top_center),
        tuple(center),
        (0, 255, 0),
        3,
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
        txtfile = stem + ".txt"

        display = img.copy()

        if os.path.exists(txtfile):
            with open(txtfile, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 6:
                        continue

                    _, xc, yc, w, h, angle = parts
                    corners = yolo_obb_to_corners(
                        img,
                        float(xc),
                        float(yc),
                        float(w),
                        float(h),
                        float(angle),
                    )

                    display = draw_obb(display, corners)

        cv2.imshow("OBB Viewer", display)

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
