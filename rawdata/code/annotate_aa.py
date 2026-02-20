import os
import glob
import numpy as np

# Convert OBB corners to AA bounding box (AABB)
def obb_to_aabb_from_array(corners):
    min_xy = corners.min(axis=0)
    max_xy = corners.max(axis=0)

    cx, cy = (min_xy + max_xy) / 2.0
    width, height = max_xy - min_xy

    W, H = 640, 360
    cx /= W
    cy /= H
    width /= W
    height /= H
    
    return cx, cy, width, height

# ---------------------------------
# Convert YOLO OBB → 4 corner points
# ---------------------------------
def yolo_obb_to_corners(x_center, y_center, width, height, angle_deg):
    w_img, h_img = 640, 360

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

    return np.array([A, B, C, D], dtype=float)



# Change this to your target directory
directory = "."

for filepath in glob.glob(os.path.join(directory, "*.txt")):
    filename = os.path.basename(filepath)
    stem, ext = os.path.splitext(filename)

    # Skip already processed files (avoid infinite loop)
    if stem.endswith("_aa"):
        continue

    with open(filepath, "r") as f:
        numbers = f.read().split()

    if len(numbers) < 5:
        print(f"Skipping {filename}: not enough numbers")
        continue

    # Take first 6 numbers (class_id, xc, yc, w, h, angle)
    id, cx, cy, w, h, angle = numbers[:6]
    corners = yolo_obb_to_corners(
                        float(cx),
                        float(cy),
                        float(w),
                        float(h),
                        float(angle),
                    )
    cx, cy, width, height = obb_to_aabb_from_array(corners)

    new_filename = os.path.join(directory, f"{stem}_aa.txt")

    with open(new_filename, "w") as f:
        f.write(f"{id} {cx:4} {cy:4} {width:4} {height:4}")

    print(f"Created {new_filename}")