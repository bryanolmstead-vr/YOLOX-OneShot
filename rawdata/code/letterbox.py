# letterbox.py

import numpy as np
import cv2
import glob
import os
import sys

def pad_640x360_to_640x640(img):
    """
    Assumes img is 640x360.
    Returns padded image.
    """
    h, w = img.shape[:2]
    assert w == 640 and h == 360, "Image must be 640x360"

    canvas = np.zeros((640, 640, 3), dtype=img.dtype)

    top = (640 - h) // 2  # 140
    canvas[top:top+h, 0:w] = img

    return canvas

def adjust_box_yolo(box):
    """
    box: [id, x, y, w, h] normalized to 640x360
    returns box normalized to 640x640
    """
    top_offset = (640 - 360) // 2  # 140

    id, x, y, w, h = box
    # Convert y to pixels
    y_px = y * 360

    # Shift downward
    y_px += top_offset

    # Re-normalize to 640
    y_new = y_px / 640
    h_new = (h * 360) / 640

    new_box = [id, x, y_new, w, h_new]

    return new_box

# ---------------------------------
# Main loop
# ---------------------------------
def letterbox(filelist):
    for fname in filelist:
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue
        if img.shape[1] != 640 or img.shape[0] != 360:
            print(f"Image {fname} is not 640x360, skipping.")
            continue

        stem = os.path.splitext(fname)[0]
        txtfile = stem + "_aa.txt"

        display = img.copy()
        canvas = pad_640x360_to_640x640(img)

        if os.path.exists(txtfile):
            with open(txtfile, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    box = [float(x) for x in parts]
                    new_box = adjust_box_yolo(box)

        id, x ,y, w, h = new_box
        id = int(id) 
        overlay = canvas.copy()
        x*=640
        y*=640
        w*=640
        h*=640  
        corners = np.array([
            [x - w/2, y - h/2],
            [x - w/2, y + h/2],
            [x + w/2, y + h/2],
            [x + w/2, y - h/2],
        ], dtype=np.int32)   

        cv2.polylines(
            overlay,
            [corners],
            isClosed=True,
            color=(0, 255, 0),
            thickness=3,
        )
        cv2.imshow("letterbox", overlay)

        imFilename = stem + "_640x640.png"
        labelFilename = stem + "_640x640_aa.txt" 

        cv2.imwrite(imFilename, canvas)

        with open(labelFilename, "w") as f:
            f.write(f"{id} {x/640:4} {y/640:4} {w/640:4} {h/640:4}")

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
    letterbox(files)