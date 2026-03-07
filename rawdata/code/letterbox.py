# letterbox.py
#
# usage:
# python letterbox.py *.png
# creates for every image: bla.png:
# assumes there is bla.txt with YOLO format boxes normalized to 640x360
# converted/bla_640x640.png
# converted/bla_640x640.txt


from pathlib import Path
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
    box: [id, x, y, w, h, theta] normalized to 640x360
    returns box normalized to 640x640
    """
    top_offset = (640 - 360) // 2  # 140

    id, x, y, w, h, theta = box
    # Convert y to pixels
    y_px = y * 360

    # Shift downward
    y_px += top_offset

    # Re-normalize to 640
    y_new = y_px / 640
    h_new = (h * 360) / 640

    new_box = [id, x, y_new, w, h_new, theta]

    return new_box

# ---------------------------------
# Main loop
# ---------------------------------
def letterbox(filelist):
    # assumes filelist of image files and corresponding txt files with YOLO format boxes normalized to 640x360
    # creates for every image: bla.png:
    # converted/bla_640x640.png, converted/bla_640x640.txt with boxes normalized to 640x640

    for fname in filelist:

        p = Path(fname)
        directory = p.parent
        stem = p.stem
        ext = p.suffix

        # look for only 640x360 images
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load {fname}")
            continue
        if img.shape[1] != 640 or img.shape[0] != 360:
            print(f"Image {fname} is not 640x360, skipping.")
            continue

        # original annotation path
        original_annotation_file = directory / f"{stem}.txt"

        # create 'converted' folder in the same directory
        converted_dir = directory / "converted"
        converted_dir.mkdir(exist_ok=True)  # make dir if it doesn't exist

        # new text file path
        new_annotation_file = converted_dir / f"{stem}_640x640.txt"

        # new image file path
        new_imgfile = converted_dir / f"{stem}_640x640{ext}"

        #print(f"Processing {fname} -> {new_imgfile} and {new_annotation_file}")

        display = img.copy()
        canvas = pad_640x360_to_640x640(img)

        new_box = []
        if os.path.exists(original_annotation_file):
            with open(original_annotation_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    # expect id, x, y, w, h, theta (6 parameters)
                    if len(parts) != 6:
                        continue
                    box = [float(x) for x in parts]
                    new_box = adjust_box_yolo(box)

        id, x ,y, w, h, theta = new_box
        id = int(id) 
        overlay = canvas.copy()
        x*=640
        y*=640
        w*=640
        h*=640 
        theta_rad = np.deg2rad(theta) 
        R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], [np.sin(theta_rad), np.cos(theta_rad)]])
        rect = np.array([
            [-w/2, -h/2],
            [-w/2, +h/2],
            [+w/2, +h/2],
            [+w/2, -h/2],
        ], dtype=np.int32)   
        rotated_rect = rect @ R.T
        corners = (rotated_rect + np.array([x, y])).astype(np.int32)

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

        cv2.imshow("letterbox", overlay)

        cv2.imwrite(new_imgfile, canvas)

        with open(new_annotation_file, "w") as f:
            f.write(f"{id} {x/640:4} {y/640:4} {w/640:4} {h/640:4} {theta:4}\n")

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