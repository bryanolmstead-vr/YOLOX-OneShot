# convert YOLO style annotations to YOLO-OBB format
# assumptions:
# images in: datasets/mydataset/train2017/
# labels in same folder OR labels/train2017/
# one .txt per image
# classes are zero-indexed

import os
import json
import cv2

# ===== EDIT THESE =====
IMAGE_DIR = "../../datasets/OBB360/val2017"
LABEL_DIR = "../../datasets/OBB360/annotations/val2017" 
OUTPUT_JSON = "../../datasets/OBB360/annotations/instances_val2017.json"
CATEGORIES = ["candy", "cards", "cheeto"] 
# ======================

images = []
annotations = []
categories = []

annotation_id = 1
image_id = 1

# Create category entries
for i, name in enumerate(CATEGORIES):
    categories.append({
        "id": i,
        "name": name,
        "supercategory": "object"
    })

for filename in os.listdir(IMAGE_DIR):
    if not filename.endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    images.append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                class_id, x_center, y_center, w, h, theta = map(float, line.strip().split())

                # convert normalized center/size to absolute coordinates
                x_ctr = x_center * width
                y_ctr = y_center * height
                box_width = w * width
                box_height = h * height

                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(class_id),
                    "bbox": [x_ctr, y_ctr, box_width, box_height],
                    "area": box_width * box_height,
                    "angle": theta,
                    "iscrowd": 0
                })

                annotation_id += 1

    image_id += 1

yolo_obb_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

with open(OUTPUT_JSON, "w") as f:
    json.dump(yolo_obb_format, f, indent=4)

print("Conversion complete.")