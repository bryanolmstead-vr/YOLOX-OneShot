# convert AABB labels into OBB format with angle from OBB dataset
# assumptions:
# AABB dataset: datasets/COCO3/annotations/instances_train2017.json
#               datasets/COCO3/annotations/instances_val2017.json
# OBB dataset:  datasets/OBB360/annotations/instances_train2017.json
#               datasets/OBB360/annotations/instances_val2017.json
# Output:       datasets/OBB360/annotations/instanecs_obb2aabb_train2017.json
#               datasets/OBB360/annotations/instanecs_obb2aabb_val2017.json

#!/usr/bin/env python3
# obb2aabb.py
import json
import sys

def main(aabb_file, obb_file, output_file):
    # Load the AABB JSON
    with open(aabb_file, "r") as f:
        aabb_data = json.load(f)

    # Load the OBB JSON
    with open(obb_file, "r") as f:
        obb_data = json.load(f)

    # Create a mapping from annotation id to AABB bbox/area
    aabb_map = {ann["id"]: (ann["bbox"], ann["area"]) for ann in aabb_data["annotations"]}

    # Replace bbox and area in OBB annotations with AABB values
    for ann in obb_data["annotations"]:
        if ann["id"] in aabb_map:
            x1, y1, w, h = aabb_map[ann["id"]][0]  # unpack the bbox
            area = aabb_map[ann["id"]][1]

            # Convert top-left (x1,y1) to center (xc,yc)
            xc = x1 + w / 2
            yc = y1 + h / 2

            # Update annotation
            ann["bbox"] = [xc, yc, w, h]
            ann["area"] = area
        else:
            print(f"Warning: annotation id {ann['id']} not found in AABB dataset")

    # Save new OBB JSON
    with open(output_file, "w") as f:
        json.dump(obb_data, f, indent=4)
    print(f"Saved hybrid OBB+AABB dataset to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python obb2aabb.py <aabb_json> <obb_json> <output_json>")
        sys.exit(1)

    aabb_file = sys.argv[1]
    obb_file = sys.argv[2]
    output_file = sys.argv[3]

    main(aabb_file, obb_file, output_file)