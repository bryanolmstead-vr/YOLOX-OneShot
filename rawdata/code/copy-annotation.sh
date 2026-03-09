#!/usr/bin/bash

# copy annotation files from the raw data directory to the annotation directory

for f in c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/train2017/*.png; do
  base=$(basename "$f" .png)
  cp "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/rawdata/AI535-Images/converted/$base.txt" \
     "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/annotations/train2017/"
done

for f in c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/train2017/*.png; do
  base=$(basename "$f" .png)
  cp "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/rawdata/AI535-Images2/converted/$base.txt" \
     "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/annotations/train2017/"
done

for f in c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/val2017/*.png; do
  base=$(basename "$f" .png)
  cp "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/rawdata/AI535-Images/converted/$base.txt" \
     "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/annotations/val2017/"
done

for f in c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/val2017/*.png; do
  base=$(basename "$f" .png)
  cp "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/rawdata/AI535-Images2/converted/$base.txt" \
     "c:/Users/Bryan/Desktop/home/OSU/YOLOX-OneShot/datasets/OBB360/annotations/val2017/"
done

# manually copy the "three" files

#         should be   is   need to add
# train:  116         111  5
# val:     29         26   3

# manually did cat candy.three.450.53_640x640.txt cards.three.450.53_640x640.txt cheetos.three.450.53_640x640.txt > thre.450.53_640x640.txt
# the "three" files that are in validation set are manually moved and are these:
# three.450.50_640x640.txt
# three.450.52_640x640.txt
# three.600.51_640x640.txt