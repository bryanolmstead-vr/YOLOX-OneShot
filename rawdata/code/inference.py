from yolox.exp import get_exp
from yolox.utils import fuse_model
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
import torch
import cv2
import numpy as np

# Load experiment
exp = get_exp(exp_file="exps/example/yolox_mydata.py")
model = exp.get_model()
checkpoint = torch.load("YOLOX_outputs/mydata/best_ckpt.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

# Optional: fuse conv + bn for faster inference
model = fuse_model(model)

# Load image
img = cv2.imread("datasets/mydataset/val2017/img10.jpg")
img_info = {"id": 0}
height, width = img.shape[:2]

# Preprocess
transform = ValTransform(legacy=False)
img_input, _ = transform(img, None, exp.test_size)

img_input = torch.from_numpy(img_input).unsqueeze(0)

# Inference
with torch.no_grad():
    outputs = model(img_input)
    # Post-process: NMS, convert coords
    from yolox.utils import postprocess
    outputs = postprocess(outputs, exp.num_classes, conf_thre=0.25, nms_thre=0.45)

for box in outputs[0]:
    x1, y1, x2, y2, score, cls_id = box
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
    cv2.putText(img, f"{COCO_CLASSES[int(cls_id)]}:{score:.2f}", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imshow("Prediction", img)
cv2.waitKey(0)