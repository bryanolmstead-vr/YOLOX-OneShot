# **AI-535 Final Project**

Bryan Olmstead

3/1/2026

## **YOLOX with One-Shot field training capability**

Real-time one-shot object detector providing full 360° oriented bounding box for vision guided robotic applications using siamese **YOLOX** network.

## **INTRODUCTION**

Vision guided robotics with a wrist-mounted camera can be used for pick and place applications for stationary or moving objects. While some applications require full 6 degree of freedom location of objects, many applications require only 5 degrees of freedom (x,y,z,w,h,θ), with the assumption that the object is rotated only around the camera axis.

An efficient image processing algorithm uses 2D image processing to determine the identity and location of objects within a 2D image with 4 degrees of freedom (x,y,w,h,θ) and uses depth from stereo to determine the distance z from the camera to the object, enabling the 3D location and orientation of the object to be determined with 5 degrees of freedom.

The **LINE2D** template-matching algorithm by [Multimodal Templates for Real-Time Detection of Texture-less Objects in Heavily Cluttered Scenes](http://www.stefan-hinterstoisser.com/papers/hinterstoisser2011linemod.pdf) by Stefan Hinterstoisser is an effective technique for finding objects in a 2D image. This algorithm can be invariant with respect to translation, rotation, and scale. It is used by **Visual Robotics** in their *Template Finder*, **MVTec** in their *HALCON* software, and in a modifed fashion by **Cognex** in their *PatMax* algorithm. 

The LINE2D algorithm is a brute-force algorithm that creates scaled and rotated templates consisting of dominant oriented edges. While it generally performs well, it can be brittle in real-world environments due to noise, illumination variations, and warping due to non-rigid objects. The algorithm is compute intensive, and is only moderately parallelizable. One of the beneficial features of the LINE2D is that templates can be created in the field, enabling a vision system to rapidly adapt to new objects.

It is desirable to develop an improved object detector that is more rugged in real-world environments - less sensitive to noise and illumination variations, and able to tolerate object deformation. It is desirable for the algorithm to be innately parallelizable so that it can run efficiently on edge-computing hardware such as a GPU. A neural network may provide that result.

## **YOLO and its limitations**

**YOLO** (You Only Look Once) is a family of CNNs designed for efficient object classification, detection, and segmentation. [YOLOv1](https://arxiv.org/abs/1506.02640) was developed by Joseph Redmon in 2016. Since then, many versions have been developed, primarily by [Ultralytics](https://docs.ultralytics.com/). YOLO provides high speed detection on **edge AI** hardware, which is beneficial for robotic pick and place applications. Most YOLO versions provide a so-called **axis-aligned bounding box (AABB)**. For robotic pick and place applications, a so-called **oriented bounding box (OBB)** is required. OBB versions of YOLO provide orientation within ±90° (180°) limits. For robotic pick and place applications, a full ±180° (360°) is required. YOLO, like most deep learning networks, requires objects to be pre-trained. This is incompatible with the desire to have a field trainable object detector. Commercial use **Ultralytics** versions of YOLO is problematic due to their use of AGPL-3.0 license, which requires open-sourcing of all source code for any project that uses it. It is desirable to develop a version of YOLO that uses a more permissive commercial friendly license, detects the orientation of objects with full 360° orientation, and supports field training of new objects.

## **YOLOX**

**YOLOX**, developed in 2021 [YOLOX GitHub Repo](https://github.com/Megvii-BaseDetection/YOLOX), was developed to be anchor-free (less restrictions on object location), high accuracy (as measured by mAP), more flexible and modular (with decoupled classification and regression heads), and uses a commercial friendly Apache-2.0 license. This forms the basis of the object detector developed in this project.

**YOLOX Paper (2021):** Zheng Ge, Songtao Liu, Feng Wang, Zeming Li, and Jian Sun. *YOLOX: Exceeding YOLO Series in 2021*. arXiv preprint arXiv:2107.08430. [PDF](https://arxiv.org/abs/2107.08430)

**One-Shot Detection**

In the deep learning community, **One-Shot Detection** is the term used to describe a network that detects new objects without being trained. One popular network architecture for implementing one-shot detection is **Siamese Networks** popularized by the paper [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf). Siamese Network techniques will be used to implement the detector for this project.

## **Project Requirements**

<table border="1">
  <tr style="background-color: lightgray;">
    <th>Parameter</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Target Hardware</td>
    <td>Movidius Myriad X (1TOPS)<br>Qualcomm QCS8550 (48TOPS)</td>
  </tr>
  <tr>
    <td>Orientation</td>
    <td>±180° (360°)</td>
  </tr>
  <tr>
    <td>Field Training</td>
    <td>100 templates</td>
  </tr>
</table>

## **Project Phases**

The project is developed in 7 phases.

<table border="1">
  <tr style="background-color: lightgray;">
    <th>Phase</th>
    <th>Description</th>
    <th style="width: 300px; word-wrap: break-word;">Details</th>
  </tr>
  <tr>
    <td>1</td>
    <td>Data set creation</td>
    <td>Create annotated dataset for novel objects</td>
  </tr>
  <tr>
    <td>2</td>
    <td>YOLOX AABB</td>
    <td>Train YOLOX AABB on novel objects as baseline</td>
  </tr>
  <tr>
    <td>3</td>
    <td>OBB 360° head</td>
    <td>Add (cosθ, sinθ) head to provide 360° OBB and train with novel objects</td>
  </tr>
  <tr>
    <td>4</td>
    <td>Cosine similarity head</td>
    <td>Add normalized dot product to compute cos(θ) similarity of template vector with feature map output</td>
  </tr>
  <tr>
    <td>5</td>
    <td>Location, orientation, NMS</td>
    <td>Process feature map to determine location and orientation while eliminating duplicate matches with NMS</td>
  </tr>
  <tr>
    <td>6</td>
    <td>Siamese network</td>
    <td>Improve performance by creating metric feature space with Siamese network</td>
  </tr>
  <tr>
    <td>7</td>
    <td>Deployment on edge hardware</td>
    <td>Deploy on MyriadX and QCS8550</td>
  </tr>
</table>

## **Phase 1 - Data Set**

<span style="color:red;">
Data set – 3 rotated and translated objects. Custom 360° OBB annotation tool. Novel objects: 3 new objects, rotated and translated. All images annotated with tool. Create AA bounding boxes for use with standard YOLOX. Letterboxing from 640x360 to 640x640. Augmentation – rotate, translate, lighting, noise. Divide into training and test set. 
</span>

## **Phase 2 - YOLOX AABB**

<span style="color:red;">
Prove that I can train the network for custom classes.
</span>

## **Phase 3 - OBB 360° head**

<span style="color:red;">
OBB head: (cos θ, sin θ) unit vector.
Loss function: dot product, how to normalize so vector tries to be unit vector.
Combined loss function.
Train existing classifier head on 3 objects. 
Train existing x,y,w,h head on 3 objects.
Configuration of network – YOLOX + OBB head.
Training hyperparameters.
Graph of training loss, test loss vs epochs.
Loss = 1 - normalized dot product = 1 – cos dθ ~ dθ
</span>


## **Phase 4 - Cosine similarity head**

<span style="color:red;">
Cosine similarity aka dot product aka 1x1 convolution.
Look at literature for other ideas.
No training needed.
Create rotated and scaled templates.
Plot heat maps vs translation, scale, rotation
How many templates are needed for good performance?
Test on novel objects.
</span>

## **Phase 5 - Location, orientation, NMS**

<span style="color:red;">
Process the similarity output to provide true object location
</span>

## **Phase 6 - Siamese network**

<span style="color:red;">
Look at literature about how to do it.
Create contrastive loss, which I think would be better than triplet loss.
Add network to end of FPN basically a metric head.
Train this network with all my data.
Test it on novel objects.
Compare performance with phase 3. 
</span>

## **Phase 7 - Deployment on edge hardware**

<span style="color:red;">
Figure out how to deploy on Movidius Myriad X.
Measure speed and performance.
Figure out how to deploy on OAK4-D.
Measure speed and performance.
</span>

## **Comments from Fuxin Li**

<span style="color:red;">
Expects 60%-70% performance compared to fully trained NN
Vision Language Model (VLM) might work better
Few-Shot is called Multi-Template Matching
See NVIDIA blog: Detecting Rotated Objects Using the NVIDIA Object Detection Toolkit _ NVIDIA Technical Blog
Look at https://github.com/NVIDIA/retinanet-examples/tree/main
</span>

## **Journal**

**2/9/26** – 2hrs

Set up YOLOX on laptop. Working in VS Code.
Created miniconda environments: 
```
opencv-env
yolox
```
Got repo from https://github.com/Megvii-BaseDetection/YOLOX

Detections: car in class, cards not in class…
![First Detection](media/first-detections-2026.02.09.jpg)

```
python tools/demo.py image -n yolox-s -c yolox_s.pth --path car.png --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
```
```
python tools/demo.py image -n yolox-s -c yolox_s.pth --path 'C:\Users\Bryan\OneDrive - Visual Robotic Systems, Inc\Visual Robotics\OSU\AI535\project\AI535-Images\cards.450.03.png' --conf 0.25 --nms 0.45 --tsize 640 --save_result --device cpu
```
   
**2/10/26** – 1hr

Created captures of M&Ms, Cards, Cheetos. About 40 images, different translation, rotation, scale. Used RPC image capture program with VIM-303.

**2/11/26** – 2hrs

Created annotation tool in Python. Annotated OBB with (cx,cy,w,h,θ).

**2/13/26**

Created GitHub repo: https://github.com/bryanolmstead-vr/YOLOX-fork 

**2/16/26** – 1hr

Gave presentation in AI535 class. Fuxin’s comment was to use Siamese networks.

**2/19/26** – 3hrs

Created the framework of original Word document.

Created AI535 project GitHub repo: https://github.com/bryanolmstead-vr/YOLOX-OneShot 

Created branch protection rule on main. Annotated 2nd set of images. Decided that letterboxing (just center the 640x360 image inside 640x640 with black otherwise) is the best plan. Need to scale annotation as well (height and Y coordinate). Letterboxed YOLOX-Std dataset with AA bounding box.

**2/20/26** – 1hr

Created data set AI535-Images3 which has a white background. Has images of 3 objects at a time: candy, cards, cheetos. Has images of new objects: vention plates, APMC 402-403 parts, cereal, rice, noodle roni, milk duds, individual playing cards, pretzels, doritos, lays.

**2/25/26** – 2hrs

Need to have multiple annotations for images that have multiple objects. 
Combined the annotations – one per line in the text file – for the multi-object images. Updated the show_annotation_aa.py script to show multiple objects and display filename. Installed albumentations (for augmentation).

```
conda activate yolox
conda install -c conda-forge albumentations
python
import albumentations as A
print(A.__version__)
2.08
```

Apparently the YOLOX training script already does augmentation.


**3/1/2026** - 2hrs

Converted Word document into this README file that will get printed as a PDF file.

**3/2/2026** - 1.5hrs

Splitting up data set between train and validation sets. converting annotation to COCO format for training.

There are 145 files: split 80% training, 20% validation.
I chose 29 files for validation.

The label files need to be turned into a COCO style file:

```
YOLO:	class_id x_center y_center width height   (normalized 0–1)
COCO:	x_min = (x_center - width/2) * image_width
y_min = (y_center - height/2) * image_height
box_width  = width * image_width
box_height = height * image_height
```

Created `yolo2coco.py` to perform the conversion

Label files were converted to COCO format using `yolo2coco.py`

Created directory structure
```
datasets/COCO
 ├── train2017/ candy.200.01_640x640.png
 ├── val2017/   candy.300.02_640x640.png
 └── annotations/
      ├── instances_train2017.json
      ├── instances_val2017.json
      ├── train2017/ candy.200.01_640x640.txt
      └── val2017/   candy.300.02_640x640.txt
```

**3/3/2026** - 1hr

My directory structure:

```
OSU
 ├── YOLOX          fork of YOLOX
 ├── YOLOX-OneShot  my repo for AI535
 ```

To start with, I will just modify the YOLOX files and point to the dataset in YOLOX-OneShot

```
YOLOX/exps/example/custom/yolox_s.py
copy to
yolox_s_3classes.py
edit:
        self.data_dir = "../../YOLOX-OneShot/datasets/COCO3"
        self.train_ann = "annotations/instances_train2017.json"
        self.val_ann = "annotations/instances_val2017.json"
        self.num_classes = 3
```
 
To train:
I have 116 images in training dataset.

Proposed hyperparameters (put in yolox_s_3classes.py):
```
Workers = 4
Epochs  = 100
Eval Interval = 1
Weight Decay = 5e-4
Warmup Epochs = 3
Input Size = (640x640)
Random Size = 18x32 to 22x32: 576x576 to 704x704      
```

Training command:

```
be in YOLOX directory:
d=1  number of gpus
b=8  batch size
fp16 precision
-o   use LARS optimizer warmup override
-c   pretrained weights

python tools/train.py -f exps/example/custom/yolox_s_3classes.py -d 1 -b 8 --fp16 -o -c yolox_s.pth
```

My first training attempt on my laptop failed with an opencv error. It can be replicated by a single import line:

```
python
Python 3.10.19 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 16:41:31) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Ctrl click to launch VS Code Native REPL
>>> from yolox.utils import configure_models


Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\Bryan\Desktop\home\OSU\yolox\yolox\utils\__init__.py", line 8, in <module>
    from .demo_utils import *
  File "C:\Users\Bryan\Desktop\home\OSU\yolox\yolox\utils\demo_utils.py", line 7, in <module>
    import cv2
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\cv2\__init__.py", line 181, in <module>
    bootstrap()
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\cv2\__init__.py", line 153, in bootstrap
    native_module = importlib.import_module("cv2")
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
ImportError: DLL load failed while importing cv2: The operating system cannot run %1.
```

ChatGPT offers this suggestion:
```
Step 1 — Install VC++ Redistributable x64: 
  Download & install: VC++ Redistributable 2015–2022 x64. 
  Reboot your computer after installation. 
  This fixes 90% of OpenCV “DLL load failed” issues on Windows.
Step 2:
  conda activate yolox
  conda remove opencv
  conda install -c conda-forge opencv
Step 3:
  cd C:\Users\Bryan\Desktop\home\OSU\yolox
  python -c "from yolox.utils import demo_utils; import cv2; print(cv2.__version__)"
  Must succeed without the DLL error.
```
I'm trying just step 2 first. then i did:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
cd YOLOX
pip3 install -v -e .
```

the pip3 install uninstalled everything i did previously

**3/4/2026** - 3:30pm

```
PS C:\Users\Bryan\Desktop\home\OSU\yolox\tools> python
Python 3.10.13 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:15:57) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Ctrl click to launch VS Code Native REPL
>>> from yolox.utils import configure_models

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "<stdin>", line 1, in <module>
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\__init__.py", line 5, in <module>
    from .boxes import *
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\boxes.py", line 7, in <module>
    import torchvision
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\__init__.py", line 6, in <module>
    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\models\__init__.py", line 2, in <module>
    from .convnext import *
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\models\convnext.py", line 8, in <module>
    from ..ops.misc import Conv2dNormActivation, Permute
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\ops\__init__.py", line 23, in <module>
    from .poolers import MultiScaleRoIAlign
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\ops\poolers.py", line 10, in <module>
    from .roi_align import roi_align
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\ops\roi_align.py", line 4, in <module>
    import torch._dynamo
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\__init__.py", line 64, in <module>
    torch.manual_seed = disable(torch.manual_seed)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\decorators.py", line 50, in disable
    return DisableContext()(fn)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\eval_frame.py", line 410, in __call__
    (filename is None or trace_rules.check(fn))
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 3378, in check
    return check_verbose(obj, is_inlined_call).skipped
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 3361, in check_verbose
    rule = torch._dynamo.trace_rules.lookup_inner(
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 3442, in lookup_inner
    rule = get_torch_obj_rule_map().get(obj, None)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 2782, in get_torch_obj_rule_map        
    obj = load_object(k)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 2811, in load_object
    val = _load_obj_from_str(x[0])
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 2795, in _load_obj_from_str
    return getattr(importlib.import_module(module), obj_name)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\nested\_internal\nested_tensor.py", line 417, in <module>
    values=torch.randn(3, 3, device="meta"),
C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\nested\_internal\nested_tensor.py:417: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at ..\torch\csrc\utils\tensor_numpy.cpp:84.)
  values=torch.randn(3, 3, device="meta"),
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\__init__.py", line 8, in <module>
    from .demo_utils import *
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\demo_utils.py", line 7, in <module>
    import cv2
ImportError: DLL load failed while importing cv2: The specified module could not be found.
```
let's debug why this is happening:

```
from yolox.utils import configure_models
__init__.py
from .boxes import *
import torchvision
from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
from .convnext import *
from ..ops.misc import Conv2dNormActivation, Permute
from .poolers import MultiScaleRoIAlign
from .roi_align import roi_align
import torch._dynamo
torch.manual_seed = disable
decorators.py", line 50, in disable: return DisableContext()(fn)
eval_frame.py", line 410, in __call__ (filename is None or trace_rules.check(fn))
trace_rules.py", line 3378, in check: return check_verbose(obj, is_inlined_call).skipped
trace_rules.py: check_verbose: rule = torch._dynamo.trace_rules.lookup_inner
trace_rules.py", line 3442, in lookup_inner: rule = get_torch_obj_rule_map().get(obj, None)
trace_rules.py", line 2782, in get_torch_obj_rule_map: obj = load_object(k)
trace_rules.py", line 2811, in load_object: val = _load_obj_from_str(x[0])
trace_rules.py", line 2795, in _load_obj_from_str: return getattr(importlib.import_module(module), obj_name)
in import_module: return _bootstrap._gcd_import(name[level:], package, level)
nested_tensor.py", line 417, in <module>: values=torch.randn(3, 3, device="meta"),
nested_tensor.py:417: UserWarning: Failed to initialize NumPy: _ARRAY_API not found 
tensor_numpy.cpp:84.): values=torch.randn(3, 3, device="meta"),
from .demo_utils import *
import cv2
ImportError: DLL load failed while importing cv2: The specified module could not be found.
```
1. wants numpy1 vs numpy2
2. seems to not have cv2

redoing YOLOX installation:
```
cd YOLOX
pip3 install -v -e .
```

gives these errors/warnings:
```
1. unable to import torch, pre-compiling ops will be disabled
2. requirement already satisfied:
     numpy 2.2.6
     torch 2.3.1
     opencv_python 4.13.0.92
     loguru 0.7.3
     tqdm 4.67.3
     torchvision 0.18.1
     thop 0.0.1
     ninja 1.13.0
     psutil 7.2.2
     pycocotools 2.0.11
     onnx 1.20.1
     onnx-simplifier 0.4.10
     rich 14.3.2
     protobuf 6.33.5
     typing_extensions 4.15.0
     ml_dtypes 0.5.4
     filelock 3.20.3
     sympy 1.40.0
     networkx 3.2.1
     jinja2 3.1.6
     fsspec 2026.2.0
     mkl 2021.4.0
     intel-openmp 2021.4.0
     tbb 2021.13.1
     MarkupSafe 3.0.2
     colorama 0.4.6
     win32-setctime 1.2.0
     markdown-it-py 4.0.0
     pygments 2.19.2
     mdurl 0.1.2
     mpmath 1.3.0
     absl-py 2.4.0
     grpcio 1.78.0
     markdown 3.10.2
     packaging 25.0
     pillow 11.1.0
     setuptools 80.10.2
     tensorboard-data-server 0.7.2
     werkzeug 3.1.5
3. warning package yolox.layers.cocoeval is absent from packages configuration
```

testing via:
```
python (3.10.13)
import cv2: DLL load failed while importing cv2
pip install opencv-python: requirement already satisfied
pip uninstall opencv-python opencv-python-headless -y
conda install -c conda-forge opencv
```
chatGPT suggested that i uninstall python 3.13 because it might be confusing the dll loading.
it didn't help. The result of a long process:
```
then it suggested i install numpy<2
pip uninstall numpy -y
conda install numpy=1.26 -c conda-forge
conda remove opencv -y
conda install -c conda-forge opencv
now it wants me to install Microsoft Visual C++ Redistributable for Visual Studio 2015–2022
https://aka.ms/vc14/vc_redist.x64.exe 
nope didn't work
conda remove opencv libopencv py-opencv -y
pip install opencv-python
python -c "import cv2; print(cv2.__version__)"
4.13.0
```
I think I'm ready to try the training script again.

In YOLOX directory:
```
python tools/train.py -f exps/example/custom/yolox_s_3classes.py -d 1 -b 8 --fp16 -o -c yolox_s.pth
```

Fails this way:
```
python tools/train.py -f exps/example/custom/yolox_s_3classes.py -d 1 -b 8 --fp16 -o -c yolox_s.pth
Traceback (most recent call last):
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\tools\train.py", line 13, in <module>
    from yolox.core import launch
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\core\__init__.py", line 5, in <module>
    from .launch import launch
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\core\launch.py", line 16, in <module>
    import yolox.utils.dist as comm
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\__init__.py", line 5, in <module>
    from .boxes import *
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\boxes.py", line 7, in <module>
    import torchvision
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\__init__.py", line 6, in <module>
    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\datasets\__init__.py", line 1, in <module>
    from ._optical_flow import FlyingChairs, FlyingThings3D, HD1K, KittiFlow, Sintel
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\datasets\_optical_flow.py", line 10, in <module>
    from PIL import Image
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\PIL\Image.py", line 100, in <module>
    from . import _imaging as core
ImportError: DLL load failed while importing _imaging: The specified module could not be found.
```
Apparently this is a pillow issue.
```
pip uninstall pillow -y
pip install --upgrade pillow
```
Didn't work
```
pip uninstall pillow -y
conda remove pillow -y
pip install --upgrade pillow
```
Had a cuda error. do this:
```
python tools/train.py -f exps/example/custom/yolox_s_3classes.py -d 1 -b 8 -o -c yolox_s.pth --fp16=False
```
This error:
```
python tools/train.py -f exps/example/custom/yolox_s_3classes.py -d 1 -b 8 -o -c yolox_s.pth --fp16=False

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.6 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\tools\train.py", line 13, in <module>
    from yolox.core import launch
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\core\__init__.py", line 5, in <module>
    from .launch import launch
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\core\launch.py", line 16, in <module>
    import yolox.utils.dist as comm
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\__init__.py", line 5, in <module>
    from .boxes import *
  File "C:\Users\Bryan\Desktop\home\OSU\YOLOX\yolox\utils\boxes.py", line 7, in <module>
    import torchvision
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\__init__.py", line 6, in <module>
    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\models\__init__.py", line 2, in <module>
    from .convnext import *
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\models\convnext.py", line 8, in <module>
    from ..ops.misc import Conv2dNormActivation, Permute
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\ops\__init__.py", line 23, in <module>
    from .poolers import MultiScaleRoIAlign
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\ops\poolers.py", line 10, in <module>
    from .roi_align import roi_align
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torchvision\ops\roi_align.py", line 4, in <module>
    import torch._dynamo
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\__init__.py", line 64, in <module>
    torch.manual_seed = disable(torch.manual_seed)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\decorators.py", line 50, in disable
    return DisableContext()(fn)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\eval_frame.py", line 410, in __call__
    (filename is None or trace_rules.check(fn))
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 3378, in check
    return check_verbose(obj, is_inlined_call).skipped
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 3361, in check_verbose
    rule = torch._dynamo.trace_rules.lookup_inner(
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 3442, in lookup_inner
    rule = get_torch_obj_rule_map().get(obj, None)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 2782, in get_torch_obj_rule_map
    obj = load_object(k)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 2811, in load_object
    val = _load_obj_from_str(x[0])
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\_dynamo\trace_rules.py", line 2795, in _load_obj_from_str
    return getattr(importlib.import_module(module), obj_name)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\importlib\__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\nested\_internal\nested_tensor.py", line 417, in <module>
    values=torch.randn(3, 3, device="meta"),
C:\Users\Bryan\miniconda3\envs\yolox\lib\site-packages\torch\nested\_internal\nested_tensor.py:417: UserWarning: Failed to initialize NumPy: _ARRAY_API not found (Triggered internally at ..\torch\csrc\utils\tensor_numpy.cpp:84.)
  values=torch.randn(3, 3, device="meta"),
usage: YOLOX train parser [-h] [-expn EXPERIMENT_NAME] [-n NAME] [--dist-backend DIST_BACKEND] [--dist-url DIST_URL] [-b BATCH_SIZE] [-d DEVICES] [-f EXP_FILE] [--resume] [-c CKPT]
                          [-e START_EPOCH] [--num_machines NUM_MACHINES] [--machine_rank MACHINE_RANK] [--fp16] [--cache [CACHE]] [-o] [-l LOGGER]
                          ...
YOLOX train parser: error: argument --fp16: ignored explicit argument 'False'
```
chatGPT says to do this:
```
pip install "numpy<2"
```
That didn't work.
All this because I really just need a GPU. Just use Google Colab!









<span style="color:red;">
to do inference:
</span>

```
YOLOX_outputs/
 └── mydata/
      └── best_ckpt.pth   # usually the best validation model

python tools/demo.py image \
    -f exps/example/yolox_mydata.py \
    -c YOLOX_outputs/mydata/best_ckpt.pth \
    --path datasets/mydataset/val2017/img10.jpg \
    --conf 0.25 \
    --nms 0.45 \
    --tsize 640 \
    --save_result
```

do an entire folder:

```
python tools/demo.py image \
    -f exps/example/yolox_mydata.py \
    -c YOLOX_outputs/mydata/best_ckpt.pth \
    --path datasets/mydataset/val2017 \
    --conf 0.25 \
    --nms 0.45 \
    --tsize 640 \
    --save_result
```