OLOv8n Industrial Fire & Smoke Detection AI

Personal Technology & Learning Reference

Purpose of this README: This is my personal reference for
understanding the technologies used in the Industrial Fire & Smoke
Detection project.
When I encounter any of these technologies in a future GitHub project,
I should be able to come back here and quickly remember what it is,
why it is used, and where it appears in this project.

1. Project at a Glance

This project is a hybrid computer-vision system for detecting:

Fire

Smoke

People

Cars

Buses

Trucks

It combines several approaches:

Video / Camera
      |
      v
OpenCV reads frames
      |
      v
NumPy image arrays
      |
      +-------------------+-------------------+
      |                   |                   |
      v                   v                   v
Fire Detector       Smoke Detector       YOLOv8n
Classical CV        Classical CV         Deep Learning
      |                   |                   |
      +-------------------+-------------------+
                          |
                          v
                   Risk Classifier
                   Rule-based logic
                          |
                          v
                    HUD Renderer
                          |
                          v
                Display + Output Video

The important idea is:

OpenCV processes pixels, NumPy represents and manipulates numerical
image data, classical CV detects fire/smoke, YOLO detects general
objects, rule-based logic determines risk, and the HUD displays the
result.

2. Technology Stack

Technology              What it is               Why this project uses
it

Python                  Programming language     Entire application

OOP                     Programming design       Separates detectors and
approach                 components

pip                     Python package installer Installs dependencies

virtual environment     Isolated Python          Keeps project
environment              dependencies separate

NumPy                   Numerical array library  Represents and
processes images

OpenCV                  Computer vision library  Reads video, processes
frames, draws UI

PyTorch                 Deep learning framework  Backend used by YOLO

CUDA                    NVIDIA GPU computing     Accelerates inference
platform                 when available

FP16                    Half-precision floating  Faster/lower-memory GPU
point                    inference

Ultralytics             YOLO                     Loads and runs YOLOv8n
implementation/package

YOLOv8n                 Object detection neural  Detects people and
network                  vehicles

HSV                     Color representation     Fire/smoke color
segmentation

Morphology              Classical image          Cleans binary masks
processing

Contours                Shape/boundary           Finds candidate
extraction               fire/smoke regions

MOG2                    Background subtraction   Helps detect moving
algorithm                smoke

Frame differencing      Temporal CV technique    Detects changes between
frames

argparse                Python CLI library       Allows configurable
command-line input

VideoWriter             OpenCV video output      Saves processed video

3. Python

What is Python?

Python is the main programming language used to build this project.

Examples from the project:

import cv2
import numpy as np
import torch

Python is responsible for:

Program flow

Functions

Classes

Object creation

Configuration

Error handling

Model execution

Video pipeline

User input

Important Python concepts used here

Variables

DW = 1280
DH = 720

A variable stores a value.

Dictionary

CFG = {
    "display_w": 1280,
    "display_h": 720,
    "yolo_conf": 0.40
}

A dictionary stores key-value pairs.

CFG["yolo_conf"]

returns:

0.40

List

[0, 2, 5, 7]

An ordered collection.

In this project these are YOLO class IDs.

Tuple

(0, 30, 255)

Used for fixed groups of values, such as OpenCV colors.

Function

def classify_risk(fire_conf, smoke_conf):
    ...

A reusable block of logic.

Class

class FireDetector:
    ...

A reusable blueprint containing data and behavior.

Object

fire_det = FireDetector()

Creates an instance of the class.

self

self.model
self._prev_gray

self refers to the current object instance.

__init__

def __init__(self):
    ...

Initialization method called when an object is created.

4. Object-Oriented Programming (OOP)

The project is organized into separate classes.

FireDetector
SmokeDetector
SceneDetector
HUDRenderer
FPSCounter
ExpSmooth

Why?

This is called separation of concerns.

Each component has a focused responsibility.

FireDetector
    -> fire detection

SmokeDetector
    -> smoke detection

SceneDetector
    -> YOLO object detection

HUDRenderer
    -> visualization

FPSCounter
    -> performance measurement

This makes the project easier to:

Understand

Debug

Test

Modify

Extend

Important OOP idea

Instead of one giant function:

main()

the project creates specialized objects:

fire_det = FireDetector()
smoke_det = SmokeDetector()
scene_det = SceneDetector()
hud = HUDRenderer(...)

Then the objects communicate through methods:

fire_det.run(frame)
smoke_det.run(frame)
scene_det.run(frame)

5. Python Modules and Imports

The project uses Python's standard library:

import sys
import subprocess
import importlib
import os
import time
import math
import argparse
import warnings

And external packages:

import cv2
import numpy as np
import torch

General idea:

Python program
     |
     +-- Standard library
     |
     +-- Third-party packages
     |
     +-- My own classes/functions

6. pip and Dependencies

The project automatically checks for packages.

_REQUIRED = {
    "cv2": "opencv-python",
    "numpy": "numpy",
    "torch": "torch",
    "torchvision": "torchvision",
    "PIL": "pillow",
    "ultralytics": "ultralytics",
}

Important distinction:

Package installed with pip      Import name

opencv-python                   cv2
pillow                          PIL
ultralytics                     ultralytics
numpy                           numpy
torch                           torch

Example:

pip install opencv-python

then:

import cv2

7. Virtual Environments

A virtual environment creates an isolated Python environment for a
project.

Typical workflow:

python -m venv .venv

Activate it, then:

pip install ultralytics opencv-python numpy

Why?

Different projects may require different package versions.

Project A
    -> package version X

Project B
    -> package version Y

A virtual environment prevents dependency conflicts.

Recommended future habit

For serious projects:

project/
├── .venv/
├── main.py
├── requirements.txt
└── README.md

8. NumPy

What is NumPy?

NumPy is a numerical computing library.

The key object is:

numpy.ndarray

Images are represented as NumPy arrays.

For example:

image.shape

might return:

(720, 1280, 3)

Meaning:

720    = height
1280   = width
3      = color channels

9. Image = Numerical Data

A digital image is essentially a grid of numbers.

One pixel in OpenCV BGR format might be:

[20, 100, 240]

Meaning approximately:

Blue  = 20
Green = 100
Red   = 240

An image can therefore be represented as:

Height x Width x Channels

For example:

720 x 1280 x 3

This concept is fundamental to computer vision.

10. OpenCV

What is OpenCV?

OpenCV is a computer-vision library.

In this project:

import cv2

It is responsible for:

Reading images

Reading videos

Reading cameras

Resizing frames

Color conversion

Masks

Morphology

Contours

Drawing boxes

Drawing text

Showing windows

Saving videos

Important functions:

cv2.imread()
cv2.VideoCapture()
cv2.resize()
cv2.cvtColor()
cv2.inRange()
cv2.findContours()
cv2.boundingRect()
cv2.contourArea()
cv2.rectangle()
cv2.putText()
cv2.imshow()
cv2.waitKey()
cv2.imwrite()
cv2.VideoWriter()

11. Video Processing

A video is processed as a sequence of frames.

Video
 |
 +-- Frame 1
 +-- Frame 2
 +-- Frame 3
 +-- Frame 4
 +-- ...

The project uses:

cap = cv2.VideoCapture(source)

Then:

ret, raw = cap.read()

Each iteration gets another frame.

The pipeline is:

Read frame
    |
Resize
    |
Fire detection
    |
Smoke detection
    |
YOLO detection
    |
Risk classification
    |
Draw HUD
    |
Display
    |
Save
    |
Next frame

12. BGR vs RGB

A common computer-vision issue:

OpenCV normally represents color as:

BGR

not:

RGB

So:

cv2.imread(...)

produces BGR data.

If another library expects RGB, conversion may be required:

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

Always check the expected color format when combining libraries.

13. HSV Color Space

The project converts:

hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

HSV means:

H = Hue
S = Saturation
V = Value

Hue

Represents the type of color.

Saturation

Represents color intensity/purity.

Value

Represents brightness.

HSV is useful for color segmentation because color and brightness are
represented separately.

14. Fire Detection

The fire detector is classical computer vision, not a neural
network.

The pipeline is:

Frame
 |
BGR -> HSV
 |
HSV thresholding
 |
Binary mask
 |
Morphological cleanup
 |
Contours
 |
Area filtering
 |
Brightness check
 |
Bounding boxes
 |
Fire confidence

Configuration:

fire_h1_lo = np.array([0, 120, 120])
fire_h1_hi = np.array([22, 255, 255])

fire_h2_lo = np.array([22, 80, 180])
fire_h2_hi = np.array([40, 255, 255])

These define HSV ranges considered fire-like.

Important limitation:

This detector does not truly understand fire. It identifies pixels and
regions that satisfy predefined visual rules.

Therefore it can produce false positives from things with similar
colors.

15. cv2.inRange()

Used to create a binary mask.

mask = cv2.inRange(
    hsv,
    lower_bound,
    upper_bound
)

Conceptually:

Pixel inside range
    -> 255

Pixel outside range
    -> 0

Result:

White = candidate
Black = rejected

This is called thresholding / segmentation.

16. Binary Mask

A mask is an image used to select regions.

Example:

0   0   255 255
0   255 255 0
0   0   0   0

Usually:

0   = background/rejected
255 = selected

Masks are heavily used in classical computer vision.

17. Morphological Operations

The project uses:

cv2.morphologyEx(...)

Morphology operates on shapes in binary images.

The project uses:

MORPH_CLOSE

Useful for closing small gaps and holes.

MORPH_OPEN

Useful for removing small noise.

General idea:

Raw mask
   |
Morphological cleanup
   |
Cleaner mask

Morphology is based on operations such as:

Erosion

Dilation

Opening

Closing

18. Contours

The project uses:

cv2.findContours(...)

A contour represents a boundary around a connected region.

Pipeline:

Binary mask
     |
findContours()
     |
Detected shapes

Then:

cv2.contourArea(cnt)

calculates area.

The project rejects tiny regions:

if area < CFG["fire_min_area"]:
    continue

This reduces noise.

19. Bounding Boxes

The project uses:

x, y, bw, bh = cv2.boundingRect(cnt)

This produces:

x      = left
y      = top
bw     = width
bh     = height

Then:

(x, y, x + bw, y + bh)

becomes:

(x1, y1, x2, y2)

This is a common representation of bounding boxes.

20. Fire Confidence

The project calculates a custom fire confidence:

raw_conf = min(
    1.0,
    total_area / max(1, h * w * 0.05)
)

This is a rule-based score.

It is NOT the same thing as neural-network confidence.

It is essentially asking:

How much of the frame is occupied by regions that satisfy my fire
rules?

The score is then smoothed.

21. Exponential Smoothing

The project contains:

class ExpSmooth:

and:

self._v = (
    self.alpha * x
    + (1.0 - self.alpha) * self._v
)

This is an exponential moving average.

Purpose:

Frame 1 -> 0.10
Frame 2 -> 0.80
Frame 3 -> 0.20
Frame 4 -> 0.70

Raw values can jump around.

Smoothing makes the output more stable.

22. Smoke Detection

Smoke detection uses a more complicated classical-CV pipeline.

It combines:

Grayscale
+
HSV
+
MOG2 background subtraction
+
Frame differencing
+
Thresholding
+
Morphology
+
Contours

23. Grayscale

gray = cv2.cvtColor(
    frame_bgr,
    cv2.COLOR_BGR2GRAY
)

Instead of three color channels, grayscale has one intensity channel.

Conceptually:

Color image
   |
   v
Brightness/intensity image

Useful when color isn't necessary for a particular operation.

24. Background Subtraction

The project uses:

cv2.createBackgroundSubtractorMOG2(...)

MOG2 is a background-subtraction technique.

The basic idea:

Learn background
      |
Compare current frame
      |
Find regions that changed

This helps identify moving regions.

25. Frame Differencing

The project also does:

diff = cv2.absdiff(
    gray,
    self._prev_gray
)

It compares:

Current frame
       vs
Previous frame

If a region changed significantly, the difference becomes larger.

Pipeline:

Previous frame
       |
       +---- compare ----+
                         |
Current frame -----------+
                         |
                         v
                    Difference

This introduces a very important concept:

Temporal information

We're no longer looking at only one image. We're looking at how images
change over time.

26. Gaussian Blur

The project uses:

cv2.GaussianBlur(...)

Blurring smooths an image and reduces high-frequency noise.

It is commonly used before thresholding or edge/difference operations.

27. Thresholding

The project uses:

cv2.threshold(...)

Conceptually:

Value > threshold
    -> 255

Value <= threshold
    -> 0

This converts continuous values into a binary mask.

28. PyTorch

The project imports:

import torch

PyTorch is a deep-learning framework.

YOLO uses a deep-learning framework underneath.

PyTorch provides concepts such as:

Tensors

Neural networks

GPU computation

Model parameters

Training

Inference

Automatic differentiation

This project primarily uses PyTorch for model execution and device
handling.

29. Tensor

A tensor is a multidimensional numerical data structure.

Think:

Scalar
   -> one number

Vector
   -> 1D numbers

Matrix
   -> 2D numbers

Tensor
   -> potentially many dimensions

Images, batches and neural-network activations are commonly represented
as tensors.

30. GPU and CUDA

The project checks:

torch.cuda.is_available()

If available:

torch.device("cuda")

is selected.

Conceptually:

CPU
 |
General-purpose computation

GPU
 |
Massively parallel numerical computation

Deep-learning inference often benefits greatly from GPU acceleration.

CUDA is NVIDIA's platform for GPU computing.

31. FP16

The project contains:

"fp16": True

and:

USE_FP16 = USE_GPU and CFG["fp16"]

FP16 means 16-bit floating point.

Traditional neural-network computation often uses FP32.

FP32 -> 32-bit floating point
FP16 -> 16-bit floating point

FP16 can reduce memory usage and can improve inference speed on
compatible GPUs, with tradeoffs in numerical precision.

The project passes:

half=USE_FP16

to YOLO inference.

32. Ultralytics

The project uses:

from ultralytics import YOLO

Ultralytics provides the Python interface used to load and run YOLO
models.

Example:

model = YOLO("yolov8n.pt")

Then:

results = model(
    frame,
    conf=0.40,
    classes=[0, 2, 5, 7]
)

Ultralytics handles much of the model inference machinery.

33. YOLOv8n

YOLO = You Only Look Once.

YOLO is an object-detection family.

The project uses:

YOLOv8n

The n means nano, referring to a small model variant.

YOLO is fundamentally different from the project's fire-color detector.

Classical fire detection

Pixel properties
     |
Rules
     |
Candidate region

YOLO

Image
  |
Neural network
  |
Learned features
  |
Object predictions
  |
Bounding boxes
+
Class
+
Confidence

34. Object Detection

Object detection answers:

What objects are present, and where are they?

Example:

PERSON 92%
CAR    87%
TRUCK  81%

Each detection has:

class
confidence
bounding box

In the project:

{
    "cls": cls,
    "conf": conf,
    "box": (x1, y1, x2, y2)
}

35. YOLO Classes

The project uses:

[0, 2, 5, 7]

and maps them to:

0 -> person
2 -> car
5 -> bus
7 -> truck

Only these classes are requested for this application's scene context.

36. YOLO Confidence

This:

conf=0.40

is a model inference threshold.

It is different from the custom fire confidence.

Important distinction:

Fire confidence
    -> calculated by this project's classical-CV rules

Smoke confidence
    -> calculated by this project's classical-CV rules

YOLO confidence
    -> associated with a neural-network detection

Do not treat them as the same measurement.

37. YOLO Architecture Concepts to Learn

For deeper YOLO understanding, learn these concepts:

Input image
     |
Backbone
     |
Neck
     |
Detection head
     |
Predictions
     |
Confidence / classes / boxes
     |
NMS
     |
Final detections

Important topics:

Backbone

Feature extraction

Neck

Detection head

Multi-scale detection

Bounding boxes

IoU

Confidence

NMS

Transfer learning

Training

Validation

Precision

Recall

mAP

These are the next YOLO topics to study after understanding the current
inference pipeline.

38. IoU

IoU = Intersection over Union

Used to compare two bounding boxes.

Conceptually:

             Intersection
IoU = ---------------------------
             Union

IoU ranges from:

0 -> no overlap
1 -> perfect overlap

IoU is important in object detection evaluation and NMS.

39. NMS

NMS = Non-Maximum Suppression.

Purpose:

Remove duplicate overlapping detections.

Example:

Person 95%
Person 89%
Person 72%

If all three boxes represent the same person, NMS can retain the
strongest detection and suppress redundant ones.

40. Rule-Based Risk Classifier

The project does not use machine learning for risk classification.

It uses explicit rules:

combined = max(
    fire_conf,
    smoke_conf * 0.7 + fire_conf * 0.3
)

Then:

>= 0.55 -> CRITICAL
>= 0.30 -> WARNING
>= 0.10 -> CAUTION
else    -> CLEAR

This is a rule-based decision system.

Important distinction:

YOLO
-> learned from data

Risk classifier
-> explicitly programmed rules

41. HUD Renderer

The HUDRenderer class is responsible for visualization.

It draws:

Fire boxes

Smoke boxes

YOLO object boxes

Confidence

Risk level

FPS

Latency

Frame number

GPU mode

Status panel

Warning banners

Scan lines

Timestamp

This is not the AI itself.

It is the presentation layer.

42. FPS

FPS = Frames Per Second

Example:

30 FPS

means approximately 30 frames processed/displayed per second.

The project uses:

class FPSCounter:

to calculate it.

43. Latency

Latency measures how long processing takes.

The project calculates:

lat_ms = (
    time.perf_counter() - t0
) * 1000.0

Example:

LAT: 35 ms

means that processing took approximately 35 milliseconds for that
measurement.

FPS and latency are related but not identical.

44. Frame Skipping

The project does YOLO every second frame:

if frame_idx % 2 == 0:
    scene_dets = scene_det.run(frame)

Why?

YOLO inference is computationally expensive.

Instead of:

Frame 1 -> YOLO
Frame 2 -> YOLO
Frame 3 -> YOLO
Frame 4 -> YOLO

it does approximately:

Frame 1 -> skip
Frame 2 -> YOLO
Frame 3 -> skip
Frame 4 -> YOLO

This is a simple performance optimization.

45. argparse

The project uses:

import argparse

This lets the program accept command-line arguments.

Example:

python main.py 0

Here:

0 -> webcam

Or:

python main.py video.mp4

Or:

python main.py video.mp4 --no-yolo

This is useful for making programs configurable without changing source
code.

46. VideoWriter

The project uses:

cv2.VideoWriter(...)

to save processed frames.

Pipeline:

Input video
    |
Process frames
    |
Draw detections
    |
VideoWriter
    |
output_detected.mp4

47. try / except / finally

The main pipeline uses:

try:
    ...
except KeyboardInterrupt:
    ...
finally:
    ...

Purpose:

try -> execute normal code

except -> handle an exception

finally -> perform cleanup regardless of errors

Cleanup includes:

cap.release()
writer.release()
cv2.destroyAllWindows()

This is important in real applications because cameras, files and
windows should be properly released.

48. Project Architecture

Current architecture:

main.py
 |
 +-- Configuration
 |
 +-- Device detection
 |
 +-- FPS utilities
 |
 +-- FireDetector
 |
 +-- SmokeDetector
 |
 +-- SceneDetector
 |
 +-- Risk classifier
 |
 +-- HUDRenderer
 |
 +-- Video pipeline

A future professional refactor could become:

fire_smoke_ai/
│
├── main.py
├── config.py
├── detectors/
│   ├── fire_detector.py
│   ├── smoke_detector.py
│   └── yolo_detector.py
│
├── vision/
│   ├── preprocessing.py
│   └── morphology.py
│
├── risk/
│   └── classifier.py
│
├── visualization/
│   └── hud.py
│
├── utils/
│   └── performance.py
│
├── models/
│   └── yolov8n.pt
│
├── requirements.txt
└── README.md

This is the direction to move toward as projects become larger.

49. Technology Relationships

Remember this map:

                    PYTHON
                       |
       +---------------+---------------+
       |               |               |
     NumPy          OpenCV          PyTorch
       |               |               |
       |               |               |
   Arrays          Classical CV     Deep Learning
                       |               |
             +---------+---------+     |
             |         |         |     |
            HSV    Morphology  MOG2   YOLO
             |         |         |     |
             +---------+---------+     |
                       |               |
                    Fire/Smoke       Objects
                       |               |
                       +-------+-------+
                               |
                        Risk Classifier
                               |
                         HUD Renderer
                               |
                         Video Output

50. Classical CV vs Deep Learning

Classical Computer Vision

Examples in this project:

HSV thresholding

inRange

Morphology

Contours

Background subtraction

Frame differencing

Thresholding

General philosophy:

Human defines rules
       |
Algorithm follows rules

Advantages:

Simple for controlled visual conditions

Fast for some operations

Doesn't require training data for the rule itself

Limitations:

Sensitive to lighting/environment

Rules may not generalize

Can produce false positives

Deep Learning

YOLO is the deep-learning part.

General philosophy:

Training data
     |
Neural network learns patterns
     |
Model
     |
New image
     |
Prediction

Advantages:

Can learn complex visual patterns

Better suited to general object recognition

Can detect objects using learned features rather than simple color
rules

Requirements:

Dataset

Training

Compute

Evaluation

Model management

51. The Most Important Mental Model

When I see a future GitHub project, I should ask:

Layer 1: Language

What programming language?

Layer 2: Environment

How are dependencies installed?
What Python version?
Virtual environment?
requirements.txt?

Layer 3: Data

What data enters the system?
Image?
Video?
Text?
Audio?
Sensor data?

Layer 4: Processing

NumPy?
OpenCV?
Pandas?
SQL?

Layer 5: AI/ML

PyTorch?
TensorFlow?
YOLO?
Transformers?
scikit-learn?

Layer 6: Logic

Rules?
ML model?
LLM?

Layer 7: Application

API?
Web app?
CLI?
GUI?
Database?

Layer 8: Deployment

Docker?
Cloud?
GPU?
FastAPI?
Kubernetes?

This is how to stop being a GitHub project runner and become someone
who can actually understand a project.

52. What I Should Learn Next

Recommended learning sequence for this project:

1. Python fundamentals
        ↓
2. OOP
        ↓
3. NumPy
        ↓
4. OpenCV fundamentals
        ↓
5. Images and pixels
        ↓
6. BGR / RGB / HSV
        ↓
7. Masks and thresholding
        ↓
8. Morphology
        ↓
9. Contours
        ↓
10. Background subtraction
        ↓
11. Frame differencing
        ↓
12. PyTorch fundamentals
        ↓
13. Tensors
        ↓
14. GPU / CUDA / FP16
        ↓
15. Object detection
        ↓
16. YOLOv8
        ↓
17. Confidence / IoU / NMS
        ↓
18. YOLO training
        ↓
19. Tracking
        ↓
20. Real-time optimization
        ↓
21. FastAPI
        ↓
22. Docker
        ↓
23. Deployment

53. Quick Reference: "What does this mean?"

If I see...          I should think...

import cv2         OpenCV
np.array()         NumPy numerical array
image.shape        Image dimensions
cv2.VideoCapture   Camera/video input
cv2.cvtColor       Color-space conversion
BGR2HSV            BGR → HSV
cv2.inRange        Color/range segmentation
mask               Selected/rejected pixel regions
MORPH_OPEN         Remove small noise
MORPH_CLOSE        Close gaps/holes
findContours       Find connected shapes
boundingRect       Bounding box around shape
MOG2               Background subtraction
absdiff            Difference between images
GaussianBlur       Smooth/reduce noise
torch              PyTorch
cuda               NVIDIA GPU computation
FP16               Half precision
YOLO()             Load/run YOLO model
conf               Detection threshold/confidence
classes            Restrict object classes
IoU                Box overlap measurement
NMS                Remove duplicate detections
argparse           Command-line arguments
VideoWriter        Save video
FPS                Frames per second
latency            Processing time
class              OOP class
self               Current object
__init__           Object initialization
try/except         Exception handling
requirements.txt   Project dependencies

54. Final Project Mental Model

If I forget everything else, remember this:

                 INDUSTRIAL FIRE/SMOKE AI
                          |
                    Python program
                          |
                  OpenCV reads video
                          |
                    NumPy frame
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
     Fire detector   Smoke detector   YOLOv8n
      Classical CV    Classical CV    Deep Learning
          |               |               |
          v               v               v
     Fire regions     Smoke regions    Objects
          |               |          Person/Car/etc.
          +---------------+---------------+
                          |
                    Rule-based risk
                          |
                 CLEAR / CAUTION /
                 WARNING / CRITICAL
                          |
                    HUD Renderer
                          |
                    Display/Video

The core lesson

This project is not "just YOLO."

It is a combination of:

Python
+ OOP
+ NumPy
+ OpenCV
+ Classical Computer Vision
+ PyTorch
+ CUDA/GPU
+ FP16
+ YOLOv8n
+ Object Detection
+ Rule-Based Logic
+ Real-Time Video Processing
+ Performance Monitoring
+ CLI Programming
+ Visualization

That is the technology map I should keep in mind when I encounter
similar GitHub projects in the future.