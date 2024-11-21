'''
Train a detection model using ultralytics library on the prepared dataset
The input is a configuration file that specifies the dataset location and the classes
The outputs are visualizations of training, metrics and the model weights
Refer to ultralytics documentations for more details
'''

import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO("yolov8x.pt")

# Train the model
results = model.train(data="larva.yaml", epochs=100)

# Validate the model
results = model.val(data="larva.yaml")