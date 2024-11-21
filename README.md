# Larva tracking

## 1 Pipeline

### 1.1 Overview
For a frame, a detection model detects larvae and output bounding boxes, these will be the input for SAM 2 model. 

SAM 2 tracks the larvae and output segmentation masks.

From the masks, required data is extracted.

### 1.2 Detection model
A detection model was trained to detect larvae, in this case YOLOv8 using ultralytics library.

#### 1.2.1 Create a dataset using SAM 2 model
A video is tracked using SAM 2, sampled frames and the associated masks are used to create a dataset using a detection model.

#### 1.2.2 Train a detection model
YOLOv8 was trained on the sampled dataset.

#### 1.2.3 Weights
The weights are shared publicly at https://drive.google.com/file/d/1qlITJxJwaoaXcwFuUL8eJXqGoPTGpw6k/view?usp=drive_link

#### 1.2.4 Notes
See 'useful_snippets' for more information.

### 1.3 SAM 2 model
This repo is forked from SAM 2 original repo (https://github.com/facebookresearch/sam2) with some hacky modifications to limit the memory usage.

### 1.4 Extracted data
Raw data: location, size, speed etc.

Aggregated data: mean size, mean speed etc.

## 2 Colab notebook

### 2.1 Overview
A colab notebook was developed to use to pipeline, see 'colab_notebook/Larva tracking.ipynb'.

The notebook and input videos should be on Google Drive, the output will also be written to Google Drive.

### 2.2 Helper functions
The notebook uses helper functions in 'sam2/utils/utils.py'. These functions are unique to the larva tracking project.
