'''
Extract images and labels from SAM 2 output to prepare a dataset to train a model using ultralytics library
'''

import pickle
import numpy as np
import torch
import os
import shutil
import cv2
from ultralytics.engine.results import Boxes

def load_pickle(file_name):
    # Open the file in read-binary mode
    with open(f'{file_name}.pkl', 'rb') as file:
        # Use pickle to load the object from the file
        loaded_video_segments = pickle.load(file)

    print("Data loaded")

    return loaded_video_segments

def mask_to_bbox(mask):
    """
    Converts a binary segmentation mask to a bounding box.
    
    Args:
    - mask (numpy array): 2D binary mask where non-zero values represent the object.
    
    Returns:
    - bbox (tuple): A tuple (x_min, y_min, x_max, y_max) representing the bounding box.
    """
    # Find the coordinates of non-zero pixels
    coords = np.argwhere(mask)
    
    # Get the minimum and maximum x and y coordinates
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Return the bounding box as (x_min, y_min, x_max, y_max)
    return x_min, y_min, x_max, y_max

def get_image_dimensions(image_path):
    img = cv2.imread(image_path)
    height, width, _ = img.shape
    return width, height

def mask_to_xywhn(mask, orig_shape):
    bbox = mask_to_bbox(mask.squeeze())
    boxes = torch.tensor([[*bbox, 0., 0]])
    detection_boxes = Boxes(boxes, orig_shape)
    xywhn = detection_boxes.xywhn
    return xywhn[0].tolist()

'''
Step 1: use SAM 2 to make predictions on a video, see SAM 2 example notebook
https://github.com/facebookresearch/sam2/blob/7e1596c0b6462eb1d1ba7e1492430fed95023598/notebooks/video_predictor_example.ipynb

Step 2: store output in a dictionary and save to a pickle file
In SAM 2 example notebook, they stored the output from SAM 2 in a dictionary called 'video_segments'
Save this dictionary as a pickle file for example 'video_segments.pkl'

Step 3: extract images and labels in the format that can be used to train a model using ultralytics library
The code below will extract images and labels to specified locations.

Step 4: prepare a configuration file for training, validation and testing, refer to ultralytics documentation
'''
file_name = 'video_segments.pkl'
steps = 30
frame_dir = 'data/frames'
dest_image_dir = 'datasets/larva/images/train'
dest_label_dir = 'datasets/larva/labels/train'

loaded_video_segments = load_pickle(file_name)

total_frames = len(os.listdir(os.path.join(frame_dir, file_name)))

for frame_number in range(0, total_frames, steps):
    # get image dimension
    frame_path = os.path.join(frame_dir, file_name, f'{frame_number:05d}.jpg')
    w, h = get_image_dimensions(frame_path)
    orig_shape = (h, w)

    # copy image to dataset location
    dest_image_path = os.path.join(dest_image_dir, f'{file_name}_{frame_number:05d}.jpg')
    shutil.copy(frame_path, dest_image_path)
    
    # get label file to write
    label_file = os.path.join(dest_label_dir, f'{file_name}_{frame_number:05d}.txt')

    # get mask of each object
    mask_data = loaded_video_segments[frame_number]
    with open(label_file, 'w') as f:
        for obj_id, label_row in mask_data.items():
            # xywhn = mask_to_xywhn(mask, orig_shape)  # convert to coco format
            label_row = [0, *label_row]  # add class always class 0
            for item in label_row:
                f.write(str(item) + ' ')
            f.write('\n')

print('Done!')
