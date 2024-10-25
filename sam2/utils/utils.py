import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import ffmpeg
import os
import torch
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from scipy.spatial import distance
import pandas as pd
from scipy import stats
import copy
from collections import defaultdict
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Masks, Boxes


def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Retrieve video properties: width, height, and frames per second
    w, h, fps, frame_count = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS, cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return w, h, fps, frame_count


def get_larva_detections(video_path, detection_model, steps=30, num_larvae=5):
    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Check if video file exists
    if not cap.isOpened():
        print("Error opening video file")

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Looking for {num_larvae} larvae in video')

    # Set frame position
    for frame_index in range(0, total_frames, steps):
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        results = detection_model.predict(frame)  # return a list of Results objects
        result = results[0]

        boxes = result.boxes
        boxes_xyxy = boxes.xyxy.cpu().numpy()

        num_detections = len(boxes_xyxy)
        
        if num_detections == num_larvae:
            print(f'Found {num_larvae} larvae at frame {frame_index}.')
            return result, boxes_xyxy, frame_index
        else:
            continue

    # Release resources
    cap.release()

    print(f'Failed to predict {num_larvae} larvae in a frame, found {num_detections} larvae at frame {frame_index}')

    return result, boxes_xyxy, frame_index


def extract_frames_ffmpeg(video_path, output_dir, quality=2, start_number=0, file_pattern='%05d.jpg'):
    """
    Uses ffmpeg-python to extract frames from a video and save as images.

    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save the extracted frames.
        quality (int): Quality level for the output images (lower is better quality).
        start_number (int): Starting number for the output image filenames.
        file_pattern (str): Pattern for naming the output images (default: '%05d.jpg').
    
    Returns:
        None
    """
    if os.path.exists(output_dir):
        print(f'{output_dir} exists')
        return
    os.makedirs(output_dir, exist_ok=True)

    # Construct output file path pattern
    output_path = f'{output_dir}/{file_pattern}'
    
    # Use ffmpeg to extract frames
    ffmpeg.input(video_path).output(
        output_path, 
        q=quality,  # Quality for the frames
        start_number=start_number  # Start number for frame file names
    ).run()


def get_track_data(mask, h, w):
    track_data = {}
    # for drawing predictions on video
    line_mask = Masks(torch.tensor(mask), (h, w)).xy[0]
    # bbox_xyxy = mask_to_bbox(mask.squeeze())
    # bbox_xywh = Boxes(torch.tensor([[*bbox_xyxy, 0., 0]]), (h, w)).xywh[0]

    track_data['line mask'] = line_mask
    # track_data['bbox_xyxy'] = bbox_xyxy
    centroid_point = Polygon(line_mask).centroid

    # for collecting data
    # centre_x, center_y = bbox_xywh[0], bbox_xywh[1]
    size_in_pixels = mask.sum()

    track_data['centre x'] = int(centroid_point.x)
    track_data['centre y'] = int(centroid_point.y)
    track_data['size (pixels)'] = size_in_pixels

    return track_data


def get_video_segments(predictor, inference_state, h, w):
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results

    # predict forwards
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=False):
        data = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            track_data = get_track_data(mask, h, w)

            # add data to an object
            data[out_obj_id] = track_data
        
        # add data to a frame
        video_segments[out_frame_idx] = data

    # predict backwards
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state, reverse=True):
        data = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy()

            track_data = get_track_data(mask, h, w)

            # add data to an object
            data[out_obj_id] = track_data
        
        # add data to a frame
        video_segments[out_frame_idx] = data

    video_segments = dict(sorted(video_segments.items()))  # sort by frame number

    return video_segments


def get_frame_data_subset(frame_dict, step=1):
    """
    Convert frame numbers to data per second.
    
    Args:
        frame_dict (dict): Dictionary with frame numbers as keys.
        fps (int): Frames per second.
    
    Returns:
        dict: Dictionary with seconds as keys.
    """
    data = {}
    for frame, value in frame_dict.items():
        second = frame // step
        if second not in data:
            data[second] = value
    return data


def smooth_distances_velocities(frame_dict, window=1, fps=30, scale_factor=0.05):
    """
    Convert frame numbers to data per second.
    
    Args:
        frame_dict (dict): Dictionary with frame numbers as keys.
        window (int): Window to average.
    
    Returns:
        dict: Dictionary with seconds as keys.
    """
    data = copy.deepcopy(frame_dict)

    first_frame = 0
    obj_ids = list(data[first_frame].keys())
    n_frames = len(data)
    
    for obj_id in obj_ids:
        distances = np.array([item[obj_id]['distance (pixels)'] for index, item in data.items()])
        distances = pd.Series(distances)
        distances_moving_averages = distances.rolling(window=window, min_periods=1).mean()
        
        velocities = np.array([item[obj_id]['velocity (pixels/frame)'] for index, item in data.items()])
        velocities = pd.Series(velocities)
        velocities_moving_averages = velocities.rolling(window=window, min_periods=1).mean()

        for frame in range(n_frames):
            data[frame][obj_id]['distance (pixels)'] = distances_moving_averages[frame]
            data[frame][obj_id]['velocity (pixels/frame)'] = velocities_moving_averages[frame]
            data[frame][obj_id]['velocity (mm/second)'] = velocities_moving_averages[frame] * fps * scale_factor
            if data[frame][obj_id]['distance (pixels)'] == 0:
                data[frame][obj_id]['is stationary'] = 1
            else:
                data[frame][obj_id]['is stationary'] = 0

    return data


def add_raw_data(data_obj, fps=30, scale_factor=0.05):
    data = copy.deepcopy(data_obj)
    
    # initialize data for frame 0
    first_frame = 0
    obj_ids = list(data[first_frame].keys())
    for obj_id in obj_ids:
        data[first_frame][obj_id]['size (mm2)'] = data[first_frame][obj_id]['size (pixels)'] * (scale_factor**2)
        data[first_frame][obj_id]['distance (pixels)'] = 0
        data[first_frame][obj_id]['distance (mm)'] = 0
        data[first_frame][obj_id]['velocity (pixels/frame)'] = 0
        data[first_frame][obj_id]['velocity (mm/second)'] = 0
        # data[first_frame][obj_id]['delta_velocity'] = 0
        # data[first_frame][obj_id]['angle (degrees)'] = 0
        # data[first_frame][obj_id]['angular velocity (degrees/frame)'] = 0
        # data[first_frame][obj_id]['angular velocity (degrees/second)'] = 0
        # data[first_frame][obj_id]['delta_angular_velocity'] = 0
        data[first_frame][obj_id]['is stationary'] = 0

    # propagate across frames
    for frame_index in range(1, len(data)):
        for obj_id in obj_ids:
            # calculated data from previous frame
            previous_frame_index = frame_index - 1

            current_frame_data = data[frame_index][obj_id]
            previous_frame_data = data[previous_frame_index][obj_id]

            current_point = current_frame_data['centre x'], current_frame_data['centre y']
            previous_point = previous_frame_data['centre x'], previous_frame_data['centre y']

            # calculate size
            current_frame_data['size (mm2)'] = current_frame_data['size (pixels)'] * (scale_factor**2)

            # calculate distance
            dist = distance.euclidean(current_point, previous_point)
            current_frame_data['distance (pixels)'] = dist
            current_frame_data['distance (mm)'] = dist * scale_factor

            # calculate velocity
            velocity = current_frame_data['distance (pixels)']
            current_frame_data['velocity (pixels/frame)'] = velocity
            current_frame_data['velocity (mm/second)'] = velocity * fps * scale_factor

            # calculate change in velocity
            # delta_velocity = current_frame_data['velocity'] - previous_frame_data['velocity']
            # current_frame_data['delta_velocity'] = delta_velocity

            # calculate angle
            # Calculate the differences in x and y
            # dx = current_point[0] - previous_point[0]
            # dy = current_point[1] - previous_point[1]

            # Calculate the angle using arctan2
            # angle = np.degrees(np.arctan2(dy, dx))
            # current_frame_data['angle (degrees)'] = angle

            # calculate angular velocity
            # angular_velocity = current_frame_data['angle (degrees)']
            # current_frame_data['angular velocity (degrees/frame)'] = angular_velocity
            # current_frame_data['angular velocity (degrees/second)'] = angular_velocity * fps

            # calculate change in angular velocity
            # delta_angular_velocity = current_frame_data['angular_velocity'] - previous_frame_data['angular_velocity']
            # current_frame_data['delta_angular_velocity'] = delta_angular_velocity

            # set is_stationary
            if current_frame_data['distance (pixels)'] == 0:
                current_frame_data['is stationary'] = 1
            else:
                current_frame_data['is stationary'] = 0

    return data
        

# random walk modeling using linear regression
def random_walk_modeling(values):
    x = values[:-2]
    y = values[1:-1]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Plot data and regression line
    # plt.scatter(x, y)
    # plt.plot(x, slope * x + intercept, 'r')
    # plt.show()

    return slope, intercept, r_value, p_value, std_err


def get_pearson_correlation_coefficient(x, y):
    corr, p_value = stats.pearsonr(x, y)

    return corr


def detect_outliers(arr, threshold=2):
    """
    Detect outliers in a NumPy array using mean and standard deviation.

    Args:
    - arr (numpy.ndarray): Input array.
    - threshold (int, optional): Number of standard deviations from mean to consider a value an outlier. Defaults to 2.

    Returns:
    - outliers (numpy.ndarray): Array of outlier values.
    - outlier_indices (numpy.ndarray): Indices of outlier values in the original array.
    """
    # Calculate mean and standard deviation
    mean = np.mean(arr)
    std_dev = np.std(arr)

    # Identify outliers
    outliers = arr[np.abs((arr - mean) / std_dev) > threshold]
    outlier_indices = np.where(np.abs((arr - mean) / std_dev) > threshold)[0]

    return outliers, outlier_indices


def get_aggregated_data(data, fps=30, scale_factor=0.05):
    # calculate aggregated data
    aggregated_data = {}

    first_frame = 0
    obj_ids = list(data[first_frame].keys())

    for obj_id in obj_ids:
        temp_data = {}

        # calculate min, max, mean and standard deviation of sizes
        sizes = np.array([item[obj_id]['size (mm2)'] for index, item in data.items()])
        min_size = np.min(sizes)
        max_size = np.max(sizes)
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        temp_data['min size (mm2)'] = min_size
        temp_data['max size (mm2)'] = max_size
        temp_data['mean size (mm2)'] = mean_size
        temp_data['std size (mm2)'] = std_size

        # calculate min, max, mean and standard deviation of speeds
        velocities = np.array([item[obj_id]['velocity (mm/second)'] for index, item in data.items()])
        # min_velocity = np.min(velocities)
        max_velocity = np.max(velocities)
        mean_velocity = np.mean(velocities)
        std_velocity = np.std(velocities)
        # temp_data['min speed (mm/second)'] = min_velocity
        temp_data['max speed (mm/second)'] = max_velocity
        temp_data['mean speed (mm/second)'] = mean_velocity
        temp_data['std speed (mm/second)'] = std_velocity

        # calculate accumulated distance
        distances = [item[obj_id]['distance (mm)'] for index, item in data.items()]
        temp_data['accumulated distance (mm)'] = sum(distances)

        # calculate time spent stationary
        is_stationary = [item[obj_id]['is stationary'] for index, item in data.items()]
        # temp_data['time spent stationary (frames)'] = sum(is_stationary)
        temp_data['time spent stationary (seconds)'] = sum(is_stationary) / fps

        # calculate coefficient of correlation for velocities
        velocities = np.array([item[obj_id]['velocity (mm/second)'] for index, item in data.items()])
        slope, intercept, r_value, p_value, std_err = random_walk_modeling(velocities)
        temp_data['correlation coefficient for velocities'] = r_value

        # velocities = np.array([item[obj_id]['velocity (pixels/frame)'] for index, item in data.items()])
        # slope, intercept, r_value, p_value, std_err = random_walk_modeling(velocities)
        # temp_data['correlation coefficient for velocities (pixels/frame)'] = r_value

        # velocities = np.array([item[obj_id]['velocity (pixels/second)'] for index, item in data.items()])
        # slope, intercept, r_value, p_value, std_err = random_walk_modeling(velocities)
        # temp_data['correlation coefficient for velocities (pixels/second)'] = r_value

        # calculate coefficient of correlation for angular velocities
        # angular_velocities = np.array([item[obj_id]['angular velocity (degrees/frame)'] for index, item in data.items()])
        # slope, intercept, r_value, p_value, std_err = random_walk_modeling(angular_velocities)
        # temp_data['correlation coefficient for angular velocities (degrees/frame)'] = r_value

        # calculate coefficient of correlation for angular velocities
        # angular_velocities = np.array([item[obj_id]['angular velocity (degrees/second)'] for index, item in data.items()])
        # slope, intercept, r_value, p_value, std_err = random_walk_modeling(angular_velocities)
        # temp_data['correlation coefficient for angular velocities (degrees/second)'] = r_value

        # calculate Pearson correlation coefficient
        # x = np.array([item[obj_id]['velocity (pixels/frame)'] for index, item in data.items()])[:-2]
        # y = np.array([item[obj_id]['velocity (pixels/frame)'] for index, item in data.items()])[1:-1]
        # r = get_pearson_correlation_coefficient(x, y)
        # temp_data['pearson correlation coefficient'] = r

        # determine if the track data is reliable using mean and standard deviation
        outliers, outlier_indices = detect_outliers(sizes, threshold=4)
        if len(outliers) > 0:
            temp_data['is data reliable'] = 0
        else:
            temp_data['is data reliable'] = 1

        aggregated_data[obj_id] = temp_data

    return aggregated_data


def write_raw_data(out_dir, csv_dir, data, index_label, exclude=['line mask']):
    os.makedirs(os.path.join(out_dir, csv_dir), exist_ok=True)
    first_frame = 0
    first_obj = 0
    obj_ids = list(data[first_frame].keys())

    for obj_id in obj_ids:
        data_to_write = {}
        for second_index in range(len(data)):
            temp_data = {}
            keys = data[first_frame][first_obj].keys()
            for key in keys:
                if key not in exclude:
                    temp_data[key] = data[second_index][obj_id][key]
            data_to_write[second_index] = temp_data
            
        df = pd.DataFrame(data_to_write)
        df = df.T
        df.to_csv(os.path.join(out_dir, csv_dir, f'{obj_id}.csv'), index=True, index_label=index_label, float_format='%.4f')


def write_aggregated_data(out_dir, aggregated_data, fname):
    df = pd.DataFrame(aggregated_data)
    df = df.T
    df.to_csv(os.path.join(out_dir, fname), index=True, index_label='track', float_format='%.4f')


def draw_track(out_dir, paths_dir, video_segments, h, w):
    os.makedirs(os.path.join(out_dir, paths_dir), exist_ok=True)

    first_frame = 0
    obj_ids = list(video_segments[first_frame].keys())

    for obj_id in obj_ids:
        x_list = []
        y_list = []

        for frame_index in range(len(video_segments)):
            x_list.append(video_segments[frame_index][obj_id]['centre x'])
            y_list.append(video_segments[frame_index][obj_id]['centre y'])

        canvas = np.ones((h, w, 3), np.int32) * 255

        pts = np.array(list(zip(x_list, y_list)), np.int32)

        cv2.polylines(canvas, [pts], isClosed=False, color=colors(obj_id+4, True))

        # Save the image
        cv2.imwrite(os.path.join(out_dir, paths_dir, f'{obj_id}.png'), canvas)


def draw_velocities(out_dir, speeds_dir, data, total_frames, tick_every_seconds=10, fps=30, scale_factor=0.05, ymax=4):
    os.makedirs(os.path.join(out_dir, speeds_dir), exist_ok=True)

    first_frame = 0
    obj_ids = list(data[first_frame].keys())

    for obj_id in obj_ids:
        velocities = []

        for frame_index in range(len(data)):
            velocities.append(data[frame_index][obj_id]['velocity (mm/second)'])

        # Turn off interactive mode
        # plt.ioff()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot velocities
        ax.plot(velocities, label='speed')

        # Calculate the number of frames per 30 seconds
        frames_per_30_sec = tick_every_seconds * fps

        # Generate the positions for x-ticks (in multiples of 30 seconds)
        x_ticks = np.arange(0, total_frames, frames_per_30_sec)
        # Generate the corresponding x-tick labels (time in seconds)
        x_labels = (x_ticks / fps).astype(int)  # Time in seconds

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        # Set title and labels
        ax.set_title(f'Speed Over Time (fps: {fps} frames/second, scale factor: {scale_factor:.4f} mm/pixel)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Speed (mm/second)')

        # Show grid
        ax.grid(True)

        ax.legend()

        ax.set_xlim(0, None)
        ax.set_ylim(0, ymax)

        # Save figure
        plt.savefig(os.path.join(out_dir, speeds_dir, f'{obj_id}.png'))

        plt.close()


def draw_sizes(out_dir, sizes_dir, data, total_frames, tick_every_seconds=10, fps=30, scale_factor=0.05):
    os.makedirs(os.path.join(out_dir, sizes_dir), exist_ok=True)

    first_frame = 0
    obj_ids = list(data[first_frame].keys())

    for obj_id in obj_ids:
        velocities = []

        for frame_index in range(len(data)):
            velocities.append(data[frame_index][obj_id]['size (mm2)'])

        outliers, outlier_indices = detect_outliers(np.array(velocities), threshold=4)

        # Turn off interactive mode
        # plt.ioff()

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot velocities
        ax.plot(velocities, label='size')

        ax.scatter(outlier_indices, outliers, color='r', label='outlier')

        # Calculate the number of frames per 30 seconds
        frames_per_30_sec = tick_every_seconds * fps

        # Generate the positions for x-ticks (in multiples of 30 seconds)
        x_ticks = np.arange(0, total_frames, frames_per_30_sec)
        # Generate the corresponding x-tick labels (time in seconds)
        x_labels = (x_ticks / fps).astype(int)  # Time in seconds

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)

        # Set title and labels
        ax.set_title(f'Size Over Time (fps: {fps} frames/second, scale factor: {scale_factor:.4f} mm/pixel)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Size (mm2)')

        # Show grid
        ax.grid(True)

        ax.legend()

        ax.set_xlim(0, None)
        # ax.set_ylim(0, None)

        # Save figure
        plt.savefig(os.path.join(out_dir, sizes_dir, f'{obj_id}.png'))

        plt.close()


def draw_on_video(video_path, out_dir, out_video_name, video_segments, fps, h, w, duration_of_tracking_path=10):
    # Dictionary to store tracking history with default empty lists
    track_history = defaultdict(lambda: [])

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize video writer to save the output video with the specified properties
    out_video_path = os.path.join(out_dir, out_video_name)
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"MP4V"), fps, (w, h))

    # Get total frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_index in tqdm(range(frame_count)):
        # Read frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()

        # Create an annotator object to draw on the frame
        annotator = Annotator(frame, line_width=2)

        results = video_segments[frame_index]

        for track_id, data in results.items():
            annotator.seg_bbox(mask=data['line mask'], mask_color=colors(track_id+4, True), label=str(track_id))

            track = track_history[track_id]
            track.append((data['centre x'], data['centre y']))
            if len(track) > fps * duration_of_tracking_path:
                track.pop(0)
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=colors(track_id+4, True), thickness=2)

            # im0 = overlay_mask(im0, mask.squeeze(), color=colors(track_id+4, True))
            # annotator.box_label(box=data['bbox_xyxy'], label=None, color=colors(track_id+4, True), txt_color=(255, 255, 255), rotated=False)

        # Write the annotated frame to the output video
        out.write(frame)

    # Release the video writer and capture objects, and close all OpenCV windows
    out.release()
    cap.release()



