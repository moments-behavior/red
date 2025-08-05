#!/usr/bin/env python3
"""
Export YOLO Pose Dataset

Usage:
    python export_yolo_pose.py -i /path/to/labeled/data -v /path/to/videos -o /path/to/output -s skeleton.json
"""

import cv2
import os
import argparse
import re
import numpy as np
import yaml
import random
import json
import shutil
from pathlib import Path
import ctypes as C
try:
    import PyNvVideoCodec as nvc
    NVCODEC_AVAILABLE = True
except ImportError:
    NVCODEC_AVAILABLE = False
    print("Warning: PyNvVideoCodec not available. Using OpenCV for video decoding.")

from utils import *


def cast_address_to_1d_bytearray(base_address, size):
    """Convert CUDA memory address to numpy array for PyNvVideoCodec."""
    return np.ctypeslib.as_array(
        C.cast(base_address, C.POINTER(C.c_uint8)), shape=(size,)
    )


def extract_frame_opencv(video_path, frame_number):
    """Extract a single frame using OpenCV (CPU fallback)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_frame_nvcodec(decoder, frame_number):
    """Extract a single frame using PyNvVideoCodec (GPU accelerated)."""
    if not NVCODEC_AVAILABLE:
        return None
    
    try:
        frames = decoder.get_batch_frames_by_index([frame_number])
        if len(frames) == 0:
            return None
        
        frame = frames[0]
        luma_base_addr = frame.GetPtrToPlane(0)
        new_array = cast_address_to_1d_bytearray(
            base_address=luma_base_addr, size=frame.framesize()
        )
        
        height = decoder.Height()
        width = decoder.Width()
        img = new_array.reshape((height, width, -1))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr
    except Exception as e:
        print(f"Error extracting frame {frame_number} with NvCodec: {e}")
        return None


def load_skeleton_config(skeleton_file):
    """Load skeleton configuration from JSON file."""
    if not os.path.exists(skeleton_file):
        # Return default human pose skeleton if no config provided
        return {
            "name": "human_pose",
            "num_nodes": 17,
            "node_names": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "edges": [
                [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9],
                [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
                [13, 15], [12, 14], [14, 16]
            ]
        }
    
    with open(skeleton_file, 'r') as f:
        return json.load(f)


def parse_bbox_csv(csv_path):
    """Parse bounding box CSV file and return frame-wise labels."""
    labels = {}
    
    if not os.path.exists(csv_path):
        return labels
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    data_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith(('skeleton', 'frame,bbox_id')):
            data_lines.append(line)
    
    for line in data_lines:
        parts = line.split(',')
        if len(parts) >= 8:
            frame_id = parts[0]
            bbox_id = int(parts[1])
            class_id = int(parts[2])
            confidence = float(parts[3])
            x_min = float(parts[4])
            y_min = float(parts[5])
            x_max = float(parts[6])
            y_max = float(parts[7])
            
            # Only include high-confidence detections (manual labels have confidence = 1.0)
            if confidence >= 0.5:
                if frame_id not in labels:
                    labels[frame_id] = []
                
                labels[frame_id].append({
                    'bbox_id': bbox_id,
                    'class_id': class_id,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': confidence,
                    'keypoints': None  # Will be filled from keypoint data
                })
    
    return labels


def parse_keypoint_csv(csv_path, skeleton_config):
    """Parse keypoint CSV file and return frame-wise keypoints."""
    keypoints = {}
    
    if not os.path.exists(csv_path):
        return keypoints
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header line
    data_lines = []
    for line in lines[1:]:  # Skip first line (skeleton name)
        line = line.strip()
        if line:
            data_lines.append(line)
    
    for line in data_lines:
        parts = line.split(',')
        if len(parts) >= 3:
            frame_id = parts[0]
            
            # Parse keypoints: each keypoint has id, x, y, z (we only use x, y)
            frame_keypoints = []
            num_nodes = skeleton_config.get('num_nodes', 17)
            
            # Expected format: frame, kp_id_0, x_0, y_0, z_0, kp_id_1, x_1, y_1, z_1, ...
            for i in range(num_nodes):
                base_idx = 1 + i * 4  # Skip frame_id, then groups of 4 (id, x, y, z)
                if base_idx + 3 < len(parts):
                    try:
                        kp_id = int(parts[base_idx])
                        x = float(parts[base_idx + 1])
                        y = float(parts[base_idx + 2])
                        z = float(parts[base_idx + 3])
                        
                        # Check if keypoint is labeled (not NaN or invalid)
                        if not (np.isnan(x) or np.isnan(y) or x == 1e7 or y == 1e7):
                            frame_keypoints.append({
                                'id': kp_id,
                                'x': x,
                                'y': y,
                                'visible': 2  # Visible and labeled
                            })
                        else:
                            frame_keypoints.append({
                                'id': kp_id,
                                'x': 0,
                                'y': 0,
                                'visible': 0  # Not labeled
                            })
                    except (ValueError, IndexError):
                        # Invalid data, mark as not labeled
                        frame_keypoints.append({
                            'id': i,
                            'x': 0,
                            'y': 0,
                            'visible': 0
                        })
            
            if frame_keypoints:
                keypoints[frame_id] = frame_keypoints
    
    return keypoints


def convert_bbox_to_yolo_format(bbox, img_width, img_height):
    """Convert bounding box from pixel coordinates to YOLO format."""
    x_min, y_min, x_max, y_max = bbox
    
    # Calculate center coordinates and dimensions
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    width = x_max - x_min
    height = y_max - y_min
    
    # Normalize to image dimensions
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return x_center, y_center, width, height


def convert_keypoints_to_yolo_format(keypoints, img_width, img_height):
    """Convert keypoints to YOLO pose format."""
    yolo_keypoints = []
    
    for kp in keypoints:
        x = kp['x'] / img_width if kp['visible'] > 0 else 0
        y = kp['y'] / img_height if kp['visible'] > 0 else 0
        v = kp['visible']
        yolo_keypoints.extend([x, y, v])
    
    return yolo_keypoints


def create_dataset_structure(output_dir):
    """Create YOLO dataset directory structure."""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    return splits


def split_data(labeled_frames, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Split labeled frames into train/validation/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    all_frames = list(labeled_frames.keys())
    random.shuffle(all_frames)
    
    n_total = len(all_frames)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_frames = all_frames[:n_train]
    val_frames = all_frames[n_train:n_train + n_val]
    test_frames = all_frames[n_train + n_val:]
    
    return {
        'train': train_frames,
        'val': val_frames,
        'test': test_frames
    }


def create_data_yaml(output_dir, skeleton_config, splits):
    """Create data.yaml file for YOLO pose training."""
    # For pose estimation, typically use single class (person/animal)
    class_names = [skeleton_config.get('name', 'pose')]
    
    # Create skeleton info for pose training
    kpt_shape = [skeleton_config.get('num_nodes', 17), 3]  # [num_keypoints, 3] for x,y,visible
    
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names,
        'kpt_shape': kpt_shape,
        'flip_idx': list(range(skeleton_config.get('num_nodes', 17))),  # No symmetric flipping by default
    }
    
    # Add skeleton-specific information
    if 'node_names' in skeleton_config:
        data_yaml['keypoint_names'] = skeleton_config['node_names']
    
    if 'edges' in skeleton_config:
        data_yaml['skeleton'] = skeleton_config['edges']
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path


def export_yolo_pose_dataset(label_dir, video_dir, output_dir, skeleton_file=None,
                            train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Export complete YOLO pose dataset."""
    
    # Load skeleton configuration
    skeleton_config = load_skeleton_config(skeleton_file)
    print(f"Using skeleton: {skeleton_config.get('name', 'unknown')}")
    print(f"Number of keypoints: {skeleton_config.get('num_nodes', 17)}")
    
    # Find the most recent label folder
    datetime_pattern = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
    matching_folders = [
        name for name in os.listdir(label_dir)
        if os.path.isdir(os.path.join(label_dir, name)) and datetime_pattern.match(name)
    ]
    
    if not matching_folders:
        raise ValueError(f"No labeled data folders found in {label_dir}")
    
    matching_folders.sort()
    label_folder = os.path.join(label_dir, matching_folders[-1])
    print(f"Using most recent labels: {matching_folders[-1]}")
    
    # Create dataset structure
    splits = create_dataset_structure(output_dir)
    
    # Find all camera files
    bbox_files = [f for f in os.listdir(label_folder) if f.endswith('_bboxes.csv')]
    keypoint_files = [f for f in os.listdir(label_folder) if f.endswith('.csv') and not f.endswith('_bboxes.csv') and f != 'keypoints3d.csv']
    
    if not bbox_files:
        raise ValueError(f"No bounding box files found in {label_folder}")
    
    print(f"Found {len(bbox_files)} bbox files and {len(keypoint_files)} keypoint files")
    
    all_labeled_frames = {}
    video_files = {}
    
    # Process each camera
    for bbox_file in bbox_files:
        camera_name = bbox_file.replace('_bboxes.csv', '')
        print(f"Processing camera: {camera_name}")
        
        # Find corresponding video file
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_path = None
        
        for ext in video_extensions:
            potential_path = os.path.join(video_dir, f"{camera_name}{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if not video_path:
            print(f"Warning: No video file found for camera {camera_name}")
            continue
        
        video_files[camera_name] = video_path
        
        # Parse bounding box labels
        bbox_csv_path = os.path.join(label_folder, bbox_file)
        bbox_labels = parse_bbox_csv(bbox_csv_path)
        
        # Parse keypoint labels
        keypoint_csv_path = os.path.join(label_folder, f"{camera_name}.csv")
        keypoint_labels = {}
        if os.path.exists(keypoint_csv_path):
            keypoint_labels = parse_keypoint_csv(keypoint_csv_path, skeleton_config)
        
        # Combine bbox and keypoint data
        for frame_id, bboxes in bbox_labels.items():
            full_frame_id = f"{camera_name}_{frame_id}"
            
            # Add keypoint data to each bounding box
            frame_keypoints = keypoint_labels.get(frame_id, [])
            
            for bbox in bboxes:
                # For pose estimation, we typically have one person per bbox
                # Assign keypoints to the bbox (in more complex scenarios, 
                # you might need to match keypoints to specific bboxes)
                bbox['keypoints'] = frame_keypoints
            
            all_labeled_frames[full_frame_id] = {
                'camera': camera_name,
                'frame_id': frame_id,
                'video_path': video_path,
                'bboxes': bboxes
            }
    
    if not all_labeled_frames:
        raise ValueError("No labeled frames found in the dataset")
    
    print(f"Found {len(all_labeled_frames)} labeled frames across {len(video_files)} cameras")
    
    # Split data into train/val/test
    data_splits = split_data(all_labeled_frames, train_ratio, val_ratio, test_ratio)
    
    print(f"Data split: Train={len(data_splits['train'])}, Val={len(data_splits['val'])}, Test={len(data_splits['test'])}")
    
    # Initialize video decoders if using NvCodec
    decoders = {}
    if NVCODEC_AVAILABLE:
        for camera_name, video_path in video_files.items():
            try:
                decoder = nvc.PyNvDecoder(video_path, nvc.PixelFormat.RGB)
                decoders[camera_name] = decoder
                print(f"Initialized NvCodec decoder for {camera_name}")
            except Exception as e:
                print(f"Failed to initialize NvCodec decoder for {camera_name}: {e}")
                decoders[camera_name] = None
    
    # Export frames and labels for each split
    for split_name, frame_ids in data_splits.items():
        print(f"\nExporting {split_name} split...")
        
        images_dir = os.path.join(output_dir, split_name, 'images')
        labels_dir = os.path.join(output_dir, split_name, 'labels')
        
        for i, frame_id in enumerate(frame_ids):
            frame_data = all_labeled_frames[frame_id]
            camera_name = frame_data['camera']
            original_frame_id = frame_data['frame_id']
            video_path = frame_data['video_path']
            bboxes = frame_data['bboxes']
            
            # Extract frame
            frame = None
            if camera_name in decoders and decoders[camera_name]:
                try:
                    frame_number = int(original_frame_id) if original_frame_id.isdigit() else i
                    frame = extract_frame_nvcodec(decoders[camera_name], frame_number)
                except:
                    pass
            
            if frame is None:
                # Fallback to OpenCV
                frame_number = int(original_frame_id) if original_frame_id.isdigit() else i
                frame = extract_frame_opencv(video_path, frame_number)
            
            if frame is None:
                print(f"Failed to extract frame {frame_id}")
                continue
            
            # Save image
            img_filename = f"{frame_id}.jpg"
            img_path = os.path.join(images_dir, img_filename)
            cv2.imwrite(img_path, frame)
            
            # Generate YOLO pose format labels
            img_height, img_width = frame.shape[:2]
            label_filename = f"{frame_id}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for bbox_data in bboxes:
                    class_id = bbox_data['class_id']
                    bbox = bbox_data['bbox']
                    keypoints = bbox_data.get('keypoints', [])
                    
                    # Convert bbox to YOLO format
                    x_center, y_center, width, height = convert_bbox_to_yolo_format(
                        bbox, img_width, img_height
                    )
                    
                    # Convert keypoints to YOLO format
                    yolo_keypoints = convert_keypoints_to_yolo_format(
                        keypoints, img_width, img_height
                    )
                    
                    # Write YOLO pose format: class_id x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    
                    if yolo_keypoints:
                        kp_str = " ".join([f"{kp:.6f}" for kp in yolo_keypoints])
                        line += f" {kp_str}"
                    
                    f.write(line + "\n")
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(frame_ids)} frames")
    
    # Create data.yaml
    create_data_yaml(output_dir, skeleton_config, splits)
    
    # Save skeleton configuration
    skeleton_output_path = os.path.join(output_dir, 'skeleton.json')
    with open(skeleton_output_path, 'w') as f:
        json.dump(skeleton_config, f, indent=2)
    
    # Create README
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# YOLO Pose Dataset\n\n")
        f.write(f"Generated from RED annotation tool\n\n")
        f.write(f"## Dataset Statistics\n")
        f.write(f"- Total frames: {len(all_labeled_frames)}\n")
        f.write(f"- Training frames: {len(data_splits['train'])}\n")
        f.write(f"- Validation frames: {len(data_splits['val'])}\n")
        f.write(f"- Test frames: {len(data_splits['test'])}\n")
        f.write(f"- Skeleton: {skeleton_config.get('name', 'unknown')}\n")
        f.write(f"- Number of keypoints: {skeleton_config.get('num_nodes', 17)}\n\n")
        
        if 'node_names' in skeleton_config:
            f.write(f"## Keypoint Names\n")
            for i, name in enumerate(skeleton_config['node_names']):
                f.write(f"{i}: {name}\n")
            f.write(f"\n")
        
        f.write(f"## Directory Structure\n")
        f.write(f"```\n")
        f.write(f"dataset/\n")
        f.write(f"├── train/\n")
        f.write(f"│   ├── images/\n")
        f.write(f"│   └── labels/\n")
        f.write(f"├── val/\n")
        f.write(f"│   ├── images/\n")
        f.write(f"│   └── labels/\n")
        f.write(f"├── test/\n")
        f.write(f"│   ├── images/\n")
        f.write(f"│   └── labels/\n")
        f.write(f"├── data.yaml\n")
        f.write(f"├── skeleton.json\n")
        f.write(f"└── README.md\n")
        f.write(f"```\n\n")
        f.write(f"## Label Format\n")
        f.write(f"Each label file contains lines in the format:\n")
        f.write(f"```\n")
        f.write(f"class_id x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...\n")
        f.write(f"```\n")
        f.write(f"Where:\n")
        f.write(f"- class_id: Object class (0 for person/animal)\n")
        f.write(f"- x_center, y_center, width, height: Normalized bounding box coordinates\n")
        f.write(f"- kpN_x, kpN_y, kpN_v: Normalized keypoint coordinates and visibility (0=not labeled, 1=labeled but occluded, 2=labeled and visible)\n")
    
    print(f"\nDataset export completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Total frames exported: {len(all_labeled_frames)}")


def main():
    parser = argparse.ArgumentParser(description='Export YOLO Pose Dataset')
    parser.add_argument('-i', '--label_dir', type=str, required=True,
                       help='Directory containing labeled data folders')
    parser.add_argument('-v', '--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='Output directory for YOLO dataset')
    parser.add_argument('-s', '--skeleton_file', type=str,
                       help='JSON file containing skeleton configuration')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Ratio of data for training (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Ratio of data for validation (default: 0.2)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Ratio of data for testing (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible splits')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate arguments
    if not os.path.exists(args.label_dir):
        print(f"Error: Label directory {args.label_dir} does not exist")
        return
    
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory {args.video_dir} does not exist")
        return
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("Error: Train, validation, and test ratios must sum to 1.0")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        export_yolo_pose_dataset(
            label_dir=args.label_dir,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            skeleton_file=args.skeleton_file,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    except Exception as e:
        print(f"Error during export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
