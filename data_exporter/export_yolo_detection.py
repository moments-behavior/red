#!/usr/bin/env python3
"""
Export YOLO Detection Dataset

Usage:
    python export_yolo_detection.py -i /path/to/labeled/data -v /path/to/videos -o /path/to/output -c class_names.txt
"""

import cv2
import os
import argparse
import re
import numpy as np
import yaml
import random
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
                    'class_id': class_id,
                    'bbox': [x_min, y_min, x_max, y_max],
                    'confidence': confidence
                })
    
    return labels


def convert_to_yolo_format(bbox, img_width, img_height):
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


def load_class_names(class_file):
    """Load class names from file."""
    if not os.path.exists(class_file):
        # Default class names for common detection tasks
        return ['person', 'animal', 'object']
    
    with open(class_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    return class_names


def create_data_yaml(output_dir, class_names, splits):
    """Create data.yaml file for YOLO training."""
    data_yaml = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")
    return yaml_path


def export_yolo_detection_dataset(label_dir, video_dir, output_dir, class_file=None, 
                                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Export complete YOLO detection dataset."""
    
    # Load class names
    if class_file and os.path.exists(class_file):
        class_names = load_class_names(class_file)
    else:
        # Try to infer class names from data or use defaults
        class_names = ['person']  # Default for pose estimation tasks
    
    print(f"Using class names: {class_names}")
    
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
    
    # Find all camera bounding box files
    bbox_files = [f for f in os.listdir(label_folder) if f.endswith('_bboxes.csv')]
    
    if not bbox_files:
        raise ValueError(f"No bounding box files found in {label_folder}")
    
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
        camera_labels = parse_bbox_csv(bbox_csv_path)
        
        # Add camera prefix to frame IDs to avoid conflicts
        for frame_id, labels in camera_labels.items():
            full_frame_id = f"{camera_name}_{frame_id}"
            all_labeled_frames[full_frame_id] = {
                'camera': camera_name,
                'frame_id': frame_id,
                'video_path': video_path,
                'labels': labels
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
            labels = frame_data['labels']
            
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
            
            # Generate YOLO format labels
            img_height, img_width = frame.shape[:2]
            label_filename = f"{frame_id}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for label in labels:
                    class_id = label['class_id']
                    bbox = label['bbox']
                    
                    # Convert to YOLO format
                    x_center, y_center, width, height = convert_to_yolo_format(
                        bbox, img_width, img_height
                    )
                    
                    # Write YOLO format: class_id x_center y_center width height
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(frame_ids)} frames")
    
    # Create data.yaml
    create_data_yaml(output_dir, class_names, splits)
    
    # Create README
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(f"# YOLO Detection Dataset\n\n")
        f.write(f"Generated from RED annotation tool\n\n")
        f.write(f"## Dataset Statistics\n")
        f.write(f"- Total frames: {len(all_labeled_frames)}\n")
        f.write(f"- Training frames: {len(data_splits['train'])}\n")
        f.write(f"- Validation frames: {len(data_splits['val'])}\n")
        f.write(f"- Test frames: {len(data_splits['test'])}\n")
        f.write(f"- Number of classes: {len(class_names)}\n")
        f.write(f"- Classes: {', '.join(class_names)}\n\n")
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
        f.write(f"└── README.md\n")
        f.write(f"```\n")
    
    print(f"\nDataset export completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Total frames exported: {len(all_labeled_frames)}")


def main():
    parser = argparse.ArgumentParser(description='Export YOLO Detection Dataset')
    parser.add_argument('-i', '--label_dir', type=str, required=True,
                       help='Directory containing labeled data folders')
    parser.add_argument('-v', '--video_dir', type=str, required=True,
                       help='Directory containing video files')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                       help='Output directory for YOLO dataset')
    parser.add_argument('-c', '--class_file', type=str,
                       help='File containing class names (one per line)')
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
        export_yolo_detection_dataset(
            label_dir=args.label_dir,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            class_file=args.class_file,
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
