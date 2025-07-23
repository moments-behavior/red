#!/usr/bin/env python3
"""
Convert RED exported CSV data to YOLOv8 compatible dataset format.
Usage: python csv_to_yolo.py <data_root_directory>
"""

import os
import sys
import csv
import glob
import subprocess
from pathlib import Path
import shutil
import random
import time

def clear_dataset_directories(output_dir):
    """Clear existing files from train and val directories."""
    directories = [
        os.path.join(output_dir, "train", "images"),
        os.path.join(output_dir, "train", "labels"),
        os.path.join(output_dir, "val", "images"),
        os.path.join(output_dir, "val", "labels")
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared files from: {directory}")


def find_most_recent_csv_files(data_root):
    """Find the most recent CSV files for each camera."""
    csv_files = {}
    
    print(f"Searching for CSV files in: {data_root}")
    
    # First, check if labeled_data directory exists
    labeled_data_dir = os.path.join(data_root, "labeled_data")
    if not os.path.exists(labeled_data_dir):
        print(f"Warning: labeled_data directory not found at: {labeled_data_dir}")
        
        # Try to find CSV files directly in the data_root
        print("Searching for CSV files directly in data_root...")
        for csv_file in glob.glob(os.path.join(data_root, "*.csv")):
            filename = os.path.basename(csv_file)
            # Try to extract camera name from filename
            if "Cam" in filename:
                cam_start = filename.find("Cam")
                cam_part = filename[cam_start:]
                import re
                match = re.match(r'Cam\d+', cam_part)
                if match:
                    cam_name = match.group(0)
                    csv_files[cam_name] = csv_file
                    print(f"Found CSV for {cam_name}: {csv_file}")
        
        return csv_files
    
    print(f"Found labeled_data directory: {labeled_data_dir}")
    print("Looking for camera subdirectories...")
    
    # Look for camera directories
    cam_dirs = []
    for item in os.listdir(labeled_data_dir):
        item_path = os.path.join(labeled_data_dir, item)
        if os.path.isdir(item_path):
            cam_dirs.append(item)
            print(f"Found camera directory: {item}")
    
    if not cam_dirs:
        print("No camera directories found in labeled_data")
        return csv_files
    
    # Look for CSV files in each camera directory
    for cam_dir in cam_dirs:
        cam_path = os.path.join(labeled_data_dir, cam_dir)
        cam_name = cam_dir
        
        print(f"Searching for CSV files in: {cam_path}")
        
        # Try multiple patterns for CSV files
        patterns = [
            os.path.join(cam_path, f"{cam_name}_*.csv"),  # Original pattern
            os.path.join(cam_path, "*.csv"),  # Any CSV file
            os.path.join(cam_path, f"*{cam_name}*.csv")   # Contains camera name
        ]
        
        csv_list = []
        for pattern in patterns:
            found_files = glob.glob(pattern)
            csv_list.extend(found_files)
            if found_files:
                print(f"Found {len(found_files)} CSV files with pattern: {pattern}")
        
        # Remove duplicates
        csv_list = list(set(csv_list))
        
        if csv_list:
            most_recent = max(csv_list, key=os.path.getctime)
            csv_files[cam_name] = most_recent
            print(f"Selected most recent CSV for {cam_name}: {most_recent}")
        else:
            print(f"No CSV files found for camera {cam_name}")
    
    return csv_files

def find_video_files(data_root, csv_files):
    """Find video files for each camera, looking 2 directories up from CSV location."""
    video_files = {}
    
    if not csv_files:
        print("No CSV files provided to determine video location")
        return video_files
    
    # Get the first CSV file to determine the structure
    first_csv = list(csv_files.values())[0]
    csv_dir = os.path.dirname(first_csv)
    
    # Go up 2 directories from CSV location to find movies
    movies_dir = os.path.join(csv_dir, "..", "..", "movies")
    movies_dir = os.path.abspath(movies_dir)  # Resolve relative path
    
    print(f"Looking for movies directory at: {movies_dir}")
    
    if not os.path.exists(movies_dir):
        print(f"Movies directory not found at: {movies_dir}")
        
        # Also try the original location as fallback
        fallback_movies_dir = os.path.join(data_root, "movies")
        print(f"Trying fallback location: {fallback_movies_dir}")
        
        if os.path.exists(fallback_movies_dir):
            movies_dir = fallback_movies_dir
            print(f"Using fallback movies directory: {movies_dir}")
        else:
            print("No movies directory found in either location")
            return video_files
    
    print(f"Found movies directory: {movies_dir}")
    
    # Look for video files
    video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    for pattern in video_patterns:
        for video_file in glob.glob(os.path.join(movies_dir, pattern)):
            filename = os.path.basename(video_file)
            print(f"Found video file: {filename}")
            
            # Extract camera name from filename (e.g., "Cam1" from "SomethingCam1Something.mp4")
            if "Cam" in filename:
                cam_start = filename.find("Cam")
                cam_part = filename[cam_start:]
                import re
                match = re.match(r'Cam\d+', cam_part)
                if match:
                    cam_name = match.group(0)
                    video_files[cam_name] = video_file
                    print(f"Mapped video {filename} to camera {cam_name}")
    
    return video_files

def extract_frame_with_ffmpeg(video_path, frame_number, output_path, target_size="640x640"):
    """Extract a specific frame from video using FFmpeg and resize to target size."""
    try:
        # Split target_size into width and height
        width, height = target_size.split('x')
        
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select=eq(n\\,{frame_number}),scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black',
            '-vframes', '1',
            '-y',  # Overwrite output file
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error for frame {frame_number}: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"Error extracting frame {frame_number}: {e}")
        return False

def parse_csv_line(line_parts):
    """Parse a CSV line to extract frame, animal, and bbox data."""
    if len(line_parts) < 3:
        return None
    
    frame_num = int(line_parts[0])
    animal_id = int(line_parts[1])
    
    # Read number of multi bboxes
    num_multi_bboxes = int(line_parts[2])
    
    bboxes = []
    idx = 3  # Start after the count
    
    for i in range(num_multi_bboxes):
        if idx + 4 >= len(line_parts):
            break
            
        class_id = int(line_parts[idx])
        x_min = float(line_parts[idx + 1])
        y_min = float(line_parts[idx + 2])
        x_max = float(line_parts[idx + 3])
        y_max = float(line_parts[idx + 4])
        
        # Only include valid bounding boxes (not placeholder values)
        if x_min != 1E7 and y_min != 1E7 and x_max != 1E7 and y_max != 1E7:
            bboxes.append({
                'class_id': class_id,
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            })
        
        idx += 5
    
    return {
        'frame': frame_num,
        'animal': animal_id,
        'bboxes': bboxes
    }

def convert_bbox_to_yolo_format(bbox, img_width, img_height, original_width, original_height):
    """Convert bbox from absolute coordinates to YOLO normalized format."""
    x_min, y_min, x_max, y_max = bbox['x_min'], bbox['y_min'], bbox['x_max'], bbox['y_max']
    
    # Calculate scaling factors for the resized image
    scale_x = img_width / original_width
    scale_y = img_height / original_height
    
    # Apply scaling to maintain aspect ratio (letterboxing)
    scale = min(scale_x, scale_y)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Calculate padding offsets
    pad_x = (img_width - new_width) // 2
    pad_y = (img_height - new_height) // 2
    
    # Scale and adjust bbox coordinates
    scaled_x_min = x_min * scale + pad_x
    scaled_y_min = y_min * scale + pad_y
    scaled_x_max = x_max * scale + pad_x
    scaled_y_max = y_max * scale + pad_y
    
    # Calculate center point and dimensions
    center_x = (scaled_x_min + scaled_x_max) / 2.0
    center_y = (scaled_y_min + scaled_y_max) / 2.0
    width = scaled_x_max - scaled_x_min
    height = scaled_y_max - scaled_y_min
    
    # Normalize to [0, 1] range
    center_x_norm = center_x / img_width
    center_y_norm = (img_height - center_y) / img_height
    width_norm = width / img_width
    height_norm = height / img_height
    
    return bbox['class_id'], center_x_norm, center_y_norm, width_norm, height_norm

def create_yolo_dataset(csv_files, video_files, data_root, output_dir):
    """Create YOLOv8 dataset structure from CSV files and extract corresponding frames."""
    clear_dataset_directories(output_dir)
    
    # Create output directories
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")

    val_labels_dir = os.path.join(output_dir, "val", "labels")
    val_images_dir = os.path.join(output_dir, "val", "images")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # Target image dimensions for YOLOv8
    target_width = 640
    target_height = 640
    
    # Auto-detect original video dimensions from the first available video
    original_width = None
    original_height = None

    # Set split for testing and validation
    training_split = 0.8  # 80% for training
    val_split = (1 - training_split)  # 20% for validation

    now = int(time.time())
    random.seed(now)  # Seed for reproducibility
    random.shuffle(list(csv_files.items()))  # Shuffle CSV files for consistent splits
    random.shuffle(list(video_files.items()))  # Shuffle video files for consistent splits

    # Create train/val directories
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    # Randomly sort csv and video files to ensure consistent splits

    
    if video_files:
        first_video = list(video_files.values())[0]
        print(f"Auto-detecting video dimensions from: {os.path.basename(first_video)}")
        
        detected_width, detected_height = get_video_dimensions(first_video)
        if detected_width and detected_height:
            original_width = detected_width
            original_height = detected_height
            print(f"Detected video dimensions: {original_width}x{original_height}")
        else:
            print("Failed to detect video dimensions, using fallback values")
            original_width = 1936  # Fallback
            original_height = 1464  # Fallback
    else:
        print("No video files available for dimension detection, using default values")
        original_width = 1936  # Default fallback
        original_height = 1464  # Default fallback
    
    processed_frames = set()
    frame_info = {}  # Store frame info for extraction
    all_class_ids = set()  # Collect all unique class IDs
    
    # First pass: collect all frame information and class IDs
    for cam_name, csv_file in csv_files.items():
        print(f"Processing CSV for {cam_name}...")
        
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            
            # Skip header line
            next(reader, None)
            
            for row in reader:
                if len(row) < 7:
                    continue
                    
                data = parse_csv_line(row)
                if not data or not data['bboxes']:
                    continue
                
                frame_num = data['frame']
                frame_key = f"{cam_name}_frame_{frame_num:06d}"
                
                # Collect all class IDs
                for bbox in data['bboxes']:
                    all_class_ids.add(bbox['class_id'])
                
                if frame_key in processed_frames:
                    continue
                    
                processed_frames.add(frame_key)
                
                # Store frame info for later extraction
                frame_info[frame_key] = {
                    'cam_name': cam_name,
                    'frame_num': frame_num,
                    'bboxes': data['bboxes']
                }
    
    # Create class mapping from original IDs to normalized IDs (0, 1, 2, ...)
    sorted_class_ids = sorted(list(all_class_ids))
    class_id_mapping = {original_id: normalized_id for normalized_id, original_id in enumerate(sorted_class_ids)}
    
    print(f"Found {len(frame_info)} unique frames to process")
    print(f"Detected class IDs: {sorted_class_ids}")
    print(f"Class mapping: {class_id_mapping}")

    frame_items = list(frame_info.items())
    random.shuffle(frame_items)
    
    total_frames = len(frame_items)
    train_count = int(total_frames * training_split)
    
    print(f"Total frames: {total_frames}, Training: {train_count}, Validation: {total_frames - train_count}")
    
    #TODO: Remove this since it probably isn't necessary

    # Copy YOLO model to dataset directory (2 directories up from CSV location)
    if csv_files:
        first_csv = list(csv_files.values())[0]
        csv_dir = os.path.dirname(first_csv)
        model_search_dir = os.path.abspath(os.path.join(csv_dir, "..", ".."))
        
        print(f"Looking for YOLO model files in: {model_search_dir}")
        
        # Look for YOLO model files (.pt, .pth, .onnx)
        model_patterns = ["*.pt", "*.pth", "*.onnx"]
        model_files = []
        
        for pattern in model_patterns:
            found_models = glob.glob(os.path.join(model_search_dir, pattern))
            model_files.extend(found_models)
        
        if model_files:
            print(f"Found {len(model_files)} model files:")
            for model_file in model_files:
                print(f"  - {os.path.basename(model_file)}")
                
                # Copy model file to dataset directory
                dest_model = os.path.join(output_dir, os.path.basename(model_file))
                shutil.copy2(model_file, dest_model)
                print(f"Copied model to: {dest_model}")
        else:
            print("No YOLO model files (.pt, .pth, .onnx) found in the expected directory")
    
    # Second pass: extract frames and create labels with normalized class IDs
    for i, (frame_key, info) in enumerate(frame_items):
        cam_name = info['cam_name']
        frame_num = info['frame_num']
        bboxes = info['bboxes']
        
        # file save directories
        if cam_name in video_files:
            if i < train_count:  # Use train_count instead of len(frame_info) * training_split
                image_file = os.path.join(train_images_dir, f"{frame_key}.jpg")
                label_file = os.path.join(train_labels_dir, f"{frame_key}.txt")
            else:
                image_file = os.path.join(val_images_dir, f"{frame_key}.jpg")
                label_file = os.path.join(val_labels_dir, f"{frame_key}.txt")

            print(f"Extracting frame {frame_num} from {cam_name}...")
            success = extract_frame_with_ffmpeg(
                video_files[cam_name], 
                frame_num, 
                image_file,
                f"{target_width}x{target_height}"
            )
            
            if not success:
                print(f"Failed to extract frame {frame_num} from {cam_name}")
                continue
        else:
            print(f"Warning: No video file found for {cam_name}, skipping frame extraction")
            continue
        
        with open(label_file, 'w') as lf:
            for bbox in bboxes:
                # Use normalized class ID
                normalized_class_id = class_id_mapping[bbox['class_id']]
                bbox_with_normalized_class = bbox.copy()
                bbox_with_normalized_class['class_id'] = normalized_class_id
                
                class_id, cx, cy, w, h = convert_bbox_to_yolo_format(
                    bbox_with_normalized_class, target_width, target_height, original_width, original_height
                )
                lf.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        print(f"Created: {frame_key}.jpg and {frame_key}.txt with {len(bboxes)} bboxes")
    
    # Create dataset.yaml file with automatically detected classes
    class_names = {}
    for original_id, normalized_id in class_id_mapping.items():
        class_names[normalized_id] = f"class_{original_id}"
    
    # Generate the names section for YAML
    names_section = "names:\n"
    for normalized_id in sorted(class_names.keys()):
        names_section += f"  {normalized_id}: {class_names[normalized_id]}\n"
    
    yaml_content = f"""# YOLOv8 dataset configuration
# Generated automatically from RED CSV export
train: train/images
val: train/images

# Classes detected from CSV data
# Original class IDs: {sorted_class_ids}
# Normalized to: {list(range(len(sorted_class_ids)))}
{names_section.rstrip()}

nc: {len(sorted_class_ids)}  # number of classes
"""
    
    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        f.write(yaml_content)
    
    # Also create a class mapping reference file
    mapping_file = os.path.join(output_dir, "class_mapping.txt")
    with open(mapping_file, 'w') as f:
        f.write("# Class ID Mapping Reference\n")
        f.write("# Original_ID -> Normalized_ID (YOLO_Class_Name)\n")
        for original_id, normalized_id in class_id_mapping.items():
            f.write(f"{original_id} -> {normalized_id} (class_{original_id})\n")
    
    print(f"\nYOLOv8 dataset created in: {output_dir}")
    print(f"Processed {len(frame_info)} frames")
    print(f"Images resized to {target_width}x{target_height} with letterboxing")
    print(f"Detected {len(sorted_class_ids)} unique classes: {sorted_class_ids}")
    print(f"Classes normalized to: {list(range(len(sorted_class_ids)))}")
    print(f"Class mapping saved to: {mapping_file}")

def get_video_dimensions(video_path):
    """Get video dimensions using FFmpeg probe."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFprobe error: {result.stderr}")
            return None, None
        
        import json
        data = json.loads(result.stdout)
        
        # Find the video stream
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                width = int(stream.get('width', 0))
                height = int(stream.get('height', 0))
                return width, height
        
        return None, None
    except Exception as e:
        print(f"Error getting video dimensions: {e}")
        return None, None

def main():
    if len(sys.argv) != 2:
        print("Usage: python csv_to_yolo.py <data_root_directory>")
        sys.exit(1)
    
    data_root = sys.argv[1]
    
    if not os.path.exists(data_root):
        print(f"Error: Directory {data_root} does not exist")
        sys.exit(1)
    
    # Check if FFmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: FFmpeg not found. Please install FFmpeg and make sure it's in your PATH")
        sys.exit(1)
    
    # Find most recent CSV files
    csv_files = find_most_recent_csv_files(data_root)
    
    if not csv_files:
        print("No CSV files found in labeled_data directories")
        sys.exit(1)
    
    # Find corresponding video files (pass csv_files to determine correct location)
    video_files = find_video_files(data_root, csv_files)
    
    if not video_files:
        print("No video files found in movies directory")
        sys.exit(1)
    
    # Check that we have matching CSV and video files
    missing_videos = set(csv_files.keys()) - set(video_files.keys())
    if missing_videos:
        print(f"Warning: Missing video files for cameras: {missing_videos}")
    
    # Create output directory
    output_dir = os.path.join(data_root, "yolo_dataset")
    
    # Convert to YOLO format and extract frames
    create_yolo_dataset(csv_files, video_files, data_root, output_dir)
    
    print(f"\nYOLOv8 dataset ready at: {output_dir}")
    print("Dataset includes:")
    print("- Extracted video frames resized to 640x640")
    print("- YOLO format labels with adjusted coordinates")
    print("- data.yaml configuration file")
    print("\nNext steps:")
    print("1. Update data.yaml with correct class names and count")
    print("2. Split into train/val datasets if needed")
    print("3. Train your YOLOv8 model!")

if __name__ == "__main__":
    main()