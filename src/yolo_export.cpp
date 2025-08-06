#include "yolo_export.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cstdlib>
#include <ctime>

namespace YoloExport {

cv::Mat resize_image_for_yolo(const cv::Mat& image, int target_size) {
    int h = image.rows;
    int w = image.cols;
    
    // Calculate scaling factor to fit image in square while maintaining aspect ratio
    float scale = static_cast<float>(target_size) / std::max(h, w);
    
    // Calculate new dimensions
    int new_w = static_cast<int>(w * scale);
    int new_h = static_cast<int>(h * scale);
    
    // Resize image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    
    // Create square canvas and center the resized image
    cv::Mat canvas = cv::Mat::zeros(target_size, target_size, CV_8UC3);
    
    // Calculate padding offsets to center the image
    int y_offset = (target_size - new_h) / 2;
    int x_offset = (target_size - new_w) / 2;
    
    // Copy resized image to center of canvas
    resized.copyTo(canvas(cv::Rect(x_offset, y_offset, new_w, new_h)));
    
    return canvas;
}

std::vector<float> adjust_bbox_for_resize(const std::vector<float>& bbox, 
                                         int original_width, int original_height, 
                                         int target_size) {
    if (bbox.size() != 4) {
        return bbox;
    }
    
    float scale = static_cast<float>(target_size) / std::max(original_height, original_width);
    int new_w = static_cast<int>(original_width * scale);
    int new_h = static_cast<int>(original_height * scale);
    int x_offset = (target_size - new_w) / 2;
    int y_offset = (target_size - new_h) / 2;
    
    std::vector<float> adjusted_bbox(4);
    adjusted_bbox[0] = bbox[0] * scale + x_offset;  // x_min
    adjusted_bbox[1] = bbox[1] * scale + y_offset;  // y_min
    adjusted_bbox[2] = bbox[2] * scale + x_offset;  // x_max
    adjusted_bbox[3] = bbox[3] * scale + y_offset;  // y_max
    
    return adjusted_bbox;
}

std::vector<std::vector<float>> adjust_keypoints_for_resize(const std::vector<std::vector<float>>& keypoints,
                                                           int original_width, int original_height,
                                                           int target_size) {
    float scale = static_cast<float>(target_size) / std::max(original_height, original_width);
    int new_w = static_cast<int>(original_width * scale);
    int new_h = static_cast<int>(original_height * scale);
    int x_offset = (target_size - new_w) / 2;
    int y_offset = (target_size - new_h) / 2;
    
    std::vector<std::vector<float>> adjusted_keypoints;
    for (const auto& kp : keypoints) {
        if (kp.size() >= 2) {
            std::vector<float> adjusted_kp = kp;
            adjusted_kp[0] = kp[0] * scale + x_offset;  // x
            adjusted_kp[1] = kp[1] * scale + y_offset;  // y
            // Keep visibility unchanged if present
            adjusted_keypoints.push_back(adjusted_kp);
        }
    }
    
    return adjusted_keypoints;
}

std::vector<float> convert_bbox_to_yolo_format(const std::vector<float>& bbox, 
                                              int img_width, int img_height) {
    if (bbox.size() != 4) {
        return {};
    }
    
    float x_min = bbox[0];
    float y_min = bbox[1];
    float x_max = bbox[2];
    float y_max = bbox[3];
    
    float x_center = (x_min + x_max) / 2.0f / img_width;
    float y_center = (y_min + y_max) / 2.0f / img_height;
    float width = (x_max - x_min) / img_width;
    float height = (y_max - y_min) / img_height;
    
    return {x_center, y_center, width, height};
}

std::vector<float> convert_keypoints_to_yolo_format(const std::vector<std::vector<float>>& keypoints,
                                                   int img_width, int img_height) {
    std::vector<float> yolo_keypoints;
    
    for (const auto& kp : keypoints) {
        if (kp.size() >= 2) {
            float x = kp[0] / img_width;
            float y = kp[1] / img_height;
            float visibility = (kp.size() >= 3) ? kp[2] : 2.0f;  // Default to visible
            
            yolo_keypoints.push_back(x);
            yolo_keypoints.push_back(y);
            yolo_keypoints.push_back(visibility);
        }
    }
    
    return yolo_keypoints;
}

cv::Mat extract_frame_ffmpeg(const std::string& video_path, int frame_number) {
    // Create a temporary directory for frame extraction
    std::string temp_dir = "/tmp/frame_extraction_" + std::to_string(std::time(nullptr));
    std::string mkdir_cmd = "mkdir -p " + temp_dir;
    if (system(mkdir_cmd.c_str()) != 0) {
        std::cerr << "Error: Failed to create temporary directory: " << temp_dir << std::endl;
        return cv::Mat();
    }
    
    // Extract the specific frame using FFmpeg
    std::string output_path = temp_dir + "/frame.png";
    std::string ffmpeg_cmd = "ffmpeg -i \"" + video_path + "\" -vf \"select=eq(n\\," + 
                            std::to_string(frame_number) + ")\" -vframes 1 \"" + output_path + "\" -y 2>/dev/null";
    
    std::cout << "Extracting frame " << frame_number << " using FFmpeg..." << std::endl;
    
    int result = system(ffmpeg_cmd.c_str());
    if (result != 0) {
        std::cerr << "Error: FFmpeg frame extraction failed for frame " << frame_number << std::endl;
        system(("rm -rf " + temp_dir).c_str());
        return cv::Mat();
    }
    
    // Load the extracted frame using OpenCV
    cv::Mat frame = cv::imread(output_path, cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Error: Could not load extracted frame from " << output_path << std::endl;
        system(("rm -rf " + temp_dir).c_str());
        return cv::Mat();
    }
    
    std::cout << "Successfully extracted frame " << frame_number << " (" << frame.cols << "x" << frame.rows << ")" << std::endl;
    
    // Clean up temporary files
    system(("rm -rf " + temp_dir).c_str());
    
    return frame;
}

std::vector<FrameAnnotation> load_frame_annotations(const std::string& label_dir) {
    std::vector<FrameAnnotation> annotations;
    std::map<std::string, std::vector<BoundingBox>> all_camera_bboxes;
    
    // Find all bbox CSV files
    for (const auto& entry : std::filesystem::recursive_directory_iterator(label_dir)) {
        if (entry.path().extension() == ".csv" && 
            entry.path().filename().string().find("_bboxes.csv") != std::string::npos) {
            
            std::string camera_name = entry.path().stem().string();
            // Remove "_bboxes" suffix
            if (camera_name.length() > 7 && camera_name.substr(camera_name.length() - 7) == "_bboxes") {
                camera_name = camera_name.substr(0, camera_name.length() - 7);
            }
            
            auto camera_bboxes = parse_bbox_csv(entry.path().string());
            
            // Convert per-camera bboxes to frame annotations with camera prefix
            for (const auto& frame_data : camera_bboxes) {
                std::string original_frame_id = frame_data.first;
                std::string full_frame_id = camera_name + "_" + original_frame_id;
                for (const auto& bbox : frame_data.second) {
                    all_camera_bboxes[full_frame_id].push_back(bbox);
                }
            }
        }
    }
    
    // Convert to FrameAnnotation format
    for (const auto& frame_data : all_camera_bboxes) {
        FrameAnnotation annotation;
        annotation.frame_id = frame_data.first;
        annotation.bboxes = frame_data.second;
        annotations.push_back(annotation);
    }
    
    return annotations;
}

std::map<std::string, std::vector<BoundingBox>> parse_bbox_csv(const std::string& csv_path) {
    std::map<std::string, std::vector<BoundingBox>> frame_bboxes;
    
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
        return frame_bboxes;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip empty lines and header lines
        if (line.empty() || line.find("skeleton") == 0 || line.find("frame,bbox_id") == 0) {
            continue;
        }
        
        // Parse CSV: frame_id,bbox_id,class_id,confidence,x_min,y_min,x_max,y_max
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 8) {
            try {
                std::string frame_id = tokens[0];
                BoundingBox bbox;
                bbox.class_id = std::stoi(tokens[2]);
                // bbox.confidence = std::stof(tokens[3]); // We don't use confidence in BoundingBox struct
                bbox.bbox = {
                    std::stof(tokens[4]),  // x_min
                    std::stof(tokens[5]),  // y_min
                    std::stof(tokens[6]),  // x_max
                    std::stof(tokens[7])   // y_max
                };
                
                frame_bboxes[frame_id].push_back(bbox);
            } catch (const std::exception& e) {
                std::cerr << "Error parsing CSV line: " << line << " - " << e.what() << std::endl;
            }
        }
    }
    
    return frame_bboxes;
}

std::map<std::string, std::vector<std::vector<float>>> parse_keypoint_csv(const std::string& csv_path, 
                                                                         const nlohmann::json& skeleton_config) {
    std::map<std::string, std::vector<std::vector<float>>> frame_keypoints;
    
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open keypoint CSV file: " << csv_path << std::endl;
        return frame_keypoints;
    }
    
    int num_nodes = 17;  // Default for COCO
    if (skeleton_config.contains("num_nodes")) {
        num_nodes = skeleton_config["num_nodes"];
    } else if (skeleton_config.contains("keypoints") && skeleton_config["keypoints"].is_array()) {
        num_nodes = skeleton_config["keypoints"].size();
    }
    
    std::string line;
    bool first_line = true;
    while (std::getline(file, line)) {
        // Skip empty lines and first line (skeleton name)
        if (line.empty() || first_line) {
            first_line = false;
            continue;
        }
        
        // Parse CSV: frame_id,kp_id_0,x_0,y_0,z_0,kp_id_1,x_1,y_1,z_1,...
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() >= 1 + num_nodes * 4) {  // frame_id + num_nodes * (id,x,y,z)
            try {
                std::string frame_id = tokens[0];
                std::vector<std::vector<float>> keypoints;
                
                for (int i = 0; i < num_nodes; ++i) {
                    int base_idx = 1 + i * 4;  // Skip frame_id, then groups of 4 (id, x, y, z)
                    
                    if (base_idx + 3 < tokens.size()) {
                        float x = std::stof(tokens[base_idx + 1]);
                        float y = std::stof(tokens[base_idx + 2]);
                        // z coordinate is stored but not used for YOLO format
                        
                        // Check if keypoint is labeled (not NaN or invalid)
                        float visibility = 0;  // Not labeled
                        if (!(std::isnan(x) || std::isnan(y) || x == 1e7 || y == 1e7)) {
                            visibility = 2;  // Visible and labeled
                        } else {
                            x = 0;
                            y = 0;
                        }
                        
                        keypoints.push_back({x, y, visibility});
                    } else {
                        // Not enough data, mark as not labeled
                        keypoints.push_back({0, 0, 0});
                    }
                }
                
                frame_keypoints[frame_id] = keypoints;
            } catch (const std::exception& e) {
                std::cerr << "Error parsing keypoint CSV line: " << line << " - " << e.what() << std::endl;
            }
        }
    }
    
    return frame_keypoints;
}

std::vector<FrameAnnotation> load_pose_annotations(const std::string& label_dir, 
                                                  const nlohmann::json& skeleton_config) {
    std::vector<FrameAnnotation> annotations;
    std::map<std::string, std::vector<BoundingBox>> all_camera_bboxes;
    std::map<std::string, std::vector<std::vector<float>>> all_camera_keypoints;
    
    // Find all bbox CSV files and keypoint CSV files
    std::vector<std::string> camera_names;
    
    for (const auto& entry : std::filesystem::recursive_directory_iterator(label_dir)) {
        if (entry.path().extension() == ".csv") {
            std::string filename = entry.path().filename().string();
            
            if (filename.find("_bboxes.csv") != std::string::npos) {
                std::string camera_name = entry.path().stem().string();
                // Remove "_bboxes" suffix
                if (camera_name.length() > 7 && camera_name.substr(camera_name.length() - 7) == "_bboxes") {
                    camera_name = camera_name.substr(0, camera_name.length() - 7);
                }
                
                auto camera_bboxes = parse_bbox_csv(entry.path().string());
                for (const auto& frame_data : camera_bboxes) {
                    std::string original_frame_id = frame_data.first;
                    std::string full_frame_id = camera_name + "_" + original_frame_id;
                    for (const auto& bbox : frame_data.second) {
                        all_camera_bboxes[full_frame_id].push_back(bbox);
                    }
                }
                
                // Add to camera names if not already present
                if (std::find(camera_names.begin(), camera_names.end(), camera_name) == camera_names.end()) {
                    camera_names.push_back(camera_name);
                }
            }
        }
    }
    
    // Load keypoints for each camera
    for (const std::string& camera_name : camera_names) {
        std::string keypoint_file = label_dir + "/" + camera_name + ".csv";
        if (std::filesystem::exists(keypoint_file)) {
            auto camera_keypoints = parse_keypoint_csv(keypoint_file, skeleton_config);
            for (const auto& frame_data : camera_keypoints) {
                std::string original_frame_id = frame_data.first;
                std::string full_frame_id = camera_name + "_" + original_frame_id;
                all_camera_keypoints[full_frame_id] = frame_data.second;
            }
        }
    }
    
    // Combine bboxes and keypoints into frame annotations
    std::set<std::string> all_frame_ids;
    for (const auto& bbox_data : all_camera_bboxes) {
        all_frame_ids.insert(bbox_data.first);
    }
    for (const auto& kp_data : all_camera_keypoints) {
        all_frame_ids.insert(kp_data.first);
    }
    
    for (const std::string& frame_id : all_frame_ids) {
        FrameAnnotation annotation;
        annotation.frame_id = frame_id;
        
        // Add bounding boxes for this frame
        if (all_camera_bboxes.find(frame_id) != all_camera_bboxes.end()) {
            for (auto& bbox : all_camera_bboxes[frame_id]) {
                // Add keypoints to the bbox if available
                if (all_camera_keypoints.find(frame_id) != all_camera_keypoints.end()) {
                    bbox.keypoints = all_camera_keypoints[frame_id];
                }
                annotation.bboxes.push_back(bbox);
            }
        }
        
        // If no bboxes but keypoints exist, create a default bbox
        if (annotation.bboxes.empty() && all_camera_keypoints.find(frame_id) != all_camera_keypoints.end()) {
            BoundingBox default_bbox;
            default_bbox.class_id = 0;  // Default to class 0 (e.g., person)
            default_bbox.bbox = {0, 0, 100, 100};  // Default bbox, will be adjusted later
            default_bbox.keypoints = all_camera_keypoints[frame_id];
            annotation.bboxes.push_back(default_bbox);
        }
        
        if (!annotation.bboxes.empty()) {
            annotations.push_back(annotation);
        }
    }
    
    return annotations;
}

std::map<std::string, std::string> find_video_files(const std::string& video_dir) {
    std::map<std::string, std::string> video_files;
    
    std::cout << "Searching for video files in: " << video_dir << std::endl;
    
    for (const auto& entry : std::filesystem::recursive_directory_iterator(video_dir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        if (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv") {
            std::string basename = entry.path().stem().string();
            std::string full_path = entry.path().string();
            video_files[basename] = full_path;
            std::cout << "Found video: " << basename << " -> " << full_path << std::endl;
        }
    }
    
    std::cout << "Total video files found: " << video_files.size() << std::endl;
    return video_files;
}

std::vector<std::string> load_class_names(const std::string& class_names_file) {
    std::vector<std::string> class_names;
    std::ifstream file(class_names_file);
    
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                class_names.push_back(line);
            }
        }
    }
    
    return class_names;
}

nlohmann::json load_skeleton_config(const std::string& skeleton_file) {
    nlohmann::json skeleton_config;
    
    std::ifstream file(skeleton_file);
    if (file.is_open()) {
        try {
            file >> skeleton_config;
        }
        catch (const std::exception& e) {
            std::cerr << "Error parsing skeleton file: " << e.what() << std::endl;
        }
    }
    
    return skeleton_config;
}

void create_dataset_directories(const std::string& output_dir) {
    std::vector<std::string> dirs = {
        output_dir,
        output_dir + "/train/images",
        output_dir + "/train/labels",
        output_dir + "/val/images", 
        output_dir + "/val/labels",
        output_dir + "/test/images",
        output_dir + "/test/labels"
    };
    
    for (const auto& dir : dirs) {
        std::filesystem::create_directories(dir);
    }
}

void split_dataset(const std::vector<std::string>& frame_ids, 
                  const DatasetSplit& split,
                  std::vector<std::string>& train_frames,
                  std::vector<std::string>& val_frames,
                  std::vector<std::string>& test_frames) {
    
    // Set random seed for reproducible splits
    std::mt19937 rng(split.seed);
    
    std::vector<std::string> shuffled_frames = frame_ids;
    std::shuffle(shuffled_frames.begin(), shuffled_frames.end(), rng);
    
    size_t total_frames = shuffled_frames.size();
    size_t train_count = static_cast<size_t>(total_frames * split.train_ratio);
    size_t val_count = static_cast<size_t>(total_frames * split.val_ratio);
    
    train_frames.assign(shuffled_frames.begin(), shuffled_frames.begin() + train_count);
    val_frames.assign(shuffled_frames.begin() + train_count, shuffled_frames.begin() + train_count + val_count);
    test_frames.assign(shuffled_frames.begin() + train_count + val_count, shuffled_frames.end());
}

void create_data_yaml(const std::string& output_dir, 
                     const std::vector<std::string>& class_names,
                     int num_keypoints) {
    std::ofstream yaml_file(output_dir + "/data.yaml");
    
    yaml_file << "# YOLO Dataset Configuration\n";
    yaml_file << "path: " << std::filesystem::absolute(output_dir).string() << "\n";
    yaml_file << "train: train/images\n";
    yaml_file << "val: val/images\n";
    yaml_file << "test: test/images\n\n";
    
    yaml_file << "# Number of classes\n";
    yaml_file << "nc: " << class_names.size() << "\n\n";
    
    yaml_file << "# Class names\n";
    yaml_file << "names:\n";
    for (size_t i = 0; i < class_names.size(); ++i) {
        yaml_file << "  " << i << ": " << class_names[i] << "\n";
    }
    
    if (num_keypoints > 0) {
        yaml_file << "\n# Keypoints for pose estimation\n";
        yaml_file << "kpt_shape: [" << num_keypoints << ", 3]  # number of keypoints, number of dimensions (x, y, visibility)\n";
    }
}

void print_progress(int current, int total, const std::string& message) {
    if ((current + 1) % 10 == 0 || current == total - 1) {
        std::cout << message << ": " << (current + 1) << "/" << total << " frames" << std::endl;
    }
}

std::string sanitize_class_name(const std::string& name) {
    std::string sanitized = name;
    std::replace(sanitized.begin(), sanitized.end(), ' ', '_');
    return sanitized;
}

bool export_yolo_detection_dataset(const ExportConfig& config) {
    std::cout << "Starting YOLO Detection Dataset Export..." << std::endl;
    
    // Validate input directories
    if (!std::filesystem::exists(config.label_dir)) {
        std::cerr << "Error: Label directory does not exist: " << config.label_dir << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(config.video_dir)) {
        std::cerr << "Error: Video directory does not exist: " << config.video_dir << std::endl;
        return false;
    }
    
    // Load annotations and find videos
    std::cout << "Loading annotations..." << std::endl;
    auto annotations = load_frame_annotations(config.label_dir);
    
    if (annotations.empty()) {
        std::cerr << "Error: No annotations found in " << config.label_dir << std::endl;
        return false;
    }
    
    std::cout << "Finding video files..." << std::endl;
    auto video_files = find_video_files(config.video_dir);
    
    if (video_files.empty()) {
        std::cerr << "Error: No video files found in " << config.video_dir << std::endl;
        return false;
    }
    
    // Load class names
    std::vector<std::string> class_names;
    if (!config.class_names_file.empty()) {
        class_names = load_class_names(config.class_names_file);
    } else {
        // Extract unique class names from annotations
        std::set<std::string> unique_classes;
        for (const auto& annotation : annotations) {
            for (const auto& bbox : annotation.bboxes) {
                unique_classes.insert("class_" + std::to_string(bbox.class_id));
            }
        }
        class_names.assign(unique_classes.begin(), unique_classes.end());
    }
    
    // Create output directories
    std::cout << "Creating output directories..." << std::endl;
    create_dataset_directories(config.output_dir);
    
    // Split dataset
    std::vector<std::string> frame_ids;
    for (const auto& annotation : annotations) {
        frame_ids.push_back(annotation.frame_id);
    }
    
    std::vector<std::string> train_frames, val_frames, test_frames;
    split_dataset(frame_ids, config.split, train_frames, val_frames, test_frames);
    
    std::cout << "Dataset split: " << train_frames.size() << " train, " 
              << val_frames.size() << " val, " << test_frames.size() << " test" << std::endl;
    
    // Process each split
    std::vector<std::pair<std::string, std::vector<std::string>>> splits = {
        {"train", train_frames},
        {"val", val_frames},
        {"test", test_frames}
    };
    
    for (const auto& [split_name, split_frames] : splits) {
        if (split_frames.empty()) continue;
        
        std::cout << "Processing " << split_name << " split..." << std::endl;
        
        std::string images_dir = config.output_dir + "/" + split_name + "/images";
        std::string labels_dir = config.output_dir + "/" + split_name + "/labels";
        
        for (size_t i = 0; i < split_frames.size(); ++i) {
            const std::string& frame_id = split_frames[i];
            
            // Find corresponding annotation
            auto it = std::find_if(annotations.begin(), annotations.end(),
                                 [&frame_id](const FrameAnnotation& ann) {
                                     return ann.frame_id == frame_id;
                                 });
            
            if (it == annotations.end()) continue;
            
            // Parse frame ID: format is {camera_name}_{original_frame_id}
            std::string camera_name;
            std::string original_frame_id;
            size_t underscore_pos = frame_id.find_last_of('_');
            
            if (underscore_pos != std::string::npos) {
                camera_name = frame_id.substr(0, underscore_pos);
                original_frame_id = frame_id.substr(underscore_pos + 1);
            } else {
                // Fallback: assume frame_id is just the frame number and find matching video
                original_frame_id = frame_id;
                // Try to find any video file that might match
                if (!video_files.empty()) {
                    camera_name = video_files.begin()->first;
                }
            }
            
            // Find video file
            auto video_it = video_files.find(camera_name);
            if (video_it == video_files.end()) {
                std::cerr << "Warning: Video file not found for camera: " << camera_name << std::endl;
                continue;
            }
            
            // Convert frame ID to frame number
            int frame_number;
            try {
                frame_number = std::stoi(original_frame_id);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid frame number: " << original_frame_id << std::endl;
                continue;
            }
            
            // Extract frame
            cv::Mat frame = extract_frame_ffmpeg(video_it->second, frame_number);
            if (frame.empty()) {
                std::cerr << "Failed to extract frame " << frame_id << std::endl;
                continue;
            }
            
            // Resize image for YOLO if needed
            int original_height = frame.rows;
            int original_width = frame.cols;
            if (original_width != config.image_size || original_height != config.image_size) {
                frame = resize_image_for_yolo(frame, config.image_size);
            }
            
            // Save image
            std::string img_filename = frame_id + ".jpg";
            std::string img_path = images_dir + "/" + img_filename;
            cv::imwrite(img_path, frame);
            
            // Generate YOLO detection format labels
            int img_height = frame.rows;
            int img_width = frame.cols;
            std::string label_filename = frame_id + ".txt";
            std::string label_path = labels_dir + "/" + label_filename;
            
            std::ofstream label_file(label_path);
            for (const auto& bbox_data : it->bboxes) {
                auto bbox = bbox_data.bbox;
                
                // Adjust bbox coordinates for resized image
                if (original_width != config.image_size || original_height != config.image_size) {
                    bbox = adjust_bbox_for_resize(bbox, original_width, original_height, config.image_size);
                }
                
                // Convert bbox to YOLO format
                auto yolo_bbox = convert_bbox_to_yolo_format(bbox, img_width, img_height);
                
                if (yolo_bbox.size() == 4) {
                    label_file << bbox_data.class_id << " " 
                              << std::fixed << std::setprecision(6)
                              << yolo_bbox[0] << " " << yolo_bbox[1] << " " 
                              << yolo_bbox[2] << " " << yolo_bbox[3] << "\n";
                }
            }
            
            print_progress(i, split_frames.size(), "  Processed");
        }
    }
    
    // Create data.yaml
    std::cout << "Creating data.yaml..." << std::endl;
    create_data_yaml(config.output_dir, class_names);
    
    std::cout << "YOLO Detection Dataset export completed successfully!" << std::endl;
    return true;
}

bool export_yolo_pose_dataset(const ExportConfig& config) {
    std::cout << "Starting YOLO Pose Dataset Export..." << std::endl;
    
    // Validate input directories
    if (!std::filesystem::exists(config.label_dir)) {
        std::cerr << "Error: Label directory does not exist: " << config.label_dir << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(config.video_dir)) {
        std::cerr << "Error: Video directory does not exist: " << config.video_dir << std::endl;
        return false;
    }
    
    // Load skeleton configuration
    nlohmann::json skeleton_config;
    int num_keypoints = 0;
    if (!config.skeleton_file.empty()) {
        skeleton_config = load_skeleton_config(config.skeleton_file);
        if (skeleton_config.contains("keypoints") && skeleton_config["keypoints"].is_array()) {
            num_keypoints = skeleton_config["keypoints"].size();
        }
    }
    
    // Load annotations and find videos
    std::cout << "Loading annotations..." << std::endl;
    auto annotations = load_pose_annotations(config.label_dir, skeleton_config);
    
    if (annotations.empty()) {
        std::cerr << "Error: No annotations found in " << config.label_dir << std::endl;
        return false;
    }
    
    std::cout << "Finding video files..." << std::endl;
    auto video_files = find_video_files(config.video_dir);
    
    if (video_files.empty()) {
        std::cerr << "Error: No video files found in " << config.video_dir << std::endl;
        return false;
    }
    
    // Extract class names (for pose, typically just "person" or similar)
    std::set<std::string> unique_classes;
    for (const auto& annotation : annotations) {
        for (const auto& bbox : annotation.bboxes) {
            unique_classes.insert("class_" + std::to_string(bbox.class_id));
        }
    }
    std::vector<std::string> class_names(unique_classes.begin(), unique_classes.end());
    
    // Create output directories
    std::cout << "Creating output directories..." << std::endl;
    create_dataset_directories(config.output_dir);
    
    // Split dataset
    std::vector<std::string> frame_ids;
    for (const auto& annotation : annotations) {
        frame_ids.push_back(annotation.frame_id);
    }
    
    std::vector<std::string> train_frames, val_frames, test_frames;
    split_dataset(frame_ids, config.split, train_frames, val_frames, test_frames);
    
    std::cout << "Dataset split: " << train_frames.size() << " train, " 
              << val_frames.size() << " val, " << test_frames.size() << " test" << std::endl;
    
    // Process each split
    std::vector<std::pair<std::string, std::vector<std::string>>> splits = {
        {"train", train_frames},
        {"val", val_frames},
        {"test", test_frames}
    };
    
    for (const auto& [split_name, split_frames] : splits) {
        if (split_frames.empty()) continue;
        
        std::cout << "Processing " << split_name << " split..." << std::endl;
        
        std::string images_dir = config.output_dir + "/" + split_name + "/images";
        std::string labels_dir = config.output_dir + "/" + split_name + "/labels";
        
        for (size_t i = 0; i < split_frames.size(); ++i) {
            const std::string& frame_id = split_frames[i];
            
            // Find corresponding annotation
            auto it = std::find_if(annotations.begin(), annotations.end(),
                                 [&frame_id](const FrameAnnotation& ann) {
                                     return ann.frame_id == frame_id;
                                 });
            
            if (it == annotations.end()) continue;
            
            // Parse frame ID: format is {camera_name}_{original_frame_id}
            std::string camera_name;
            std::string original_frame_id;
            size_t underscore_pos = frame_id.find_last_of('_');
            
            if (underscore_pos != std::string::npos) {
                camera_name = frame_id.substr(0, underscore_pos);
                original_frame_id = frame_id.substr(underscore_pos + 1);
            } else {
                // Fallback: assume frame_id is just the frame number and find matching video
                original_frame_id = frame_id;
                // Try to find any video file that might match
                if (!video_files.empty()) {
                    camera_name = video_files.begin()->first;
                }
            }
            
            // Find video file
            auto video_it = video_files.find(camera_name);
            if (video_it == video_files.end()) {
                std::cerr << "Warning: Video file not found for camera: " << camera_name << std::endl;
                continue;
            }
            
            // Convert frame ID to frame number
            int frame_number;
            try {
                frame_number = std::stoi(original_frame_id);
            } catch (const std::exception& e) {
                std::cerr << "Warning: Invalid frame number: " << original_frame_id << std::endl;
                continue;
            }
            
            // Extract frame
            cv::Mat frame = extract_frame_ffmpeg(video_it->second, frame_number);
            if (frame.empty()) {
                std::cerr << "Failed to extract frame " << frame_id << std::endl;
                continue;
            }
            
            // Resize image for YOLO if needed
            int original_height = frame.rows;
            int original_width = frame.cols;
            if (original_width != config.image_size || original_height != config.image_size) {
                frame = resize_image_for_yolo(frame, config.image_size);
            }
            
            // Save image
            std::string img_filename = frame_id + ".jpg";
            std::string img_path = images_dir + "/" + img_filename;
            cv::imwrite(img_path, frame);
            
            // Generate YOLO pose format labels
            int img_height = frame.rows;
            int img_width = frame.cols;
            std::string label_filename = frame_id + ".txt";
            std::string label_path = labels_dir + "/" + label_filename;
            
            std::ofstream label_file(label_path);
            for (const auto& bbox_data : it->bboxes) {
                auto bbox = bbox_data.bbox;
                auto keypoints = bbox_data.keypoints;
                
                // Adjust coordinates for resized image
                if (original_width != config.image_size || original_height != config.image_size) {
                    bbox = adjust_bbox_for_resize(bbox, original_width, original_height, config.image_size);
                    if (!keypoints.empty()) {
                        keypoints = adjust_keypoints_for_resize(keypoints, original_width, original_height, config.image_size);
                    }
                }
                
                // Convert bbox to YOLO format
                auto yolo_bbox = convert_bbox_to_yolo_format(bbox, img_width, img_height);
                
                // Convert keypoints to YOLO format
                auto yolo_keypoints = convert_keypoints_to_yolo_format(keypoints, img_width, img_height);
                
                if (yolo_bbox.size() == 4) {
                    label_file << bbox_data.class_id << " " 
                              << std::fixed << std::setprecision(6)
                              << yolo_bbox[0] << " " << yolo_bbox[1] << " " 
                              << yolo_bbox[2] << " " << yolo_bbox[3];
                    
                    // Add keypoints if available
                    if (!yolo_keypoints.empty()) {
                        for (float kp : yolo_keypoints) {
                            label_file << " " << std::fixed << std::setprecision(6) << kp;
                        }
                    }
                    
                    label_file << "\n";
                }
            }
            
            print_progress(i, split_frames.size(), "  Processed");
        }
    }
    
    // Create data.yaml
    std::cout << "Creating data.yaml..." << std::endl;
    create_data_yaml(config.output_dir, class_names, num_keypoints);
    
    std::cout << "YOLO Pose Dataset export completed successfully!" << std::endl;
    return true;
}

} // namespace YoloExport
