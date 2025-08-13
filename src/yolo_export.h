#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include "json.hpp"

namespace YoloExport {

// Structure to hold bounding box data
struct BoundingBox {
    int class_id;
    std::vector<float> bbox;  // [x_min, y_min, x_max, y_max]
    std::vector<std::vector<float>> keypoints;  // For pose: [[x, y, visibility], ...]
    std::vector<float> obb_corners;  // For OBB: [x1, y1, x2, y2, x3, y3, x4, y4] normalized
};

// Structure to hold frame annotation data
struct FrameAnnotation {
    std::string frame_id;
    std::vector<BoundingBox> bboxes;
};

// Structure for dataset split ratios
struct DatasetSplit {
    float train_ratio = 0.7f;
    float val_ratio = 0.2f;
    float test_ratio = 0.1f;
    int seed = 42;
};

// Structure for export configuration
struct ExportConfig {
    std::string label_dir;
    std::string video_dir;
    std::string output_dir;
    std::string skeleton_file;  // For pose datasets
    std::string class_names_file;  // For detection datasets
    int image_size = 640;
    DatasetSplit split;
    bool use_gpu_decode = false;  // For future GPU acceleration
};

// Image processing functions
cv::Mat resize_image_for_yolo(const cv::Mat& image, int target_size = 640);
std::vector<float> adjust_bbox_for_resize(const std::vector<float>& bbox, 
                                         int original_width, int original_height, 
                                         int target_size);
std::vector<std::vector<float>> adjust_keypoints_for_resize(const std::vector<std::vector<float>>& keypoints,
                                                           int original_width, int original_height,
                                                           int target_size);

// Coordinate conversion functions
std::vector<float> convert_bbox_to_yolo_format(const std::vector<float>& bbox, 
                                              int img_width, int img_height);
std::vector<float> convert_keypoints_to_yolo_format(const std::vector<std::vector<float>>& keypoints,
                                                   int img_width, int img_height,
                                                   int expected_num_keypoints = -1);

// Frame extraction
cv::Mat extract_frame_opencv(const std::string& video_path, int frame_number);

// Data loading functions
std::vector<FrameAnnotation> load_frame_annotations(const std::string& label_dir);
std::vector<FrameAnnotation> load_pose_annotations(const std::string& label_dir, 
                                                  const nlohmann::json& skeleton_config);
std::vector<FrameAnnotation> load_obb_annotations(const std::string& label_dir);
std::map<std::string, std::vector<BoundingBox>> parse_bbox_csv(const std::string& csv_path);
std::pair<std::map<std::string, std::vector<BoundingBox>>, std::map<std::string, std::vector<std::vector<float>>>> 
parse_bbox_keypoints_csv(const std::string& csv_path, const nlohmann::json& skeleton_config);
std::map<std::string, std::vector<std::vector<float>>> parse_keypoint_csv(const std::string& csv_path, 
                                                                         const nlohmann::json& skeleton_config);
std::map<std::string, std::vector<BoundingBox>> parse_obb_csv(const std::string& csv_path);
std::map<std::string, std::string> find_video_files(const std::string& video_dir);
std::vector<std::string> load_class_names(const std::string& class_names_file);
nlohmann::json load_skeleton_config(const std::string& skeleton_file);

// Dataset creation functions
void create_dataset_directories(const std::string& output_dir);
void split_dataset(const std::vector<std::string>& frame_ids, 
                  const DatasetSplit& split,
                  std::vector<std::string>& train_frames,
                  std::vector<std::string>& val_frames,
                  std::vector<std::string>& test_frames);
void create_data_yaml(const std::string& output_dir, 
                     const std::vector<std::string>& class_names,
                     int num_keypoints = 0,
                     const nlohmann::json& skeleton_config = nlohmann::json());

// Main export functions
bool export_yolo_detection_dataset(const ExportConfig& config, std::string* status = nullptr);
bool export_yolo_pose_dataset(const ExportConfig& config, std::string* status = nullptr);
bool export_yolo_obb_dataset(const ExportConfig& config, std::string* status = nullptr);

// Utility functions
void print_progress(int current, int total, const std::string& message = "Processing", std::string* status = nullptr);
std::string sanitize_class_name(const std::string& name);

} // namespace YoloExport
