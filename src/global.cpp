#include "global.h"
#include <string>

std::unordered_map<std::string, std::atomic<bool>> window_need_decoding;
std::unordered_map<std::string, std::atomic<int>> latest_decoded_frame;
std::map<int, int> g_yolo_class_map;
std::map<int, int> g_reverse_yolo_class_map;
int next_class_id = 0;
float confidence_threshold = 0.5f;
float iou_threshold = 0.4f;
std::vector<std::vector<YoloBBox>> yolo_bboxes(MAX_VIEWS);
std::vector<std::vector<YoloPrediction>> yolo_predictions(MAX_VIEWS);
std::string yolo_model_path = "";
std::vector<std::vector<BoundingBox>> yolo_drag_boxes(MAX_VIEWS);
std::vector<int> yolo_active_bbox_idx(MAX_VIEWS, -1);
std::vector<int> user_active_bbox_idx(MAX_VIEWS, -1);

// JARVIS + CoTracker globals
std::string jarvis_center_engine_path = "";
std::string jarvis_kp_engine_path     = "";
std::string cotracker_model_path      = "";
float jarvis_confidence_threshold     = 0.3f;
JarvisInfer    *g_jarvis_infer    = nullptr;
CoTrackerInfer *g_cotracker_infer = nullptr;
