#include "global.h"

std::unordered_map<std::string, std::atomic<bool>> window_need_decoding;
std::unordered_map<std::string, std::atomic<int>> latest_decoded_frame;
std::map<int, int> g_yolo_class_map;
std::map<int, int> g_reverse_yolo_class_map;
int next_class_id = 0;
float confidence_threshold = 0.5f;
float iou_threshold = 0.4f;
#ifndef __APPLE__
std::vector<std::vector<YoloBBox>> yolo_bboxes(MAX_VIEWS);
std::vector<std::vector<YoloPrediction>> yolo_predictions(MAX_VIEWS);
#endif
std::string yolo_model_path = "";
std::vector<std::vector<BoundingBox>> yolo_drag_boxes(MAX_VIEWS);
std::vector<int> yolo_active_bbox_idx(MAX_VIEWS, -1);
std::vector<int> user_active_bbox_idx(MAX_VIEWS, -1);
