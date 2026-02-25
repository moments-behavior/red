#ifndef RED_GLOBAL
#define RED_GLOBAL

#define MAX_VIEWS 20

#include "skeleton.h"
#ifndef __APPLE__
#include "yolo_torch.h"
#endif
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

extern std::unordered_map<std::string, std::atomic<bool>> window_need_decoding;
extern std::unordered_map<std::string, std::atomic<int>> latest_decoded_frame;
extern std::map<int, int> g_yolo_class_map;
extern std::map<int, int> g_reverse_yolo_class_map;
extern int next_class_id;
extern float confidence_threshold;
extern float iou_threshold;
#ifndef __APPLE__
extern std::vector<std::vector<YoloBBox>> yolo_bboxes;
extern std::vector<std::vector<YoloPrediction>> yolo_predictions;
#endif
extern std::string yolo_model_path;
extern std::vector<std::vector<BoundingBox>> yolo_drag_boxes;
extern std::vector<int> yolo_active_bbox_idx;
extern std::vector<int> user_active_bbox_idx;
#endif
