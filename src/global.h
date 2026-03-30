#ifndef RED_GLOBAL
#define RED_GLOBAL

#define MAX_VIEWS 20

#include "jarvis_infer.h"
#include "cotracker_infer.h"
#include "skeleton.h"
#include "yolo_torch.h"
#include <atomic>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

extern std::unordered_map<std::string, std::atomic<bool>> window_need_decoding;
extern std::unordered_map<std::string, std::atomic<int>> latest_decoded_frame;
extern std::map<int, int> g_yolo_class_map;
extern std::map<int, int> g_reverse_yolo_class_map;
extern int next_class_id;
extern float confidence_threshold;
extern float iou_threshold;
extern std::vector<std::vector<YoloBBox>> yolo_bboxes;
extern std::vector<std::vector<YoloPrediction>> yolo_predictions;
extern std::string yolo_model_path;
extern std::vector<std::vector<BoundingBox>> yolo_drag_boxes;
extern std::vector<int> yolo_active_bbox_idx;
extern std::vector<int> user_active_bbox_idx;

// JARVIS + CoTracker globals
extern std::string jarvis_center_engine_path;
extern std::string jarvis_kp_engine_path;
extern std::string cotracker_model_path;
extern float jarvis_confidence_threshold;
extern JarvisInfer *g_jarvis_infer;
extern CoTrackerInfer *g_cotracker_infer;
#endif
