#ifndef RED_SKELETON
#define RED_SKELETON
#include "imgui.h"
#include "implot.h"
#include "json.hpp"
#include "render.h"
#include "types.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct KeyPoints2D {
    tuple_d position;
    bool is_labeled;
    bool is_triangulated;
};

enum RectState { RectNull, RectOnePoint, RectTwoPoints };

enum OBBState {
    OBBNull,
    OBBFirstAxisPoint,
    OBBSecondAxisPoint,
    OBBThirdPoint,
    OBBComplete
};

struct BoundingBox {
    ImPlotRect *rect;
    RectState state;
    int class_id;
    int id;
    float confidence;

    KeyPoints2D **bbox_keypoints2d; // Per-camera keypoints for this bbox
    bool has_bbox_keypoints;
    u32 *active_kp_id; // Active keypoint ID per camera for this bbox
};

struct OrientedBoundingBox {
    ImVec2 axis_point1;  // First axis point
    ImVec2 axis_point2;  // Second axis point
    ImVec2 corner_point; // Third point to define width/height
    ImVec2 center;       // Center of the OBB
    float width;         // Width of the OBB
    float height;        // Height of the OBB
    float rotation;      // Rotation angle in radians
    OBBState state;      // Current drawing state
    int class_id;
    int id;
    float confidence;
};

struct KeyPoints {
    triple_d *keypoints3d;
    KeyPoints2D **keypoints2d;
    u32 *active_id;
    std::vector<std::vector<BoundingBox>> bbox2d_list;
    std::vector<std::vector<OrientedBoundingBox>> obb2d_list;
};

struct SkeletonContext {
    int num_nodes;
    int num_edges;
    std::vector<ImVec4> node_colors;
    std::vector<tuple_i> edges;
    std::vector<std::string> node_names;
    std::string name;
    bool has_bbox;
    bool has_obb;
    bool has_skeleton;
};

enum SkeletonPrimitive {
    Target,
    Rat7Target,
    Rat10Target2,
    RatTarget,
    Rat3Target,
    Rat4Target,
    Rat6Target,
    Rat6Target2,
    Rat6,
    Rat4,
    Rat4Box,
    Rat4Box3Ball,
    Table3Corners,
    Rat22,
    Rat20,
    Rat24,
    Rat20Target,
    Rat24Target,
    SP_BBOX,
    SP_OBB,
    SP_SIMPLE_BBOX_SKELETON,
    SP_LOAD
};

std::map<std::string, SkeletonPrimitive> skeleton_get_all();
bool has_labeled_frames(const std::map<u32, KeyPoints *> &keypoints_map,
                        SkeletonContext *skeleton);
void load_skeleton_json(std::string file_name, SkeletonContext *skeleton);
void skeleton_initialize(std::string name, std::string skeleton_file_name,
                         SkeletonContext *skeleton,
                         SkeletonPrimitive skeleton_type);
void allocate_keypoints(KeyPoints *keypoints, render_scene *scene,
                        SkeletonContext *skeleton);
void free_bbox_keypoints(BoundingBox *bbox, render_scene *scene);
void free_keypoints(KeyPoints *keypoints, render_scene *scene);
void allocate_bbox_keypoints(BoundingBox *bbox, render_scene *scene,
                             SkeletonContext *skeleton);
void constrain_keypoint_to_bbox(KeyPoints2D *keypoint, ImPlotRect *bbox_rect);
bool is_point_in_bbox(double x, double y, ImPlotRect *bbox_rect);
void scale_bbox_keypoints(BoundingBox *bbox, render_scene *scene,
                          SkeletonContext *skeleton, ImPlotRect *old_rect,
                          ImPlotRect *new_rect);
void free_all_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                        render_scene *scene);
float calculate_distance(ImVec2 p1, ImVec2 p2);
float calculate_angle(ImVec2 p1, ImVec2 p2);
void cleanup_skeleton_data(std::map<u32, KeyPoints *> &keypoints_map,
                           render_scene *scene);
#endif
