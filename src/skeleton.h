#ifndef RED_SKELETON
#define RED_SKELETON
#include "imgui.h"
#include "json.hpp"
#include "render.h"
#include "types.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct KeyPoints2D {
    tuple_d position;
    tuple_d last_position;
    bool is_labeled;
    bool last_is_labeled;
    float confidence;
};

struct KeyPoints3D {
    triple_d position;
    bool is_triangulated;
    float confidence;
};

struct KeyPoints {
    KeyPoints3D *kp3d;
    KeyPoints2D **kp2d;
    u32 *active_id;
};

struct SkeletonContext {
    int num_nodes;
    int num_edges;
    std::vector<ImVec4> node_colors;
    std::vector<tuple_i> edges;
    std::vector<std::string> node_names;
    std::string name;
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
    Table3Corners,
    Rat22,
    Rat20,
    Rat24,
    Rat20Target,
    Rat24Target,
    SP_LOAD
};

std::map<std::string, SkeletonPrimitive> skeleton_get_all();
bool has_labeled_frames(const std::map<u32, KeyPoints *> &keypoints_map,
                        SkeletonContext *skeleton);
void load_skeleton_json(std::string file_name, SkeletonContext *skeleton);
void skeleton_initialize(std::string name, SkeletonContext *skeleton,
                         SkeletonPrimitive skeleton_type);
void allocate_keypoints(KeyPoints *keypoints, RenderScene *scene,
                        SkeletonContext *skeleton);
void free_keypoints(KeyPoints *keypoints, RenderScene *scene);
void free_all_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                        RenderScene *scene);
void cleanup_skeleton_data(std::map<u32, KeyPoints *> &keypoints_map,
                           RenderScene *scene);

bool has_any_labels(const KeyPoints *keypoints, const SkeletonContext &skeleton,
                    const RenderScene *scene, float yolo_thresh = 0.0f);

void copy_keypoints(KeyPoints *dst, const KeyPoints *src,
                    const RenderScene *scene, const SkeletonContext *skeleton);
#endif
