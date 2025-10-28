#include "skeleton.h"
#include "global.h"
#include "render.h"

std::map<std::string, SkeletonPrimitive> skeleton_get_all() {
    std::map<std::string, SkeletonPrimitive> skeleton_all = {
        {"Target", Target},
        {"Rat7Target", Rat7Target},
        {"Rat10Target2", Rat10Target2},
        {"RatTarget", RatTarget},
        {"Rat3Target", Rat3Target},
        {"Rat4Target", Rat4Target},
        {"Rat6Target", Rat6Target},
        {"Rat6Target2", Rat6Target2},
        {"Rat6", Rat6},
        {"Rat4", Rat4},
        {"Rat4Box", Rat4Box},
        {"Rat4Box3Ball", Rat4Box3Ball},
        {"Table3Corners", Table3Corners},
        {"Rat22", Rat22},
        {"Rat20", Rat20},
        {"Rat24", Rat24},
        {"Rat20Target", Rat20Target},
        {"Rat24Target", Rat24Target},
        {"BoundingBox", SP_BBOX},
        {"OrientedBoundingBox", SP_OBB},
        {"Simple BBox+Skeleton", SP_SIMPLE_BBOX_SKELETON}};
    return skeleton_all;
}

void load_skeleton_json(std::string file_name, SkeletonContext *skeleton) {
    std::ifstream f(file_name);
    nlohmann::json s_config = nlohmann::json::parse(f);
    skeleton->name = file_name;
    skeleton->has_skeleton = s_config.contains("has_skeleton")
                                 ? s_config["has_skeleton"].get<bool>()
                                 : true;
    skeleton->has_bbox = s_config.contains("has_bbox")
                             ? s_config["has_bbox"].get<bool>()
                             : false;
    skeleton->has_obb =
        s_config.contains("has_obb") ? s_config["has_obb"].get<bool>() : false;
    skeleton->num_nodes = s_config["num_nodes"];
    skeleton->num_edges = s_config["num_edges"];

    for (int i = 0; i < s_config["node_names"].size(); i++) {
        skeleton->node_names.push_back(s_config["node_names"][i]);
    }

    for (int i = 0; i < s_config["edges"].size(); i++) {
        tuple_i edge_start_end = {s_config["edges"][i][0],
                                  s_config["edges"][i][1]};
        skeleton->edges.push_back(edge_start_end);
    }
    for (int i = 0; i < skeleton->num_nodes; i++) {
        ImVec4 color =
            (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes, 1.0f, 1.0f);
        skeleton->node_colors.push_back(color);
    }
}

void skeleton_initialize(std::string name, SkeletonContext *skeleton,
                         SkeletonPrimitive skeleton_type) {
    skeleton->has_skeleton = true;
    skeleton->has_bbox = false;
    skeleton->has_obb = false;

    switch (skeleton_type) {
    case Table3Corners:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->has_obb = false;
        skeleton->num_nodes = 3;
        skeleton->num_edges = 2;

        skeleton->node_names = {"BottomLeft", "BottomRight", "TopLeft"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                0.8f, 0.8f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {
            {0, 1},
            {0, 2},
        };
        break;

    case Target:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 1;
        skeleton->num_edges = 0;
        skeleton->node_names = {"Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        break;

    case Rat7Target:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 9;
        skeleton->num_edges = 8;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",
                                "Neck",   "SpineF",   "SpineM",
                                "SpineL", "TailBase", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3},
                           {3, 4}, {4, 5}, {5, 6}, {6, 7}};
        break;

    case RatTarget:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 2;
        skeleton->num_edges = 1;
        skeleton->node_names = {"Snout", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}};

        break;

    case Rat3Target:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = false;
        skeleton->num_nodes = 4;
        skeleton->num_edges = 2;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}, {0, 2}};
        break;

    case Rat4Target:
        skeleton->name = name;
        skeleton->num_nodes = 5;
        skeleton->num_edges = 4;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Tail", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        break;

    case Rat4:
        skeleton->name = name;
        skeleton->num_nodes = 4;
        skeleton->num_edges = 4;
        skeleton->node_names = {"Snout", "EarL", "EarR", "Tail"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }
        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};
        break;

    case Rat6Target:
        skeleton->name = name;
        skeleton->num_nodes = 7;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",  "Neck",
                                "SpineL", "TailBase", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat6Target2:
        skeleton->name = name;
        skeleton->num_nodes = 8;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",    "Neck",
                                "SpineL", "TailBase", "Target1", "Target2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat6:
        skeleton->name = name;
        skeleton->num_nodes = 6;
        skeleton->num_edges = 6;
        skeleton->node_names = {"Snout", "EarL",   "EarR",
                                "Neck",  "SpineL", "TailBase"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}};
        break;

    case Rat4Box:
        skeleton->name = name;
        skeleton->num_nodes = 6;
        skeleton->num_edges = 3;
        skeleton->node_names = {"EarR", "EarL",    "Snout",
                                "Tail", "TopLeft", "BottomRight"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 2}, {1, 2}, {2, 3}};
        break;

    case Rat4Box3Ball:
        skeleton->name = name;
        skeleton->num_nodes = 9;
        skeleton->num_edges = 3;
        skeleton->node_names = {"EarR",  "EarL",    "Snout",
                                "Tail",  "TopLeft", "BottomRight",
                                "Ball0", "Ball1",   "Ball2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 2}, {1, 2}, {2, 3}};
        break;

    case Rat10Target2:
        skeleton->name = name;
        skeleton->num_nodes = 12;
        skeleton->num_edges = 10;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",    "Neck",
                                "SpineL", "TailBase", "HandL",   "HandR",
                                "FootL",  "FootR",    "Target1", "Target2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {3, 4},
                           {4, 5}, {3, 6}, {3, 7}, {4, 8}, {4, 9}};
        break;

    case Rat20:
        skeleton->name = name;
        skeleton->num_nodes = 20;
        skeleton->num_edges = 20;
        skeleton->node_names = {"Snout",  "EarL",     "EarR",      "Neck",
                                "SpineL", "TailBase", "ShoulderL", "ElbowL",
                                "WristL", "HandL",    "ShoulderR", "ElbowR",
                                "WristR", "HandR",    "KneeL",     "AnkleL",
                                "FootL",  "KneeR",    "AnkleR",    "FootR"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19}};
        break;

    case Rat24:
        skeleton->name = name;
        skeleton->num_nodes = 24;
        skeleton->num_edges = 24;
        skeleton->node_names = {"Snout",   "EarL",     "EarR",      "Neck",
                                "SpineL",  "TailBase", "ShoulderL", "ElbowL",
                                "WristL",  "HandL",    "ShoulderR", "ElbowR",
                                "WristR",  "HandR",    "KneeL",     "AnkleL",
                                "FootL",   "KneeR",    "AnkleR",    "FootR",
                                "TailTip", "TailMid",  "Tail1Q",    "Tail3Q"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19},
                           {5, 22},  {22, 21}, {21, 23}, {23, 20}};
        break;

    case Rat20Target:
        skeleton->name = name;
        skeleton->num_nodes = 21;
        skeleton->num_edges = 20;
        skeleton->node_names = {
            "Snout",     "EarL",   "EarR",   "Neck",   "SpineL",    "TailBase",
            "ShoulderL", "ElbowL", "WristL", "HandL",  "ShoulderR", "ElbowR",
            "WristR",    "HandR",  "KneeL",  "AnkleL", "FootL",     "KneeR",
            "AnkleR",    "FootR",  "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19}};
        break;

    case Rat24Target:
        skeleton->name = name;
        skeleton->num_nodes = 25;
        skeleton->num_edges = 24;
        skeleton->node_names = {
            "Snout",     "EarL",      "EarR",   "Neck",   "SpineL",
            "TailBase",  "ShoulderL", "ElbowL", "WristL", "HandL",
            "ShoulderR", "ElbowR",    "WristR", "HandR",  "KneeL",
            "AnkleL",    "FootL",     "KneeR",  "AnkleR", "FootR",
            "TailTip",   "TailMid",   "Tail1Q", "Tail3Q", "Target"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {4, 17},  {17, 18}, {18, 19},
                           {5, 22},  {22, 21}, {21, 23}, {23, 20}};
        break;

    case Rat22:
        skeleton->name = name;
        skeleton->num_nodes = 22;
        skeleton->num_edges = 22;
        skeleton->node_names = {
            "Snout",     "EarL",   "EarR",   "Neck",  "SpineL",    "TailBase",
            "ShoulderL", "ElbowL", "WristL", "HandL", "ShoulderR", "ElbowR",
            "WristR",    "HandR",  "HipL",   "KneeL", "AnkleL",    "FootL",
            "HipR",      "KneeR",  "AnkleR", "FootR"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1},   {0, 2},   {1, 3},   {2, 3},   {3, 4},
                           {4, 5},   {3, 6},   {6, 7},   {7, 8},   {8, 9},
                           {3, 10},  {10, 11}, {11, 12}, {12, 13}, {4, 14},
                           {14, 15}, {15, 16}, {16, 17}, {4, 18},  {18, 19},
                           {19, 20}, {20, 21}};
        break;

    case SP_BBOX:
        skeleton->name = name;
        skeleton->has_bbox = true;
        skeleton->has_obb = false;
        skeleton->has_skeleton = false;
        break;

    case SP_OBB:
        skeleton->name = name;
        skeleton->has_bbox = false;
        skeleton->has_obb = true;
        skeleton->has_skeleton = false;
        break;

    case SP_SIMPLE_BBOX_SKELETON:
        skeleton->name = name;
        skeleton->has_skeleton = true;
        skeleton->has_bbox = true;
        skeleton->has_obb = false;
        skeleton->num_nodes = 2;
        skeleton->num_edges = 1;
        skeleton->node_names = {"Point1", "Point2"};

        for (int i = 0; i < skeleton->num_nodes; i++) {
            ImVec4 color = (ImVec4)ImColor::HSV(i / (float)skeleton->num_nodes,
                                                1.0f, 1.0f);
            skeleton->node_colors.push_back(color);
        }

        skeleton->edges = {{0, 1}};
        break;
    }
}

bool has_labeled_frames(const std::map<u32, KeyPoints *> &keypoints_map,
                        SkeletonContext *skeleton) {
    for (const auto &[frame_num, keypoints] : keypoints_map) {
        if (!keypoints)
            continue;

        if (skeleton->has_skeleton) {
            for (size_t cam_id = 0;
                 cam_id < keypoints->bbox2d_list.size() && cam_id < MAX_VIEWS;
                 cam_id++) {
                for (int kp_id = 0; kp_id < skeleton->num_nodes; kp_id++) {
                    if (keypoints->kp2d[cam_id][kp_id].is_labeled) {
                        return true;
                    }
                }
            }
        }

        if (skeleton->has_bbox) {
            for (size_t cam_id = 0; cam_id < keypoints->bbox2d_list.size();
                 cam_id++) {
                for (const auto &bbox : keypoints->bbox2d_list[cam_id]) {
                    if (bbox.state == RectTwoPoints) {
                        return true;
                    }
                }
            }
        }

        if (skeleton->has_obb) {
            for (size_t cam_id = 0; cam_id < keypoints->obb2d_list.size();
                 cam_id++) {
                for (const auto &obb : keypoints->obb2d_list[cam_id]) {
                    if (obb.state == OBBComplete) {
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

void allocate_keypoints(KeyPoints *keypoints, RenderScene *scene,
                        SkeletonContext *skeleton) {
    // allocate memory for storing keypoints
    keypoints->active_id = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    keypoints->cam_supress = (bool *)malloc(sizeof(u32) * scene->num_cams);

    new (&keypoints->bbox2d_list) std::vector<std::vector<BoundingBox>>();
    new (&keypoints->obb2d_list)
        std::vector<std::vector<OrientedBoundingBox>>();

    if (skeleton->has_bbox) {
        keypoints->bbox2d_list.resize(scene->num_cams);
        for (u32 j = 0; j < scene->num_cams; j++) {
            // Initialize with a default BoundingBox
            BoundingBox default_bbox;
            default_bbox.rect = NULL;
            default_bbox.state = RectNull;
            default_bbox.class_id = -1;
            default_bbox.confidence = 0.0f;
            default_bbox.has_bbox_keypoints = false;
            default_bbox.bbox_keypoints2d = nullptr;
            default_bbox.active_kp_id = nullptr;
            keypoints->bbox2d_list[j].push_back(default_bbox);
        }
    } else {
        keypoints->bbox2d_list.resize(scene->num_cams);
    }

    if (skeleton->has_obb) {
        keypoints->obb2d_list.resize(scene->num_cams);
        for (u32 j = 0; j < scene->num_cams; j++) {
            // Initialize with a default OrientedBoundingBox
            OrientedBoundingBox default_obb;
            default_obb.axis_point1 = ImVec2(0, 0);
            default_obb.axis_point2 = ImVec2(0, 0);
            default_obb.corner_point = ImVec2(0, 0);
            default_obb.center = ImVec2(0, 0);
            default_obb.width = 0;
            default_obb.height = 0;
            default_obb.rotation = 0;
            default_obb.state = OBBNull;
            default_obb.class_id = -1;
            default_obb.confidence = 0.0f;
            keypoints->obb2d_list[j].push_back(default_obb);
        }
    } else {
        keypoints->obb2d_list.resize(scene->num_cams);
    }

    if (skeleton->has_skeleton) {
        keypoints->kp3d =
            (KeyPoints3D *)malloc(sizeof(KeyPoints3D) * skeleton->num_nodes);
        keypoints->kp2d =
            (KeyPoints2D **)malloc(sizeof(KeyPoints2D *) * scene->num_cams);
        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->kp2d[j] = (KeyPoints2D *)malloc(sizeof(KeyPoints2D) *
                                                       skeleton->num_nodes);
        }

        // initialize to big number
        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->active_id[j] = 0;
            for (u32 k = 0; k < skeleton->num_nodes; k++) {
                keypoints->kp2d[j][k].is_labeled = false;
                keypoints->kp2d[j][k].position.x = 1E7;
                keypoints->kp2d[j][k].position.y = 1E7;
                keypoints->kp2d[j][k].last_position =
                    keypoints->kp2d[j][k].position;
                keypoints->kp2d[j][k].last_is_labeled = false;
            }
        }

        for (u32 k = 0; k < skeleton->num_nodes; k++) {
            keypoints->kp3d[k].position.x = 1E7;
            keypoints->kp3d[k].position.y = 1E7;
            keypoints->kp3d[k].position.z = 1E7;
            keypoints->kp3d[k].is_triangulated = false;
        }
    } else {
        keypoints->kp3d = nullptr;
        keypoints->kp2d = nullptr;
        // Still need to initialize active_id array even without skeleton
        for (u32 j = 0; j < scene->num_cams; j++) {
            keypoints->active_id[j] = 0;
        }
    }
}

void free_bbox_keypoints(BoundingBox *bbox, RenderScene *scene) {
    if (!bbox->has_bbox_keypoints)
        return;

    if (bbox->bbox_keypoints2d) {
        for (u32 j = 0; j < scene->num_cams; j++) {
            if (bbox->bbox_keypoints2d[j]) {
                free(bbox->bbox_keypoints2d[j]);
            }
        }
        free(bbox->bbox_keypoints2d);
        bbox->bbox_keypoints2d = nullptr;
    }
    if (bbox->active_kp_id) {
        free(bbox->active_kp_id);
        bbox->active_kp_id = nullptr;
    }
    bbox->has_bbox_keypoints = false;
}

void free_keypoints(KeyPoints *keypoints, RenderScene *scene) {
    if (!keypoints)
        return;

    if (keypoints->kp2d) {
        for (u32 j = 0; j < scene->num_cams; ++j) {
            free(keypoints->kp2d[j]);
        }
        free(keypoints->kp2d);
    }

    free(keypoints->active_id);
    free(keypoints->kp3d);

    // Free bounding boxes
    for (u32 j = 0; j < scene->num_cams; j++) {
        for (auto &bbox : keypoints->bbox2d_list[j]) {
            if (bbox.rect) {
                delete bbox.rect;
            }
            free_bbox_keypoints(&bbox, scene);
        }
    }

    keypoints->bbox2d_list.clear();
    keypoints->obb2d_list.clear();

    free(keypoints); // finally free the KeyPoints struct itself
}

void allocate_bbox_keypoints(BoundingBox *bbox, RenderScene *scene,
                             SkeletonContext *skeleton) {
    if (!skeleton->has_skeleton) {
        bbox->has_bbox_keypoints = false;
        return;
    }

    bbox->has_bbox_keypoints = true;
    bbox->active_kp_id = (u32 *)malloc(sizeof(u32) * scene->num_cams);
    bbox->bbox_keypoints2d =
        (KeyPoints2D **)malloc(sizeof(KeyPoints2D *) * scene->num_cams);

    for (u32 j = 0; j < scene->num_cams; j++) {
        bbox->bbox_keypoints2d[j] =
            (KeyPoints2D *)malloc(sizeof(KeyPoints2D) * skeleton->num_nodes);
        bbox->active_kp_id[j] = 0;

        for (u32 k = 0; k < skeleton->num_nodes; k++) {
            bbox->bbox_keypoints2d[j][k].is_labeled = false;
            bbox->bbox_keypoints2d[j][k].position.x = 1E7;
            bbox->bbox_keypoints2d[j][k].position.y = 1E7;
        }
    }
}

void constrain_keypoint_to_bbox(KeyPoints2D *keypoint, ImPlotRect *bbox_rect) {
    if (!bbox_rect || !keypoint->is_labeled)
        return;

    if (keypoint->position.x < bbox_rect->X.Min) {
        keypoint->position.x = bbox_rect->X.Min;
    } else if (keypoint->position.x > bbox_rect->X.Max) {
        keypoint->position.x = bbox_rect->X.Max;
    }

    if (keypoint->position.y < bbox_rect->Y.Min) {
        keypoint->position.y = bbox_rect->Y.Min;
    } else if (keypoint->position.y > bbox_rect->Y.Max) {
        keypoint->position.y = bbox_rect->Y.Max;
    }
}
bool is_point_in_bbox(double x, double y, ImPlotRect *bbox_rect) {
    if (!bbox_rect)
        return false;
    return (x >= bbox_rect->X.Min && x <= bbox_rect->X.Max &&
            y >= bbox_rect->Y.Min && y <= bbox_rect->Y.Max);
}
void scale_bbox_keypoints(BoundingBox *bbox, RenderScene *scene,
                          SkeletonContext *skeleton, ImPlotRect *old_rect,
                          ImPlotRect *new_rect) {
    if (!bbox->has_bbox_keypoints || !old_rect || !new_rect)
        return;

    double old_width = old_rect->X.Max - old_rect->X.Min;
    double old_height = old_rect->Y.Max - old_rect->Y.Min;
    double new_width = new_rect->X.Max - new_rect->X.Min;
    double new_height = new_rect->Y.Max - new_rect->Y.Min;

    if (old_width <= 0 || old_height <= 0)
        return;

    double old_min_x = std::min(old_rect->X.Min, old_rect->X.Max);
    double old_min_y = std::min(old_rect->Y.Min, old_rect->Y.Max);
    double old_max_x = std::max(old_rect->X.Min, old_rect->X.Max);
    double old_max_y = std::max(old_rect->Y.Min, old_rect->Y.Max);

    double new_min_x = std::min(new_rect->X.Min, new_rect->X.Max);
    double new_min_y = std::min(new_rect->Y.Min, new_rect->Y.Max);
    double new_max_x = std::max(new_rect->X.Min, new_rect->X.Max);
    double new_max_y = std::max(new_rect->Y.Min, new_rect->Y.Max);

    double normalized_old_width = old_max_x - old_min_x;
    double normalized_old_height = old_max_y - old_min_y;

    double normalized_new_width = new_max_x - new_min_x;
    double normalized_new_height = new_max_y - new_min_y;

    if (normalized_old_width <= 0 || normalized_old_height <= 0)
        return;

    double scale_x = normalized_new_width / normalized_old_width;
    double scale_y = normalized_new_height / normalized_old_height;

    for (u32 j = 0; j < scene->num_cams; j++) {
        for (u32 node = 0; node < skeleton->num_nodes; node++) {
            if (bbox->bbox_keypoints2d[j][node].is_labeled) {
                // Get relative position in old bbox
                double rel_x =
                    (bbox->bbox_keypoints2d[j][node].position.x - old_min_x) /
                    normalized_old_width;
                double rel_y =
                    (bbox->bbox_keypoints2d[j][node].position.y - old_min_y) /
                    normalized_old_height;

                // Scale to new bbox
                bbox->bbox_keypoints2d[j][node].position.x =
                    new_min_x + rel_x * normalized_new_width;
                bbox->bbox_keypoints2d[j][node].position.y =
                    new_min_y + rel_y * normalized_new_height;

                constrain_keypoint_to_bbox(&bbox->bbox_keypoints2d[j][node],
                                           new_rect);
            }
        }
    }
}

void free_all_keypoints(std::map<u32, KeyPoints *> &keypoints_map,
                        RenderScene *scene) {
    for (auto &[frame, kp] : keypoints_map)
        free_keypoints(kp, scene);
    keypoints_map.clear();
}

// Oriented Bounding Box utility functions
float calculate_distance(ImVec2 p1, ImVec2 p2) {
    float dx = p2.x - p1.x;
    float dy = p2.y - p1.y;
    return sqrtf(dx * dx + dy * dy);
}

float calculate_angle(ImVec2 p1, ImVec2 p2) {
    return atan2f(p2.y - p1.y, p2.x - p1.x);
}

void cleanup_skeleton_data(std::map<u32, KeyPoints *> &keypoints_map,
                           RenderScene *scene) {
    // Free all existing keypoints
    if (!keypoints_map.empty()) {
        free_all_keypoints(keypoints_map, scene);
        keypoints_map.clear();
    }
}

bool has_any_labels(const KeyPoints *keypoints, const SkeletonContext &skeleton,
                    const RenderScene *scene, float yolo_thresh) {

    if (!keypoints)
        return false;

    // Skeleton keypoints
    if (skeleton.has_skeleton) {
        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS;
             ++cam_id) {
            for (int kp_id = 0; kp_id < skeleton.num_nodes; ++kp_id) {
                if (keypoints->kp2d[cam_id][kp_id].is_labeled)
                    return true;
            }
        }
    }

    // Bounding boxes
    if (skeleton.has_bbox) {
        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS;
             ++cam_id) {
            for (const auto &bbox : keypoints->bbox2d_list[cam_id]) {
                // Manual or YOLO rectangles (RectTwoPoints) with confidence
                // >= yolo_thresh
                if (bbox.state == RectTwoPoints &&
                    bbox.confidence >= yolo_thresh)
                    return true;

                // BBox keypoints
                if (bbox.has_bbox_keypoints && bbox.bbox_keypoints2d &&
                    bbox.bbox_keypoints2d[cam_id]) {
                    for (int kp_id = 0; kp_id < skeleton.num_nodes; ++kp_id) {
                        if (bbox.bbox_keypoints2d[cam_id][kp_id].is_labeled)
                            return true;
                    }
                }
            }
        }
    }

    // OBBs
    if (skeleton.has_obb) {
        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS;
             ++cam_id) {
            for (const auto &obb : keypoints->obb2d_list[cam_id]) {
                if (obb.state == OBBComplete)
                    return true;
            }
        }
    }

    return false;
}

bool has_any_labels(const KeyPoints *keypoints, const SkeletonContext &skeleton,
                    const RenderScene *scene) {
    if (!keypoints)
        return false;

    // --- Skeleton keypoints ---
    if (skeleton.has_skeleton) {
        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS;
             ++cam_id) {
            for (int kp_id = 0; kp_id < skeleton.num_nodes; ++kp_id) {
                if (keypoints->kp2d[cam_id][kp_id].is_labeled)
                    return true;
            }
        }
    }

    // --- Bounding boxes ---
    if (skeleton.has_bbox) {
        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS;
             ++cam_id) {
            for (const auto &bbox : keypoints->bbox2d_list[cam_id]) {
                if ((bbox.state == RectTwoPoints && bbox.confidence >= 0.0f) ||
                    (bbox.confidence >= 1.0f && bbox.has_bbox_keypoints)) {

                    if (bbox.state == RectTwoPoints)
                        return true;

                    if (bbox.has_bbox_keypoints && bbox.bbox_keypoints2d &&
                        bbox.bbox_keypoints2d[cam_id]) {
                        for (int kp_id = 0; kp_id < skeleton.num_nodes;
                             ++kp_id) {
                            if (bbox.bbox_keypoints2d[cam_id][kp_id].is_labeled)
                                return true;
                        }
                    }
                }
            }
        }
    }

    // --- OBBs ---
    if (skeleton.has_obb) {
        for (int cam_id = 0; cam_id < scene->num_cams && cam_id < MAX_VIEWS;
             ++cam_id) {
            for (const auto &obb : keypoints->obb2d_list[cam_id]) {
                if (obb.state == OBBComplete)
                    return true;
            }
        }
    }

    return false;
}

void copy_keypoints(KeyPoints *dst, const KeyPoints *src,
                    const RenderScene *scene, const SkeletonContext *skeleton) {
    if (!dst || !src)
        return;

    // Copy simple pointers and allocate new memory where necessary

    // --- Copy 3D keypoints ---
    if (src->kp3d) {
        dst->kp3d = (KeyPoints3D *)malloc(sizeof(KeyPoints3D));
        *dst->kp3d =
            *src->kp3d; // assuming it's a POD / trivially copyable type
    } else {
        dst->kp3d = nullptr;
    }

    // --- Copy triangulation flags ---
    if (src->is_triangulated) {
        size_t num_nodes = skeleton->num_nodes;
        dst->is_triangulated = (bool *)malloc(sizeof(bool) * num_nodes);
        std::memcpy(dst->is_triangulated, src->is_triangulated,
                    sizeof(bool) * num_nodes);
    } else {
        dst->is_triangulated = nullptr;
    }

    // --- Copy active ID ---
    if (src->active_id) {
        dst->active_id = (u32 *)malloc(sizeof(u32));
        *dst->active_id = *src->active_id;
    } else {
        dst->active_id = nullptr;
    }

    // --- Copy camera suppression flags ---
    if (src->cam_supress) {
        size_t num_cams = scene->num_cams;
        dst->cam_supress = (bool *)malloc(sizeof(bool) * num_cams);
        std::memcpy(dst->cam_supress, src->cam_supress,
                    sizeof(bool) * num_cams);
    } else {
        dst->cam_supress = nullptr;
    }

    // --- Copy 2D keypoints ---
    if (src->kp2d) {
        int num_cams = scene->num_cams;
        int num_nodes = skeleton->num_nodes;
        dst->kp2d = (KeyPoints2D **)malloc(sizeof(KeyPoints2D *) * num_cams);
        for (int cam = 0; cam < num_cams; ++cam) {
            if (src->kp2d[cam]) {
                dst->kp2d[cam] =
                    (KeyPoints2D *)malloc(sizeof(KeyPoints2D) * num_nodes);
                std::memcpy(dst->kp2d[cam], src->kp2d[cam],
                            sizeof(KeyPoints2D) * num_nodes);
            } else {
                dst->kp2d[cam] = nullptr;
            }
        }
    } else {
        dst->kp2d = nullptr;
    }

    // --- Copy 2D bounding boxes ---
    dst->bbox2d_list =
        src->bbox2d_list; // std::vector performs deep copy of POD types

    // --- Copy oriented bounding boxes ---
    dst->obb2d_list = src->obb2d_list; // std::vector deep-copies too
}
