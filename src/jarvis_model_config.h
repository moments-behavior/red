#pragma once
// jarvis_model_config.h — Shared model configuration for JARVIS inference backends
//
// Parsed once from model_info.json and passed to both ONNX and CoreML init.

#include "json.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

struct JarvisModelConfig {
    int center_input_size = 320;   // CenterDetect input (square)
    int keypoint_input_size = 704; // KeypointDetect input (square)
    int num_joints = 24;
    std::string project_name;
    // Display metadata (parsed alongside config)
    std::string architecture;     // e.g., "EfficientTrack-medium"
    std::string precision;        // e.g., "float16"
    bool imagenet_norm = false;   // ImageNet normalization baked into CoreML model
    // HybridNet 3D volumetric stage (only present for 3D models). roi_cube_size
    // and grid_spacing are physical, in the inference calibration's world units;
    // grid_in/grid_out are the V2VNet input/output voxel-grid sides (dimensionless).
    bool has_hybridnet = false;
    int hn_num_cameras = 16;
    float hn_roi_cube = 200.0f;
    float hn_grid_spacing = 2.0f;
    int hn_grid_in = 100;
    int hn_grid_out = 50;
};

// Parse model_info.json once into a unified config. Safe to call with nullptr.
inline JarvisModelConfig parse_jarvis_model_info(const char *path) {
    JarvisModelConfig cfg;
    if (!path || !std::filesystem::exists(path)) return cfg;
    try {
        std::ifstream f(path);
        nlohmann::json j;
        f >> j;
        if (j.contains("project_name"))
            cfg.project_name = j["project_name"].get<std::string>();
        if (j.contains("center_detect")) {
            auto &cd = j["center_detect"];
            if (cd.contains("input_size"))
                cfg.center_input_size = cd["input_size"].get<int>();
            if (cd.contains("model_size"))
                cfg.architecture = "EfficientTrack-" + cd["model_size"].get<std::string>();
        }
        if (j.contains("keypoint_detect")) {
            auto &kd = j["keypoint_detect"];
            if (kd.contains("input_size"))
                cfg.keypoint_input_size = kd["input_size"].get<int>();
            if (kd.contains("num_joints"))
                cfg.num_joints = kd["num_joints"].get<int>();
        }
        if (j.contains("coreml_info")) {
            auto &ci = j["coreml_info"];
            if (ci.contains("precision"))
                cfg.precision = ci["precision"].get<std::string>();
            if (ci.contains("note")) {
                std::string note = ci["note"].get<std::string>();
                if (note.find("ImageType") != std::string::npos)
                    cfg.imagenet_norm = true;
            }
        }
        if (j.contains("hybridnet")) {
            auto &h = j["hybridnet"];
            cfg.has_hybridnet = true;
            cfg.hn_num_cameras  = h.value("num_cameras",  cfg.hn_num_cameras);
            cfg.hn_roi_cube     = h.value("roi_cube_size", cfg.hn_roi_cube);
            cfg.hn_grid_spacing = h.value("grid_spacing",  cfg.hn_grid_spacing);
            cfg.hn_grid_in      = h.value("grid_in",       cfg.hn_grid_in);
            cfg.hn_grid_out     = h.value("grid_out",      cfg.hn_grid_out);
        }
    } catch (const std::exception &e) {
        // A malformed model_info.json must not look identical to a missing one:
        // surface it so a silently-wrong default config can be diagnosed.
        std::fprintf(stderr, "[jarvis] failed to parse %s: %s\n",
                     path ? path : "(null)", e.what());
    }
    return cfg;
}
