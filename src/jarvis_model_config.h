#pragma once
// jarvis_model_config.h — Shared model configuration for JARVIS inference backends
//
// Parsed once from model_info.json and passed to both ONNX and CoreML init.

#include "json.hpp"
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
    } catch (...) {}
    return cfg;
}
