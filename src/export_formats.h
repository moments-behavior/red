#pragma once
// export_formats.h — Multi-format export dispatcher
//
// Single entry point for exporting annotations to various training frameworks.
// Each format reads from the same AnnotationMap. The JARVIS exporter
// (jarvis_export.h) is called through this dispatcher for JARVIS format.

#include "annotation.h"
#include "jarvis_export.h"
#include "json.hpp"
#include "opencv_yaml_io.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ExportFormats {

enum Format {
    JARVIS,
    JARVIS_TR,
    COCO,
    DEEPLABCUT,
    YOLO_POSE,
    YOLO_DETECT,
    FORMAT_COUNT
};

inline const char *format_name(Format f) {
    switch (f) {
    case JARVIS:      return "JARVIS";
    case JARVIS_TR:   return "JARVIS (with video index)";
    case COCO:        return "COCO Keypoints";
    case DEEPLABCUT:  return "DeepLabCut";
    case YOLO_POSE:   return "YOLO Pose";
    case YOLO_DETECT: return "YOLO Detection";
    default:          return "Unknown";
    }
}

struct ExportConfig {
    Format format = COCO;

    // Paths
    std::string label_folder;       // labeled_data/<timestamp>/
    std::string calibration_folder; // calibration YAMLs
    std::string media_folder;       // video mp4s
    std::string output_folder;      // where to write output

    // Project info
    std::vector<std::string> camera_names;
    std::string skeleton_name;
    std::vector<std::string> node_names;
    std::vector<std::pair<int, int>> edges;
    int num_keypoints = 0;

    // Export options
    float bbox_margin = 50.0f;
    float train_ratio = 0.9f;
    int seed = 42;
    int jpeg_quality = 95;
};

// ── Train/val split helper ──
inline void split_train_val(const std::vector<u32> &frames, float train_ratio,
                            int seed, std::vector<u32> &train, std::vector<u32> &val) {
    std::vector<size_t> indices(frames.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    int n_train = (int)std::floor(frames.size() * train_ratio);
    train.clear(); val.clear();
    for (int i = 0; i < n_train; ++i) train.push_back(frames[indices[i]]);
    for (size_t i = n_train; i < indices.size(); ++i) val.push_back(frames[indices[i]]);
    std::sort(train.begin(), train.end());
    std::sort(val.begin(), val.end());
}

// ── Get labeled frames from AnnotationMap ──
inline std::vector<u32> get_labeled_frames(const AnnotationMap &amap) {
    std::vector<u32> frames;
    for (const auto &[f, fa] : amap)
        if (frame_has_any_labels(fa)) frames.push_back(f);
    return frames;
}

// ═══════════════════════════════════════════════════════════════════════════
// COCO Keypoints export
// ═══════════════════════════════════════════════════════════════════════════
// Standard COCO format compatible with mmpose, detectron2, SLEAP import.
// One JSON per split (train/val), images extracted per camera.

inline nlohmann::json build_coco_json(
    const AnnotationMap &amap, const std::vector<u32> &frames,
    const ExportConfig &cfg, int cam_idx, const std::string &cam_name,
    int img_w, int img_h) {

    nlohmann::json images = nlohmann::json::array();
    nlohmann::json annotations = nlohmann::json::array();
    int img_id = 0, ann_id = 0;

    for (u32 frame : frames) {
        auto it = amap.find(frame);
        if (it == amap.end()) continue;

        std::string filename = cam_name + "/Frame_" + std::to_string(frame) + ".jpg";
        nlohmann::json img;
        img["id"] = img_id;
        img["file_name"] = filename;
        img["width"] = img_w;
        img["height"] = img_h;
        images.push_back(img);

        for (const auto &inst : it->second.instances) {
            if (cam_idx >= (int)inst.cameras.size()) continue;
            const auto &cam = inst.cameras[cam_idx];

            // Count visible keypoints
            int num_visible = 0;
            for (size_t k = 0; k < cam.kp_labeled.size(); ++k)
                if (cam.kp_labeled[k]) ++num_visible;
            if (num_visible == 0) continue;

            // Build flat keypoints array [x,y,v, x,y,v, ...]
            nlohmann::json kp_flat = nlohmann::json::array();
            double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
            for (size_t k = 0; k < cam.keypoints.size(); ++k) {
                if (cam.kp_labeled[k]) {
                    double x = cam.keypoints[k].x;
                    double y = img_h - cam.keypoints[k].y; // ImPlot Y-flip
                    kp_flat.push_back(x); kp_flat.push_back(y); kp_flat.push_back(2);
                    x_min = std::min(x_min, x); x_max = std::max(x_max, x);
                    y_min = std::min(y_min, y); y_max = std::max(y_max, y);
                } else {
                    kp_flat.push_back(0); kp_flat.push_back(0); kp_flat.push_back(0);
                }
            }

            // Bbox from keypoints + margin (or from explicit bbox)
            double bx, by, bw, bh;
            if (cam.has_bbox) {
                bx = cam.bbox_x; by = cam.bbox_y;
                bw = cam.bbox_w; bh = cam.bbox_h;
            } else {
                bx = std::max(x_min - cfg.bbox_margin, 0.0);
                by = std::max(y_min - cfg.bbox_margin, 0.0);
                bw = std::min(x_max + cfg.bbox_margin, (double)img_w) - bx;
                bh = std::min(y_max + cfg.bbox_margin, (double)img_h) - by;
            }

            // Segmentation (mask polygons if available)
            nlohmann::json seg = nlohmann::json::array();
            if (cam.has_mask) {
                for (const auto &poly : cam.mask_polygons) {
                    nlohmann::json flat_poly = nlohmann::json::array();
                    for (const auto &pt : poly) {
                        flat_poly.push_back(pt.x);
                        flat_poly.push_back(img_h - pt.y);
                    }
                    seg.push_back(flat_poly);
                }
            }

            nlohmann::json ann;
            ann["id"] = ann_id++;
            ann["image_id"] = img_id;
            ann["category_id"] = inst.category_id;
            ann["keypoints"] = kp_flat;
            ann["num_keypoints"] = num_visible;
            ann["bbox"] = {bx, by, bw, bh};
            ann["area"] = bw * bh;
            ann["iscrowd"] = 0;
            ann["segmentation"] = seg;
            annotations.push_back(ann);
        }
        img_id++;
    }

    // Build skeleton edges for COCO
    nlohmann::json skel_arr = nlohmann::json::array();
    for (const auto &[a, b] : cfg.edges)
        skel_arr.push_back({a + 1, b + 1}); // COCO uses 1-indexed

    nlohmann::json categories = nlohmann::json::array();
    nlohmann::json cat;
    cat["id"] = 0;
    cat["name"] = cfg.skeleton_name;
    cat["supercategory"] = "animal";
    cat["keypoints"] = cfg.node_names;
    cat["skeleton"] = skel_arr;
    categories.push_back(cat);

    nlohmann::json root;
    root["images"] = images;
    root["annotations"] = annotations;
    root["categories"] = categories;
    return root;
}

inline bool export_coco(const ExportConfig &cfg, const AnnotationMap &amap,
                        std::string *status) {
    namespace fs = std::filesystem;
    auto labeled = get_labeled_frames(amap);
    if (labeled.empty()) {
        if (status) *status = "Error: No labeled frames found.";
        return false;
    }

    std::vector<u32> train, val;
    split_train_val(labeled, cfg.train_ratio, cfg.seed, train, val);

    // Read image dimensions from calibration
    std::map<std::string, int> img_w, img_h;
    for (const auto &cam : cfg.camera_names) {
        std::string path = cfg.calibration_folder + "/" + cam + ".yaml";
        try {
            auto yaml = opencv_yaml::read(path);
            img_w[cam] = yaml.getInt("image_width");
            img_h[cam] = yaml.getInt("image_height");
        } catch (...) {
            if (status) *status = "Error: Cannot read calibration: " + path;
            return false;
        }
    }

    fs::create_directories(cfg.output_folder + "/annotations");

    // One JSON per camera per split
    for (int ci = 0; ci < (int)cfg.camera_names.size(); ++ci) {
        const auto &cam = cfg.camera_names[ci];

        auto train_json = build_coco_json(amap, train, cfg, ci, cam, img_w[cam], img_h[cam]);
        auto val_json   = build_coco_json(amap, val, cfg, ci, cam, img_w[cam], img_h[cam]);

        std::string prefix = cfg.output_folder + "/annotations/";
        {
            std::ofstream f(prefix + cam + "_train.json");
            f << train_json.dump(2);
        }
        {
            std::ofstream f(prefix + cam + "_val.json");
            f << val_json.dump(2);
        }
    }

    if (status)
        *status = "COCO export complete: " + std::to_string(train.size()) +
                  " train, " + std::to_string(val.size()) + " val frames";
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// YOLO Pose / Detection export
// ═══════════════════════════════════════════════════════════════════════════
// YOLO format: data.yaml + images/ + labels/ with .txt per image.
// Each line: <class> <cx> <cy> <w> <h> [<kp_x> <kp_y> <vis> ...]

inline bool export_yolo(const ExportConfig &cfg, const AnnotationMap &amap,
                        bool include_keypoints, std::string *status) {
    namespace fs = std::filesystem;
    auto labeled = get_labeled_frames(amap);
    if (labeled.empty()) {
        if (status) *status = "Error: No labeled frames found.";
        return false;
    }

    std::vector<u32> train, val;
    split_train_val(labeled, cfg.train_ratio, cfg.seed, train, val);

    // Read image dims
    std::map<std::string, int> img_w, img_h;
    for (const auto &cam : cfg.camera_names) {
        std::string path = cfg.calibration_folder + "/" + cam + ".yaml";
        try {
            auto yaml = opencv_yaml::read(path);
            img_w[cam] = yaml.getInt("image_width");
            img_h[cam] = yaml.getInt("image_height");
        } catch (...) {
            if (status) *status = "Error: Cannot read calibration: " + path;
            return false;
        }
    }

    auto write_split = [&](const std::vector<u32> &frames, const std::string &split) {
        for (int ci = 0; ci < (int)cfg.camera_names.size(); ++ci) {
            const auto &cam = cfg.camera_names[ci];
            int w = img_w[cam], h = img_h[cam];

            std::string img_dir = cfg.output_folder + "/images/" + split + "/" + cam;
            std::string lbl_dir = cfg.output_folder + "/labels/" + split + "/" + cam;
            fs::create_directories(img_dir);
            fs::create_directories(lbl_dir);

            for (u32 frame : frames) {
                auto it = amap.find(frame);
                if (it == amap.end()) continue;

                std::string fname = "Frame_" + std::to_string(frame);
                std::ofstream lbl(lbl_dir + "/" + fname + ".txt");

                for (const auto &inst : it->second.instances) {
                    if (ci >= (int)inst.cameras.size()) continue;
                    const auto &c2d = inst.cameras[ci];

                    // Compute bbox (normalized)
                    double bx, by, bw, bh;
                    if (c2d.has_bbox) {
                        bx = c2d.bbox_x; by = c2d.bbox_y;
                        bw = c2d.bbox_w; bh = c2d.bbox_h;
                    } else {
                        // Derive from keypoints
                        double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
                        bool any = false;
                        for (size_t k = 0; k < c2d.kp_labeled.size(); ++k) {
                            if (!c2d.kp_labeled[k]) continue;
                            double x = c2d.keypoints[k].x;
                            double y = h - c2d.keypoints[k].y; // Y-flip
                            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
                            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
                            any = true;
                        }
                        if (!any) continue;
                        bx = std::max(xmin - cfg.bbox_margin, 0.0);
                        by = std::max(ymin - cfg.bbox_margin, 0.0);
                        bw = std::min(xmax + cfg.bbox_margin, (double)w) - bx;
                        bh = std::min(ymax + cfg.bbox_margin, (double)h) - by;
                    }

                    // YOLO format: cx cy w h (all normalized 0-1)
                    double cx = (bx + bw / 2.0) / w;
                    double cy = (by + bh / 2.0) / h;
                    double nw = bw / w;
                    double nh = bh / h;

                    lbl << inst.category_id << " "
                        << std::fixed << std::setprecision(6)
                        << cx << " " << cy << " " << nw << " " << nh;

                    if (include_keypoints) {
                        for (size_t k = 0; k < c2d.keypoints.size(); ++k) {
                            if (c2d.kp_labeled[k]) {
                                double kx = c2d.keypoints[k].x / w;
                                double ky = (h - c2d.keypoints[k].y) / h; // Y-flip
                                lbl << " " << kx << " " << ky << " 2";
                            } else {
                                lbl << " 0 0 0";
                            }
                        }
                    }
                    lbl << "\n";
                }
            }
        }
    };

    write_split(train, "train");
    write_split(val, "val");

    // Write data.yaml
    {
        std::ofstream f(cfg.output_folder + "/data.yaml");
        f << "path: " << cfg.output_folder << "\n";
        f << "train: images/train\n";
        f << "val: images/val\n";
        f << "nc: " << 1 << "\n"; // TODO: multi-class from AnnotationConfig
        f << "names: ['" << cfg.skeleton_name << "']\n";
        if (include_keypoints) {
            f << "kpt_shape: [" << cfg.num_keypoints << ", 3]\n";
        }
    }

    std::string fmt = include_keypoints ? "YOLO Pose" : "YOLO Detection";
    if (status)
        *status = fmt + " export complete: " + std::to_string(train.size()) +
                  " train, " + std::to_string(val.size()) + " val frames";
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepLabCut CSV export
// ═══════════════════════════════════════════════════════════════════════════
// DLC format: per-camera CollectedData CSV with multi-level header.
// Row format: frame_path, x1, y1, x2, y2, ...

inline bool export_deeplabcut(const ExportConfig &cfg, const AnnotationMap &amap,
                              std::string *status) {
    namespace fs = std::filesystem;
    auto labeled = get_labeled_frames(amap);
    if (labeled.empty()) {
        if (status) *status = "Error: No labeled frames found.";
        return false;
    }

    // Read image dims
    std::map<std::string, int> img_h;
    for (const auto &cam : cfg.camera_names) {
        std::string path = cfg.calibration_folder + "/" + cam + ".yaml";
        try {
            auto yaml = opencv_yaml::read(path);
            img_h[cam] = yaml.getInt("image_height");
        } catch (...) {
            if (status) *status = "Error: Cannot read calibration: " + path;
            return false;
        }
    }

    for (int ci = 0; ci < (int)cfg.camera_names.size(); ++ci) {
        const auto &cam = cfg.camera_names[ci];
        int h = img_h[cam];

        std::string dir = cfg.output_folder + "/" + cam;
        fs::create_directories(dir);

        std::ofstream f(dir + "/CollectedData.csv");

        // 3-row DLC header
        // Row 1: scorer
        f << "scorer";
        for (const auto &node : cfg.node_names) {
            (void)node;
            f << ",RED,RED";
        }
        f << "\n";

        // Row 2: bodyparts
        f << "bodyparts";
        for (const auto &node : cfg.node_names)
            f << "," << node << "," << node;
        f << "\n";

        // Row 3: coords
        f << "coords";
        for (size_t k = 0; k < cfg.node_names.size(); ++k)
            f << ",x,y";
        f << "\n";

        // Data rows
        for (u32 frame : labeled) {
            auto it = amap.find(frame);
            if (it == amap.end()) continue;
            if (it->second.instances.empty()) continue;
            const auto &inst = it->second.instances[0];
            if (ci >= (int)inst.cameras.size()) continue;
            const auto &c2d = inst.cameras[ci];

            f << "labeled-data/" << cam << "/Frame_" << frame << ".png";
            for (size_t k = 0; k < c2d.keypoints.size(); ++k) {
                if (c2d.kp_labeled[k]) {
                    double x = c2d.keypoints[k].x;
                    double y = h - c2d.keypoints[k].y; // Y-flip
                    f << "," << std::fixed << std::setprecision(2) << x << "," << y;
                } else {
                    f << ",,";
                }
            }
            f << "\n";
        }
    }

    // Write minimal DLC config.yaml
    {
        std::ofstream f(cfg.output_folder + "/config.yaml");
        f << "Task: " << cfg.skeleton_name << "\n";
        f << "scorer: RED\n";
        f << "bodyparts:\n";
        for (const auto &node : cfg.node_names)
            f << "- " << node << "\n";
        f << "skeleton:\n";
        for (const auto &[a, b] : cfg.edges)
            f << "- [" << cfg.node_names[a] << ", " << cfg.node_names[b] << "]\n";
        f << "numframes2pick: " << labeled.size() << "\n";
    }

    if (status)
        *status = "DeepLabCut export complete: " + std::to_string(labeled.size()) +
                  " frames x " + std::to_string(cfg.camera_names.size()) + " cameras";
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// JARVIS export — delegates to existing jarvis_export.h
// ═══════════════════════════════════════════════════════════════════════════
inline bool export_jarvis(const ExportConfig &cfg, std::string *status) {
    JarvisExport::ExportConfig jcfg;
    jcfg.label_folder       = cfg.label_folder;
    jcfg.calibration_folder = cfg.calibration_folder;
    jcfg.media_folder       = cfg.media_folder;
    jcfg.output_folder      = cfg.output_folder;
    jcfg.camera_names       = cfg.camera_names;
    jcfg.skeleton_name      = cfg.skeleton_name;
    jcfg.node_names         = cfg.node_names;
    jcfg.edges              = cfg.edges;
    jcfg.num_keypoints      = cfg.num_keypoints;
    jcfg.margin_pixel       = cfg.bbox_margin;
    jcfg.train_ratio        = cfg.train_ratio;
    jcfg.seed               = cfg.seed;
    jcfg.jpeg_quality       = cfg.jpeg_quality;
    return JarvisExport::export_jarvis_dataset(jcfg, status);
}

// ═══════════════════════════════════════════════════════════════════════════
// JARVIS-TR export — JARVIS + video_index.json for unlabeled frames
// ═══════════════════════════════════════════════════════════════════════════
inline bool export_jarvis_tr(const ExportConfig &cfg, std::string *status) {
    // First do standard JARVIS export
    if (!export_jarvis(cfg, status)) return false;

    // Then write video_index.json pointing to source videos
    namespace fs = std::filesystem;
    std::string output_dir = cfg.output_folder;
    // Find the timestamped subfolder (JARVIS creates one)
    std::string latest;
    for (auto &entry : fs::directory_iterator(output_dir)) {
        if (entry.is_directory()) {
            std::string name = entry.path().filename().string();
            if (name > latest) latest = name;
        }
    }
    if (latest.empty()) latest = output_dir;
    else latest = output_dir + "/" + latest;

    nlohmann::json vid_index;
    for (const auto &cam : cfg.camera_names) {
        vid_index[cam] = cfg.media_folder + "/" + cam + ".mp4";
    }

    std::ofstream f(latest + "/video_index.json");
    f << vid_index.dump(2);

    if (status) {
        std::string prev = *status;
        *status = prev + " (+ video_index.json for JARVIS-TR)";
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main dispatch
// ═══════════════════════════════════════════════════════════════════════════
inline bool export_dataset(Format fmt, const ExportConfig &cfg,
                           const AnnotationMap &amap, std::string *status) {
    namespace fs = std::filesystem;
    fs::create_directories(cfg.output_folder);

    switch (fmt) {
    case JARVIS:      return export_jarvis(cfg, status);
    case JARVIS_TR:   return export_jarvis_tr(cfg, status);
    case COCO:        return export_coco(cfg, amap, status);
    case YOLO_POSE:   return export_yolo(cfg, amap, true, status);
    case YOLO_DETECT: return export_yolo(cfg, amap, false, status);
    case DEEPLABCUT:  return export_deeplabcut(cfg, amap, status);
    default:
        if (status) *status = "Error: Unknown export format";
        return false;
    }
}

} // namespace ExportFormats
