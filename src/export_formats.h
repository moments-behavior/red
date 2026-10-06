#pragma once
#include "video_files.h"
// export_formats.h — Multi-format export dispatcher
//
// Single entry point for exporting annotations to various training frameworks.
// Each format reads from the same AnnotationMap. The JARVIS exporter
// (jarvis_export.h) is called through this dispatcher for JARVIS format.

#include "annotation.h"
#include "tailcycle_export.h"
#include "camera.h"
#include "ffmpeg_frame_reader.h"
#include "jarvis_export.h"
#include "json.hpp"
#include "opencv_yaml_io.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
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
    NERFSTUDIO,
    TAILCYCLE,
    FORMAT_COUNT
};

inline const char *format_name(Format f) {
    switch (f) {
    case JARVIS:      return "JARVIS";
    case TAILCYCLE:   return "tailcycle-dataset";
    case JARVIS_TR:   return "JARVIS (with video index)";
    case COCO:        return "COCO Keypoints";
    case DEEPLABCUT:  return "DeepLabCut";
    case YOLO_POSE:   return "YOLO Pose";
    case YOLO_DETECT: return "YOLO Detection";
    case NERFSTUDIO:  return "Nerfstudio / 3DGS";
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
    // Box classes by number (the Bbox tool's, saved with the labels). Empty:
    // one class, named after the skeleton.
    std::vector<std::string> class_names;
    std::vector<std::string> node_names;
    std::vector<std::pair<int, int>> edges;
    int num_keypoints = 0;

    // Camera params (loaded from project, indexed parallel to camera_names)
    std::vector<CameraParams> camera_params;

    // Telecentric (DLT) calibration. When true the calibration folder has
    // <cam>_dlt.csv (no <cam>.yaml); JARVIS export converts those to a
    // projectionMatrix <cam>.yaml and sources image dims from the video
    // (image_width/image_height below).
    bool telecentric = false;
    // Unit scaling factor for the JARVIS calibration — telecentric and
    // perspective alike (see JarvisExport::ExportConfig::scale_factor).
    int scale_factor = 1;

    // Per-camera image dimensions (parallel to camera_names). Supplied from the
    // loaded video so 2D / uncalibrated projects — which have no calibration
    // YAML to read dims from — can still export. Empty/<=0 => fall back to YAML.
    std::vector<int> image_width;
    std::vector<int> image_height;

    // Export options
    float bbox_margin = 50.0f;
    float train_ratio = 0.9f;
    int seed = 42;
    int jpeg_quality = 95;

    // Nerfstudio-specific: frame list (if empty, uses annotated frames)
    std::vector<int> nerfstudio_frames;

    // tailcycle-dataset. The split is a directory level rather than a field,
    // and the session id becomes the folder name -- so both are structural,
    // not metadata. n_frames and fps come from the media, not the annotation
    // range: the format validates every frame index against n_frames.
    std::string tailcycle_split = "train";      // train | val | test
    std::string tailcycle_session_id;
    int tailcycle_n_frames = 0;                 // frames in the MEDIA
    float tailcycle_fps = 0.0f;
    int tailcycle_frame_start = 0;              // inclusive
    int tailcycle_frame_end = 0;                // inclusive; 0 = to the end
    int tailcycle_layers = 0;   // 0 = 2D, 1 = 2D+3D, 2 = 3D only
    int tailcycle_window = 0;   // >0: only clips of this many frames around labels
};

// ── Per-camera image-size resolver ──
// Fills img_w/img_h keyed by camera name. Prefers dimensions supplied in the
// config (from the loaded video — used by 2D/uncalibrated projects); otherwise
// reads them from the calibration YAML (calibrated projects). Returns false and
// sets *status if a camera's size can't be determined either way.
inline bool resolve_image_dims(const ExportConfig &cfg,
                               std::map<std::string, int> &img_w,
                               std::map<std::string, int> &img_h,
                               std::string *status) {
    const bool have_cfg_dims =
        cfg.image_width.size() == cfg.camera_names.size() &&
        cfg.image_height.size() == cfg.camera_names.size();
    for (size_t ci = 0; ci < cfg.camera_names.size(); ++ci) {
        const auto &cam = cfg.camera_names[ci];
        if (have_cfg_dims && cfg.image_width[ci] > 0 &&
            cfg.image_height[ci] > 0) {
            img_w[cam] = cfg.image_width[ci];
            img_h[cam] = cfg.image_height[ci];
            continue;
        }
        std::string path = cfg.calibration_folder + "/" + cam + ".yaml";
        try {
            auto yaml = opencv_yaml::read(path);
            img_w[cam] = yaml.getInt("image_width");
            img_h[cam] = yaml.getInt("image_height");
        } catch (...) {
            if (status)
                *status = "Error: cannot determine image size for camera '" +
                          cam + "' (no video dims and no calibration YAML at " +
                          path + ")";
            return false;
        }
    }
    return true;
}

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

// ── Get annotated frames from AnnotationMap (any annotation type) ──
inline std::vector<u32> get_labeled_frames(const AnnotationMap &amap) {
    std::vector<u32> frames;
    for (const auto &[f, fis] : amap)
        if (any_instance_has_labels(fis)) frames.push_back(f);
    return frames;
}

// ── Get frames with keypoints only (for keypoint-only exporters) ──
inline std::vector<u32> get_keypoint_frames(const AnnotationMap &amap) {
    std::vector<u32> frames;
    for (const auto &[f, fis] : amap)
        if (any_instance_has_keypoints(fis)) frames.push_back(f);
    return frames;
}

// ── Shared image extraction for all exporters ──
// Extracts JPEG frames from video files, one thread per camera.
// Creates <output>/{train,val}/<cam>/Frame_<N>.jpg for each frame.
// Uses JarvisExport::extract_jpegs_for_camera (already thread-safe).
// trial_name: subfolder between split and camera (empty string = none).
inline bool extract_images(const ExportConfig &cfg,
                           const std::vector<u32> &train,
                           const std::vector<u32> &val,
                           const std::string &trial_name,
                           std::string *status,
                           std::atomic<int> *images_saved_counter = nullptr) {
    if (cfg.media_folder.empty()) return true; // no video → skip silently

    // Build frame→mode map
    std::map<int, std::string> frame_to_mode;
    std::vector<int> train_int, val_int;
    for (u32 f : train) { frame_to_mode[(int)f] = "train"; train_int.push_back((int)f); }
    for (u32 f : val)   { frame_to_mode[(int)f] = "val";   val_int.push_back((int)f); }

    std::mutex status_mutex;
    std::vector<std::thread> threads;

    for (const auto &cam : cfg.camera_names) {
        std::string video_path = camera_video_path(cfg.media_folder, cam);
        if (!std::filesystem::exists(video_path)) continue;
        threads.emplace_back(
            JarvisExport::extract_jpegs_for_camera,
            cam, trial_name, video_path, cfg.output_folder,
            train_int, val_int, frame_to_mode,
            status, &status_mutex, images_saved_counter, cfg.jpeg_quality);
    }
    for (auto &t : threads) t.join();

    if (status && status->find("Error") != std::string::npos)
        return false;
    return true;
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
        if (it->second.empty()) continue;
        const auto &fa = it->second.front();

        std::string filename = cam_name + "/Frame_" + std::to_string(frame) + ".jpg";
        nlohmann::json img;
        img["id"] = img_id;
        img["file_name"] = filename;
        img["width"] = img_w;
        img["height"] = img_h;
        images.push_back(img);

        if (cam_idx >= (int)fa.cameras.size()) { img_id++; continue; }
        const auto &cam = fa.cameras[cam_idx];

        // Count visible keypoints
        int num_visible = 0;
        for (size_t k = 0; k < cam.keypoints.size(); ++k)
            if (cam.keypoints[k].usable()) ++num_visible;

        // Skip frames with no keypoints
        if (num_visible == 0) { img_id++; continue; }

        // Build flat keypoints array [x,y,v, x,y,v, ...]
        nlohmann::json kp_flat = nlohmann::json::array();
        double x_min = 1e9, x_max = -1e9, y_min = 1e9, y_max = -1e9;
        for (size_t k = 0; k < cam.keypoints.size(); ++k) {
            if (cam.keypoints[k].usable()) {
                double x = cam.keypoints[k].x;
                double y = img_h - cam.keypoints[k].y; // ImPlot Y-flip
                kp_flat.push_back(x); kp_flat.push_back(y); kp_flat.push_back(2);
                x_min = std::min(x_min, x); x_max = std::max(x_max, x);
                y_min = std::min(y_min, y); y_max = std::max(y_max, y);
            } else {
                kp_flat.push_back(0); kp_flat.push_back(0); kp_flat.push_back(0);
            }
        }

        // Segmentation: always empty (COCO schema compatibility)
        nlohmann::json seg = nlohmann::json::array();

        // Bbox: from explicit bbox, or keypoint bounds + margin
        double bx, by, bw, bh;
        if (cam.has_bbox()) {
            bx = cam.extras->bbox_x; by = cam.extras->bbox_y;
            bw = cam.extras->bbox_w; bh = cam.extras->bbox_h;
        } else {
            bx = std::max(x_min - cfg.bbox_margin, 0.0);
            by = std::max(y_min - cfg.bbox_margin, 0.0);
            bw = std::min(x_max + cfg.bbox_margin, (double)img_w) - bx;
            bh = std::min(y_max + cfg.bbox_margin, (double)img_h) - by;
        }

        // Area from the bbox
        double area = bw * bh;

        nlohmann::json ann;
        ann["id"] = ann_id++;
        ann["image_id"] = img_id;
        ann["category_id"] = fa.category_id;
        ann["segmentation"] = seg;
        ann["bbox"] = {bx, by, bw, bh};
        ann["area"] = area;
        ann["iscrowd"] = 0;
        ann["keypoints"] = kp_flat;
        ann["num_keypoints"] = num_visible;
        annotations.push_back(ann);

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
                        std::string *status,
                        std::atomic<int> *img_counter = nullptr) {
    namespace fs = std::filesystem;
    auto labeled = get_labeled_frames(amap);
    if (labeled.empty()) {
        if (status) *status = "Error: No labeled frames found.";
        return false;
    }

    std::vector<u32> train, val;
    split_train_val(labeled, cfg.train_ratio, cfg.seed, train, val);

    // Per-camera image dimensions (video dims for 2D, else calibration YAML)
    std::map<std::string, int> img_w, img_h;
    if (!resolve_image_dims(cfg, img_w, img_h, status))
        return false;

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

    // Extract images from video (if media_folder available)
    if (!cfg.media_folder.empty()) {
        if (status) *status = "Extracting images...";
        if (!extract_images(cfg, train, val, "", status, img_counter))
            return false;
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
                        bool include_keypoints, std::string *status,
                        std::atomic<int> *img_counter = nullptr) {
    namespace fs = std::filesystem;
    // Frames with keypoints or a box -- a detection set can be boxes alone --
    // or an instance marked absent: an image with nothing to find, written
    // with an empty label file (a background image to YOLO).
    std::vector<u32> labeled;
    for (const auto &[f, fis] : amap) {
        bool any = any_instance_has_keypoints(fis);
        for (const auto &fa : fis)
            for (const auto &cam : fa.cameras)
                any = any || cam.has_bbox() || cam.is_absent();
        if (any) labeled.push_back(f);
    }
    if (labeled.empty()) {
        if (status) *status = "Error: No labeled frames found.";
        return false;
    }

    std::vector<u32> train, val;
    split_train_val(labeled, cfg.train_ratio, cfg.seed, train, val);

    // Per-camera image dimensions (video dims for 2D, else calibration YAML)
    std::map<std::string, int> img_w, img_h;
    if (!resolve_image_dims(cfg, img_w, img_h, status))
        return false;

    // A camera view with nothing labelled in it -- no box, no keypoints, no
    // absent mark -- gets neither a label file nor an image: to YOLO an image
    // without labels is a background ("nothing here"), which is only true
    // where someone said so. Their images are extracted with the frame's
    // others and removed after.
    std::vector<std::string> unlabelled_images;
    // Written views where an instance has neither a box (or keypoints) nor an
    // absent mark: YOLO reads the missing line as "nothing there". Exported
    // as they are; the status says how many, so they can be finished.
    int incomplete_views = 0;
    auto write_split = [&](const std::vector<u32> &frames, const std::string &split) {
        for (int ci = 0; ci < (int)cfg.camera_names.size(); ++ci) {
            const auto &cam_name = cfg.camera_names[ci];
            int w = img_w[cam_name], h = img_h[cam_name];

            std::string img_dir = cfg.output_folder + "/images/" + split + "/" + cam_name;
            std::string lbl_dir = cfg.output_folder + "/labels/" + split + "/" + cam_name;
            fs::create_directories(img_dir);
            fs::create_directories(lbl_dir);

            for (u32 frame : frames) {
                auto it = amap.find(frame);
                if (it == amap.end()) continue;
                if (it->second.empty()) continue;

                std::string fname = "Frame_" + std::to_string(frame);
                std::ostringstream lbl;
                bool absent_here = false;

                // One line per instance: every object in the image.
                for (const auto &fa : it->second) {
                    if (ci >= (int)fa.cameras.size()) continue;
                    const auto &c2d = fa.cameras[ci];
                    if (c2d.is_absent()) {   // not in this image: no line
                        absent_here = true;
                        continue;
                    }

                    // Compute bbox (normalized)
                    double bx, by, bw, bh;
                    if (c2d.has_bbox()) {
                        bx = c2d.extras->bbox_x; by = c2d.extras->bbox_y;
                        bw = c2d.extras->bbox_w; bh = c2d.extras->bbox_h;
                    } else {
                        // Derive from keypoints
                        double xmin = 1e9, xmax = -1e9, ymin = 1e9, ymax = -1e9;
                        bool any = false;
                        for (size_t k = 0; k < c2d.keypoints.size(); ++k) {
                            if (!c2d.keypoints[k].usable()) continue;
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

                    lbl << fa.category_id << " "
                        << std::fixed << std::setprecision(6)
                        << cx << " " << cy << " " << nw << " " << nh;

                    if (include_keypoints) {
                        for (size_t k = 0; k < c2d.keypoints.size(); ++k) {
                            if (c2d.keypoints[k].usable()) {
                                double kx = c2d.keypoints[k].x / w;
                                double ky = (h - c2d.keypoints[k].y) / h; // Y-flip
                                lbl << " " << kx << " " << ky << " 2";
                            } else {
                                lbl << " 0 0 0";
                            }
                        }
                    }
                    lbl << "\n";
                } // instances

                const std::string text = lbl.str();
                if (!text.empty() || absent_here) {
                    std::ofstream(lbl_dir + "/" + fname + ".txt") << text;
                    for (const auto &fa : it->second) {
                        if (ci >= (int)fa.cameras.size()) continue;
                        const auto &c = fa.cameras[ci];
                        bool kp = false;
                        for (const auto &k : c.keypoints) kp = kp || k.usable();
                        if (!c.has_bbox() && !c.is_absent() && !kp) {
                            ++incomplete_views;
                            break;
                        }
                    }
                }
                else
                    unlabelled_images.push_back(img_dir + "/" + fname + ".jpg");
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
        // One name per class number the labels use: the Bbox tool's classes,
        // or the skeleton's name for a project without any.
        std::vector<std::string> names = cfg.class_names;
        if (names.empty()) names.push_back(cfg.skeleton_name);
        for (const auto &[fnum, fis] : amap)
            for (const auto &fa : fis)
                while (fa.category_id >= (int)names.size())
                    names.push_back("class_" + std::to_string(names.size()));
        f << "nc: " << names.size() << "\n";
        f << "names: [";
        for (size_t i = 0; i < names.size(); ++i) {
            std::string n = names[i];
            for (size_t p = 0; (p = n.find('\'', p)) != std::string::npos; p += 2)
                n.insert(p, 1, '\'');   // YAML single-quote escape
            f << (i ? ", " : "") << "'" << n << "'";
        }
        f << "]\n";
        if (include_keypoints) {
            f << "kpt_shape: [" << cfg.num_keypoints << ", 3]\n";
        }
    }

    // Extract images from video (if media_folder available)
    if (!cfg.media_folder.empty()) {
        if (status) *status = "Extracting images...";
        // YOLO images go to images/{train,val}/<cam>/Frame_N.jpg
        // We need to use output_folder + "/images" as the extraction root
        ExportConfig img_cfg = cfg;
        img_cfg.output_folder = cfg.output_folder + "/images";
        if (!extract_images(img_cfg, train, val, "", status, img_counter))
            return false;
        std::error_code ec;
        for (const auto &p : unlabelled_images) fs::remove(p, ec);
    }

    std::string fmt = include_keypoints ? "YOLO Pose" : "YOLO Detection";
    if (status) {
        *status = fmt + " export complete: " + std::to_string(train.size()) +
                  " train, " + std::to_string(val.size()) + " val frames";
        if (incomplete_views > 0)
            *status += ". Note: " + std::to_string(incomplete_views) + " camera view" +
                       (incomplete_views == 1 ? " has" : "s have") +
                       " an instance with neither a box nor an absent mark; YOLO "
                       "will treat it as background there.";
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// DeepLabCut CSV export
// ═══════════════════════════════════════════════════════════════════════════
// DLC format: per-camera CollectedData CSV with multi-level header.
// Row format: frame_path, x1, y1, x2, y2, ...

inline bool export_deeplabcut(const ExportConfig &cfg, const AnnotationMap &amap,
                              std::string *status,
                              std::atomic<int> *img_counter = nullptr) {
    namespace fs = std::filesystem;
    auto labeled = get_keypoint_frames(amap);
    if (labeled.empty()) {
        if (status) *status = "Error: No labeled frames found.";
        return false;
    }

    // Per-camera image dimensions (video dims for 2D, else calibration YAML)
    std::map<std::string, int> img_w, img_h;
    if (!resolve_image_dims(cfg, img_w, img_h, status))
        return false;

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
            if (it->second.empty()) continue;
            const auto &fa = it->second.front();
            if (ci >= (int)fa.cameras.size()) continue;
            const auto &c2d = fa.cameras[ci];

            f << "labeled-data/" << cam << "/Frame_" << frame << ".jpg";
            for (size_t k = 0; k < c2d.keypoints.size(); ++k) {
                if (c2d.keypoints[k].usable()) {
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

    // Extract images from video (if media_folder available)
    // DLC puts all images in labeled-data/<cam>/Frame_N.jpg (no train/val split)
    if (!cfg.media_folder.empty()) {
        if (status) *status = "Extracting images...";
        // Use extract_images with all frames as "train" and empty trial_name.
        // Output root = <export>/labeled-data → images at labeled-data/train/<cam>/
        // Then rename train/ → ./ to get labeled-data/<cam>/
        // Simpler: just use the JARVIS helper directly with mode="labeled-data"
        std::vector<int> frame_ints;
        for (u32 f : labeled) frame_ints.push_back((int)f);
        std::map<int, std::string> frame_to_mode;
        for (int f : frame_ints) frame_to_mode[f] = "labeled-data";
        std::mutex smtx;
        std::vector<int> empty_int;
        std::vector<std::thread> threads;
        for (const auto &cam : cfg.camera_names) {
            std::string vid = camera_video_path(cfg.media_folder, cam);
            if (!std::filesystem::exists(vid)) continue;
            // path: <output>/labeled-data/<cam>/Frame_N.jpg (trial="" so no extra subdir)
            threads.emplace_back(
                JarvisExport::extract_jpegs_for_camera,
                cam, "", vid, cfg.output_folder,
                frame_ints, empty_int, frame_to_mode,
                status, &smtx, img_counter, cfg.jpeg_quality);
        }
        for (auto &t : threads) t.join();
    }

    if (status)
        *status = "DeepLabCut export complete: " + std::to_string(labeled.size()) +
                  " frames x " + std::to_string(cfg.camera_names.size()) + " cameras";
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// JARVIS export — delegates to existing jarvis_export.h
// ═══════════════════════════════════════════════════════════════════════════
inline bool export_jarvis(const ExportConfig &cfg, const AnnotationMap &amap,
                          std::string *status,
                          std::atomic<int> *img_counter = nullptr) {
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
    jcfg.telecentric        = cfg.telecentric;
    jcfg.scale_factor       = cfg.scale_factor;
    // Forward per-camera video dims as overrides so telecentric (no YAML) works
    // and calibrated projects can still fall back to YAML inside the exporter.
    for (size_t i = 0; i < cfg.camera_names.size(); ++i) {
        int w = (i < cfg.image_width.size()) ? cfg.image_width[i] : 0;
        int h = (i < cfg.image_height.size()) ? cfg.image_height[i] : 0;
        if (w > 0) jcfg.image_width_override[cfg.camera_names[i]] = w;
        if (h > 0) jcfg.image_height_override[cfg.camera_names[i]] = h;
    }
    return JarvisExport::export_jarvis_dataset(jcfg, amap, status, img_counter);
}

// ═══════════════════════════════════════════════════════════════════════════
// JARVIS-TR export — JARVIS + video_index.json for unlabeled frames
// ═══════════════════════════════════════════════════════════════════════════
inline bool export_jarvis_tr(const ExportConfig &cfg, const AnnotationMap &amap,
                             std::string *status,
                             std::atomic<int> *img_counter = nullptr) {
    // First do standard JARVIS export
    if (!export_jarvis(cfg, amap, status, img_counter)) return false;

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
        vid_index[cam] = camera_video_path(cfg.media_folder, cam);
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
// Nerfstudio / 3DGS export
// ═══════════════════════════════════════════════════════════════════════════
// Exports camera calibration as transforms.json and extracts JPEG frames
// for use with nerfstudio (splatfacto), 3D Gaussian Splatting, or similar
// novel-view-synthesis / 3D reconstruction tools.
//
// Output structure:
//   <output>/
//     transforms.json
//     images/
//       <Cam>_<Frame>.jpg
//
// Camera convention: nerfstudio expects camera-to-world matrices in OpenGL
// convention (Y-up, Z-back). RED stores world-to-camera in OpenCV convention
// (Y-down, Z-forward). The conversion is:
//   c2w = [R^T | -R^T t]   (invert w2c)
//   c2w[:3, 1:3] *= -1     (flip Y and Z columns: OpenCV → OpenGL)

inline void extract_jpegs_flat(
    const std::string &cam,
    const std::string &video_path,
    const std::string &output_dir,
    const std::vector<int> &frames,
    std::string *status, std::mutex *status_mutex,
    std::atomic<int> *images_saved_counter = nullptr,
    int jpeg_quality = 95) {

    namespace fs = std::filesystem;
    fs::create_directories(output_dir);

    std::vector<int> sorted_frames = frames;
    std::sort(sorted_frames.begin(), sorted_frames.end());

    ffmpeg_reader::FrameReader reader;
    // Use software decode for batch extraction — avoids VideoToolbox session
    // limits and transient reconfig errors when cold-starting multiple decoders.
    // HW decode is faster but only reliable when decoders are warmed up
    // (e.g., during interactive playback in the main app).
    if (!reader.open(video_path, false)) {
        if (status && status_mutex) {
            std::lock_guard<std::mutex> lock(*status_mutex);
            *status = "Error: Cannot open video: " + video_path;
        }
        return;
    }

    double fps = reader.fps();
    int w = reader.width();
    int h = reader.height();

    int pts_offset = JarvisExport::detect_negative_pts_offset(video_path, fps);

    for (int frame_num : sorted_frames) {
        int seek_frame = frame_num - pts_offset;
        if (seek_frame < 0) continue;

        const uint8_t *rgb = reader.readFrame(seek_frame);
        if (!rgb) continue;

        std::string filename = output_dir + "/" + cam + "_" +
                               std::to_string(frame_num) + ".jpg";
        JarvisExport::write_jpeg(filename.c_str(), w, h, 3, rgb, jpeg_quality);
        if (images_saved_counter)
            images_saved_counter->fetch_add(1, std::memory_order_relaxed);
    }
}

inline bool export_nerfstudio(const ExportConfig &cfg, const AnnotationMap &amap,
                               std::string *status,
                               std::atomic<int> *img_counter = nullptr) {
    namespace fs = std::filesystem;

    // Validate calibration
    if (cfg.camera_params.empty()) {
        if (status) *status = "Error: No camera calibration loaded.";
        return false;
    }
    if (cfg.camera_params.size() != cfg.camera_names.size()) {
        if (status) *status = "Error: Camera params / names size mismatch.";
        return false;
    }

    // Determine which frames to export
    std::vector<int> frames;
    if (!cfg.nerfstudio_frames.empty()) {
        frames = cfg.nerfstudio_frames;
    } else {
        auto labeled = get_labeled_frames(amap);
        for (u32 f : labeled) frames.push_back((int)f);
    }
    if (frames.empty()) {
        if (status) *status = "Error: No frames to export.";
        return false;
    }
    std::sort(frames.begin(), frames.end());

    // Read per-camera image dimensions from calibration
    std::map<std::string, std::pair<int, int>> cam_dims;
    for (const auto &cam : cfg.camera_names) {
        std::string path = cfg.calibration_folder + "/" + cam + ".yaml";
        try {
            auto yaml = opencv_yaml::read(path);
            cam_dims[cam] = {yaml.getInt("image_width"),
                             yaml.getInt("image_height")};
        } catch (...) {}
    }
    if (cam_dims.empty()) {
        if (status) *status = "Error: Cannot read image dimensions from calibration.";
        return false;
    }

    // Build transforms.json
    nlohmann::json transforms;
    transforms["camera_model"] = "OPENCV";
    nlohmann::json jframes = nlohmann::json::array();

    for (int frame_num : frames) {
        for (size_t ci = 0; ci < cfg.camera_names.size(); ++ci) {
            const auto &cam_name = cfg.camera_names[ci];
            const auto &cp = cfg.camera_params[ci];

            // Skip telecentric cameras (not supported by nerfstudio)
            if (cp.telecentric) continue;

            auto dim_it = cam_dims.find(cam_name);
            if (dim_it == cam_dims.end()) continue;
            int img_w = dim_it->second.first;
            int img_h = dim_it->second.second;

            // world-to-camera: R, t
            Eigen::Matrix3d R = cp.r;
            Eigen::Vector3d t = cp.tvec;

            // Invert to camera-to-world
            Eigen::Matrix3d R_c2w = R.transpose();
            Eigen::Vector3d t_c2w = -R.transpose() * t;

            // Build 4x4 c2w matrix
            Eigen::Matrix4d c2w = Eigen::Matrix4d::Identity();
            c2w.block<3, 3>(0, 0) = R_c2w;
            c2w.block<3, 1>(0, 3) = t_c2w;

            // OpenCV -> OpenGL: negate Y and Z columns
            c2w.block<3, 1>(0, 1) *= -1.0;
            c2w.block<3, 1>(0, 2) *= -1.0;

            // Serialize as row-major nested array
            nlohmann::json mat = nlohmann::json::array();
            for (int r = 0; r < 4; r++) {
                nlohmann::json row = nlohmann::json::array();
                for (int c = 0; c < 4; c++)
                    row.push_back(c2w(r, c));
                mat.push_back(row);
            }

            nlohmann::json entry;
            entry["file_path"] = "images/" + cam_name + "_" +
                                 std::to_string(frame_num) + ".jpg";
            entry["transform_matrix"] = mat;
            entry["fl_x"] = cp.k(0, 0);
            entry["fl_y"] = cp.k(1, 1);
            entry["cx"] = cp.k(0, 2);
            entry["cy"] = cp.k(1, 2);
            entry["w"] = img_w;
            entry["h"] = img_h;
            entry["k1"] = cp.dist_coeffs(0);
            entry["k2"] = cp.dist_coeffs(1);
            entry["p1"] = cp.dist_coeffs(2);
            entry["p2"] = cp.dist_coeffs(3);
            entry["k3"] = cp.dist_coeffs(4);

            jframes.push_back(entry);
        }
    }
    transforms["frames"] = jframes;

    // Write transforms.json
    fs::create_directories(cfg.output_folder);
    {
        std::ofstream f(cfg.output_folder + "/transforms.json");
        if (!f.is_open()) {
            if (status) *status = "Error: Cannot write transforms.json";
            return false;
        }
        f << transforms.dump(2);
    }

    // Extract frames in parallel (software decode, no VT session limit)
    if (!cfg.media_folder.empty()) {
        if (status) *status = "Extracting images...";
        std::string img_dir = cfg.output_folder + "/images";
        std::mutex status_mutex;

        // Collect camera/video pairs
        std::vector<std::pair<std::string, std::string>> cam_vids;
        for (const auto &cam : cfg.camera_names) {
            std::string video_path = camera_video_path(cfg.media_folder, cam);
            if (fs::exists(video_path))
                cam_vids.push_back({cam, video_path});
        }

        // Process all cameras concurrently (SW decode has no session limit)
        const size_t batch_size = cam_vids.size();
        for (size_t start = 0; start < cam_vids.size(); start += batch_size) {
            size_t end = std::min(start + batch_size, cam_vids.size());
            std::vector<std::thread> threads;
            for (size_t i = start; i < end; ++i) {
                threads.emplace_back(
                    extract_jpegs_flat,
                    cam_vids[i].first, cam_vids[i].second,
                    img_dir, frames,
                    status, &status_mutex, img_counter, cfg.jpeg_quality);
            }
            for (auto &t : threads) t.join();
        }

        if (status && status->find("Error") != std::string::npos)
            return false;
    }

    if (status)
        *status = "Nerfstudio export complete: " +
                  std::to_string(frames.size()) + " frames x " +
                  std::to_string(cfg.camera_names.size()) + " cameras -> " +
                  cfg.output_folder;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Main dispatch
// ═══════════════════════════════════════════════════════════════════════════
// Red frames in [start, end] carrying an assessed keypoint, a box or an absent
// mark (in camera
// `cam`, or any camera if cam < 0), or a 3D point unless the export is 2D only.
inline std::vector<int> tailcycle_labelled_frames(const AnnotationMap &amap, int cam,
                                                  int layers, int start, int end) {
    std::vector<int> out;
    for (const auto &[fnum, fis] : amap) {
        if ((int)fnum < start || (int)fnum > end) continue;
        bool any = false;
        for (const FrameAnnotation &fa : fis) {
            for (size_t ci = 0; ci < fa.cameras.size(); ci++) {
                if (cam >= 0 && (int)ci != cam) continue;
                const CameraAnnotation &c = fa.cameras[ci];
                any |= c.has_bbox() && c.extras->bbox_w > 0 && c.extras->bbox_h > 0;
                any |= c.is_absent();   // "not here" is a label too
                if (layers != 2)
                    for (const auto &kp : c.keypoints) any |= keypoint2d_assessed(kp);
            }
            if (layers != 0)
                for (const auto &k3 : fa.kp3d) any |= k3.exist;
        }
        if (any) out.push_back((int)fnum);
    }
    return out;
}

// Inclusive [first, last] clips of `window` frames centred on each labelled
// frame, shifted inward at the ends of [start, end] so they keep their length.
// Clips that overlap or touch merge.
inline std::vector<std::pair<int, int>> tailcycle_clips(const std::vector<int> &labelled,
                                                        int start, int end, int window) {
    std::vector<std::pair<int, int>> out;
    const int span = std::min(window, end - start + 1);
    for (int f : labelled) {
        const int a = std::max(start, std::min(f - (span - 1) / 2, end - span + 1));
        if (!out.empty() && a <= out.back().second + 1)
            out.back().second = std::max(out.back().second, a + span - 1);
        else
            out.push_back({a, a + span - 1});
    }
    return out;
}

// Images an export of red frames [start, end] extracts (for the progress bar).
inline long long tailcycle_image_estimate(const AnnotationMap &amap, int n_cams, int layers,
                                          int start, int end, int window) {
    long long n = 0;
    for (int ci = 0; ci < n_cams; ci++) {
        const auto labelled =
            tailcycle_labelled_frames(amap, layers == 0 ? ci : -1, layers, start, end);
        if (labelled.empty()) continue;
        if (window <= 0) n += end - start + 1;
        else for (const auto &[a, b] : tailcycle_clips(labelled, start, end, window)) n += b - a + 1;
    }
    return n;
}

// ── tailcycle-dataset ────────────────────────────────────────────────────────
// Adapts the shared ExportConfig onto TailcycleExport's own, which stays
// separate because it carries structural fields (split, group, frame rebasing)
// that no other exporter has.
//
// One call writes ONE session, covering one frame range. Several splits means
// several calls -- the format makes split a directory level so a session
// belongs wholly to one, which is what stops a shuffled train/val split from
// putting near-identical adjacent frames on both sides (rule 14).
//
// A 2D export writes one session per video with labels (a 2D session is one
// camera, rule 5), named after the video.
inline bool export_tailcycle(const ExportConfig &cfg, const AnnotationMap &amap,
                             std::string *status,
                             std::atomic<int> *img_counter = nullptr,
                             std::atomic<bool> *cancel = nullptr) {
    namespace fs = std::filesystem;
    if (!TailcycleExport::available()) {
        if (status)
            *status = "Error: this build has no Parquet support (Arrow was not "
                      "found at configure time).";
        return false;
    }
    if (cfg.tailcycle_layers == 0 && cfg.camera_names.size() > 1) {
        const int last = cfg.tailcycle_frame_end > 0 ? cfg.tailcycle_frame_end : INT_MAX;
        int written = 0;
        std::string skipped;
        for (size_t i = 0; i < cfg.camera_names.size(); i++) {
            if (tailcycle_labelled_frames(amap, (int)i, 0, cfg.tailcycle_frame_start, last).empty()) {
                skipped += " " + cfg.camera_names[i];
                continue;
            }
            ffmpeg_reader::FrameReader reader;
            if (!reader.open(camera_video_path(cfg.media_folder, cfg.camera_names[i]))) {
                if (status) *status = "Error: cannot open the video for camera " + cfg.camera_names[i];
                return false;
            }
            ExportConfig one = cfg;
            one.camera_names = {cfg.camera_names[i]};
            one.camera_params = {i < cfg.camera_params.size() ? cfg.camera_params[i] : CameraParams{}};
            one.image_width = {reader.width()};
            one.image_height = {reader.height()};
            one.tailcycle_n_frames = reader.frameCount();
            one.tailcycle_fps = (float)reader.fps();
            AnnotationMap one_amap = amap;
            for (auto &[f, fis] : one_amap)
                for (FrameAnnotation &fa : fis) {
                    CameraAnnotation cam = i < fa.cameras.size() ? fa.cameras[i] : CameraAnnotation{};
                    fa.cameras.clear();
                    fa.cameras.push_back(std::move(cam));
                }
            if (!export_tailcycle(one, one_amap, status, img_counter, cancel)) return false;
            written++;
        }
        if (written == 0) {
            if (status) *status = "Error: no video has labelled points or boxes in the frame range.";
            return false;
        }
        if (status)
            *status = "Wrote " + std::to_string(written) + " 2D session(s)" +
                      (skipped.empty() ? "" : "; skipped videos with no labels:" + skipped);
        return true;
    }
    if (cfg.tailcycle_n_frames <= 0) {
        if (status) *status = "Error: no media loaded, so the group length is unknown.";
        return false;
    }

    const int total = cfg.tailcycle_n_frames;
    const int start = std::max(0, cfg.tailcycle_frame_start);
    const int end = cfg.tailcycle_frame_end > 0
                        ? std::min(cfg.tailcycle_frame_end, total - 1)
                        : total - 1;
    if (end < start) {
        if (status) *status = "Error: frame range ends before it starts.";
        return false;
    }
    const int n = end - start + 1;

    TailcycleExport::ExportConfig tc;
    tc.output_folder = cfg.output_folder;
    tc.split = cfg.tailcycle_split;
    // The session id becomes a single folder name, so it cannot carry
    // separators -- a path-like project name would otherwise nest the session
    // several directories deep and break the <split>/<session>/ layout.
    tc.session_id = cfg.tailcycle_session_id.empty() ? cfg.skeleton_name
                                                     : cfg.tailcycle_session_id;
    const bool one_video = cfg.tailcycle_layers == 0 && cfg.camera_names.size() == 1;
    if (one_video) tc.session_id = cfg.camera_names[0];
    for (char &c : tc.session_id)
        if (c == '/' || c == '\\') c = '_';
    tc.camera_names = cfg.camera_names;
    tc.calibration = cfg.camera_params;
    tc.node_names = cfg.node_names;
    tc.edges = cfg.edges;
    tc.n_frames = n;
    tc.fps = cfg.tailcycle_fps;
    tc.source_frame_start = start;
    tc.layers = (TailcycleExport::ExportConfig::Layers)cfg.tailcycle_layers;
    tc.provenance_source = cfg.label_folder;

    // filename() returns empty when the path ends in a separator, which would
    // leave the group id as a bare "_ix<start>".
    std::string stem;
    if (one_video) {
        stem = cfg.camera_names[0];
    } else if (!cfg.media_folder.empty()) {
        fs::path mp(cfg.media_folder);
        if (mp.filename().empty()) mp = mp.parent_path();
        stem = mp.filename().string();
    }
    if (stem.empty()) stem = tc.session_id;
    tc.source_video = stem;
    // Encodes the offset the way johnson-mouse-tracked does: <recording>_ix<start>.
    // Two ranges of one recording then have distinct group ids.
    tc.group_id = stem + "_ix" + std::to_string(start);

    std::vector<std::pair<int, int>> clips{{start, end}};
    if (cfg.tailcycle_window > 0) {
        clips = tailcycle_clips(tailcycle_labelled_frames(amap, -1, cfg.tailcycle_layers, start, end),
                                start, end, cfg.tailcycle_window);
        if (clips.empty()) {
            if (status) *status = "Error: nothing labelled in the frame range.";
            return false;
        }
        for (const auto &[a, b] : clips)
            tc.groups.push_back({stem + "_ix" + std::to_string(a), b - a + 1, a});
    }

    for (size_t i = 0; i < tc.calibration.size(); i++) {
        if (i < cfg.image_width.size() && cfg.image_width[i] > 0)
            tc.calibration[i].image_width = cfg.image_width[i];
        if (i < cfg.image_height.size() && cfg.image_height[i] > 0)
            tc.calibration[i].image_height = cfg.image_height[i];
    }
    // A consumer reads group frame f as the f-th frame of the media in the
    // group folder -- source_frame_start is provenance, not an indexing
    // instruction. So only a whole-recording group may link the video; a
    // sub-range must carry exactly its own frames, or every index is off by
    // `start`. That is why johnson-mouse-tracked ships extracted JPEGs.
    TailcycleExport::ExportStats st;
    if (!TailcycleExport::export_session(tc, amap, &st, status)) return false;

    // A project with both hand-placed and predicted labels writes <id>_annotated
    // and <id>_tracked; both need the pixels.
    for (const std::string &sdir : st.sessions) {
        for (size_t i = 0; i < cfg.camera_names.size(); i++) {
            const std::string vpath = camera_video_path(cfg.media_folder, cfg.camera_names[i]);
            if (!fs::exists(vpath)) {
                if (status) *status = "Error: no video for camera " + cfg.camera_names[i];
                return false;
            }
            ffmpeg_reader::FrameReader reader;
            if (!reader.open(vpath)) {
                if (status)
                    *status = "Error: cannot open " + vpath + " to extract frames.";
                return false;
            }
            for (size_t g = 0; g < clips.size(); g++) {
                const auto [a, b] = clips[g];
                const fs::path cdir = fs::path(sdir) / "groups" /
                                      (tc.groups.empty() ? tc.group_id : tc.groups[g].id) /
                                      cfg.camera_names[i];
                fs::create_directories(cdir);
                for (int f = a; f <= b; f++) {
                    // Cancelling leaves a group with fewer frames than groups.pq
                    // declares, which is worse than no session at all -- so remove
                    // what was written rather than leaving something that looks
                    // complete.
                    if (cancel && cancel->load(std::memory_order_relaxed)) {
                        std::error_code ec;
                        for (const std::string &d : st.sessions) fs::remove_all(d, ec);
                        if (status) *status = "Export cancelled; partial session removed.";
                        return false;
                    }
                    const uint8_t *rgb = reader.readFrame(f);
                    if (!rgb) {
                        if (status)
                            *status = "Error: frame " + std::to_string(f) + " of " +
                                      cfg.camera_names[i] + " could not be decoded.";
                        return false;
                    }
                    char name[32];
                    snprintf(name, sizeof(name), "%06d.jpg", f - a);
                    if (!JarvisExport::write_jpeg((cdir / name).string().c_str(),
                                                  reader.width(), reader.height(), 3,
                                                  rgb, cfg.jpeg_quality)) {
                        if (status) *status = "Error: could not write " + (cdir / name).string();
                        return false;
                    }
                    if (img_counter) img_counter->fetch_add(1);
                }
            }
        }
    }

    if (status) {
        int frames = 0;
        for (const auto &[a, b] : clips) frames += b - a + 1;
        *status = "Wrote " + tc.split + "/" + tc.session_id + " (" +
                  std::to_string(frames) + " frames, " + std::to_string(st.keypoint_rows) +
                  " 2D rows, " + std::to_string(st.points3d_rows) + " 3D rows)" +
                  ", frames extracted";
    }
    return true;
}

inline bool export_dataset(Format fmt, const ExportConfig &cfg,
                           const AnnotationMap &amap, std::string *status,
                           std::atomic<int> *img_counter = nullptr,
                           std::atomic<bool> *cancel = nullptr) {
    namespace fs = std::filesystem;
    fs::create_directories(cfg.output_folder);

    switch (fmt) {
    case JARVIS:      return export_jarvis(cfg, amap, status, img_counter);
    case JARVIS_TR:   return export_jarvis_tr(cfg, amap, status, img_counter);
    case COCO:        return export_coco(cfg, amap, status, img_counter);
    case YOLO_POSE:   return export_yolo(cfg, amap, true, status, img_counter);
    case YOLO_DETECT: return export_yolo(cfg, amap, false, status, img_counter);
    case DEEPLABCUT:  return export_deeplabcut(cfg, amap, status, img_counter);
    case NERFSTUDIO:  return export_nerfstudio(cfg, amap, status, img_counter);
    case TAILCYCLE:   return export_tailcycle(cfg, amap, status, img_counter, cancel);
    default:
        if (status) *status = "Error: Unknown export format";
        return false;
    }
}

} // namespace ExportFormats
