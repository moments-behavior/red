#pragma once
// jarvis_merge.h — Group JARVIS export: merge many JARVIS datasets into one
// master dataset.
//
// A JARVIS dataset JSON is already multi-trial: `calibrations` and `framesets`
// are keyed by trial/dataset name, and image file_names are namespaced by trial
// ("<trial>/<cam>/Frame_<N>.jpg"). So merging N datasets is just: concatenate
// `images`/`annotations` (re-indexing ids), union `calibrations`/`framesets`,
// re-split train/val globally, and copy/produce the image + calib files.
//
// Two source kinds (see SourceInfo::Kind):
//   Project — a live RED project (.redproj). We build its instances JSON in
//             memory with JarvisExport::generate_annotation_json_from_amap and
//             decode images from the project's videos.
//   Dataset — an already-exported JARVIS dataset folder (annotations/,
//             train/, val/, calib_params/). We parse its instances JSONs and
//             copy its JPEGs / calibration YAMLs directly.
//
// Requirements (per feature decisions):
//   * All sources must share an identical `keypoint_names` list (same order —
//     keypoints are positional everywhere in RED). Mismatches are rejected.
//   * Train/val is re-split globally across all pooled framesets.
//   * Trial-name collisions across sources are auto-suffixed (__2, __3, ...).

#include "annotation.h"
#include "annotation_csv.h"
#include "jarvis_export.h"   // ExportConfig + JSON/calib/JPEG helpers (reused)
#include "json.hpp"
#include "opencv_yaml_io.h"
#include "project.h"         // ProjectManager, load_project_manager_json, reload_skeleton
#include "skeleton.h"

#include <algorithm>
#include <atomic>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace JarvisMerge {

// ---------------------------------------------------------------------------
// Public config + source description
// ---------------------------------------------------------------------------
struct MergeConfig {
    std::string output_folder;   // master output root; a timestamped subfolder is created inside
    float train_ratio = 0.9f;
    int seed = 42;
    float margin_pixel = 50.0f;  // used only for Project sources (dataset bboxes are kept as-is)
    int jpeg_quality = 95;       // used only for Project sources (dataset JPEGs are copied as-is)
    bool scale_10x = false;      // telecentric fly x10; applied to Project-source projectionMatrix
};

struct SourceInfo {
    enum Kind { Project, Dataset };
    Kind kind = Project;
    std::string path;                       // .redproj path (Project) or dataset folder (Dataset)
    std::string display_name;

    // Populated by scan_*; used both for the UI and the merge.
    std::vector<std::string> keypoint_names;
    std::vector<std::string> trials;        // trial/dataset names contained
    int frame_count = 0;                    // valid framesets (multi-view frames)
    int image_count = 0;                    // total per-camera images (for progress total)
    bool valid = false;
    std::string message;                    // human-readable status / error

    // Project-source resolved details (unused for Dataset sources).
    std::vector<std::string> camera_names;
    std::string calibration_folder;
    std::string media_folder;
    std::string label_folder;
    std::string skeleton_name;
    int num_nodes = 0;
    std::vector<std::pair<int, int>> edges;
    bool telecentric = false;               // DLT calibration (no per-cam YAML)
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

// Resolve a project's calibration folder, honoring the timestamped-subfolder
// convention used by setup_project(): if <folder>/<cam>.yaml is missing but a
// single newest YYYY_MM_DD_* subdir exists, use that instead.
inline std::string resolve_calib_folder(const std::string &folder,
                                        const std::vector<std::string> &cams) {
    namespace fs = std::filesystem;
    if (folder.empty() || !fs::is_directory(folder)) return folder;
    if (!cams.empty() &&
        (fs::exists(fs::path(folder) / (cams.front() + ".yaml")) ||
         fs::exists(fs::path(folder) / (cams.front() + "_dlt.csv"))))
        return folder;
    std::string newest;
    for (const auto &e : fs::directory_iterator(folder)) {
        if (!e.is_directory()) continue;
        std::string n = e.path().filename().string();
        if (n.size() >= 10 && n[4] == '_' && n[7] == '_' && n > newest)
            newest = n;
    }
    if (!newest.empty()) return (fs::path(folder) / newest).string();
    return folder;
}

// Parse a JARVIS image file_name "<trial>/<cam>/Frame_<N>.jpg" into its parts.
inline bool parse_file_name(const std::string &fn, std::string &trial,
                            std::string &cam, int &frame) {
    auto s1 = fn.find('/');
    if (s1 == std::string::npos) return false;
    auto s2 = fn.find('/', s1 + 1);
    if (s2 == std::string::npos) return false;
    trial = fn.substr(0, s1);
    cam = fn.substr(s1 + 1, s2 - s1 - 1);
    std::string leaf = fn.substr(s2 + 1); // Frame_<N>.jpg
    auto us = leaf.rfind('_');
    auto dot = leaf.rfind('.');
    if (us == std::string::npos || dot == std::string::npos || dot <= us + 1)
        return false;
    try { frame = std::stoi(leaf.substr(us + 1, dot - us - 1)); }
    catch (...) { return false; }
    return true;
}

// ---------------------------------------------------------------------------
// Scanning — fast validation to populate the UI. No image work.
// ---------------------------------------------------------------------------

inline SourceInfo scan_project(const std::string &redproj_path,
                               const std::map<std::string, SkeletonPrimitive> &skeleton_map) {
    namespace fs = std::filesystem;
    SourceInfo s;
    s.kind = SourceInfo::Project;
    s.path = redproj_path;
    s.display_name = fs::path(redproj_path).stem().string();

    ProjectManager pm;
    std::string err;
    if (!load_project_manager_json(&pm, redproj_path, &err)) {
        s.message = "Cannot read project: " + err;
        return s;
    }
    // Gate on the *persisted* annotation_2d flag, NOT project_is_2d(): the latter
    // also treats an empty camera_params as 2D, but camera_params is rebuilt by
    // setup_project() at load and is never serialized — load_project_manager_json()
    // leaves it empty, so project_is_2d() would reject every calibrated project.
    // Real calibration presence is enforced by the camera-count + calibration-file
    // checks below.
    if (pm.annotation_2d) {
        s.message = "2D annotation projects are not supported (JARVIS needs "
                    "multi-view calibration).";
        return s;
    }
    if (pm.camera_names.size() < 2) {
        s.message = "Project has fewer than 2 cameras.";
        return s;
    }
    s.telecentric = pm.telecentric;

    // Rebuild the skeleton to read its node names.
    SkeletonContext skel;
    if (!reload_skeleton(pm, skel, skeleton_map, &err)) {
        s.message = "Cannot load skeleton: " + err;
        return s;
    }
    s.keypoint_names = skel.node_names;
    s.skeleton_name = skel.name;
    s.num_nodes = skel.num_nodes;
    for (const auto &e : skel.edges) s.edges.push_back({e.x, e.y});

    s.camera_names = pm.camera_names;
    s.media_folder = pm.media_folder;
    s.calibration_folder = resolve_calib_folder(pm.calibration_folder, pm.camera_names);

    // Validate calibration files exist for every camera (DLT csv when telecentric,
    // perspective YAML otherwise).
    for (const auto &cam : pm.camera_names) {
        std::string need = s.calibration_folder + "/" + cam +
                           (s.telecentric ? "_dlt.csv" : ".yaml");
        if (!std::filesystem::exists(need)) {
            s.message = "Missing calibration file: " + need;
            return s;
        }
    }

    // Most-recent label folder.
    std::string labels;
    if (AnnotationCSV::find_most_recent_labels(pm.keypoints_root_folder, labels, err) != 0) {
        s.message = "No labeled data found: " + err;
        return s;
    }
    s.label_folder = labels;

    // Load annotations to count fully-triangulated frames.
    AnnotationMap amap;
    if (AnnotationCSV::load_all(labels, amap, skel.name, skel.num_nodes,
                                (int)pm.camera_names.size(), pm.camera_names, err) != 0) {
        s.message = "Cannot load annotations: " + err;
        return s;
    }
    int valid = 0;
    for (const auto &[fid, fa] : amap)
        if (frame_is_fully_triangulated(fa, skel.num_nodes)) ++valid;
    if (valid == 0) {
        s.message = "No fully-triangulated frames.";
        return s;
    }

    // Trial name = project name (fallback to label-folder name).
    std::string trial = pm.project_name.empty()
                            ? fs::path(labels).filename().string()
                            : pm.project_name;
    s.trials = {trial};
    s.frame_count = valid;
    s.image_count = valid * (int)pm.camera_names.size();
    s.valid = true;
    s.message = std::to_string(valid) + " frames x " +
                std::to_string(pm.camera_names.size()) + " cams";
    return s;
}

inline SourceInfo scan_dataset(const std::string &dataset_folder) {
    namespace fs = std::filesystem;
    SourceInfo s;
    s.kind = SourceInfo::Dataset;
    s.path = dataset_folder;
    s.display_name = fs::path(dataset_folder).filename().string();

    std::string train_path = dataset_folder + "/annotations/instances_train.json";
    if (!fs::exists(train_path)) {
        s.message = "Not a JARVIS dataset (missing annotations/instances_train.json).";
        return s;
    }
    try {
        std::set<std::string> trials;
        int framesets = 0, images = 0;
        for (const char *mode : {"train", "val"}) {
            std::string p = dataset_folder + "/annotations/instances_" + mode + ".json";
            if (!fs::exists(p)) continue;
            std::ifstream f(p);
            nlohmann::json j;
            f >> j;
            if (mode == std::string("train") && j.contains("keypoint_names"))
                s.keypoint_names = j["keypoint_names"].get<std::vector<std::string>>();
            if (j.contains("framesets")) framesets += (int)j["framesets"].size();
            if (j.contains("images")) images += (int)j["images"].size();
            if (j.contains("calibrations"))
                for (auto it = j["calibrations"].begin(); it != j["calibrations"].end(); ++it)
                    trials.insert(it.key());
        }
        if (s.keypoint_names.empty()) {
            s.message = "Dataset has no keypoint_names.";
            return s;
        }
        s.trials.assign(trials.begin(), trials.end());
        s.frame_count = framesets;
        s.image_count = images;
        s.valid = true;
        s.message = std::to_string(framesets) + " framesets, " +
                    std::to_string(images) + " images";
    } catch (const std::exception &e) {
        s.message = std::string("Failed to parse dataset JSON: ") + e.what();
    }
    return s;
}

// ---------------------------------------------------------------------------
// Merge internals
// ---------------------------------------------------------------------------
namespace detail {

// One per-camera image plus its annotations, carried through the merge.
struct ImgRec {
    int src_index = 0;
    nlohmann::json image;                  // COCO image entry (id + file_name already rewritten)
    std::vector<nlohmann::json> anns;      // COCO annotation entries (ids already rewritten)
    std::string trial;                     // renamed trial
    std::string cam;
    int frame = 0;
    // For Dataset sources: where the source JPEG lives.
    std::string orig_trial;
    std::string orig_mode;                 // "train" / "val"
};

// A COCO chunk (one instances JSON) tagged with its source + orig mode.
struct Chunk {
    nlohmann::json json;
    int src_index;
    std::string orig_mode; // for Dataset sources; empty for Project
};

} // namespace detail

// Build the in-memory instances JSON for a Project source (all valid frames in
// one trial). Returns false with *err set on failure. Also fills image dims.
inline bool build_project_json(const SourceInfo &src, float margin_pixel,
                               const std::string &trial_name,
                               nlohmann::json &out_json,
                               std::map<std::string, int> &img_w,
                               std::map<std::string, int> &img_h,
                               std::string *err) {
    // Resolve image dims. Perspective projects read them from the calibration
    // YAML; telecentric projects have no YAML, so open each camera video.
    for (const auto &cam : src.camera_names) {
        if (src.telecentric) {
            ffmpeg_reader::FrameReader reader;
            if (!reader.open(src.media_folder + "/" + cam + ".mp4")) {
                if (err) *err = "Cannot open video for dims: " + cam + ".mp4";
                return false;
            }
            img_w[cam] = reader.width();
            img_h[cam] = reader.height();
        } else {
            try {
                auto yaml = opencv_yaml::read(src.calibration_folder + "/" + cam + ".yaml");
                img_w[cam] = yaml.getInt("image_width");
                img_h[cam] = yaml.getInt("image_height");
            } catch (const std::exception &e) {
                if (err) *err = "Cannot read calibration for " + cam + ": " + e.what();
                return false;
            }
        }
    }

    AnnotationMap amap;
    std::string load_err;
    if (AnnotationCSV::load_all(src.label_folder, amap, src.skeleton_name,
                                src.num_nodes, (int)src.camera_names.size(),
                                src.camera_names, load_err) != 0) {
        if (err) *err = "Cannot load annotations: " + load_err;
        return false;
    }

    std::vector<int> frames;
    for (const auto &[fid, fa] : amap)
        if (frame_is_fully_triangulated(fa, src.num_nodes)) frames.push_back((int)fid);
    std::sort(frames.begin(), frames.end());

    JarvisExport::ExportConfig cfg;
    cfg.camera_names = src.camera_names;
    cfg.skeleton_name = src.skeleton_name;
    cfg.node_names = src.keypoint_names;
    cfg.edges = src.edges;
    cfg.num_keypoints = src.num_nodes;
    cfg.margin_pixel = margin_pixel;
    cfg.telecentric = src.telecentric; // makes calibrations reference <cam>.yaml

    out_json = JarvisExport::generate_annotation_json_from_amap(
        trial_name, frames, amap, cfg, img_w, img_h, nullptr, nullptr);
    return true;
}

// Main merge. Runs on a worker thread. `status` is a thread-local string.
inline bool merge_datasets(const MergeConfig &cfg_in,
                           const std::vector<SourceInfo> &sources,
                           std::string *status,
                           std::atomic<int> *images_saved = nullptr) {
    namespace fs = std::filesystem;
    using nlohmann::json;

    std::vector<const SourceInfo *> srcs;
    for (const auto &s : sources)
        if (s.valid) srcs.push_back(&s);
    if (srcs.empty()) {
        if (status) *status = "Error: no valid sources to merge";
        return false;
    }

    // 1. Verify identical keypoint_names across all sources.
    const auto &ref_kp = srcs.front()->keypoint_names;
    for (const auto *s : srcs) {
        if (s->keypoint_names != ref_kp) {
            if (status)
                *status = "Error: skeleton mismatch in '" + s->display_name +
                          "' (keypoint names differ from the first source)";
            return false;
        }
    }

    // 2. Resolve trial-name collisions -> per-source rename maps.
    std::set<std::string> used;
    std::vector<std::map<std::string, std::string>> rename(srcs.size());
    for (size_t i = 0; i < srcs.size(); ++i) {
        for (const auto &t : srcs[i]->trials) {
            std::string name = t;
            int n = 2;
            while (used.count(name)) name = t + "__" + std::to_string(n++);
            used.insert(name);
            rename[i][t] = name;
        }
    }

    // 3. Timestamped master output folder.
    std::string out;
    {
        time_t now = time(0);
        struct tm ts = *localtime(&now);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y_%m_%d_%H_%M_%S", &ts);
        out = cfg_in.output_folder + "/" + buf;
    }
    std::error_code ec;
    fs::create_directories(out + "/annotations", ec);
    if (ec) {
        if (status) *status = "Error: cannot create output folder: " + ec.message();
        return false;
    }

    if (status) *status = "Building annotations...";

    // 4. Collect COCO chunks from every source.
    //    Project -> one generated JSON. Dataset -> its train + val JSONs.
    std::vector<detail::Chunk> chunks;
    // Project sources also need image dims for the calib writer later; capture per source.
    std::map<int, std::string> project_trial; // src_index -> renamed trial
    for (size_t i = 0; i < srcs.size(); ++i) {
        const SourceInfo &s = *srcs[i];
        if (s.kind == SourceInfo::Project) {
            std::string trial = rename[i].begin()->second; // single trial
            project_trial[(int)i] = trial;
            json pj;
            std::map<std::string, int> w, h;
            std::string err;
            if (!build_project_json(s, cfg_in.margin_pixel, trial, pj, w, h, &err)) {
                if (status) *status = "Error: " + s.display_name + ": " + err;
                return false;
            }
            chunks.push_back({std::move(pj), (int)i, ""});
        } else {
            for (const char *mode : {"train", "val"}) {
                std::string p = s.path + "/annotations/instances_" + mode + ".json";
                if (!fs::exists(p)) continue;
                std::ifstream f(p);
                json dj;
                try { f >> dj; }
                catch (const std::exception &e) {
                    if (status) *status = "Error: parse " + p + ": " + e.what();
                    return false;
                }
                chunks.push_back({std::move(dj), (int)i, mode});
            }
        }
    }

    // 5. Merge chunks: re-index ids, rename trials, union calibrations.
    json merged_kp = ref_kp;
    json merged_skeleton = chunks.front().json.value("skeleton", json::array());
    json merged_categories = chunks.front().json.value("categories", json::array());
    json calibrations = json::object();

    std::vector<detail::ImgRec> recs;
    int next_img = 0, next_ann = 0;

    for (auto &ch : chunks) {
        const auto &rmap = rename[ch.src_index];
        std::map<int, size_t> id_to_rec; // old image id -> index into recs

        if (ch.json.contains("images")) {
            for (auto &img : ch.json["images"]) {
                std::string trial, cam;
                int frame = 0;
                std::string fn = img.value("file_name", "");
                if (!parse_file_name(fn, trial, cam, frame)) continue;
                std::string new_trial = trial;
                auto rit = rmap.find(trial);
                if (rit != rmap.end()) new_trial = rit->second;

                int old_id = img.value("id", -1);
                int new_id = next_img++;

                detail::ImgRec rec;
                rec.src_index = ch.src_index;
                rec.trial = new_trial;
                rec.cam = cam;
                rec.frame = frame;
                rec.orig_trial = trial;
                rec.orig_mode = ch.orig_mode;

                img["id"] = new_id;
                img["file_name"] = new_trial + "/" + cam + "/Frame_" +
                                   std::to_string(frame) + ".jpg";
                rec.image = img;

                id_to_rec[old_id] = recs.size();
                recs.push_back(std::move(rec));
            }
        }
        if (ch.json.contains("annotations")) {
            for (auto &ann : ch.json["annotations"]) {
                int old_img = ann.value("image_id", -1);
                auto it = id_to_rec.find(old_img);
                if (it == id_to_rec.end()) continue;
                ann["image_id"] = recs[it->second].image["id"];
                ann["id"] = next_ann++;
                recs[it->second].anns.push_back(ann);
            }
        }
        if (ch.json.contains("calibrations")) {
            for (auto it = ch.json["calibrations"].begin();
                 it != ch.json["calibrations"].end(); ++it) {
                std::string new_trial = it.key();
                auto rit = rmap.find(it.key());
                if (rit != rmap.end()) new_trial = rit->second;
                json cams = json::object();
                for (auto cit = it.value().begin(); cit != it.value().end(); ++cit) {
                    // Preserve the source filename (e.g. <cam>.yaml OR <cam>_dlt.csv);
                    // only swap the trial directory. Forcing ".yaml" here would break
                    // telecentric datasets whose calibrations point at _dlt.csv.
                    std::string base =
                        std::filesystem::path(cit.value().get<std::string>()).filename().string();
                    if (base.empty()) base = cit.key() + ".yaml";
                    cams[cit.key()] = "calib_params/" + new_trial + "/" + base;
                }
                calibrations[new_trial] = cams;
            }
        }
    }

    if (recs.empty()) {
        if (status) *status = "Error: no images found across sources";
        return false;
    }

    // 6. Global re-split at frameset granularity (trial|frame).
    std::map<std::string, std::vector<size_t>> framesets; // key -> rec indices
    std::vector<std::string> fs_keys;
    for (size_t i = 0; i < recs.size(); ++i) {
        std::string key = recs[i].trial + "|" + std::to_string(recs[i].frame);
        auto it = framesets.find(key);
        if (it == framesets.end()) { framesets[key] = {}; fs_keys.push_back(key); }
        framesets[key].push_back(i);
    }
    std::vector<int> fs_idx(fs_keys.size());
    std::iota(fs_idx.begin(), fs_idx.end(), 0);
    std::vector<int> train_fs, val_fs;
    JarvisExport::split_frames(fs_idx, cfg_in.train_ratio, cfg_in.seed, train_fs, val_fs);
    std::vector<char> is_val(fs_keys.size(), 0);
    for (int i : val_fs) is_val[i] = 1;
    std::map<std::string, bool> rec_is_val; // rec key -> val?
    for (size_t k = 0; k < fs_keys.size(); ++k) rec_is_val[fs_keys[k]] = is_val[k] != 0;

    // 7. Build merged train/val JSONs.
    // Image ids are re-numbered 0..N-1 contiguously *within each split*. JARVIS's
    // dataset loader (datasetBase.py) indexes `self.image_ids[id]`, i.e. it treats
    // an image's `id` as its positional index, so each instances_*.json must have
    // ids 0..N-1. The global ids assigned in step 5 are split across train/val and
    // are therefore non-contiguous per split — renumber them here, remapping each
    // annotation's image_id and every frameset frame id to match.
    auto build_split = [&](bool want_val) {
        json images = json::array(), anns = json::array(), fsets = json::object();
        int next_id = 0, next_ann_id = 0;
        for (const auto &rec : recs) {
            std::string key = rec.trial + "|" + std::to_string(rec.frame);
            if (rec_is_val[key] != want_val) continue;
            int local_id = next_id++;
            json img = rec.image;
            img["id"] = local_id;
            images.push_back(std::move(img));
            for (auto a : rec.anns) {
                a["image_id"] = local_id;
                a["id"] = next_ann_id++;
                anns.push_back(std::move(a));
            }
            std::string fkey = rec.trial + "/Frame_" + std::to_string(rec.frame);
            if (!fsets.contains(fkey)) {
                fsets[fkey] = {{"datasetName", rec.trial}, {"frames", json::array()}};
            }
            fsets[fkey]["frames"].push_back(local_id);
        }
        json root;
        root["keypoint_names"] = merged_kp;
        root["skeleton"] = merged_skeleton;
        root["categories"] = merged_categories;
        root["annotations"] = anns;
        root["images"] = images;
        root["calibrations"] = calibrations;
        root["framesets"] = fsets;
        return root;
    };

    json train_json = build_split(false);
    json val_json = build_split(true);

    {
        std::ofstream f(out + "/annotations/instances_train.json");
        if (!f) { if (status) *status = "Error: cannot write instances_train.json"; return false; }
        f << train_json.dump(4);
    }
    {
        std::ofstream f(out + "/annotations/instances_val.json");
        if (!f) { if (status) *status = "Error: cannot write instances_val.json"; return false; }
        f << val_json.dump(4);
    }

    // 8. Calibration files.
    if (status) *status = "Writing calibrations...";
    for (size_t i = 0; i < srcs.size(); ++i) {
        const SourceInfo &s = *srcs[i];
        if (s.kind == SourceInfo::Dataset) {
            for (const auto &[orig, ren] : rename[i]) {
                std::string from = s.path + "/calib_params/" + orig;
                std::string to = out + "/calib_params/" + ren;
                if (fs::exists(from)) {
                    fs::create_directories(to, ec);
                    fs::copy(from, to,
                             fs::copy_options::recursive |
                                 fs::copy_options::overwrite_existing, ec);
                }
            }
        } else {
            // Project: write a JARVIS-readable <cam>.yaml. Telecentric emits a
            // projectionMatrix (dims from video, x10 optional); perspective
            // converts the RED intrinsic/R/T (dims from the source YAML).
            std::string trial = project_trial[(int)i];
            JarvisExport::ExportConfig ccfg;
            ccfg.camera_names = s.camera_names;
            ccfg.calibration_folder = s.calibration_folder;
            ccfg.output_folder = out;
            ccfg.telecentric = s.telecentric;
            ccfg.scale_10x = cfg_in.scale_10x;
            std::string cerr;
            bool ok;
            std::map<std::string, int> w, h;
            if (s.telecentric) {
                for (const auto &cam : s.camera_names) {
                    ffmpeg_reader::FrameReader reader;
                    if (!reader.open(s.media_folder + "/" + cam + ".mp4")) {
                        if (status)
                            *status = "Error: cannot open video for calib dims: " + cam + ".mp4";
                        return false;
                    }
                    w[cam] = reader.width();
                    h[cam] = reader.height();
                }
                ok = JarvisExport::write_projection_yaml(ccfg, trial, w, h, &cerr);
            } else {
                for (const auto &cam : s.camera_names) {
                    try {
                        auto yaml = opencv_yaml::read(s.calibration_folder + "/" + cam + ".yaml");
                        w[cam] = yaml.getInt("image_width");
                        h[cam] = yaml.getInt("image_height");
                    } catch (...) {}
                }
                ok = JarvisExport::write_calibration_yamls(ccfg, trial, w, h, &cerr);
            }
            if (!ok) {
                if (status) *status = "Error: " + cerr;
                return false;
            }
        }
    }

    // 9. Produce images: copy for Dataset sources, decode for Project sources.
    if (status) *status = "Extracting images...";
    std::mutex status_mutex;
    for (size_t i = 0; i < srcs.size(); ++i) {
        const SourceInfo &s = *srcs[i];
        if (s.kind == SourceInfo::Dataset) {
            for (const auto &rec : recs) {
                if (rec.src_index != (int)i) continue;
                std::string mode = rec_is_val[rec.trial + "|" + std::to_string(rec.frame)]
                                       ? "val" : "train";
                std::string from = s.path + "/" + rec.orig_mode + "/" + rec.orig_trial +
                                   "/" + rec.cam + "/Frame_" + std::to_string(rec.frame) + ".jpg";
                std::string to_dir = out + "/" + mode + "/" + rec.trial + "/" + rec.cam;
                fs::create_directories(to_dir, ec);
                std::string to = to_dir + "/Frame_" + std::to_string(rec.frame) + ".jpg";
                fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
                if (images_saved) images_saved->fetch_add(1, std::memory_order_relaxed);
            }
        } else {
            // Group this project's frames by assigned mode; decode per camera.
            std::string trial = project_trial[(int)i];
            std::vector<int> train_frames, val_frames;
            std::map<int, std::string> frame_to_mode;
            std::set<int> seen;
            for (const auto &rec : recs) {
                if (rec.src_index != (int)i) continue;
                if (seen.count(rec.frame)) continue;
                seen.insert(rec.frame);
                bool v = rec_is_val[rec.trial + "|" + std::to_string(rec.frame)];
                if (v) { val_frames.push_back(rec.frame); frame_to_mode[rec.frame] = "val"; }
                else   { train_frames.push_back(rec.frame); frame_to_mode[rec.frame] = "train"; }
            }
            std::sort(train_frames.begin(), train_frames.end());
            std::sort(val_frames.begin(), val_frames.end());

            std::vector<std::thread> threads;
            for (const auto &cam : s.camera_names) {
                std::string video = s.media_folder + "/" + cam + ".mp4";
                threads.emplace_back(JarvisExport::extract_jpegs_for_camera, cam, trial,
                                     video, out, train_frames, val_frames, frame_to_mode,
                                     status, &status_mutex, images_saved, cfg_in.jpeg_quality);
            }
            for (auto &t : threads) t.join();
        }
    }

    if (status && status->rfind("Error", 0) == 0) return false;

    int total_fs = (int)fs_keys.size();
    if (status)
        *status = "Merge complete! " + std::to_string(srcs.size()) + " sources, " +
                  std::to_string(total_fs) + " framesets (" +
                  std::to_string(train_fs.size()) + " train / " +
                  std::to_string(val_fs.size()) + " val), " +
                  std::to_string(recs.size()) + " images -> " + out;
    return true;
}

} // namespace JarvisMerge
