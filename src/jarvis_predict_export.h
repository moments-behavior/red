#pragma once
// jarvis_predict_export.h — write JARVIS-CLI-compatible 3D prediction output
// from red's in-memory AnnotationMap.
//
// Produces a Predictions_3D_<YYYYMMDD-HHMMSS>/ folder holding:
//   data3D.csv  — two header rows (keypoint name ×4, then x,y,z,confidence per
//                 keypoint) followed by one row per *consecutive* frame.
//                 Frames with no 3D (never predicted, e.g. skipped by the batch
//                 Step, or a failed detection) are written as an all-"NaN" row,
//                 so row index i == frame_start + i — exactly as the JARVIS CLI
//                 (jarvis/prediction/predict3D.py) writes it.
//   info.yaml   — recording_path / dataset_name / frame_start / number_frames.
//
// This mirrors the CLI so downstream JARVIS tooling (create_videos3D, the
// analysis GUI, predictions_index scripts) reads red's output unchanged.
// See predict3D.py: create_header / create_info_file / the predict loop.

#include "annotation.h"
#include "project.h"
#include "skeleton.h"
#include "prediction_store.h"

#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>

struct JarvisExportResult {
    bool ok = false;
    std::string output_dir;   // absolute path of the folder created
    std::string message;      // human-readable status (success or failure)
    int rows_written = 0;     // total frames = number_frames
    int frames_with_data = 0; // frames that had at least one valid 3D keypoint
};

// Build <project_path>/predictions/predictions3D/Predictions_3D_<timestamp>.
// Timestamp format (%Y%m%d-%H%M%S) matches the CLI's folder naming exactly.
inline std::string jarvis_make_predictions_dir(const std::string &project_path) {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", &tm_buf);
    std::filesystem::path dir = std::filesystem::path(project_path) /
                                "predictions" / "predictions3D" /
                                (std::string("Predictions_3D_") + ts);
    return dir.string();
}

// Write one coordinate value with enough precision to round-trip a float32
// (9 significant digits). NaN sentinel matches the CLI's literal "NaN".
inline void jarvis_write_coord(std::ofstream &f, double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    f << buf;
}

// Export [frame_start, frame_start + number_frames) from `amap` as a
// JARVIS-compatible Predictions_3D folder at `output_dir`.
//
// A keypoint is written when its 3D source is set and its value is real
// (not the UNLABELED sentinel); otherwise that keypoint's four cells are NaN.
// A frame with no valid keypoints at all is written as a full NaN row.
inline JarvisExportResult jarvis_export_predictions3D(
    const std::string &output_dir,
    const AnnotationMap &amap,
    const SkeletonContext &skel,
    const ProjectManager &pm,
    int frame_start,
    int number_frames)
{
    namespace fs = std::filesystem;
    JarvisExportResult r;

    const int nj = skel.num_nodes;
    if (nj <= 0) {
        r.message = "No skeleton loaded — nothing to export.";
        return r;
    }
    if (number_frames <= 0) {
        r.message = "Empty frame range — nothing to export.";
        return r;
    }

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        r.message = "Failed to create " + output_dir + ": " + ec.message();
        return r;
    }

    // --- info.yaml ---
    // Key order matches predict3D.create_info_file. recording_path is where
    // red loads the videos from; dataset_name is the calibration used.
    {
        std::ofstream y(fs::path(output_dir) / "info.yaml");
        if (!y) {
            r.message = "Cannot open info.yaml for writing.";
            return r;
        }
        y << "recording_path: " << pm.media_folder << "\n";
        y << "dataset_name: " << pm.calibration_folder << "\n";
        y << "frame_start: " << frame_start << "\n";
        y << "number_frames: " << number_frames << "\n";
    }

    // --- data3D.csv ---
    // Open in binary mode and terminate every row with CRLF, byte-matching
    // Python's csv.writer (predict3D.py) — which emits \r\n regardless of OS.
    // (info.yaml above stays LF: the CLI writes it via ruamel.yaml, which is
    // \n.) The content is identical either way and every CSV parser accepts
    // both; this just makes red's output byte-for-byte identical to the CLI.
    std::ofstream f(fs::path(output_dir) / "data3D.csv", std::ios::binary);
    if (!f) {
        r.message = "Cannot open data3D.csv for writing.";
        return r;
    }
    static constexpr const char *CRLF = "\r\n";

    // Header row 1: each keypoint name repeated 4×.
    for (int j = 0; j < nj; ++j) {
        std::string name = (j < (int)skel.node_names.size())
                               ? skel.node_names[j]
                               : ("kp" + std::to_string(j));
        for (int k = 0; k < 4; ++k) {
            f << name;
            if (!(j == nj - 1 && k == 3)) f << ",";
        }
    }
    f << CRLF;
    // Header row 2: x,y,z,confidence per keypoint.
    for (int j = 0; j < nj; ++j) {
        f << "x,y,z,confidence";
        if (j != nj - 1) f << ",";
    }
    f << CRLF;

    // Data rows: one per consecutive frame.
    for (int i = 0; i < number_frames; ++i) {
        u32 frame = (u32)(frame_start + i);
        auto it = amap.find(frame);

        const std::vector<Keypoint3D> *kp3d = nullptr;
        bool frame_has_3d = false;
        if (it != amap.end()) {
            kp3d = &it->second.kp3d;
            for (const auto &kp : *kp3d) {
                if (kp.source != Kp3DSource::None && kp.x != UNLABELED) {
                    frame_has_3d = true;
                    break;
                }
            }
        }

        for (int j = 0; j < nj; ++j) {
            bool valid = frame_has_3d && kp3d && j < (int)kp3d->size() &&
                         (*kp3d)[j].source != Kp3DSource::None &&
                         (*kp3d)[j].x != UNLABELED;
            if (valid) {
                const Keypoint3D &p = (*kp3d)[j];
                jarvis_write_coord(f, p.x); f << ",";
                jarvis_write_coord(f, p.y); f << ",";
                jarvis_write_coord(f, p.z); f << ",";
                jarvis_write_coord(f, p.confidence);
            } else {
                f << "NaN,NaN,NaN,NaN";
            }
            if (j != nj - 1) f << ",";
        }
        f << CRLF;

        r.rows_written++;
        if (frame_has_3d) r.frames_with_data++;
    }
    f.close();

    r.ok = true;
    r.output_dir = output_dir;
    r.message = "Exported " + std::to_string(r.rows_written) + " frames (" +
                std::to_string(r.frames_with_data) + " with 3D) to " +
                fs::path(output_dir).filename().string();
    return r;
}

// Same JARVIS-CLI-compatible output, but sourced from a memory-mapped
// PredictionReader (Batch Predict "Store" mode, where predictions live in the
// .rpred store rather than the AnnotationMap). Rows absent from the store are
// written as all-NaN, so row index == frame - frame_start, matching the CLI.
inline JarvisExportResult jarvis_export_predictions3D_from_reader(
    const std::string &output_dir,
    const predstore::PredictionReader &reader,
    const SkeletonContext &skel,
    const ProjectManager &pm,
    int frame_start,
    int number_frames)
{
    namespace fs = std::filesystem;
    JarvisExportResult r;

    const int nj = skel.num_nodes;
    if (nj <= 0) { r.message = "No skeleton loaded — nothing to export."; return r; }
    if (number_frames <= 0) { r.message = "Empty frame range — nothing to export."; return r; }
    if (!reader.is_open()) { r.message = "Prediction store not open."; return r; }

    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) { r.message = "Failed to create " + output_dir + ": " + ec.message(); return r; }

    {
        std::ofstream y(fs::path(output_dir) / "info.yaml");
        if (!y) { r.message = "Cannot open info.yaml for writing."; return r; }
        y << "recording_path: " << pm.media_folder << "\n";
        y << "dataset_name: " << pm.calibration_folder << "\n";
        y << "frame_start: " << frame_start << "\n";
        y << "number_frames: " << number_frames << "\n";
    }

    std::ofstream f(fs::path(output_dir) / "data3D.csv", std::ios::binary);
    if (!f) { r.message = "Cannot open data3D.csv for writing."; return r; }
    static constexpr const char *CRLF = "\r\n";

    for (int j = 0; j < nj; ++j) {
        std::string name = (j < (int)skel.node_names.size())
                               ? skel.node_names[j] : ("kp" + std::to_string(j));
        for (int k = 0; k < 4; ++k) { f << name; if (!(j == nj - 1 && k == 3)) f << ","; }
    }
    f << CRLF;
    for (int j = 0; j < nj; ++j) { f << "x,y,z,confidence"; if (j != nj - 1) f << ","; }
    f << CRLF;

    const int epf = reader.elements_per_frame();
    for (int i = 0; i < number_frames; ++i) {
        const float *row = reader.frame((uint32_t)(frame_start + i));
        bool has = (row != nullptr);
        for (int j = 0; j < nj; ++j) {
            bool valid = has && (j * 4 + 3) < epf && !std::isnan(row[j * 4]);
            if (valid) {
                jarvis_write_coord(f, row[j * 4 + 0]); f << ",";
                jarvis_write_coord(f, row[j * 4 + 1]); f << ",";
                jarvis_write_coord(f, row[j * 4 + 2]); f << ",";
                jarvis_write_coord(f, row[j * 4 + 3]);
            } else {
                f << "NaN,NaN,NaN,NaN";
            }
            if (j != nj - 1) f << ",";
        }
        f << CRLF;
        r.rows_written++;
        if (has) r.frames_with_data++;
    }
    f.close();

    r.ok = true;
    r.output_dir = output_dir;
    r.message = "Exported " + std::to_string(r.rows_written) + " frames (" +
                std::to_string(r.frames_with_data) + " with 3D) to " +
                fs::path(output_dir).filename().string();
    return r;
}
