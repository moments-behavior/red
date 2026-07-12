#pragma once
// jarvis_predict_import.h — import a JARVIS-CLI 3D prediction folder
// (data3D.csv [+ info.yaml]) into red's memory-mapped .rpred prediction store,
// so cluster-produced predictions get the SAME downstream treatment as red's
// in-app Batch Predict "Store" mode: read-only video overlay, Pose Stats,
// bouts, and flag-for-improvement.
//
// This is the inverse of jarvis_predict_export.h. The row/frame contract is the
// same one both the JARVIS CLI (jarvis/prediction/predict3D.py) and red's
// exporter use:
//   * data3D.csv has two header rows: keypoint name ×4 per keypoint, then
//     "x,y,z,confidence" per keypoint. Data rows follow, one per *consecutive*
//     video frame; a frame with no 3D is written as an all-"NaN" row.
//   * info.yaml carries frame_start / number_frames (and recording_path /
//     dataset_name), so row index i corresponds to video frame frame_start + i.
//
// We reproduce red's in-app stash semantics exactly (see extract_pred_row /
// stash_prediction in red.cpp): a keypoint's four values are stored verbatim
// when its x is a real number, NaN otherwise; a frame is written to the sparse
// store only if it has at least one valid keypoint (all-NaN rows are absent,
// exactly as a never-predicted frame would be). The result is byte-compatible
// with a store red itself produced — the "Saved Predictions" picker, the
// keypoint-count guard, Pose Stats and Bouts all treat it identically.

#include "prediction_store.h"
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

struct JarvisImportResult {
    bool ok = false;
    std::string store_path;    // absolute .rpred written (empty on failure)
    std::string message;       // human-readable status
    int num_keypoints = 0;     // keypoints found in data3D.csv (columns / 4)
    int frame_start = 0;
    int number_frames = 0;     // total consecutive frames covered (rows)
    int frames_stored = 0;     // rows with >=1 valid keypoint (present in store)
    int total_frames = 0;      // header total_video_frames written
    int fps = 0;
    std::string media_folder;  // recording_path from info.yaml (if present)
    std::string dataset_name;  // dataset_name from info.yaml (if present)
    std::vector<std::string> keypoint_names;  // from header row 1
};

namespace jarvis_import_detail {

// Trim ASCII whitespace (incl. a trailing '\r' from CRLF lines) in place.
inline std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// Split a CSV line on ',' — data3D cells are numbers or simple node names, so
// no quoting/escaping is expected (matches Python csv.writer's default output).
inline std::vector<std::string> split_csv(const std::string &line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') { out.push_back(cur); cur.clear(); }
        else if (c != '\r') cur.push_back(c);
    }
    out.push_back(cur);
    return out;
}

// Parse one numeric cell. Empty / "NaN" (any case) → quiet NaN, matching the
// all-NaN sentinel rows the CLI and red's exporter write for missing 3D.
inline float parse_cell(const std::string &raw) {
    std::string s = trim(raw);
    if (s.empty()) return std::numeric_limits<float>::quiet_NaN();
    // strtof parses "nan"/"NaN"/"inf" per C99; guard the empty case above.
    const char *p = s.c_str();
    char *end = nullptr;
    float v = std::strtof(p, &end);
    if (end == p) return std::numeric_limits<float>::quiet_NaN();
    return v;
}

// Read a first non-empty, non-header data line from a stream, returning false
// at EOF.
inline bool read_line(std::istream &in, std::string &line) {
    return (bool)std::getline(in, line);
}

// Minimal "key: value" reader for info.yaml (the CLI writes flat scalars via
// ruamel.yaml; we only need a handful of top-level keys, so no YAML dependency).
inline std::string yaml_scalar(const std::string &line, const std::string &key) {
    std::string t = trim(line);
    if (t.rfind(key + ":", 0) != 0) return {};
    std::string v = trim(t.substr(key.size() + 1));
    // Strip surrounding quotes if present.
    if (v.size() >= 2 && (v.front() == '"' || v.front() == '\'') &&
        v.back() == v.front())
        v = v.substr(1, v.size() - 2);
    return v;
}

inline std::string timestamp_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", &tm_buf);
    return ts;
}

}  // namespace jarvis_import_detail

// Import a JARVIS 3D prediction folder (or a direct data3D.csv path) into a new
// .rpred store under `out_store_dir` (typically <project>/predictions/red_store),
// plus a provenance sidecar .json identical in schema to the one Batch Predict
// writes. On success, `result.store_path` is a store the "Saved Predictions"
// picker will list and the main loop can load through the normal guarded path.
//
// `csv_or_dir`  — path to data3D.csv, or a folder containing it (info.yaml is
//                 read from alongside the CSV when present).
// `media_folder`/`model_name`/`skeleton_name` — provenance for the sidecar;
//                 media_folder falls back to info.yaml's recording_path when
//                 the caller passes it empty.
// `fps_hint`    — video fps for bout durations (0 if unknown; info.yaml has none).
// `total_frames_hint` — full video length for the store header / Pose Stats
//                 X-axis (<=0 → derive as frame_start + number_frames).
inline JarvisImportResult jarvis_import_predictions3D(
    const std::string &csv_or_dir,
    const std::string &out_store_dir,
    const std::string &media_folder,
    const std::string &model_name,
    const std::string &skeleton_name,
    int fps_hint,
    int total_frames_hint)
{
    namespace fs = std::filesystem;
    using namespace jarvis_import_detail;
    JarvisImportResult r;

    // --- Resolve the CSV path and its sibling info.yaml. ---
    std::error_code ec;
    fs::path csv_path = csv_or_dir;
    if (fs::is_directory(csv_path, ec))
        csv_path = csv_path / "data3D.csv";
    if (!fs::exists(csv_path, ec)) {
        r.message = "No data3D.csv found at " + csv_path.string();
        return r;
    }
    fs::path info_path = csv_path.parent_path() / "info.yaml";

    // --- info.yaml (optional): frame_start / number_frames / recording_path. ---
    int frame_start = 0;
    int number_frames_declared = -1;
    std::string recording_path, dataset_name;
    {
        std::ifstream y(info_path);
        if (y) {
            std::string line;
            while (std::getline(y, line)) {
                std::string v;
                if (!(v = yaml_scalar(line, "frame_start")).empty())
                    frame_start = std::atoi(v.c_str());
                else if (!(v = yaml_scalar(line, "number_frames")).empty())
                    number_frames_declared = std::atoi(v.c_str());
                else if (!(v = yaml_scalar(line, "recording_path")).empty())
                    recording_path = v;
                else if (!(v = yaml_scalar(line, "dataset_name")).empty())
                    dataset_name = v;
            }
        }
    }
    r.frame_start = frame_start;
    r.media_folder = media_folder.empty() ? recording_path : media_folder;
    r.dataset_name = dataset_name;

    // --- data3D.csv header: two rows; column count / 4 = num_keypoints. ---
    std::ifstream f(csv_path, std::ios::binary);
    if (!f) { r.message = "Cannot open " + csv_path.string(); return r; }

    std::string header1, header2;
    if (!read_line(f, header1) || !read_line(f, header2)) {
        r.message = "data3D.csv is missing its two header rows.";
        return r;
    }
    std::vector<std::string> h1 = split_csv(header1);
    if (h1.empty() || h1.size() % 4 != 0) {
        r.message = "data3D.csv header has " + std::to_string(h1.size()) +
                    " columns (expected a multiple of 4).";
        return r;
    }
    const int nkp = (int)(h1.size() / 4);
    r.num_keypoints = nkp;
    r.keypoint_names.reserve(nkp);
    for (int j = 0; j < nkp; ++j)
        r.keypoint_names.push_back(trim(h1[(size_t)j * 4]));

    // --- Open the store writer. Name pred_<ts>.rpred so the picker's timestamp
    //     label parser and newest-first sort work exactly as for in-app stores. ---
    fs::create_directories(out_store_dir, ec);
    if (ec) {
        r.message = "Cannot create store dir " + out_store_dir + ": " + ec.message();
        return r;
    }
    fs::path store_path;
    {
        std::string ts = timestamp_now();
        store_path = fs::path(out_store_dir) / ("pred_" + ts + ".rpred");
        // Collision guard (two imports within the same second): add a suffix.
        for (int k = 2; fs::exists(store_path, ec); ++k)
            store_path = fs::path(out_store_dir) /
                         ("pred_" + ts + "-" + std::to_string(k) + ".rpred");
    }

    predstore::PredictionWriter w;
    if (!w.open(store_path.string(), nkp)) {
        r.message = "Cannot open store for writing: " + store_path.string();
        return r;
    }

    // --- Stream data rows. Row index i → video frame frame_start + i. Store a
    //     frame only if >=1 keypoint is valid (x is real), NaN-filling the rest —
    //     byte-identical to extract_pred_row/stash_prediction in red.cpp. ---
    std::vector<float> buf((size_t)nkp * 4);
    int rows = 0, stored = 0;
    std::string line;
    while (read_line(f, line)) {
        // A trailing blank line (common with CRLF files) is not a frame.
        if (trim(line).empty()) continue;
        std::vector<std::string> cells = split_csv(line);

        bool any = false;
        for (int j = 0; j < nkp; ++j) {
            float x = std::numeric_limits<float>::quiet_NaN();
            float y = x, z = x, c = x;
            const size_t base = (size_t)j * 4;
            if (base + 0 < cells.size()) x = parse_cell(cells[base + 0]);
            if (base + 1 < cells.size()) y = parse_cell(cells[base + 1]);
            if (base + 2 < cells.size()) z = parse_cell(cells[base + 2]);
            if (base + 3 < cells.size()) c = parse_cell(cells[base + 3]);
            if (!std::isnan(x)) {
                buf[base + 0] = x; buf[base + 1] = y;
                buf[base + 2] = z; buf[base + 3] = c;
                any = true;
            } else {
                buf[base + 0] = buf[base + 1] = buf[base + 2] = buf[base + 3] =
                    std::numeric_limits<float>::quiet_NaN();
            }
        }

        if (any) {
            if (!w.add_frame((uint32_t)(frame_start + rows), buf.data())) {
                w.close();
                fs::remove(store_path, ec);
                r.message = "Write failed at row " + std::to_string(rows) +
                            " — store discarded.";
                return r;
            }
            ++stored;
        }
        ++rows;
    }
    f.close();

    if (rows == 0) {
        w.close();
        fs::remove(store_path, ec);
        r.message = "data3D.csv has no data rows.";
        return r;
    }

    const int number_frames =
        (number_frames_declared > 0) ? number_frames_declared : rows;
    const int total_frames =
        (total_frames_hint > 0) ? total_frames_hint : (frame_start + number_frames);
    const int fps = (fps_hint > 0) ? fps_hint : 0;

    if (!w.finalize((uint32_t)total_frames, (uint32_t)fps)) {
        fs::remove(store_path, ec);
        r.message = "Failed to finalize the prediction store.";
        return r;
    }

    // --- Provenance sidecar (schema identical to Batch Predict's, so the picker
    //     tooltip / scan reads it the same way). ---
    {
        nlohmann::json meta = {
            {"store_file", store_path.filename().string()},
            {"media_folder", r.media_folder},
            {"model_name", model_name},
            {"skeleton", skeleton_name},
            {"num_keypoints", nkp},
            {"frame_start", frame_start},
            {"frame_end", frame_start + number_frames - 1},
            {"n_stored", stored},
            {"fps", fps},
            {"imported_from", csv_path.string()},
        };
        fs::path side = store_path;
        side.replace_extension(".json");
        std::ofstream sf(side);
        if (sf) sf << meta.dump(2) << "\n";
    }

    r.ok = true;
    r.store_path = store_path.string();
    r.number_frames = number_frames;
    r.frames_stored = stored;
    r.total_frames = total_frames;
    r.fps = fps;
    r.message = "Imported " + std::to_string(stored) + " frames (of " +
                std::to_string(number_frames) + ", " + std::to_string(nkp) +
                " keypoints) → " + store_path.filename().string();
    return r;
}
