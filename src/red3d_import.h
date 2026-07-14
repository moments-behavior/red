#pragma once
/*  red3d_import.h — Import a red_csv v2 3D keypoints file into the prediction
 *  store (NOT the editable Labeling Tool).
 *
 *  The cluster JARVIS pipeline emits keypoints3d.csv in RED's native #red_csv v2
 *  3D format (frame, then x,y,z,c per keypoint). A whole-video prediction is far
 *  too large to live in the in-memory AnnotationMap — merging it into the
 *  Labeling Tool floods `annotations` and makes every render iterate hundreds of
 *  thousands of frames, so the UI crawls.
 *
 *  Instead we stream the 3D straight into a sparse, memory-mapped .rpred store —
 *  the same "promotion of frames" pattern Batch Predict "Store" mode and the
 *  JARVIS-CLI importer use. Predictions then appear as a read-only viewport
 *  overlay and feed Pose Stats / Bouts, while the user promotes individual
 *  frames into the editable Labeling Tool on demand (Pose Stats "Fix this
 *  frame", which reprojects the stored 3D to 2D for that one frame). Nothing is
 *  held in memory beyond the tiny frame index the writer keeps.
 *
 *  The file's #skeleton header (which carries the MODEL name, e.g. "fly44_l_V4")
 *  is intentionally ignored; the store is written at the project skeleton's node
 *  count so it matches the active skeleton's load guard and promote path.
 */

#include "annotation_csv.h"   // parse_csv_double
#include "prediction_store.h"

#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace Red3DImport {

struct ImportStats {
    int frames_total = 0;       // data rows with >= 1 valid 3D keypoint
    int frames_stored = 0;      // frames actually written to the store
    long long kp3d_placed = 0;  // 3D keypoints written
    int num_nodes = 0;
    std::string store_path;     // absolute .rpred written (empty on failure)
    std::string error;
};

// Local timestamp for the pred_<ts>.rpred filename — a copy of
// jarvis_import_detail::timestamp_now(), duplicated so this header does not
// transitively pull in json.hpp via jarvis_predict_import.h.
inline std::string store_timestamp_now() {
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

// Stream-parse a red_csv v2 keypoints3d.csv directly into a new sparse .rpred
// store under `out_store_dir` (typically <project>/predictions/red_store).
//   num_nodes          — store keypoint count (use the project skeleton's, so
//                        the load guard and Pose Stats "Fix frame" promote match).
//                        Extra columns in the file are ignored; missing columns
//                        leave those keypoints NaN.
//   fps                — video fps for bout durations (0 if unknown).
//   total_video_frames — full video length for the Pose Stats X-axis (0 → derive
//                        as max stored frame + 1).
// A frame is stored only if it has >= 1 valid 3D keypoint; per-keypoint values
// are (x,y,z,conf) when valid, NaN otherwise. A keypoint present but without a
// confidence column defaults to 1.0 (a placed point counts as fully confident,
// so Bouts / Pose Stats stay meaningful on confidence-less files).
//
// Comment (#...) and the "frame," column-header line are skipped. Rows must be
// in ascending frame order (RED's save_all and the cluster pipeline both write
// sorted); an out-of-order row is reported as an error and the store discarded,
// because the reader's index binary-search requires sorted frames.
inline ImportStats import_red3d_csv_to_store(const std::string &path,
                                             int num_nodes,
                                             const std::string &out_store_dir,
                                             uint32_t fps,
                                             uint32_t total_video_frames) {
    namespace fs = std::filesystem;
    ImportStats st;
    st.num_nodes = num_nodes;

    if (num_nodes <= 0) {
        st.error = "Project has no skeleton (num_nodes = 0)";
        return st;
    }
    std::ifstream fin(path);
    if (!fin) {
        st.error = "Failed to open: " + path;
        return st;
    }

    std::error_code ec;
    fs::create_directories(out_store_dir, ec);
    if (ec) {
        st.error = "Cannot create store dir " + out_store_dir + ": " + ec.message();
        return st;
    }

    // pred_<ts>.rpred, with a collision suffix (two imports within one second)
    // so the "Saved Predictions" picker's timestamp label / newest-first sort
    // stay correct — mirrors jarvis_import_predictions3D.
    std::string ts = store_timestamp_now();
    fs::path store_path = fs::path(out_store_dir) / ("pred_" + ts + ".rpred");
    for (int k = 2; fs::exists(store_path, ec); ++k)
        store_path = fs::path(out_store_dir) /
                     ("pred_" + ts + "-" + std::to_string(k) + ".rpred");

    predstore::PredictionWriter w;
    if (!w.open(store_path.string(), num_nodes)) {
        st.error = "Cannot open store for writing: " + store_path.string();
        return st;
    }

    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> buf((size_t)num_nodes * 4);
    bool have_prev = false;
    uint32_t prev_frame = 0, max_frame = 0;

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;                                  // comment/header
        if (line.size() >= 6 && line.compare(0, 6, "frame,") == 0)     // column header
            continue;

        const char *ptr = line.c_str();
        double frame_d;
        if (!AnnotationCSV::parse_csv_double(ptr, frame_d)) continue;
        uint32_t frame = (uint32_t)frame_d;

        bool any = false;
        for (int k = 0; k < num_nodes; ++k) {
            const size_t base = (size_t)k * 4;
            double x, y, z, c;
            bool hx = AnnotationCSV::parse_csv_double(ptr, x);
            bool hy = AnnotationCSV::parse_csv_double(ptr, y);
            bool hz = AnnotationCSV::parse_csv_double(ptr, z);
            bool hc = AnnotationCSV::parse_csv_double(ptr, c);
            if (hx && hy && hz && !std::isnan(x) && !std::isnan(y) && !std::isnan(z)) {
                buf[base + 0] = (float)x;
                buf[base + 1] = (float)y;
                buf[base + 2] = (float)z;
                buf[base + 3] = (float)(hc ? c : 1.0);
                any = true;
                st.kp3d_placed++;
            } else {
                buf[base + 0] = buf[base + 1] = buf[base + 2] = buf[base + 3] = kNaN;
            }
        }
        if (!any) continue;

        if (have_prev && frame <= prev_frame) {
            w.close();
            fs::remove(store_path, ec);
            st.kp3d_placed = 0;
            st.error = "keypoints3d.csv is not in ascending frame order (frame " +
                       std::to_string(frame) + " after " +
                       std::to_string(prev_frame) + ").";
            return st;
        }
        if (!w.add_frame(frame, buf.data())) {
            w.close();
            fs::remove(store_path, ec);
            st.kp3d_placed = 0;
            st.error = "Write failed at frame " + std::to_string(frame) +
                       " — store discarded.";
            return st;
        }
        prev_frame = frame;
        have_prev = true;
        if (frame > max_frame) max_frame = frame;
        st.frames_total++;
        st.frames_stored++;
    }

    if (st.frames_stored == 0) {
        w.close();
        fs::remove(store_path, ec);
        st.kp3d_placed = 0;
        st.error = "No frames with valid 3D keypoints found.";
        return st;
    }

    const uint32_t total =
        (total_video_frames > 0) ? total_video_frames : (max_frame + 1);
    if (!w.finalize(total, fps)) {
        fs::remove(store_path, ec);
        st.kp3d_placed = 0;
        st.error = "Failed to finalize the prediction store.";
        return st;
    }

    st.store_path = store_path.string();
    return st;
}

}  // namespace Red3DImport
