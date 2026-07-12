#pragma once
/*  red3d_import.h — Import a red_csv v2 3D keypoints file into the Labeling Tool
 *
 *  The cluster JARVIS pipeline emits keypoints3d.csv in RED's native #red_csv v2
 *  3D format (frame, then x,y,z,c per keypoint). Two things stop it from loading
 *  through AnnotationCSV::load_all directly:
 *    1. Its #skeleton header carries the MODEL name (e.g. "fly44_l_V4"), which
 *       fails load_all's exact skeleton-name check against the project skeleton.
 *    2. It has 3D only — no per-camera 2D — so there is nothing to view/edit in
 *       the camera viewports.
 *
 *  This importer parses the 3D directly (ignoring the file's #skeleton header and
 *  using the project skeleton), marks each 3D keypoint Imported/awaiting-review,
 *  and reprojects it into every camera as an editable Predicted-source 2D point.
 *  The result is an AnnotationMap identical in shape to what a JARVIS batch
 *  predict → Labeling Tool produces, so it can be saved via AnnotationCSV and
 *  edited the same way. Mirrors the Pose-Stats "promote" reprojection in
 *  red.cpp / reproject_3d_to_cam() in gui/gui_keypoints.h, minus the GUI deps.
 */

#include "annotation.h"
#include "annotation_csv.h"   // parse_csv_double
#include "camera.h"
#include "prediction_store.h"
#include "red_math.h"

#include <Eigen/Core>
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
    int frames_total = 0;       // data rows seen (excluding all-empty frames)
    int frames_imported = 0;    // frames with >= 1 valid 3D keypoint
    long long kp3d_placed = 0;  // 3D keypoints written
    long long kp2d_placed = 0;  // reprojected 2D keypoints written (across cams)
    int num_nodes = 0;
    std::string error;
};

// Reproject one 3D world point into camera `cp`'s image, returning ImPlot coords
// (Y=0 at bottom, matching the 2D label convention). Faithful copy of
// reproject_3d_to_cam() in gui/gui_keypoints.h without pulling in ImPlot/render.
// Returns false if the point is behind the camera or falls outside the image.
inline bool reproject_point(const Eigen::Vector3d &p3d, const CameraParams &cp,
                            int W, int H, double &out_x, double &out_y) {
    if (cp.telecentric) {
        auto rp = red_math::projectPointTelecentric(
            p3d, cp.projection_mat, cp.k, cp.dist_coeffs, cp.dist_center);
        out_x = rp(0);
        out_y = (double)H - rp(1);
    } else {
        Eigen::Vector3d cam_pt = cp.r * p3d + cp.tvec;
        if (cam_pt(2) <= 0) return false;  // behind camera
        auto rp = red_math::projectPointR(p3d, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
        out_x = rp(0);
        out_y = (double)H - rp(1);
    }
    return (out_x > 0 && out_x < W && out_y > 0 && out_y < H);
}

// Stream-parse a red_csv v2 keypoints3d.csv into `amap`, reprojecting every 3D
// keypoint into each camera as an editable Predicted 2D label.
//   num_nodes      — project skeleton node count. Extra columns in the file are
//                    ignored; missing columns leave those keypoints unlabeled.
//   camera_params  — per-camera calibration (size = num_cameras). May be empty
//                    (then only 3D is imported, no 2D reprojection).
//   img_w / img_h  — per-camera pixel dimensions (size = num_cameras).
// Comment (#...) and the "frame," column-header line are skipped, so the file's
// #skeleton header is intentionally not consulted.
inline ImportStats import_red3d_csv(const std::string &path, AnnotationMap &amap,
                                    int num_nodes,
                                    const std::vector<CameraParams> &camera_params,
                                    const std::vector<int> &img_w,
                                    const std::vector<int> &img_h) {
    ImportStats st;
    st.num_nodes = num_nodes;

    std::ifstream fin(path);
    if (!fin) {
        st.error = "Failed to open: " + path;
        return st;
    }
    if (num_nodes <= 0) {
        st.error = "Project has no skeleton (num_nodes = 0)";
        return st;
    }

    const int num_cams = (int)camera_params.size();

    // Per-row scratch, reused across rows to avoid per-frame allocation.
    std::vector<double> vx(num_nodes), vy(num_nodes), vz(num_nodes), vc(num_nodes);
    std::vector<char> valid(num_nodes);

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;                                  // comment/header
        if (line.size() >= 6 && line.compare(0, 6, "frame,") == 0)     // column header
            continue;

        const char *ptr = line.c_str();
        double frame_d;
        if (!AnnotationCSV::parse_csv_double(ptr, frame_d)) continue;
        u32 frame = (u32)frame_d;

        // Parse the whole row first so we can skip all-empty (out-of-view)
        // frames without allocating a FrameAnnotation for them.
        bool any = false;
        for (int k = 0; k < num_nodes; ++k) {
            double x, y, z, c;
            bool hx = AnnotationCSV::parse_csv_double(ptr, x);
            bool hy = AnnotationCSV::parse_csv_double(ptr, y);
            bool hz = AnnotationCSV::parse_csv_double(ptr, z);
            bool hc = AnnotationCSV::parse_csv_double(ptr, c);
            if (hx && hy && hz && !std::isnan(x) && !std::isnan(y) && !std::isnan(z)) {
                vx[k] = x; vy[k] = y; vz[k] = z;
                vc[k] = hc ? c : 1.0;
                valid[k] = 1;
                any = true;
            } else {
                valid[k] = 0;
            }
        }
        if (!any) continue;

        st.frames_total++;
        FrameAnnotation &fa = get_or_create_frame(amap, frame, num_nodes, num_cams);
        for (int k = 0; k < num_nodes; ++k) {
            if (!valid[k]) continue;
            fa.kp3d[k].x = vx[k];
            fa.kp3d[k].y = vy[k];
            fa.kp3d[k].z = vz[k];
            fa.kp3d[k].set_imported((float)vc[k]);   // predicted, awaiting review
            st.kp3d_placed++;

            Eigen::Vector3d p3d(vx[k], vy[k], vz[k]);
            for (int cam = 0; cam < num_cams; ++cam) {
                double px, py;
                if (reproject_point(p3d, camera_params[cam], img_w[cam], img_h[cam],
                                    px, py)) {
                    auto &kp2d = fa.cameras[cam].keypoints[k];
                    kp2d.x = px;
                    kp2d.y = py;
                    kp2d.labeled = true;
                    kp2d.confidence = (float)vc[k];
                    kp2d.source = LabelSource::Predicted;
                    st.kp2d_placed++;
                }
            }
        }
        st.frames_imported++;
    }

    return st;
}

struct StoreWriteResult {
    bool ok = false;
    std::string path;       // absolute .rpred written (empty on failure)
    int frames_stored = 0;  // frames with >= 1 valid keypoint (present in store)
    std::string error;
};

// Local timestamp for the pred_<ts>.rpred filename — a copy of
// jarvis_import_detail::timestamp_now(), duplicated here so this header does not
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

// Stream a finished AnnotationMap into a sparse .rpred prediction store so the
// Bouts / Pose Stats tools (which read only from predstore::PredictionReader)
// can see 3D-imported labels. Byte-compatible with the stores red's Batch
// Predict and the JARVIS-CLI importer produce.
//   num_nodes          — store keypoint count (use the project skeleton's, so
//                        the load guard and Pose Stats "Fix frame" promote match).
//   out_store_dir      — <project>/predictions/red_store; created if absent.
//   fps                — video fps for bout durations (0 if unknown).
//   total_video_frames — full video length for the Pose Stats X-axis (0 → derive
//                        as max stored frame + 1).
// The map is a std::map<u32,...>, so iteration is ascending — satisfying the
// writer's strictly-increasing frame requirement. A frame is stored only if it
// has >= 1 valid 3D keypoint; per-keypoint values are (x,y,z,conf) when
// kp3d[k].source != None, NaN otherwise (matching extract_pred_row in red.cpp).
inline StoreWriteResult write_store_from_map(
    const AnnotationMap &amap, int num_nodes,
    const std::string &out_store_dir,
    uint32_t fps, uint32_t total_video_frames) {
    namespace fs = std::filesystem;
    StoreWriteResult r;

    if (num_nodes <= 0) { r.error = "Project has no skeleton (num_nodes = 0)"; return r; }
    if (amap.empty())   { r.error = "No frames to store"; return r; }

    std::error_code ec;
    fs::create_directories(out_store_dir, ec);
    if (ec) {
        r.error = "Cannot create store dir " + out_store_dir + ": " + ec.message();
        return r;
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
        r.error = "Cannot open store for writing: " + store_path.string();
        return r;
    }

    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> buf((size_t)num_nodes * 4);
    int stored = 0;
    u32 max_frame = 0;
    for (const auto &[frame, fa] : amap) {
        bool any = false;
        for (int k = 0; k < num_nodes; ++k) {
            const size_t base = (size_t)k * 4;
            const bool valid = k < (int)fa.kp3d.size() &&
                               fa.kp3d[k].source != Kp3DSource::None;
            if (valid) {
                const auto &p = fa.kp3d[k];
                buf[base + 0] = (float)p.x;
                buf[base + 1] = (float)p.y;
                buf[base + 2] = (float)p.z;
                buf[base + 3] = p.confidence;
                any = true;
            } else {
                buf[base + 0] = buf[base + 1] = buf[base + 2] = buf[base + 3] = kNaN;
            }
        }
        if (!any) continue;
        if (!w.add_frame(frame, buf.data())) {
            w.close();
            fs::remove(store_path, ec);
            r.error = "Write failed at frame " + std::to_string(frame) +
                      " — store discarded.";
            return r;
        }
        if (frame > max_frame) max_frame = frame;
        ++stored;
    }

    if (stored == 0) {
        w.close();
        fs::remove(store_path, ec);
        r.error = "No frames with valid 3D keypoints to store.";
        return r;
    }

    const uint32_t total =
        (total_video_frames > 0) ? total_video_frames : (max_frame + 1);
    if (!w.finalize(total, fps)) {
        fs::remove(store_path, ec);
        r.error = "Failed to finalize the prediction store.";
        return r;
    }

    r.ok = true;
    r.path = store_path.string();
    r.frames_stored = stored;
    return r;
}

}  // namespace Red3DImport
