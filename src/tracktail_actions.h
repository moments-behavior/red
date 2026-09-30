#pragma once
// tracktail_actions.h — main-loop side of the tracktail panel.
//
// Consumes the request flags set by DrawTracktailWindow (gui/tracktail_window.h)
// and owns the HTTP state. Called once per render tick from red.cpp:
//
//     tracktail_handle_requests(win.tracktail, tracktail_rt, ctx);
//
// Seed = the active animal's triangulated 3D keypoints on the current frame
// (2D labels are triangulated first when the panel's checkbox is on). The
// prediction for frames [current+1 .. current+N] is written into that same
// animal's FrameAnnotation (matched by instance_id, created if missing) as 3D
// + reprojected 2D marked LabelSource::Predicted. Other animals in those
// frames are untouched.

#include "app_context.h"
#include "gui/gui_keypoints.h"   // reprojection() (triangulate), reproject_3d_to_cam()
#include "gui/tracktail_window.h"
#include "tracktail_server_client.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

// Server state. One instance in main(), outlives every project.
struct TracktailRuntime {
    TracktailServerState server;
};

namespace tracktail_actions_detail {

// Host pointer to camera `cam`'s frame `frame_off` frames after the current
// one, or nullptr if it isn't staged. Looks the ring buffer up by frame
// number first (robust to how the ring is laid out), falling back to the
// read_head + pause_selected arithmetic the transport uses. GPU-resident
// buffers are copied down into `scratch`, which the caller keeps alive for
// as long as the pointers are used.
inline const uint8_t *pull_frame_host(AppContext &ctx, int cam, int frame_off,
                                      std::deque<std::vector<uint8_t>> &scratch) {
    auto *scene = ctx.scene;
    auto &ps = ctx.ps;
    if (cam < 0 || cam >= (int)scene->num_cams) return nullptr;
    const int buf_size = (int)scene->size_of_buffer;
    if (buf_size <= 0) return nullptr;
    const int want = ctx.current_frame_num + frame_off;

    int slot = -1;
    for (int s = 0; s < buf_size; ++s) {
        auto &pb = scene->display_buffer[cam][s];
        if (pb.frame && pb.frame_number.load() == want &&
            !pb.available_to_write.load()) {
            slot = s;
            break;
        }
    }
    if (slot < 0) slot = (ps.read_head + ps.pause_selected + frame_off) % buf_size;
    auto &pb = scene->display_buffer[cam][slot];
    if (!pb.frame) return nullptr;
    if (scene->use_cpu_buffer) return (const uint8_t *)pb.frame;
#ifdef TRACKTAIL_HAS_CUDA
    size_t npix = (size_t)scene->image_width[cam] * scene->image_height[cam];
    scratch.emplace_back(npix * 4);
    if (cudaMemcpy(scratch.back().data(), pb.frame, npix * 4,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        scratch.pop_back();
        return nullptr;
    }
    return scratch.back().data();
#else
    (void)scratch;
    return nullptr;
#endif
}

// Write one predicted 3D point into `fa` (3D + reprojected 2D per camera).
inline void write_prediction(FrameAnnotation &fa, int k, const Eigen::Vector3d &X,
                             AppContext &ctx) {
    if (k < 0 || k >= (int)fa.kp3d.size()) return;
    fa.kp3d[k].x = X(0);
    fa.kp3d[k].y = X(1);
    fa.kp3d[k].z = X(2);
    fa.kp3d[k].set_triangulated();
    const int num_cams = (int)ctx.scene->num_cams;
    for (int v = 0; v < num_cams && v < (int)ctx.pm.camera_params.size(); ++v) {
        if (v >= (int)fa.cameras.size()) continue;
        if (k >= (int)fa.cameras[v].keypoints.size()) continue;
        double px, py;
        if (reproject_3d_to_cam(X, ctx.pm.camera_params[v],
                                (int)ctx.scene->image_width[v],
                                (int)ctx.scene->image_height[v], px, py)) {
            auto &kp = fa.cameras[v].keypoints[k];
            kp.x = px;
            kp.y = py;
            kp.labeled = true;
            kp.source = LabelSource::Predicted;
        }
    }
}

// Collect the seed from the active animal on the current frame. Triangulates
// first when asked and there is no 3D yet. Returns false (with a message)
// when there is nothing to seed from.
inline bool collect_seed(TracktailWindowState &st, AppContext &ctx,
                         std::vector<Eigen::Vector3d> &seed,
                         std::vector<int> &seed_node_idx, int &instance_id,
                         std::string &why_not) {
    seed.clear();
    seed_node_idx.clear();
    auto it = ctx.annotations.find((u32)ctx.current_frame_num);
    if (it == ctx.annotations.end() || it->second.empty()) {
        why_not = "Current frame has no annotation -- label it first";
        return false;
    }
    FrameAnnotation &fa = instance_or_first(it->second, ctx.active_instance);
    instance_id = fa.instance_id;

    auto count_3d = [&]() {
        int n = 0;
        for (int k = 0; k < (int)fa.kp3d.size() && k < ctx.skeleton.num_nodes; ++k)
            if (fa.kp3d[k].triangulated) n++;
        return n;
    };
    if (count_3d() == 0 && st.auto_triangulate) {
        // Same call as pressing T in the Labeling Tool.
        reprojection(fa, &ctx.skeleton, ctx.pm.camera_params, ctx.scene);
        printf("[tracktail] Triangulated frame %d (animal id %d) for the seed\n",
               ctx.current_frame_num, instance_id);
    }
    for (int k = 0; k < (int)fa.kp3d.size() && k < ctx.skeleton.num_nodes; ++k) {
        if (!fa.kp3d[k].triangulated) continue;
        seed.emplace_back(fa.kp3d[k].x, fa.kp3d[k].y, fa.kp3d[k].z);
        seed_node_idx.push_back(k);
    }
    if (seed.empty()) {
        why_not = "No triangulated keypoints on the current frame -- label "
                  "it in >= 2 cameras and press T";
        return false;
    }
    return true;
}

}  // namespace tracktail_actions_detail

inline void tracktail_handle_requests(TracktailWindowState &st,
                                      TracktailRuntime &rt, AppContext &ctx) {
    using namespace tracktail_actions_detail;
    auto *scene = ctx.scene;
    auto &pm = ctx.pm;

    // ── Server: Probe /info ──
    if (st.server_probe_requested) {
        st.server_probe_requested = false;
        rt.server.url = st.server_url;
        bool ok = tracktail_server_probe(rt.server);
        st.server_status = rt.server.status;
        if (ok) {
            st.server_n_frames = rt.server.n_frames;
            st.server_image_size = rt.server.image_size;
            st.server_device = rt.server.device;
            st.server_mode_3d = rt.server.mode_3d;
        }
        printf("[tracktail/server] %s\n", rt.server.status.c_str());
    }

    if (!st.forward_requested) return;
    st.forward_requested = false;

    // ── Common preconditions + seed ──
    if (pm.camera_params.empty()) {
        st.last_result = "No calibration loaded";
        st.last_result_ok = false;
        ctx.toasts.push(st.last_result, Toast::Warning, 3.0f);
        return;
    }
    if (!scene || scene->num_cams == 0 || !ctx.ps.video_loaded) {
        st.last_result = "No videos loaded";
        st.last_result_ok = false;
        ctx.toasts.push(st.last_result, Toast::Warning, 3.0f);
        return;
    }
    std::vector<Eigen::Vector3d> seed;
    std::vector<int> seed_node_idx;
    int instance_id = 0;
    std::string why_not;
    if (!collect_seed(st, ctx, seed, seed_node_idx, instance_id, why_not)) {
        st.last_result = why_not;
        st.last_result_ok = false;
        ctx.toasts.push(why_not, Toast::Warning, 4.0f);
        printf("[tracktail] %s\n", why_not.c_str());
        return;
    }

    const int num_cams = (int)scene->num_cams;
    const int joints_total = ctx.skeleton.num_nodes;
    std::vector<int> widths(num_cams), heights(num_cams);
    std::vector<std::string> cam_names(num_cams);
    for (int c = 0; c < num_cams; ++c) {
        widths[c] = (int)scene->image_width[c];
        heights[c] = (int)scene->image_height[c];
        cam_names[c] = c < (int)pm.camera_names.size() && !pm.camera_names[c].empty()
                           ? pm.camera_names[c]
                           : std::to_string(c);
    }
    auto future_frame = [&](int t) -> FrameAnnotation & {
        return get_or_create_frame(ctx.annotations,
                                   (u32)(ctx.current_frame_num + t),
                                   joints_total, num_cams, instance_id);
    };

    // ── One chunk of n_frames (the model's, from /info) ──
    // Probe first if this URL has not been probed: the chunk length and crop
    // size come from the server, not from red.
    if (rt.server.url != st.server_url || rt.server.n_frames <= 0) {
        rt.server.url = st.server_url;
        bool ok = tracktail_server_probe(rt.server);
        st.server_status = rt.server.status;
        st.server_n_frames = rt.server.n_frames;
        st.server_image_size = rt.server.image_size;
        st.server_device = rt.server.device;
        st.server_mode_3d = rt.server.mode_3d;
        if (!ok) {
            st.last_result = rt.server.status;
            st.last_result_ok = false;
            ctx.toasts.pushError(st.last_result);
            return;
        }
    }
    const int T = rt.server.n_frames;

    std::deque<std::vector<uint8_t>> scratch;
    std::vector<const uint8_t *> frames((size_t)num_cams * T, nullptr);
    int missing = 0;
    for (int c = 0; c < num_cams; ++c)
        for (int t = 0; t < T; ++t) {
            frames[c * T + t] = pull_frame_host(ctx, c, t, scratch);
            if (!frames[c * T + t]) missing++;
        }
    if (missing)
        printf("[tracktail/server] WARNING: %d of %d (cam, frame) slots not "
               "staged; sending grey for those\n", missing, num_cams * T);

    printf("[tracktail/server] Sending %d cams x %d frames x %d queries "
           "(animal id %d) to %s\n", num_cams, T, (int)seed.size(),
           instance_id, rt.server.url.c_str());
    auto t_start = std::chrono::steady_clock::now();
    TracktailChunkResult chunk = tracktail_server_predict_chunk(
        rt.server, frames, widths, heights, pm.camera_params, seed,
        /*seed_t=*/0, cam_names);
    float ms = std::chrono::duration<float, std::milli>(
                   std::chrono::steady_clock::now() - t_start).count();
    st.server_last_total_ms = rt.server.last_total_ms;
    st.server_last_request_ms = rt.server.last_request_ms;
    st.server_last_encode_ms = rt.server.last_encode_ms;
    st.server_last_decode_ms = rt.server.last_decode_ms;

    if (!chunk.ok) {
        st.server_status = "Predict failed: " + chunk.error;
        st.last_result = st.server_status;
        st.last_result_ok = false;
        ctx.toasts.pushError(st.last_result);
        printf("[tracktail/server] FAILED: %s\n", chunk.error.c_str());
        return;
    }
    // t=0 is the seed; write t=1..n_keep. A chunk holds T-1 future frames.
    const int n_keep = std::clamp(st.server_n_keep, 1, T - 1);
    if (n_keep < st.server_n_keep)
        printf("[tracktail/server] Model predicts %d frames ahead; keeping %d "
               "of the %d asked for\n", T - 1, n_keep, st.server_n_keep);
    for (int t = 1; t <= n_keep && t < (int)chunk.kp3d.size(); ++t) {
        FrameAnnotation &fa = future_frame(t);
        for (int q = 0; q < (int)seed_node_idx.size() &&
                        q < (int)chunk.kp3d[t].size(); ++q)
            write_prediction(fa, seed_node_idx[q], chunk.kp3d[t][q], ctx);
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "OK: %d frame%s x %d kp in %.0f ms (server total %.0f ms)",
                  n_keep, n_keep == 1 ? "" : "s", (int)seed.size(), ms,
                  rt.server.last_total_ms);
    st.server_status = buf;
    st.last_result = buf;
    st.last_result_ok = true;
    ctx.toasts.pushSuccess("tracktail: +" + std::to_string(n_keep) +
                           " frames (animal id " +
                           std::to_string(instance_id) + ")");
    printf("[tracktail/server] %s\n", buf);
}
