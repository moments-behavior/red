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
// prediction for frames [current+1 .. current+N] (or [current-N .. current-1]
// with Predict backwards) is written into that same animal's FrameAnnotation
// (matched by instance_id, created if missing) as 3D + reprojected 2D, both
// marked Predicted. Other animals in those frames are untouched, and so are
// hand-placed keypoints unless the panel says to overwrite them.
//
// Every image sent is looked up in the display buffer by frame number. The
// buffer only holds frames from the last seek onwards, so when the clip is
// not all there (always the case backwards) the request seeks to the clip's
// first frame, waits over the next ticks for every camera to decode it, then
// returns to the seed frame and runs.

#include "app_context.h"
#include "gui/gui_keypoints.h"   // reprojection() (triangulate), reproject_3d_to_cam()
#include "gui/tracktail_window.h"
#include "tracktail_server_client.h"
#include "utils.h"               // seek_all_cameras

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

// A request whose frames are still being decoded into the display buffer.
struct TracktailPendingRun {
    bool active = false;
    int anchor = 0;            // seed frame
    int first = 0, last = 0;   // clip, absolute frame numbers
    bool backwards = false;    // request settings, frozen while waiting
    int n_keep = 0;
    std::chrono::steady_clock::time_point deadline;
};

// Server state. One instance in main(), outlives every project.
struct TracktailRuntime {
    TracktailServerState server;
    TracktailPendingRun pending;
};

namespace tracktail_actions_detail {

// Ring slot holding camera `cam`'s frame `frame`, or -1 if it isn't staged.
// Slots marked available_to_write are the decoder's to overwrite.
inline int find_staged_slot(const AppContext &ctx, int cam, int frame) {
    auto *scene = ctx.scene;
    if (cam < 0 || cam >= (int)scene->num_cams) return -1;
    for (int s = 0; s < (int)scene->size_of_buffer; ++s) {
        auto &pb = scene->display_buffer[cam][s];
        if (pb.frame && pb.frame_number.load() == frame &&
            !pb.available_to_write.load())
            return s;
    }
    return -1;
}

// (camera, frame) pairs in [first, last] that are not staged.
inline int count_unstaged(const AppContext &ctx, int first, int last) {
    int missing = 0;
    for (int c = 0; c < (int)ctx.scene->num_cams; ++c)
        for (int f = first; f <= last; ++f)
            if (find_staged_slot(ctx, c, f) < 0) missing++;
    return missing;
}

// Host pointer to camera `cam`'s frame `frame`, or nullptr if that exact
// frame isn't staged. GPU-resident buffers are copied down into `scratch`,
// which the caller keeps alive for as long as the pointers are used.
inline const uint8_t *pull_frame_host(AppContext &ctx, int cam, int frame,
                                      std::deque<std::vector<uint8_t>> &scratch) {
    auto *scene = ctx.scene;
    const int slot = find_staged_slot(ctx, cam, frame);
    if (slot < 0) return nullptr;
    auto &pb = scene->display_buffer[cam][slot];
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

// Put the paused view on `frame`: a pause_selected offset when it is in the
// buffer (the slot at read_head holds to_display_frame_number), else a seek.
inline void show_frame(AppContext &ctx, int frame) {
    auto &ps = ctx.ps;
    const int off = frame - ps.to_display_frame_number;
    if (off >= 0 && off < (int)ctx.scene->size_of_buffer &&
        find_staged_slot(ctx, 0, frame) >= 0) {
        ps.pause_selected = off;
    } else {
        seek_all_cameras(ctx.scene, frame, ctx.dc_context->video_fps, ps, true);
        ps.pause_seeked = true;
        for (auto &[name, flag] : window_need_decoding) flag.store(true);
    }
    ctx.current_frame_num = frame;
}

// Write one predicted 3D point into `fa` (3D + reprojected 2D per camera).
// A keypoint placed by hand in any camera is left alone, and cameras where it
// is marked occluded keep that, unless `overwrite_manual`. Returns false when
// the keypoint was skipped.
inline bool write_prediction(FrameAnnotation &fa, int k, const Eigen::Vector3d &X,
                             AppContext &ctx, bool overwrite_manual) {
    if (k < 0 || k >= (int)fa.kp3d.size()) return false;
    if (!overwrite_manual)
        for (const auto &cam : fa.cameras)
            if (k < (int)cam.keypoints.size() && cam.keypoints[k].is_manual() &&
                keypoint2d_assessed(cam.keypoints[k]))
                return false;
    fa.kp3d[k].x = X(0);
    fa.kp3d[k].y = X(1);
    fa.kp3d[k].z = X(2);
    fa.kp3d[k].set_predicted();  // tracker output, awaiting review
    const int num_cams = (int)ctx.scene->num_cams;
    for (int v = 0; v < num_cams && v < (int)ctx.pm.camera_params.size(); ++v) {
        if (v >= (int)fa.cameras.size()) continue;
        if (k >= (int)fa.cameras[v].keypoints.size()) continue;
        auto &kp = fa.cameras[v].keypoints[k];
        if (kp.is_occluded() && !overwrite_manual) continue;
        double px, py;
        if (reproject_3d_to_cam(X, ctx.pm.camera_params[v],
                                (int)ctx.scene->image_width[v],
                                (int)ctx.scene->image_height[v], px, py)) {
            kp.x = px;
            kp.y = py;
            kp.vis = Keypoint2D::Vis::Unknown;
            kp.set_predicted();
            kp.reprojected = true;
        }
    }
    return true;
}

// Clip for one request, as offsets from the seed frame: the seed plus N
// predictions on one side, and one more frame when the count would be odd
// (the server wants an even one) -- past the predictions if the video has
// it, else from the other side. N is shortened at the start/end of the video
// and to fit `max_frames`. Returns false when there is nothing to predict.
inline bool plan_clip(int anchor, int last_frame, int n_req, bool backwards,
                      int max_frames, int &first_off, int &last_off,
                      int &n_keep) {
    const int after = last_frame - anchor, before = anchor;
    const int side = backwards ? before : after;
    const int other = backwards ? after : before;
    n_keep = std::clamp(n_req, 1, std::max(1, max_frames - 1));
    n_keep = std::min(n_keep, side);
    int span_side = n_keep, span_other = 0;
    if ((n_keep + 1) % 2 != 0) {
        if (n_keep + 2 > max_frames) { n_keep--; span_side--; }
        else if (side > n_keep) span_side++;
        else if (other > 0) span_other++;
        else { n_keep--; span_side--; }
    }
    first_off = backwards ? -span_side : -span_other;
    last_off = backwards ? span_other : span_side;
    return n_keep >= 1;
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
            if (fa.kp3d[k].exist) n++;
        return n;
    };
    if (count_3d() == 0 && st.auto_triangulate) {
        // Same call as pressing T in the Labeling Tool.
        reprojection(fa, &ctx.skeleton, ctx.pm.camera_params, ctx.scene);
        printf("[tracktail] Triangulated frame %d (animal id %d) for the seed\n",
               ctx.current_frame_num, instance_id);
    }
    for (int k = 0; k < (int)fa.kp3d.size() && k < ctx.skeleton.num_nodes; ++k) {
        if (!fa.kp3d[k].exist) continue;
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

    // ── A request waiting for its frames: when they are all decoded, go back
    // to the seed frame and run it from there (it replans the same clip).
    bool resuming = false;
    if (rt.pending.active) {
        TracktailPendingRun &p = rt.pending;
        st.forward_requested = false;
        if (!st.staging) {  // panel was reset (project closed)
            p.active = false;
            return;
        }
        std::string cancel;
        if (!scene || !ctx.ps.video_loaded) cancel = "videos were unloaded";
        else if (ctx.ps.play_video) cancel = "playback started";
        else if (std::chrono::steady_clock::now() > p.deadline)
            cancel = "timed out loading frames " + std::to_string(p.first) +
                     ".." + std::to_string(p.last);
        if (!cancel.empty()) {
            p.active = false;
            st.staging = false;
            st.last_result = "tracktail cancelled: " + cancel;
            st.last_result_ok = false;
            ctx.toasts.push(st.last_result, Toast::Warning, 4.0f);
            printf("[tracktail] %s\n", st.last_result.c_str());
            if (!ctx.ps.play_video && ctx.ps.video_loaded) show_frame(ctx, p.anchor);
            return;
        }
        const int missing = count_unstaged(ctx, p.first, p.last);
        if (missing > 0) {
            st.staging_msg = "Loading frames " + std::to_string(p.first) + ".." +
                             std::to_string(p.last) + " (" +
                             std::to_string(missing) + " images left)";
            return;
        }
        p.active = false;
        st.staging = false;
        show_frame(ctx, p.anchor);
        resuming = true;
    } else {
        if (!st.forward_requested) return;
        st.forward_requested = false;
    }

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
    const int anchor = ctx.current_frame_num;
    auto frame_at = [&](int offset) -> FrameAnnotation & {
        return get_or_create_frame(ctx.annotations, (u32)(anchor + offset),
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
    // Only the frames needed are sent: the seed + n_keep on one side, rounded
    // up to an even count, at most the model's n_frames. The seed is at t=0
    // forward and at the end of the clip backwards; query_times says which.
    const bool backwards = resuming ? rt.pending.backwards : st.predict_backwards;
    const int n_req = resuming ? rt.pending.n_keep : st.server_n_keep;
    const int dc_total = ctx.dc_context ? ctx.dc_context->total_num_frame : 0;
    const int last_frame = dc_total > 0 && dc_total < INT_MAX ? dc_total - 1 : INT_MAX;
    int first_off = 0, last_off = 0, n_keep = 0;
    if (!plan_clip(anchor, last_frame, n_req, backwards,
                   std::min(rt.server.n_frames, (int)scene->size_of_buffer),
                   first_off, last_off, n_keep)) {
        st.last_result = backwards ? "No frames before the current frame"
                                   : "No frames after the current frame";
        st.last_result_ok = false;
        ctx.toasts.push(st.last_result, Toast::Warning, 3.0f);
        return;
    }
    if (n_keep < n_req)
        printf("[tracktail/server] Keeping %d of the %d frames asked for (model "
               "chunk / start or end of the video)\n", n_keep, n_req);
    const int T = last_off - first_off + 1, query_t = -first_off;
    const int first = anchor + first_off;

    // Not all decoded yet: seek to the clip's first frame and come back when
    // every camera has it (see the top of this function).
    if (count_unstaged(ctx, first, first + T - 1) > 0) {
        if (resuming) {
            st.last_result = "tracktail: frames went missing while loading";
            st.last_result_ok = false;
            ctx.toasts.pushError(st.last_result);
            return;
        }
        printf("[tracktail] Loading frames %d..%d before predicting from %d\n",
               first, first + T - 1, anchor);
        ctx.ps.play_video = false;
        seek_all_cameras(scene, first, ctx.dc_context->video_fps, ctx.ps, true);
        ctx.ps.pause_seeked = true;
        for (auto &[name, flag] : window_need_decoding) flag.store(true);
        rt.pending = {true, anchor, first, first + T - 1, backwards, n_req,
                      std::chrono::steady_clock::now() + std::chrono::seconds(20)};
        st.staging = true;
        st.staging_msg = "Loading frames " + std::to_string(first) + ".." +
                         std::to_string(first + T - 1) + "...";
        return;
    }

    std::deque<std::vector<uint8_t>> scratch;
    std::vector<const uint8_t *> frames((size_t)num_cams * T, nullptr);
    int missing = 0;
    for (int c = 0; c < num_cams; ++c)
        for (int t = 0; t < T; ++t) {
            frames[c * T + t] = pull_frame_host(ctx, c, first + t, scratch);
            if (!frames[c * T + t]) missing++;
        }
    if (missing) {
        st.last_result = "tracktail: " + std::to_string(missing) +
                         " (camera, frame) images could not be read";
        st.last_result_ok = false;
        ctx.toasts.pushError(st.last_result);
        return;
    }

    printf("[tracktail/server] Sending %d cams x %d frames [%d..%d] (seed %d at "
           "t=%d) x %d queries (animal id %d) to %s\n", num_cams, T, first,
           first + T - 1, anchor, query_t, (int)seed.size(), instance_id,
           rt.server.url.c_str());
    auto t_start = std::chrono::steady_clock::now();
    TracktailChunkResult chunk = tracktail_server_predict_chunk(
        rt.server, frames, widths, heights, pm.camera_params, seed,
        query_t, cam_names, T);
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
    // Write offsets 1..n_keep (or -1..-n_keep); the seed frame stays as it is.
    int kept_manual = 0;
    for (int i = 1; i <= n_keep; ++i) {
        const int off = backwards ? -i : i, t = off - first_off;
        if (t < 0 || t >= (int)chunk.kp3d.size()) continue;
        FrameAnnotation &fa = frame_at(off);
        for (int q = 0; q < (int)seed_node_idx.size() &&
                        q < (int)chunk.kp3d[t].size(); ++q)
            if (!write_prediction(fa, seed_node_idx[q], chunk.kp3d[t][q], ctx,
                                  st.overwrite_manual))
                kept_manual++;
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf),
                  "OK: %d %s frame%s x %d kp in %.0f ms (server total %.0f ms)",
                  n_keep, backwards ? "previous" : "future",
                  n_keep == 1 ? "" : "s", (int)seed.size(), ms,
                  rt.server.last_total_ms);
    st.server_status = buf;
    st.last_result = buf;
    if (kept_manual)
        st.last_result += ", kept " + std::to_string(kept_manual) +
                          " hand-placed keypoint" + (kept_manual == 1 ? "" : "s");
    st.last_result_ok = true;
    ctx.toasts.pushSuccess(std::string("tracktail: ") + (backwards ? "-" : "+") +
                           std::to_string(n_keep) +
                           " frames (animal id " +
                           std::to_string(instance_id) + ")");
    printf("[tracktail/server] %s\n", buf);
}
