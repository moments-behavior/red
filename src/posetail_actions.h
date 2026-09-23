#pragma once
// posetail_actions.h — main-loop side of the PoseTail Tracker panel.
//
// Consumes the request flags set by DrawPosetailWindow (gui/posetail_window.h)
// and owns the model / HTTP state. Called once per render tick from red.cpp:
//
//     posetail_handle_requests(win.posetail, posetail_rt, ctx);
//
// Seed = the active animal's triangulated 3D keypoints on the current frame
// (2D labels are triangulated first when the panel's checkbox is on). The
// prediction for frames [current+1 .. current+N] is written into that same
// animal's FrameAnnotation (matched by instance_id, created if missing) as 3D
// + reprojected 2D marked LabelSource::Predicted. Other animals in those
// frames are untouched.

#include "app_context.h"
#include "gui/gui_keypoints.h"   // reprojection() (triangulate), reproject_3d_to_cam()
#include "gui/posetail_window.h"
#include "posetail_infer.h"
#include "posetail_server_client.h"

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <string>
#include <vector>

// Model + server state. One instance in main(), outlives every project.
struct PosetailRuntime {
    PosetailState local;
    PosetailServerState server;
};

namespace posetail_actions_detail {

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
#ifdef POSETAIL_HAS_CUDA
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
inline bool collect_seed(PosetailWindowState &st, AppContext &ctx,
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
        printf("[PoseTail] Triangulated frame %d (animal id %d) for the seed\n",
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

}  // namespace posetail_actions_detail

inline void posetail_handle_requests(PosetailWindowState &st,
                                     PosetailRuntime &rt, AppContext &ctx) {
    using namespace posetail_actions_detail;
    auto *scene = ctx.scene;
    auto &pm = ctx.pm;

    // ── Server: Probe /info ──
    if (st.server_probe_requested) {
        st.server_probe_requested = false;
        rt.server.url = st.server_url;
        bool ok = posetail_server_probe(rt.server);
        st.server_status = rt.server.status;
        if (ok) {
            st.server_n_frames = rt.server.n_frames;
            st.server_image_size = rt.server.image_size;
            st.server_device = rt.server.device;
            st.server_mode_3d = rt.server.mode_3d;
        }
        printf("[PoseTail/server] %s\n", rt.server.status.c_str());
    }

    // ── Local ONNX: Load model ──
    if (st.load_requested) {
        st.load_requested = false;
        if (st.onnx_path.empty()) {
            st.status = "No ONNX path set";
        } else {
#ifdef POSETAIL_HAS_CUDA
            // Auto-pick the GPU with the most free VRAM right now (= the one
            // NOT holding the display buffer / NvDecoder). CUDA device ids
            // are not nvidia-smi ids unless CUDA_DEVICE_ORDER=PCI_BUS_ID, so
            // the name is printed to disambiguate.
            if (!st.use_cpu) {
                int n_gpus = 0;
                cudaGetDeviceCount(&n_gpus);
                int best = st.gpu_id;
                size_t best_free = 0;
                for (int g = 0; g < n_gpus; ++g) {
                    if (cudaSetDevice(g) != cudaSuccess) continue;
                    size_t f = 0, t = 0;
                    cudaMemGetInfo(&f, &t);
                    cudaDeviceProp prop{};
                    cudaGetDeviceProperties(&prop, g);
                    if (f > best_free) { best_free = f; best = g; }
                    fprintf(stderr, "[PoseTail] CUDA device %d (%s): %.2f / %.2f GB free\n",
                            g, prop.name, f / 1e9, t / 1e9);
                }
                cudaSetDevice(0);  // restore main thread default
                if (best != st.gpu_id) {
                    fprintf(stderr, "[PoseTail] Auto-picking CUDA device %d (largest free VRAM)\n", best);
                    st.gpu_id = best;
                }
            }
#endif
            posetail_init(rt.local, st.onnx_path, /*force_cpu=*/st.use_cpu,
                          /*gpu_id=*/st.gpu_id);
            st.status = rt.local.status;
            printf("[PoseTail] %s\n", rt.local.status.c_str());
#ifdef POSETAIL_HAS_CUDA
            cudaSetDevice(0);
#endif
        }
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
        printf("[PoseTail] %s\n", why_not.c_str());
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

    // ── Server: one 16-frame chunk ──
    if (st.use_server) {
        rt.server.url = st.server_url;
        const int T = posetail_detail::T_CHUNK;

        std::deque<std::vector<uint8_t>> scratch;
        std::vector<const uint8_t *> frames((size_t)num_cams * T, nullptr);
        int missing = 0;
        for (int c = 0; c < num_cams; ++c)
            for (int t = 0; t < T; ++t) {
                frames[c * T + t] = pull_frame_host(ctx, c, t, scratch);
                if (!frames[c * T + t]) missing++;
            }
        if (missing)
            printf("[PoseTail/server] WARNING: %d of %d (cam, frame) slots not "
                   "staged; sending grey for those\n", missing, num_cams * T);

        printf("[PoseTail/server] Sending %d cams x %d frames x %d queries "
               "(animal id %d) to %s\n", num_cams, T, (int)seed.size(),
               instance_id, rt.server.url.c_str());
        auto t_start = std::chrono::steady_clock::now();
        PosetailChunkResult chunk = posetail_server_predict_chunk(
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
            printf("[PoseTail/server] FAILED: %s\n", chunk.error.c_str());
            return;
        }
        // t=0 is the seed; write t=1..n_keep.
        const int n_keep = std::clamp(st.server_n_keep, 1, T - 1);
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
        ctx.toasts.pushSuccess("PoseTail: +" + std::to_string(n_keep) +
                               " frames (animal id " +
                               std::to_string(instance_id) + ")");
        printf("[PoseTail/server] %s\n", buf);
        return;
    }

    // ── Local ONNX: chain chunks to +N ──
    if (!rt.local.loaded) {
        st.last_result = "PoseTail not loaded; click Load first";
        st.last_result_ok = false;
        ctx.toasts.push(st.last_result, Toast::Warning, 3.0f);
        return;
    }
    const int n_fwd = st.n_forward;
    // Cap queries per Run() so single-tensor allocations stay below the GPU
    // per-alloc ceiling. 24 keypoints -> 2 passes of 12 by default.
    const int max_q = std::max(1, st.max_queries);
    const int n_total = (int)seed.size();
    const int n_passes = (n_total + max_q - 1) / max_q;
    printf("[PoseTail] Using all %d cameras, %d keypoints (animal id %d) in "
           "%d pass(es) of up to %d each (backend=%s)\n",
           num_cams, n_total, instance_id, n_passes, max_q,
           rt.local.backend.c_str());

    std::deque<std::vector<uint8_t>> scratch;
    auto pull_frame = [&](int cam_idx, int frame_off) -> const uint8_t * {
        return pull_frame_host(ctx, cam_idx, frame_off, scratch);
    };

    auto t_start = std::chrono::steady_clock::now();
    bool any_failed = false;
    std::string fail_msg;
    int produced_min = INT_MAX;
    for (int p = 0; p < n_passes && !any_failed; ++p) {
        const int q0 = p * max_q;
        const int q1 = std::min(n_total, q0 + max_q);
        std::vector<Eigen::Vector3d> sub_seed(seed.begin() + q0, seed.begin() + q1);
        std::vector<int> sub_idx(seed_node_idx.begin() + q0, seed_node_idx.begin() + q1);

        scratch.clear();
        auto fwd = posetail_forward(rt.local, widths, heights, pm.camera_params,
                                    sub_seed, n_fwd, pull_frame, 2);
#ifdef POSETAIL_HAS_CUDA
        cudaSetDevice(0);
#endif
        if (!fwd.ok) {
            any_failed = true;
            fail_msg = fwd.error;
            printf("[PoseTail] Pass %d/%d failed: %s\n", p + 1, n_passes,
                   fwd.error.c_str());
            break;
        }
        const int produced = (int)fwd.kp3d.size();
        produced_min = std::min(produced_min, produced);
        for (int i = 0; i < produced; ++i) {
            FrameAnnotation &fa = future_frame(1 + i);
            for (int q = 0; q < (int)sub_idx.size() && q < (int)fwd.kp3d[i].size(); ++q)
                write_prediction(fa, sub_idx[q], fwd.kp3d[i][q], ctx);
        }
        printf("[PoseTail] Pass %d/%d: %d queries, %d frames produced\n",
               p + 1, n_passes, (int)sub_seed.size(), produced);
    }
    float ms = std::chrono::duration<float, std::milli>(
                   std::chrono::steady_clock::now() - t_start).count();
    if (any_failed) {
        st.last_result = "Forward failed: " + fail_msg;
        st.last_result_ok = false;
        ctx.toasts.pushError(st.last_result);
    } else {
        const int produced = produced_min == INT_MAX ? 0 : produced_min;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "OK: %d frames x %d keypoints in %.0f ms (%.1f ms/frame)",
                      produced, n_total, ms, produced > 0 ? ms / produced : 0.0f);
        st.last_result = buf;
        st.last_result_ok = true;
        ctx.toasts.pushSuccess("PoseTail: +" + std::to_string(produced) +
                               " frames (animal id " +
                               std::to_string(instance_id) + ")");
        printf("[PoseTail] %s\n", buf);
    }

    // Always drop the session after a Run, success or not. Reusing one
    // session across Forward clicks gave slightly different, drifting
    // predictions (cuDNN workspace / algorithm cache inside ORT's CUDA EP).
    // Reloading is ~1 s on GPU, much less than the run itself. The reload
    // is queued so the next tick does it before another Forward can fire.
    posetail_cleanup(rt.local);
    st.load_requested = true;
    st.status = "Reloading PoseTail for fresh state...";
}
