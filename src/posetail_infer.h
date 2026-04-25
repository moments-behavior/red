#pragma once
// posetail_infer.h — Multi-camera temporal pose tracker (PoseTail / TrackerEncoder)
//
// Wraps the exported tracker_encoder.onnx. Given:
//   - 16 consecutive frames per camera (RGBA from red's display buffer)
//   - the current-frame 3D keypoints (from JARVIS Predict + Triangulate or
//     manual labels)
//   - calibration (K, R|t, distortion) for every camera
// produces predicted 3D positions for every keypoint at every frame in the
// 16-frame chunk. Sliding-window across multiple chunks gives forward
// tracking for arbitrary horizons.
//
// Optional dependency: compile with -DRED_HAS_ONNXRUNTIME. Without ONNX
// Runtime, all functions are stubs that return false.
//
// Design mirrors jarvis_inference.h: header-only, plain Eigen + ONNX Runtime.

#include "annotation.h"
#include "camera.h"
#include "red_math.h"
#include "render.h"
#include "skeleton.h"
#include "types.h"
#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#ifdef RED_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// ── Fixed model dimensions (from ONNX_INTERFACE.md) ──
namespace posetail_detail {
static constexpr int T_CHUNK = 16;     // frames per chunk
static constexpr int CROP_SIZE = 256;  // model input H == W
static constexpr int CROP_PAD = 20;    // bbox pad before expanding to CROP_SIZE
}  // namespace posetail_detail

struct PosetailState {
    bool loaded = false;
#ifdef RED_HAS_ONNXRUNTIME
    bool available = true;
#else
    bool available = false;
#endif
    std::string status;
    std::string model_path;
    std::string backend;  // "CUDA" or "CPU"

#ifdef RED_HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
#endif

    // Timing for the most recent chunk inference (milliseconds).
    float last_total_ms = 0.0f;
    float last_inference_ms = 0.0f;

    // Last chunk's confidence/visibility — useful for masking out frames
    // where the tracker isn't sure. Layout [T_CHUNK][N].
    std::vector<std::vector<float>> last_vis;
    std::vector<std::vector<float>> last_conf;
};

// One chunk of predictions: T_CHUNK frames × N query points × 3D.
struct PosetailChunkResult {
    bool ok = false;
    std::string error;
    // [T_CHUNK][N] 3D positions in world space.
    std::vector<std::vector<Eigen::Vector3d>> kp3d;
    // [T_CHUNK][N] visibility ∈ [0, 1].
    std::vector<std::vector<float>> vis;
    // [T_CHUNK][N] 2D-localization confidence ∈ [0, 1].
    std::vector<std::vector<float>> conf;
};

// ─────────────────────────────────────────────────────────────────────────────
// Initialization / cleanup
// ─────────────────────────────────────────────────────────────────────────────

inline bool posetail_init(PosetailState &s, const std::string &onnx_path,
                          bool force_cpu = false, int gpu_id = 0) {
    s.loaded = false;
    s.model_path = onnx_path;
    s.backend = "none";

#ifdef RED_HAS_ONNXRUNTIME
    try {
        if (!s.env) {
            s.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING,
                                               "red_posetail");
        }

        Ort::SessionOptions opts;
        // Disable graph optimization. Some Reshape ops in this ONNX export
        // produce garbage int64 shape tensors after the optimizer fuses
        // them — keeping the graph as-emitted-by-the-exporter is safer.
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        opts.SetIntraOpNumThreads(std::max(1u,
            std::thread::hardware_concurrency()));

        s.backend = "CPU";
#ifndef __APPLE__
        if (!force_cpu) {
            try {
                OrtCUDAProviderOptions cuda_opts{};
                // Let the caller pick which GPU. Default 0 matches the main
                // render context; for multi-GPU workstations the user can
                // direct PoseTail to an idle card (e.g. GPU 1) where the
                // display buffer and NvDecoder aren't competing for VRAM.
                cuda_opts.device_id = gpu_id;
                cuda_opts.arena_extend_strategy = 1;  // kSameAsRequested
                // No hard gpu_mem_limit — let the arena grow to the model's
                // actual need (attention MatMul wants ~4 GB for this setup).
                cuda_opts.cudnn_conv_algo_search =
                    OrtCudnnConvAlgoSearchHeuristic;
                cuda_opts.do_copy_in_default_stream = 1;
                opts.AppendExecutionProvider_CUDA(cuda_opts);
                s.backend = "CUDA:" + std::to_string(gpu_id);
            } catch (const Ort::Exception &e) {
                fprintf(stderr,
                        "[PoseTail] CUDA EP unavailable, using CPU: %s\n",
                        e.what());
            }
        }
#else
        (void)force_cpu; (void)gpu_id;
#endif
        s.session =
            std::make_unique<Ort::Session>(*s.env, onnx_path.c_str(), opts);
        s.loaded = true;
        s.status = "PoseTail loaded (" + s.backend + ")";
        return true;
    } catch (const Ort::Exception &e) {
        s.status = std::string("PoseTail ONNX load error: ") + e.what();
        s.loaded = false;
        return false;
    } catch (const std::exception &e) {
        s.status = std::string("PoseTail load error: ") + e.what();
        s.loaded = false;
        return false;
    }
#else
    (void)onnx_path; (void)force_cpu;
    s.status = "ONNX Runtime not available";
    return false;
#endif
}

inline void posetail_cleanup(PosetailState &s) {
#ifdef RED_HAS_ONNXRUNTIME
    s.session.reset();
    s.loaded = false;
#endif
    (void)s;
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — crop window per camera, frame preprocessing
// ─────────────────────────────────────────────────────────────────────────────

namespace posetail_detail {

// One per-camera crop window, in image pixel coordinates.
// The model receives the crop resized to CROP_SIZE × CROP_SIZE, plus the
// crop's top-left as cam_offset, plus K scaled by CROP_SIZE/crop_w in fx,cx
// and CROP_SIZE/crop_h in fy,cy.
struct CropBox {
    int x0, y0, w, h;
    Eigen::Matrix3d K_scaled;  // K' = S · K with S = diag(sx, sy, 1)
    Eigen::Vector2f offset;    // [x0, y0] for the model's cam_offset input
};

// Project all 3D queries onto a camera, build a crop box that covers the
// projected bbox + padding, expand to at least CROP_SIZE on each side,
// and clamp to the image rectangle. If clamping shrinks below CROP_SIZE the
// model still works but loses some context — we keep going.
inline CropBox compute_crop_box(const std::vector<Eigen::Vector3d> &queries_3d,
                                const CameraParams &cam, int img_w, int img_h) {
    CropBox box;

    // Project queries to image-space pixels.
    double xmin = (double)img_w, ymin = (double)img_h;
    double xmax = 0.0, ymax = 0.0;
    int n_in = 0;
    for (const auto &X : queries_3d) {
        Eigen::Vector2d p = red_math::projectPoint(
            X, cam.rvec, cam.tvec, cam.k, cam.dist_coeffs);
        if (!std::isfinite(p(0)) || !std::isfinite(p(1))) continue;
        if (p(0) < 0 || p(0) >= img_w || p(1) < 0 || p(1) >= img_h) {
            // Project still counts toward the bbox so a partially-visible
            // animal keeps its center-of-mass in the crop.
        }
        xmin = std::min(xmin, p(0));
        ymin = std::min(ymin, p(1));
        xmax = std::max(xmax, p(0));
        ymax = std::max(ymax, p(1));
        n_in++;
    }
    if (n_in == 0) {
        // No valid projection — fall back to image center.
        xmin = ymin = 0;
        xmax = (double)img_w;
        ymax = (double)img_h;
    }

    // Pad and expand to ≥ CROP_SIZE on each side.
    double cx = 0.5 * (xmin + xmax);
    double cy = 0.5 * (ymin + ymax);
    double half = std::max(0.5 * (xmax - xmin), 0.5 * (ymax - ymin)) + CROP_PAD;
    double crop_dim = std::max((double)CROP_SIZE, 2.0 * half);

    // Centered square crop, clamped to image bounds.
    double x0 = cx - crop_dim * 0.5;
    double y0 = cy - crop_dim * 0.5;
    double x1 = cx + crop_dim * 0.5;
    double y1 = cy + crop_dim * 0.5;
    if (x0 < 0) { x1 -= x0; x0 = 0; }
    if (y0 < 0) { y1 -= y0; y0 = 0; }
    if (x1 > img_w) { x0 -= (x1 - img_w); x1 = img_w; }
    if (y1 > img_h) { y0 -= (y1 - img_h); y1 = img_h; }
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > img_w) x1 = img_w;
    if (y1 > img_h) y1 = img_h;

    box.x0 = (int)std::round(x0);
    box.y0 = (int)std::round(y0);
    box.w = std::max(1, (int)std::round(x1 - x0));
    box.h = std::max(1, (int)std::round(y1 - y0));

    // Scale K so projection produces pixels inside the resized 256-crop.
    // Original projection p in image space → after subtracting (x0, y0) the
    // pixel is in crop space → after scaling by 256/crop_dim the pixel is in
    // model space. We bake the scale into K (model multiplies internally:
    // K_scaled[*, 2] - cam_offset = pixel in 256 crop). The offset is
    // handled by the cam_offset input separately.
    double sx = (double)CROP_SIZE / (double)box.w;
    double sy = (double)CROP_SIZE / (double)box.h;
    box.K_scaled = cam.k;
    box.K_scaled(0, 0) *= sx;
    box.K_scaled(1, 1) *= sy;
    box.K_scaled(0, 2) *= sx;
    box.K_scaled(1, 2) *= sy;
    box.offset = Eigen::Vector2f((float)box.x0 * (float)sx,
                                 (float)box.y0 * (float)sy);
    return box;
}

// Bilinear resize from RGBA (src_w × src_h) to RGB (CROP_SIZE × CROP_SIZE)
// float in [0, 1], cropped to (cx, cy, cw, ch). Output channel order is RGB.
// Result is written CHW-flat-ish but the model expects HWC, so we write HWC.
inline void crop_and_resize_rgba_to_rgb01(const uint8_t *rgba, int src_w,
                                          int src_h, int cx, int cy, int cw,
                                          int ch, float *out_hwc /*256*256*3*/) {
    const int out_w = CROP_SIZE;
    const int out_h = CROP_SIZE;
    const float sx = (float)cw / (float)out_w;
    const float sy = (float)ch / (float)out_h;
    for (int y = 0; y < out_h; ++y) {
        float fy = (y + 0.5f) * sy - 0.5f + (float)cy;
        int y0 = std::max(0, (int)std::floor(fy));
        int y1 = std::min(src_h - 1, y0 + 1);
        float wy = std::clamp(fy - y0, 0.0f, 1.0f);
        for (int x = 0; x < out_w; ++x) {
            float fx = (x + 0.5f) * sx - 0.5f + (float)cx;
            int x0 = std::max(0, (int)std::floor(fx));
            int x1 = std::min(src_w - 1, x0 + 1);
            float wx = std::clamp(fx - x0, 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) {
                float v00 = rgba[(y0 * src_w + x0) * 4 + c];
                float v01 = rgba[(y0 * src_w + x1) * 4 + c];
                float v10 = rgba[(y1 * src_w + x0) * 4 + c];
                float v11 = rgba[(y1 * src_w + x1) * 4 + c];
                float v = (1 - wy) * ((1 - wx) * v00 + wx * v01) +
                          wy * ((1 - wx) * v10 + wx * v11);
                out_hwc[(y * out_w + x) * 3 + c] = v / 255.0f;
            }
        }
    }
}

}  // namespace posetail_detail

// ─────────────────────────────────────────────────────────────────────────────
// Inference: run one 16-frame chunk
// ─────────────────────────────────────────────────────────────────────────────
//
//  frames_rgba_per_cam_per_t : flat layout [cams * T_CHUNK] of RGBA uint8
//                              buffers. frames_rgba_per_cam_per_t[c*T+t] is
//                              the t-th frame for camera c (already on host,
//                              caller is responsible for any GPU→CPU copy).
//  cam_widths / cam_heights : full-image dimensions per camera (NOT crop).
//  cams : per-camera CameraParams in the same order.
//  seed_3d : initial 3D query positions, [N, 3] in world space.
//  seed_t : frame index within the chunk where seed observations apply
//           (typically 0 for the first chunk, n_overlap-1 for subsequent).
inline PosetailChunkResult posetail_predict_chunk(
    PosetailState &s,
    const std::vector<const uint8_t *> &frames_rgba_per_cam_per_t,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const std::vector<CameraParams> &cams,
    const std::vector<Eigen::Vector3d> &seed_3d,
    int seed_t = 0) {
    PosetailChunkResult r;
#ifdef RED_HAS_ONNXRUNTIME
    using namespace posetail_detail;
    if (!s.loaded || !s.session) {
        r.error = "PoseTail not loaded";
        return r;
    }
    int num_cams = (int)cams.size();
    int N = (int)seed_3d.size();
    if (num_cams == 0 || N == 0) {
        r.error = "No cameras or queries";
        return r;
    }
    if ((int)frames_rgba_per_cam_per_t.size() != num_cams * T_CHUNK) {
        r.error = "frames buffer size mismatch (need cams*16)";
        return r;
    }
    auto t0 = std::chrono::steady_clock::now();

    // Build per-camera crop boxes from the seed 3D points.
    std::vector<CropBox> boxes(num_cams);
    for (int c = 0; c < num_cams; ++c) {
        boxes[c] = compute_crop_box(seed_3d, cams[c], cam_widths[c],
                                    cam_heights[c]);
    }

    // ── views: [cams, B=1, T, H, W, 3] float32 in [0, 1] ──
    std::vector<int64_t> views_shape = {num_cams, 1, T_CHUNK, CROP_SIZE,
                                         CROP_SIZE, 3};
    size_t views_count = (size_t)num_cams * T_CHUNK * CROP_SIZE * CROP_SIZE * 3;
    std::vector<float> views(views_count);
    for (int c = 0; c < num_cams; ++c) {
        const CropBox &box = boxes[c];
        for (int t = 0; t < T_CHUNK; ++t) {
            const uint8_t *rgba = frames_rgba_per_cam_per_t[c * T_CHUNK + t];
            float *dst = views.data() +
                         ((size_t)c * T_CHUNK + t) * CROP_SIZE * CROP_SIZE * 3;
            if (rgba) {
                crop_and_resize_rgba_to_rgb01(rgba, cam_widths[c],
                                              cam_heights[c], box.x0, box.y0,
                                              box.w, box.h, dst);
            } else {
                std::memset(dst, 0, CROP_SIZE * CROP_SIZE * 3 * sizeof(float));
            }
        }
    }

    // ── coords: [B=1, N, 3] float32 ──
    std::vector<int64_t> coords_shape = {1, N, 3};
    std::vector<float> coords((size_t)N * 3);
    for (int n = 0; n < N; ++n) {
        coords[n * 3 + 0] = (float)seed_3d[n](0);
        coords[n * 3 + 1] = (float)seed_3d[n](1);
        coords[n * 3 + 2] = (float)seed_3d[n](2);
    }

    // ── query_times: [B=1, N] int64 ──
    std::vector<int64_t> qt_shape = {1, N};
    std::vector<int64_t> qtimes(N, (int64_t)seed_t);

    // ── cam_ext: [cams, 4, 4] float32 (world→camera) ──
    std::vector<int64_t> ext_shape = {num_cams, 4, 4};
    std::vector<float> ext((size_t)num_cams * 16, 0.0f);
    // ── cam_mat: [cams, 3, 3] float32 (scaled K) ──
    std::vector<int64_t> mat_shape = {num_cams, 3, 3};
    std::vector<float> mat((size_t)num_cams * 9, 0.0f);
    // ── cam_dist: [cams, 5] float32 ──
    std::vector<int64_t> dist_shape = {num_cams, 5};
    std::vector<float> dist((size_t)num_cams * 5, 0.0f);
    // ── cam_offset: [cams, 2] float32 ──
    std::vector<int64_t> off_shape = {num_cams, 2};
    std::vector<float> offset((size_t)num_cams * 2, 0.0f);

    for (int c = 0; c < num_cams; ++c) {
        const auto &cam = cams[c];
        const auto &box = boxes[c];
        // Build [R | t; 0 0 0 1] row-major
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                ext[c * 16 + i * 4 + j] = (float)cam.r(i, j);
            }
            ext[c * 16 + i * 4 + 3] = (float)cam.tvec(i);
        }
        ext[c * 16 + 12] = 0.0f;
        ext[c * 16 + 13] = 0.0f;
        ext[c * 16 + 14] = 0.0f;
        ext[c * 16 + 15] = 1.0f;

        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                mat[c * 9 + i * 3 + j] = (float)box.K_scaled(i, j);

        for (int i = 0; i < 5; ++i) dist[c * 5 + i] = (float)cam.dist_coeffs(i);
        offset[c * 2 + 0] = box.offset(0);
        offset[c * 2 + 1] = box.offset(1);
    }

    // ── Run session ──
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value views_t =
        Ort::Value::CreateTensor<float>(mem, views.data(), views.size(),
                                         views_shape.data(), views_shape.size());
    Ort::Value coords_t =
        Ort::Value::CreateTensor<float>(mem, coords.data(), coords.size(),
                                         coords_shape.data(),
                                         coords_shape.size());
    Ort::Value qt_t = Ort::Value::CreateTensor<int64_t>(
        mem, qtimes.data(), qtimes.size(), qt_shape.data(), qt_shape.size());
    Ort::Value ext_t = Ort::Value::CreateTensor<float>(
        mem, ext.data(), ext.size(), ext_shape.data(), ext_shape.size());
    Ort::Value mat_t = Ort::Value::CreateTensor<float>(
        mem, mat.data(), mat.size(), mat_shape.data(), mat_shape.size());
    Ort::Value dist_t = Ort::Value::CreateTensor<float>(
        mem, dist.data(), dist.size(), dist_shape.data(), dist_shape.size());
    Ort::Value off_t = Ort::Value::CreateTensor<float>(
        mem, offset.data(), offset.size(), off_shape.data(), off_shape.size());

    const char *input_names[] = {"views",   "coords",   "query_times",
                                  "cam_ext", "cam_mat",  "cam_dist",
                                  "cam_offset"};
    Ort::Value inputs[] = {std::move(views_t), std::move(coords_t),
                            std::move(qt_t),    std::move(ext_t),
                            std::move(mat_t),   std::move(dist_t),
                            std::move(off_t)};

    // We only need: coords_pred (idx 0), vis_pred (idx 8), conf_pred (idx 9).
    const char *output_names[] = {"coords_pred", "vis_pred", "conf_pred"};

    // --- Debug dump of the inputs we're about to send ---
    {
        auto finite_all = [&](const std::vector<float> &v, const char *nm) {
            bool ok = true;
            float mn = std::numeric_limits<float>::max();
            float mx = std::numeric_limits<float>::lowest();
            for (float x : v) {
                if (!std::isfinite(x)) {
                    ok = false;
                    break;
                }
                if (x < mn) mn = x;
                if (x > mx) mx = x;
            }
            fprintf(stderr, "[PoseTail]   %s: size=%zu finite=%d range=[%g,%g]\n",
                    nm, v.size(), (int)ok, ok ? mn : 0.0, ok ? mx : 0.0);
            return ok;
        };
        fprintf(stderr,
                "[PoseTail] feed: cams=%d T=%d N=%d seed_t=%d CROP=%d\n",
                num_cams, T_CHUNK, N, seed_t, CROP_SIZE);
        finite_all(views, "views");
        finite_all(coords, "coords");
        fprintf(stderr, "[PoseTail]   qtimes[0..%d]=[", std::min(N, 5));
        for (int i = 0; i < std::min(N, 5); ++i)
            fprintf(stderr, "%lld%s", (long long)qtimes[i],
                    i + 1 < std::min(N, 5) ? "," : "");
        fprintf(stderr, "]\n");
        finite_all(ext, "cam_ext");
        finite_all(mat, "cam_mat");
        finite_all(dist, "cam_dist");
        finite_all(offset, "cam_offset");
        // Print first crop box and first cam's K+ext to eyeball
        fprintf(stderr, "[PoseTail]   cam[0] box=[%d,%d,%d,%d]\n",
                boxes[0].x0, boxes[0].y0, boxes[0].w, boxes[0].h);
        fprintf(stderr, "[PoseTail]   cam[0] K_scaled=[[%g,%g,%g],[%g,%g,%g]]\n",
                mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]);
        fprintf(stderr, "[PoseTail]   cam[0] ext[0..12]=[%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g,%g]\n",
                ext[0], ext[1], ext[2], ext[3], ext[4], ext[5],
                ext[6], ext[7], ext[8], ext[9], ext[10], ext[11]);
        fprintf(stderr, "[PoseTail]   seed[0]=(%g,%g,%g) seed[N-1]=(%g,%g,%g)\n",
                coords[0], coords[1], coords[2],
                coords[(N - 1) * 3 + 0], coords[(N - 1) * 3 + 1],
                coords[(N - 1) * 3 + 2]);
    }

    auto t1 = std::chrono::steady_clock::now();
    std::vector<Ort::Value> out;
    try {
        out = s.session->Run(Ort::RunOptions{nullptr}, input_names, inputs, 7,
                              output_names, 3);
    } catch (const Ort::Exception &e) {
        r.error = std::string("PoseTail forward failed: ") + e.what();
        return r;
    }
    auto t2 = std::chrono::steady_clock::now();
    s.last_inference_ms =
        std::chrono::duration<float, std::milli>(t2 - t1).count();

    // coords_pred: [B=1, T, N, 3]
    const float *coords_data = out[0].GetTensorData<float>();
    const float *vis_data = out[1].GetTensorData<float>();
    const float *conf_data = out[2].GetTensorData<float>();

    r.kp3d.assign(T_CHUNK, std::vector<Eigen::Vector3d>(N));
    r.vis.assign(T_CHUNK, std::vector<float>(N, 0.0f));
    r.conf.assign(T_CHUNK, std::vector<float>(N, 0.0f));
    for (int t = 0; t < T_CHUNK; ++t) {
        for (int n = 0; n < N; ++n) {
            const float *p = coords_data + ((t * N) + n) * 3;
            r.kp3d[t][n] = Eigen::Vector3d(p[0], p[1], p[2]);
            r.vis[t][n] = vis_data[(t * N + n) * 1];
            r.conf[t][n] = conf_data[(t * N + n) * 1];
        }
    }

    auto t3 = std::chrono::steady_clock::now();
    s.last_total_ms = std::chrono::duration<float, std::milli>(t3 - t0).count();
    s.last_vis = r.vis;
    s.last_conf = r.conf;
    r.ok = true;
    return r;
#else
    (void)s; (void)frames_rgba_per_cam_per_t; (void)cam_widths;
    (void)cam_heights; (void)cams; (void)seed_3d; (void)seed_t;
    r.error = "ONNX Runtime not available";
    return r;
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-chunk forward inference: predict the next `n_forward` frames from the
// current frame's 3D keypoints by chaining T_CHUNK=16 frame chunks with
// overlap. Returns `n_forward` rows of 3D + visibility per query point.
// ─────────────────────────────────────────────────────────────────────────────
struct PosetailForwardResult {
    bool ok = false;
    std::string error;
    // [n_forward][N] 3D positions in world space, indexed from current+1.
    std::vector<std::vector<Eigen::Vector3d>> kp3d;
    std::vector<std::vector<float>> vis;
    std::vector<std::vector<float>> conf;
};

// `pull_frame(camera_idx, frame_offset)` should return an RGBA uint8 buffer
// for the requested (camera, frame_offset) where frame_offset is 0…
// (T_CHUNK + n_forward) measured forward from the current frame. Returning
// nullptr indicates the frame isn't available — that camera/time slot will
// be zero-filled.
template <typename PullFrameFn>
PosetailForwardResult posetail_forward(
    PosetailState &s, const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const std::vector<CameraParams> &cams,
    const std::vector<Eigen::Vector3d> &seed_3d_at_current,
    int n_forward, PullFrameFn pull_frame, int n_overlap = 2) {
    using namespace posetail_detail;
    PosetailForwardResult out;
    if (n_forward <= 0) {
        out.ok = true;
        return out;
    }
    if (!s.loaded) {
        out.error = "PoseTail not loaded";
        return out;
    }
    int num_cams = (int)cams.size();
    int N = (int)seed_3d_at_current.size();
    if (num_cams == 0 || N == 0) {
        out.error = "No cameras or queries";
        return out;
    }

    out.kp3d.reserve(n_forward);
    out.vis.reserve(n_forward);
    out.conf.reserve(n_forward);

    // Frame index relative to "current". We use DISJOINT chunks: chunk k
    // covers frame offsets [k*T .. k*T + T - 1], the seed for chunk k+1 is
    // the previous chunk's last predicted 3D (frame offset k*T + T - 1),
    // and seed_t is always 0.
    //
    // The Python reference uses a 2-frame overlap with seed_t = n_overlap-1
    // on subsequent chunks, but the exported tracker_encoder.onnx has a
    // graph bug that produces a garbage Reshape shape whenever qtimes != 0
    // (Reshape node "node_view_341" sees float bits reinterpreted as int64).
    // Disjoint chunks always call the model with qtimes = 0, which is the
    // only code path the export seems to handle cleanly. Downside: a tiny
    // discontinuity at chunk boundaries because we drop the 2-frame overlap
    // that smoothed transitions; acceptable for short +N predictions.
    (void)n_overlap;
    int chunk_start = 0;
    int produced = 0;
    std::vector<Eigen::Vector3d> seed = seed_3d_at_current;

    while (produced < n_forward) {
        std::vector<const uint8_t *> frames(num_cams * T_CHUNK, nullptr);
        for (int c = 0; c < num_cams; ++c) {
            for (int t = 0; t < T_CHUNK; ++t) {
                frames[c * T_CHUNK + t] = pull_frame(c, chunk_start + t);
            }
        }

        PosetailChunkResult chunk = posetail_predict_chunk(
            s, frames, cam_widths, cam_heights, cams, seed, /*seed_t=*/0);
        if (!chunk.ok) {
            out.error = chunk.error;
            return out;
        }

        // First chunk: skip t=0 (the seed itself); write t=1..T-1.
        // Subsequent chunks: write t=0..T-1 (no overlap to discard).
        int t_begin = (chunk_start == 0) ? 1 : 0;
        for (int t = t_begin; t < T_CHUNK && produced < n_forward; ++t) {
            out.kp3d.push_back(chunk.kp3d[t]);
            out.vis.push_back(chunk.vis[t]);
            out.conf.push_back(chunk.conf[t]);
            produced++;
        }
        if (produced >= n_forward) break;

        seed = chunk.kp3d[T_CHUNK - 1];
        chunk_start += T_CHUNK;  // disjoint: no overlap
    }

    out.ok = true;
    return out;
}
