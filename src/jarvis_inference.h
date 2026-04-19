#pragma once
// jarvis_inference.h — JARVIS pose estimation via ONNX Runtime
//
// Runs CenterDetect + KeypointDetect on camera frames to predict
// 2D keypoints, then triangulates to 3D using RED's DLT.
//
// Optional dependency: compile with -DRED_HAS_ONNXRUNTIME to enable.
// Without ONNX Runtime, all functions are stubs that return false.

#include "annotation.h"
#include "camera.h"
#include "red_math.h"
#include "render.h"
#include "skeleton.h"
#include "types.h"
#include "jarvis_model_config.h"
#include "json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifdef RED_HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

// ── Types ──

struct JarvisState {
    bool loaded = false;
#ifdef RED_HAS_ONNXRUNTIME
    bool available = true;
#else
    bool available = false;
#endif

    std::string status;
    JarvisModelConfig config;

    // Timing
    float last_center_ms = 0;
    float last_keypoint_ms = 0;
    float last_triangulate_ms = 0;
    float last_total_ms = 0;

#ifdef RED_HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> center_session;
    std::unique_ptr<Ort::Session> keypoint_session;
#endif
};

// ── Preprocessing ──

namespace jarvis_detail {

// ImageNet normalization constants
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float INV_STD[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

// Bilinear resize + ImageNet normalize → CHW float32
inline std::vector<float> preprocess(const uint8_t *rgb, int src_w, int src_h,
                                      int dst_size) {
    std::vector<float> result(3 * dst_size * dst_size);
    float sx = (float)src_w / dst_size;
    float sy = (float)src_h / dst_size;

    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < dst_size; ++y) {
            float fy = (y + 0.5f) * sy - 0.5f;
            int y0 = std::max(0, (int)std::floor(fy));
            int y1 = std::min(src_h - 1, y0 + 1);
            float wy = fy - y0;
            for (int x = 0; x < dst_size; ++x) {
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = std::max(0, (int)std::floor(fx));
                int x1 = std::min(src_w - 1, x0 + 1);
                float wx = fx - x0;
                // Bilinear interpolation on the channel
                float v = (1 - wy) * ((1 - wx) * rgb[(y0 * src_w + x0) * 3 + c] +
                                       wx * rgb[(y0 * src_w + x1) * 3 + c]) +
                          wy * ((1 - wx) * rgb[(y1 * src_w + x0) * 3 + c] +
                                 wx * rgb[(y1 * src_w + x1) * 3 + c]);
                // Scale to [0,1] then normalize
                result[c * dst_size * dst_size + y * dst_size + x] =
                    (v / 255.0f - MEAN[c]) * INV_STD[c];
            }
        }
    }
    return result;
}

// Preprocess a crop from a larger image
inline std::vector<float> preprocess_crop(const uint8_t *rgb, int img_w, int img_h,
                                           int cx, int cy, int crop_size, int dst_size) {
    // Extract crop, clamp to image bounds
    int half = crop_size / 2;
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(img_w, cx + half);
    int y1 = std::min(img_h, cy + half);
    int cw = x1 - x0;
    int ch = y1 - y0;

    // Copy crop to contiguous buffer
    std::vector<uint8_t> crop(cw * ch * 3);
    for (int y = 0; y < ch; ++y) {
        const uint8_t *src_row = rgb + ((y0 + y) * img_w + x0) * 3;
        uint8_t *dst_row = crop.data() + y * cw * 3;
        std::memcpy(dst_row, src_row, cw * 3);
    }

    return preprocess(crop.data(), cw, ch, dst_size);
}

// Argmax on a 2D heatmap → (x, y, confidence)
struct HeatmapPeak { float x, y, confidence; };

inline HeatmapPeak heatmap_argmax(const float *heatmap, int h, int w) {
    int max_idx = 0;
    float max_val = heatmap[0];
    for (int i = 1; i < h * w; ++i) {
        if (heatmap[i] > max_val) {
            max_val = heatmap[i];
            max_idx = i;
        }
    }
    HeatmapPeak peak;
    peak.x = (float)(max_idx % w) * 2.0f; // 2x scale (heatmap is half resolution)
    peak.y = (float)(max_idx / w) * 2.0f;
    peak.confidence = std::min(max_val, 255.0f) / 255.0f;
    return peak;
}

} // namespace jarvis_detail

// ── Public API ──

inline bool jarvis_init(JarvisState &s, const char *center_onnx,
                         const char *keypoint_onnx,
                         const JarvisModelConfig &cfg = {}) {
#ifdef RED_HAS_ONNXRUNTIME
    s.center_session.reset();
    s.keypoint_session.reset();
    s.loaded = false;
    s.available = true;
    s.config = cfg;

    try {
        if (!s.env)
            s.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "red_jarvis");

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(4);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // On Linux with CUDA, attach the CUDA execution provider so inference
        // runs on the GPU. Falls back silently to CPU if the CUDA EP binary
        // isn't in lib/onnxruntime/lib (onnxruntime-linux-x64-gpu tarball
        // required). On macOS we stay on CPU (validated faster than CoreML
        // for this model).
        std::string backend = "CPU";
#ifndef __APPLE__
        try {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            opts.AppendExecutionProvider_CUDA(cuda_opts);
            backend = "CUDA";
        } catch (const Ort::Exception &e) {
            fprintf(stderr,
                    "[JARVIS] CUDA EP unavailable, using CPU: %s\n", e.what());
        }
#endif

        s.center_session = std::make_unique<Ort::Session>(
            *s.env, center_onnx, opts);
        s.keypoint_session = std::make_unique<Ort::Session>(
            *s.env, keypoint_onnx, opts);

        s.status = "JARVIS loaded (" + std::to_string(s.config.num_joints) +
                   " joints, " + backend + ")";
        s.loaded = true;
        return true;
    } catch (const Ort::Exception &e) {
        s.status = std::string("ONNX error: ") + e.what();
        s.loaded = false;
        return false;
    } catch (const std::exception &e) {
        s.status = std::string("Error: ") + e.what();
        s.loaded = false;
        return false;
    }
#else
    (void)center_onnx; (void)keypoint_onnx; (void)cfg;
    s.available = false;
    s.status = "ONNX Runtime not available";
    return false;
#endif
}

// Run CenterDetect on a single camera image.
// Returns center in image coordinates (Y=0 at top).
inline jarvis_detail::HeatmapPeak jarvis_detect_center(
    JarvisState &s, const uint8_t *rgb, int w, int h) {
    jarvis_detail::HeatmapPeak result = {0, 0, 0};
#ifdef RED_HAS_ONNXRUNTIME
    if (!s.loaded || !s.center_session) return result;

    int input_size = s.config.center_input_size;
    float ds_x = (float)w / input_size; // downsampling scale
    float ds_y = (float)h / input_size;

    auto input = jarvis_detail::preprocess(rgb, w, h, input_size);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> shape = {1, 3, input_size, input_size};

    // Get input/output names from model. Number of outputs varies by export:
    // CoreML ships two heatmaps (low + high res); the generic ONNX export
    // ships just one. Prefer the highest-index output (higher resolution).
    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = s.center_session->GetInputNameAllocated(0, allocator);
    size_t n_outputs = s.center_session->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_name_holders;
    std::vector<const char *> output_names;
    out_name_holders.reserve(n_outputs);
    output_names.reserve(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        out_name_holders.push_back(s.center_session->GetOutputNameAllocated(i, allocator));
        output_names.push_back(out_name_holders.back().get());
    }

    const char *input_names[] = {in_name.get()};

    Ort::Value input_val = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), shape.data(), shape.size());

    auto outputs = s.center_session->Run(
        Ort::RunOptions{nullptr}, input_names, &input_val, 1,
        output_names.data(), n_outputs);

    // Pick the highest-index output — that's the high-res heatmap when two
    // are present, or the only tensor for a single-output export.
    auto &hm = outputs[n_outputs - 1];
    auto hm_shape = hm.GetTensorTypeAndShapeInfo().GetShape();
    int hm_h = (int)hm_shape[2];
    int hm_w = (int)hm_shape[3];
    const float *hm_data = hm.GetTensorData<float>();

    auto peak = jarvis_detail::heatmap_argmax(hm_data, hm_h, hm_w);
    // Scale from heatmap coords to original image coords
    result.x = peak.x * ds_x;
    result.y = peak.y * ds_y;
    result.confidence = peak.confidence;
#else
    (void)rgb; (void)w; (void)h;
#endif
    return result;
}

// Run KeypointDetect on a cropped region around a center point.
// Returns keypoints in image coordinates (Y=0 at top).
inline std::vector<jarvis_detail::HeatmapPeak> jarvis_detect_keypoints(
    JarvisState &s, const uint8_t *rgb, int img_w, int img_h,
    int center_x, int center_y) {
    std::vector<jarvis_detail::HeatmapPeak> result;
#ifdef RED_HAS_ONNXRUNTIME
    if (!s.loaded || !s.keypoint_session) return result;

    int input_size = s.config.keypoint_input_size;
    int bbox_hw = input_size / 2;

    // Clamp center to keep crop within image
    center_x = std::clamp(center_x, bbox_hw, img_w - bbox_hw);
    center_y = std::clamp(center_y, bbox_hw, img_h - bbox_hw);

    auto input = jarvis_detail::preprocess_crop(
        rgb, img_w, img_h, center_x, center_y, input_size, input_size);

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> shape = {1, 3, input_size, input_size};

    Ort::AllocatorWithDefaultOptions allocator;
    auto in_name = s.keypoint_session->GetInputNameAllocated(0, allocator);
    size_t n_outputs = s.keypoint_session->GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> out_name_holders;
    std::vector<const char *> output_names;
    out_name_holders.reserve(n_outputs);
    output_names.reserve(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        out_name_holders.push_back(s.keypoint_session->GetOutputNameAllocated(i, allocator));
        output_names.push_back(out_name_holders.back().get());
    }

    const char *input_names[] = {in_name.get()};

    Ort::Value input_val = Ort::Value::CreateTensor<float>(
        mem, input.data(), input.size(), shape.data(), shape.size());

    auto outputs = s.keypoint_session->Run(
        Ort::RunOptions{nullptr}, input_names, &input_val, 1,
        output_names.data(), n_outputs);

    // Pick the highest-index output (high-res heatmap or sole output).
    auto &hm = outputs[n_outputs - 1];
    auto hm_shape = hm.GetTensorTypeAndShapeInfo().GetShape();
    int n_joints = (int)hm_shape[1];
    int hm_h = (int)hm_shape[2];
    int hm_w = (int)hm_shape[3];
    const float *hm_data = hm.GetTensorData<float>();

    result.resize(n_joints);
    for (int j = 0; j < n_joints; ++j) {
        auto peak = jarvis_detail::heatmap_argmax(
            hm_data + j * hm_h * hm_w, hm_h, hm_w);
        // Convert from crop-local to full-image coordinates
        result[j].x = peak.x + (float)(center_x - bbox_hw);
        result[j].y = peak.y + (float)(center_y - bbox_hw);
        result[j].confidence = peak.confidence;
    }
#else
    (void)rgb; (void)img_w; (void)img_h; (void)center_x; (void)center_y;
#endif
    return result;
}

// Run full JARVIS prediction on one frame across all cameras.
// Stores results as LabelSource::Predicted in the AnnotationMap.
// rgb_per_cam: array of RGB buffers, one per camera (image coords, Y=0 top).
inline bool jarvis_predict_frame(
    JarvisState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<const uint8_t *> &rgb_per_cam,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &skeleton,
    const std::vector<CameraParams> &camera_params,
    RenderScene *scene,
    float confidence_threshold = 0.1f) {
#ifdef RED_HAS_ONNXRUNTIME
    if (!s.loaded) { s.status = "Not loaded"; return false; }

    auto t0 = std::chrono::steady_clock::now();
    int num_cams = (int)rgb_per_cam.size();
    int num_joints = s.config.num_joints;

    // Get or create frame annotation
    auto &fa = get_or_create_frame(amap, frame_num,
                                    std::min(num_joints, skeleton.num_nodes),
                                    num_cams);

    // Phase 1: CenterDetect on all cameras
    auto t1 = std::chrono::steady_clock::now();
    std::vector<jarvis_detail::HeatmapPeak> centers(num_cams);
    for (int c = 0; c < num_cams; ++c) {
        if (!rgb_per_cam[c]) continue;
        centers[c] = jarvis_detect_center(s, rgb_per_cam[c],
                                           cam_widths[c], cam_heights[c]);
    }
    auto t2 = std::chrono::steady_clock::now();
    s.last_center_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

    // Phase 2: KeypointDetect on all cameras (around detected center)
    for (int c = 0; c < num_cams; ++c) {
        if (!rgb_per_cam[c] || centers[c].confidence < confidence_threshold)
            continue;

        auto keypoints = jarvis_detect_keypoints(
            s, rgb_per_cam[c], cam_widths[c], cam_heights[c],
            (int)centers[c].x, (int)centers[c].y);

        // Store in AnnotationMap (convert image coords → ImPlot coords)
        int n = std::min((int)keypoints.size(), (int)fa.cameras[c].keypoints.size());
        for (int k = 0; k < n; ++k) {
            auto &kp = fa.cameras[c].keypoints[k];
            kp.x = keypoints[k].x;
            kp.y = cam_heights[c] - keypoints[k].y; // image → ImPlot
            kp.labeled = (keypoints[k].confidence >= confidence_threshold);
            kp.confidence = keypoints[k].confidence;
            kp.source = LabelSource::Predicted;
        }
    }
    auto t3 = std::chrono::steady_clock::now();
    s.last_keypoint_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();

    // Phase 3: Triangulate using RED's DLT
    reprojection(fa, const_cast<SkeletonContext *>(&skeleton),
                 camera_params, scene);
    auto t4 = std::chrono::steady_clock::now();
    s.last_triangulate_ms = std::chrono::duration<float, std::milli>(t4 - t3).count();
    s.last_total_ms = std::chrono::duration<float, std::milli>(t4 - t0).count();

    s.status = "Predicted " + std::to_string(num_joints) + " joints on " +
               std::to_string(num_cams) + " cameras in " +
               std::to_string((int)s.last_total_ms) + " ms";
    return true;
#else
    (void)amap; (void)frame_num; (void)rgb_per_cam; (void)cam_widths;
    (void)cam_heights; (void)skeleton; (void)camera_params; (void)scene;
    (void)confidence_threshold;
    s.status = "ONNX Runtime not available";
    return false;
#endif
}

inline void jarvis_cleanup(JarvisState &s) {
#ifdef RED_HAS_ONNXRUNTIME
    s.keypoint_session.reset();
    s.center_session.reset();
    s.loaded = false;
#endif
}
