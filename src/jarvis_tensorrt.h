#pragma once
// jarvis_tensorrt.h — JARVIS pose estimation via NVIDIA TensorRT
//
// Loads serialized .engine files and runs CenterDetect + KeypointDetect
// using TensorRT with CUDA device memory. Designed for RTX 3090 with FP16.
//
// Windows only. Preferred over ONNX Runtime on NVIDIA GPUs (~2-5ms/frame).
// Falls back gracefully if engine files are missing or TensorRT is unavailable.
//
// Two-stage pipeline (matching CoreML / ONNX backends):
//   1. CenterDetect:   320x320 input  → heatmap  → center (x,y) per camera
//   2. KeypointDetect: 704x704 crop   → heatmaps → N keypoints per camera
//
// Build with -DUSE_TENSORRT to enable. Without it, all functions are stubs.

#ifdef _WIN32

#include "annotation.h"
#include "jarvis_model_config.h"
#include "skeleton.h"
#include "types.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef USE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#endif

// ── TensorRT Logger (required by NvInfer API) ──

#ifdef USE_TENSORRT
namespace trt_detail {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        // Only print warnings and errors — suppress info/verbose chatter
        if (severity <= Severity::kWARNING)
            fprintf(stderr, "[TensorRT] %s\n", msg);
    }
};

// Shared CUDA helpers
inline bool cuda_check(cudaError_t err, const char *context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[TensorRT] CUDA error in %s: %s\n",
                context, cudaGetErrorString(err));
        return false;
    }
    return true;
}

} // namespace trt_detail
#endif // USE_TENSORRT

// ── Engine wrapper (manages one TensorRT engine + execution context) ──

#ifdef USE_TENSORRT
struct TRTEngine {
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;

    // Device memory for input/output bindings
    void *d_input = nullptr;
    void *d_output = nullptr;          // heatmap_high (the one we use)
    void *d_output_low = nullptr;      // heatmap_low (unused, but must be bound)

    // Dimensions (set after engine load)
    int input_h = 0, input_w = 0;     // e.g. 320 or 704
    int out_c = 0, out_h = 0, out_w = 0;  // heatmap_high shape [1, C, H, W]
    int out_low_c = 0, out_low_h = 0, out_low_w = 0;

    size_t input_bytes = 0;
    size_t output_bytes = 0;
    size_t output_low_bytes = 0;

    cudaStream_t stream = nullptr;

    bool loaded = false;

    ~TRTEngine() { release(); }

    void release() {
        if (d_input)      { cudaFree(d_input);      d_input = nullptr; }
        if (d_output)     { cudaFree(d_output);     d_output = nullptr; }
        if (d_output_low) { cudaFree(d_output_low); d_output_low = nullptr; }
        if (stream)       { cudaStreamDestroy(stream); stream = nullptr; }
        if (context)      { delete context; context = nullptr; }
        if (engine)       { delete engine;  engine = nullptr; }
        if (runtime)      { delete runtime; runtime = nullptr; }
        loaded = false;
    }
};
#endif // USE_TENSORRT

// ── State ──

struct JarvisTensorRTState {
    bool loaded = false;
#ifdef USE_TENSORRT
    bool available = true;
#else
    bool available = false;
#endif

    std::string status;
    JarvisModelConfig config;

    // Flat copies for hot-path inference access (avoid indirection)
    int center_input_size = 320;
    int keypoint_input_size = 704;
    int num_joints = 24;

    // Timing (per predict_frame call)
    float last_center_ms = 0;
    float last_keypoint_ms = 0;
    float last_total_ms = 0;

#ifdef USE_TENSORRT
    std::unique_ptr<TRTEngine> center_engine;
    std::unique_ptr<TRTEngine> keypoint_engine;
    trt_detail::Logger logger;

    // Mutex for thread safety in multi-camera scenarios where
    // different threads might call predict concurrently.
    std::mutex inference_mutex;
#endif
};

// ── Preprocessing (CPU-side, matching ONNX/CoreML backends) ──

namespace trt_detail {

// ImageNet normalization constants (identical to jarvis_inference.h)
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float INV_STD[3] = {1.0f / 0.229f, 1.0f / 0.224f, 1.0f / 0.225f};

// Bilinear resize + ImageNet normalize → CHW float32 (NCHW layout for TensorRT)
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
                float v = (1 - wy) * ((1 - wx) * rgb[(y0 * src_w + x0) * 3 + c] +
                                       wx * rgb[(y0 * src_w + x1) * 3 + c]) +
                          wy * ((1 - wx) * rgb[(y1 * src_w + x0) * 3 + c] +
                                 wx * rgb[(y1 * src_w + x1) * 3 + c]);
                result[c * dst_size * dst_size + y * dst_size + x] =
                    (v / 255.0f - MEAN[c]) * INV_STD[c];
            }
        }
    }
    return result;
}

// Preprocess a crop from a larger image (identical to jarvis_inference.h)
inline std::vector<float> preprocess_crop(const uint8_t *rgb, int img_w, int img_h,
                                           int cx, int cy, int crop_size, int dst_size) {
    int half = crop_size / 2;
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(img_w, cx + half);
    int y1 = std::min(img_h, cy + half);
    int cw = x1 - x0;
    int ch = y1 - y0;

    std::vector<uint8_t> crop(cw * ch * 3);
    for (int y = 0; y < ch; ++y) {
        const uint8_t *src_row = rgb + ((y0 + y) * img_w + x0) * 3;
        uint8_t *dst_row = crop.data() + y * cw * 3;
        std::memcpy(dst_row, src_row, cw * 3);
    }

    return preprocess(crop.data(), cw, ch, dst_size);
}

// Argmax on a 2D heatmap → (x, y, confidence)
// Identical to jarvis_inference.h — heatmap is half resolution (stride 2)
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

} // namespace trt_detail

// ── Engine loading ──

#ifdef USE_TENSORRT
// Load a serialized TensorRT engine from disk and allocate device memory.
// Engine files are built offline with convert_pth_to_trt.py or trtexec.
//
// Expected engine I/O:
//   Input:  "image"        — [1, 3, H, W] float32 or float16
//   Output: "heatmap_low"  — [1, C, H/4, W/4] float32
//   Output: "heatmap_high" — [1, C, H/2, W/2] float32 (used for argmax)
//
// Returns true on success. On failure, prints warning and returns false
// (caller should fall back to ONNX Runtime).
inline bool trt_load_engine(TRTEngine &eng, const std::string &engine_path,
                             nvinfer1::ILogger &logger) {
    eng.release();

    if (!std::filesystem::exists(engine_path)) {
        fprintf(stderr, "[TensorRT] Engine file not found: %s\n", engine_path.c_str());
        return false;
    }

    // Read serialized engine into memory
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        fprintf(stderr, "[TensorRT] Cannot open engine file: %s\n", engine_path.c_str());
        return false;
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    if (!file.read(data.data(), size)) {
        fprintf(stderr, "[TensorRT] Failed to read engine file: %s\n", engine_path.c_str());
        return false;
    }
    file.close();

    // Deserialize
    eng.runtime = nvinfer1::createInferRuntime(logger);
    if (!eng.runtime) {
        fprintf(stderr, "[TensorRT] Failed to create runtime\n");
        return false;
    }

    eng.engine = eng.runtime->deserializeCudaEngine(data.data(), data.size());
    if (!eng.engine) {
        fprintf(stderr, "[TensorRT] Failed to deserialize engine: %s\n", engine_path.c_str());
        eng.release();
        return false;
    }

    eng.context = eng.engine->createExecutionContext();
    if (!eng.context) {
        fprintf(stderr, "[TensorRT] Failed to create execution context\n");
        eng.release();
        return false;
    }

    // Create CUDA stream for async execution
    if (!trt_detail::cuda_check(cudaStreamCreate(&eng.stream), "cudaStreamCreate")) {
        eng.release();
        return false;
    }

    // Resolve binding indices and shapes.
    // TensorRT 8.x+ uses name-based tensor I/O.
    // We look for "image" (input), "heatmap_high" and "heatmap_low" (outputs).
    // Fall back to index-based if names differ.
    int nb = eng.engine->getNbIOTensors();

    int input_idx = -1, output_high_idx = -1, output_low_idx = -1;
    for (int i = 0; i < nb; ++i) {
        const char *name = eng.engine->getIOTensorName(i);
        auto mode = eng.engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_idx = i;
        } else {
            // Prefer "heatmap_high" as primary output; "heatmap_low" as secondary.
            // If names don't match, use the larger spatial output as "high".
            std::string sname(name);
            if (sname.find("high") != std::string::npos) {
                output_high_idx = i;
            } else if (sname.find("low") != std::string::npos) {
                output_low_idx = i;
            } else if (output_high_idx < 0) {
                output_high_idx = i;
            } else {
                output_low_idx = i;
            }
        }
    }

    if (input_idx < 0 || output_high_idx < 0) {
        fprintf(stderr, "[TensorRT] Could not identify input/output tensors in engine\n");
        eng.release();
        return false;
    }

    // Read shapes
    {
        const char *in_name = eng.engine->getIOTensorName(input_idx);
        auto in_dims = eng.engine->getTensorShape(in_name);
        // Expected: [1, 3, H, W]
        eng.input_h = in_dims.d[2];
        eng.input_w = in_dims.d[3];
        eng.input_bytes = sizeof(float) * 1 * 3 * eng.input_h * eng.input_w;
    }
    {
        const char *out_name = eng.engine->getIOTensorName(output_high_idx);
        auto out_dims = eng.engine->getTensorShape(out_name);
        // Expected: [1, C, H, W]
        eng.out_c = out_dims.d[1];
        eng.out_h = out_dims.d[2];
        eng.out_w = out_dims.d[3];
        eng.output_bytes = sizeof(float) * 1 * eng.out_c * eng.out_h * eng.out_w;
    }
    if (output_low_idx >= 0) {
        const char *out_name = eng.engine->getIOTensorName(output_low_idx);
        auto out_dims = eng.engine->getTensorShape(out_name);
        eng.out_low_c = out_dims.d[1];
        eng.out_low_h = out_dims.d[2];
        eng.out_low_w = out_dims.d[3];
        eng.output_low_bytes = sizeof(float) * 1 * eng.out_low_c * eng.out_low_h * eng.out_low_w;
    }

    // Allocate device memory
    if (!trt_detail::cuda_check(cudaMalloc(&eng.d_input, eng.input_bytes), "alloc input") ||
        !trt_detail::cuda_check(cudaMalloc(&eng.d_output, eng.output_bytes), "alloc output")) {
        eng.release();
        return false;
    }
    if (eng.output_low_bytes > 0) {
        if (!trt_detail::cuda_check(cudaMalloc(&eng.d_output_low, eng.output_low_bytes), "alloc output_low")) {
            eng.release();
            return false;
        }
    }

    // Bind tensors to device memory addresses
    {
        const char *in_name = eng.engine->getIOTensorName(input_idx);
        eng.context->setTensorAddress(in_name, eng.d_input);
    }
    {
        const char *out_name = eng.engine->getIOTensorName(output_high_idx);
        eng.context->setTensorAddress(out_name, eng.d_output);
    }
    if (output_low_idx >= 0 && eng.d_output_low) {
        const char *out_name = eng.engine->getIOTensorName(output_low_idx);
        eng.context->setTensorAddress(out_name, eng.d_output_low);
    }

    eng.loaded = true;
    return true;
}

// Run inference on preprocessed CHW float32 input.
// Copies input to device, runs engine, copies heatmap_high output back.
// Returns host-side heatmap data (caller owns the vector).
inline std::vector<float> trt_infer(TRTEngine &eng, const float *input_chw) {
    std::vector<float> output;
    if (!eng.loaded) return output;

    // H2D: copy preprocessed input to device
    cudaMemcpyAsync(eng.d_input, input_chw, eng.input_bytes,
                    cudaMemcpyHostToDevice, eng.stream);

    // Execute
    eng.context->enqueueV3(eng.stream);

    // D2H: copy heatmap_high output back
    output.resize(eng.out_c * eng.out_h * eng.out_w);
    cudaMemcpyAsync(output.data(), eng.d_output, eng.output_bytes,
                    cudaMemcpyDeviceToHost, eng.stream);

    // Synchronize — ensures output is ready before we return
    cudaStreamSynchronize(eng.stream);

    return output;
}
#endif // USE_TENSORRT

// ── Public API ──

// Check if TensorRT is available on this system
inline bool jarvis_tensorrt_available() {
#ifdef USE_TENSORRT
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

// Initialize: load serialized .engine files from model_dir.
// Expects center_detect.engine and keypoint_detect.engine.
// Pass pre-parsed config to avoid redundant JSON I/O.
inline bool jarvis_tensorrt_init(JarvisTensorRTState &s,
                                  const std::string &model_dir,
                                  const JarvisModelConfig &cfg) {
#ifdef USE_TENSORRT
    s.loaded = false;
    s.available = jarvis_tensorrt_available();
    if (!s.available) {
        s.status = "TensorRT: no CUDA device found";
        return false;
    }

    // Store config
    s.config = cfg;
    s.center_input_size = cfg.center_input_size;
    s.keypoint_input_size = cfg.keypoint_input_size;
    s.num_joints = cfg.num_joints;

    std::string cd_path = model_dir + "/center_detect.engine";
    std::string kd_path = model_dir + "/keypoint_detect.engine";

    if (!std::filesystem::exists(cd_path)) {
        s.status = "TensorRT: center_detect.engine not found in " + model_dir;
        fprintf(stderr, "[TensorRT] %s\n", s.status.c_str());
        return false;
    }
    if (!std::filesystem::exists(kd_path)) {
        s.status = "TensorRT: keypoint_detect.engine not found in " + model_dir;
        fprintf(stderr, "[TensorRT] %s\n", s.status.c_str());
        return false;
    }

    // Load engines
    s.center_engine = std::make_unique<TRTEngine>();
    s.status = "Loading CenterDetect engine...";
    if (!trt_load_engine(*s.center_engine, cd_path, s.logger)) {
        s.status = "TensorRT: failed to load center_detect.engine";
        s.center_engine.reset();
        return false;
    }

    s.keypoint_engine = std::make_unique<TRTEngine>();
    s.status = "Loading KeypointDetect engine...";
    if (!trt_load_engine(*s.keypoint_engine, kd_path, s.logger)) {
        s.status = "TensorRT: failed to load keypoint_detect.engine";
        s.center_engine.reset();
        s.keypoint_engine.reset();
        return false;
    }

    s.loaded = true;
    s.status = "TensorRT loaded (" + std::to_string(s.num_joints) +
               " joints, FP16 GPU)";
    return true;
#else
    (void)model_dir; (void)cfg;
    s.available = false;
    s.status = "TensorRT not available (compile with USE_TENSORRT)";
    return false;
#endif
}

// Run CenterDetect on a single camera image.
// Returns center in image coordinates (Y=0 at top).
inline trt_detail::HeatmapPeak jarvis_tensorrt_detect_center(
    JarvisTensorRTState &s, const uint8_t *rgb, int w, int h) {
    trt_detail::HeatmapPeak result = {0, 0, 0};
#ifdef USE_TENSORRT
    if (!s.loaded || !s.center_engine || !s.center_engine->loaded) return result;

    int input_size = s.center_input_size;
    float ds_x = (float)w / input_size;
    float ds_y = (float)h / input_size;

    auto input = trt_detail::preprocess(rgb, w, h, input_size);
    auto hm_data = trt_infer(*s.center_engine, input.data());
    if (hm_data.empty()) return result;

    int hm_h = s.center_engine->out_h;
    int hm_w = s.center_engine->out_w;

    auto peak = trt_detail::heatmap_argmax(hm_data.data(), hm_h, hm_w);
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
inline std::vector<trt_detail::HeatmapPeak> jarvis_tensorrt_detect_keypoints(
    JarvisTensorRTState &s, const uint8_t *rgb, int img_w, int img_h,
    int center_x, int center_y) {
    std::vector<trt_detail::HeatmapPeak> result;
#ifdef USE_TENSORRT
    if (!s.loaded || !s.keypoint_engine || !s.keypoint_engine->loaded) return result;

    int input_size = s.keypoint_input_size;
    int bbox_hw = input_size / 2;

    center_x = std::clamp(center_x, bbox_hw, img_w - bbox_hw);
    center_y = std::clamp(center_y, bbox_hw, img_h - bbox_hw);

    auto input = trt_detail::preprocess_crop(
        rgb, img_w, img_h, center_x, center_y, input_size, input_size);

    auto hm_data = trt_infer(*s.keypoint_engine, input.data());
    if (hm_data.empty()) return result;

    int n_joints = s.keypoint_engine->out_c;
    int hm_h = s.keypoint_engine->out_h;
    int hm_w = s.keypoint_engine->out_w;

    result.resize(n_joints);
    for (int j = 0; j < n_joints; ++j) {
        auto peak = trt_detail::heatmap_argmax(
            hm_data.data() + j * hm_h * hm_w, hm_h, hm_w);
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
//
// This function mirrors jarvis_predict_frame() in jarvis_inference.h and
// jarvis_coreml_predict_frame() in jarvis_coreml.mm.
inline bool jarvis_tensorrt_predict_frame(
    JarvisTensorRTState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<const uint8_t *> &rgb_per_cam,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &skeleton,
    int num_cameras,
    float confidence_threshold = 0.1f) {
#ifdef USE_TENSORRT
    if (!s.loaded) { s.status = "Not loaded"; return false; }

    // Thread safety: serialize inference calls (TensorRT contexts are not
    // thread-safe, and we share CUDA device memory per engine).
    std::lock_guard<std::mutex> lock(s.inference_mutex);

    auto t0 = std::chrono::steady_clock::now();
    int num_cams = (int)rgb_per_cam.size();
    int num_joints = s.num_joints;

    auto &fa = get_or_create_frame(amap, frame_num,
                                    std::min(num_joints, skeleton.num_nodes),
                                    num_cams);

    // Per-camera pipeline: CenterDetect → crop → KeypointDetect
    // (same sequential-per-camera approach as CoreML for cache locality)
    float center_ms_total = 0, kp_ms_total = 0;

    for (int c = 0; c < num_cams; ++c) {
        if (!rgb_per_cam[c]) continue;

        // --- CenterDetect ---
        auto tc0 = std::chrono::steady_clock::now();
        auto center = jarvis_tensorrt_detect_center(
            s, rgb_per_cam[c], cam_widths[c], cam_heights[c]);
        auto tc1 = std::chrono::steady_clock::now();
        center_ms_total += std::chrono::duration<float, std::milli>(tc1 - tc0).count();

        if (center.confidence < confidence_threshold) continue;

        // --- KeypointDetect ---
        auto tk0 = std::chrono::steady_clock::now();
        auto keypoints = jarvis_tensorrt_detect_keypoints(
            s, rgb_per_cam[c], cam_widths[c], cam_heights[c],
            (int)center.x, (int)center.y);

        int n = std::min((int)keypoints.size(), (int)fa.cameras[c].keypoints.size());
        for (int k = 0; k < n; ++k) {
            auto &kp = fa.cameras[c].keypoints[k];
            kp.x = keypoints[k].x;
            kp.y = cam_heights[c] - keypoints[k].y; // image → ImPlot (flip Y)
            kp.labeled = (keypoints[k].confidence >= confidence_threshold);
            kp.confidence = keypoints[k].confidence;
            kp.source = LabelSource::Predicted;
        }
        auto tk1 = std::chrono::steady_clock::now();
        kp_ms_total += std::chrono::duration<float, std::milli>(tk1 - tk0).count();
    }

    s.last_center_ms = center_ms_total;
    s.last_keypoint_ms = kp_ms_total;

    auto t4 = std::chrono::steady_clock::now();
    s.last_total_ms = std::chrono::duration<float, std::milli>(t4 - t0).count();
    s.status = "Predicted " + std::to_string(num_joints) + " joints on " +
               std::to_string(num_cams) + " cameras in " +
               std::to_string((int)s.last_total_ms) + " ms (TensorRT)";
    return true;
#else
    (void)amap; (void)frame_num; (void)rgb_per_cam; (void)cam_widths;
    (void)cam_heights; (void)skeleton; (void)num_cameras;
    (void)confidence_threshold;
    s.status = "TensorRT not available";
    return false;
#endif
}

inline void jarvis_tensorrt_cleanup(JarvisTensorRTState &s) {
#ifdef USE_TENSORRT
    s.center_engine.reset();
    s.keypoint_engine.reset();
    s.loaded = false;
#endif
}

#endif // _WIN32
