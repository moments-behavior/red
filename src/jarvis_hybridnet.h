#pragma once
// jarvis_hybridnet.h — Full 3D HybridNet pose estimation via TensorRT
// ─────────────────────────────────────────────────────────────────────────
// Linux/Windows backend for the JARVIS 3-stage pipeline:
//
//   per-cam (16x):  raw RGB → resize 320² → CenterDetect → 2D peak
//   once:           triangulate via red_math DLT → center_3D (world mm)
//   per-cam (16x):  reproject center_3D → centerHM (native pixel)
//                   crop 704² at centerHM → effTrack → heatmaps (24, 352, 352)
//   once:           F.pad heatmaps → (1, 16, 24, 354, 354)
//                   build P=K·[R|t] (or telecentric DLT) per cam → cameraMatrices (1, 16, 4, 3)
//                   Hybrid3D → points3D (1, 24, 3) in world mm, confidences (1, 24)
//
// All three stages run on TensorRT engines compiled offline via
// scripts/compile_tensorrt_engines.sh. Engines are GPU-architecture and
// TRT-version specific; re-run the script per rig. predict_frame takes
// host RGB (CPU Buffer mode); predict_frame_device takes device RGBA
// (GPU Buffer mode) and skips host preprocessing entirely via CUDA kernels.
//
// Mac retains its existing CoreML 2D+triangulate shortcut. This file does
// not compile on Apple.
// ─────────────────────────────────────────────────────────────────────────

#if defined(__linux__) || defined(_WIN32)
#ifdef RED_HAS_TENSORRT_HN

#include "annotation.h"
#include "camera.h"
#include "red_math.h"
#include "skeleton.h"
#include "types.h"
#include "json.hpp"

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "jarvis_hybridnet_cuda.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────
// TensorRT direct-runtime engine wrapper. Multi-input / multi-output and
// indexes bindings by tensor name — Hybrid3D has 4 inputs (heatmaps_padded,
// centerHM, center3D, cameraMatrices) and ≥2 outputs (points3D,
// confidences). All bindings get device buffers allocated up front from
// engine shapes; per-predict work is just H↔D memcpy + enqueueV3 + stream
// sync (the kernel preprocessing path skips the input H2D).
// ─────────────────────────────────────────────────────────────────────────
namespace jarvis_hn_trt {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[HN-TRT] %s\n", msg);
    }
};

inline bool cuda_ok(cudaError_t err, const char *where) {
    if (err == cudaSuccess) return true;
    std::fprintf(stderr, "[HN-TRT] CUDA error in %s: %s\n", where, cudaGetErrorString(err));
    return false;
}

struct Binding {
    nvinfer1::Dims dims{};
    size_t bytes = 0;          // element size * volume
    void *d_ptr = nullptr;     // device buffer
    bool is_input = false;
};

struct Engine {
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    cudaStream_t stream = nullptr;
    std::unordered_map<std::string, Binding> bindings;  // by tensor name
    bool loaded = false;

    ~Engine() { release(); }

    void release() {
        for (auto &kv : bindings) {
            if (kv.second.d_ptr) cudaFree(kv.second.d_ptr);
        }
        bindings.clear();
        if (stream)  { cudaStreamDestroy(stream); stream = nullptr; }
        if (context) { delete context; context = nullptr; }
        if (engine)  { delete engine;  engine = nullptr; }
        if (runtime) { delete runtime; runtime = nullptr; }
        loaded = false;
    }

    Binding *get(const std::string &name) {
        auto it = bindings.find(name);
        return it == bindings.end() ? nullptr : &it->second;
    }
};

inline size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        default:                          return 4;
    }
}

inline size_t volume(const nvinfer1::Dims &d) {
    size_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) v *= static_cast<size_t>(d.d[i]);
    return v;
}

// One-shot plugin registry init. Required for engines built with NMS,
// reproLayer, or any other op that ships as a TRT plugin — without this
// deserialization fails with "Cannot deserialize plugin since corresponding
// IPluginCreator not found in Plugin Registry". Safe to call multiple times.
inline void ensure_plugins_registered(nvinfer1::ILogger &logger) {
    static bool initialized = false;
    if (initialized) return;
    if (!initLibNvInferPlugins(&logger, "")) {
        std::fprintf(stderr, "[HN-TRT] WARN: initLibNvInferPlugins returned false "
                             "(continuing, but plugin-bearing engines may fail to load)\n");
    }
    initialized = true;
}

// Deserialize an .engine file, create an execution context, allocate device
// memory for every I/O tensor, and bind tensor addresses. On failure logs
// and returns false; partial state is cleaned up.
inline bool load_engine(Engine &eng, const std::string &engine_path,
                        nvinfer1::ILogger &logger) {
    ensure_plugins_registered(logger);
    eng.release();
    namespace fs = std::filesystem;
    if (!fs::exists(engine_path)) {
        std::fprintf(stderr, "[HN-TRT] engine not found: %s\n", engine_path.c_str());
        return false;
    }
    std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::fprintf(stderr, "[HN-TRT] cannot open engine: %s\n", engine_path.c_str());
        return false;
    }
    size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(sz);
    if (!f.read(blob.data(), sz)) {
        std::fprintf(stderr, "[HN-TRT] failed to read engine: %s\n", engine_path.c_str());
        return false;
    }
    f.close();

    eng.runtime = nvinfer1::createInferRuntime(logger);
    if (!eng.runtime) {
        std::fprintf(stderr, "[HN-TRT] createInferRuntime failed for %s\n", engine_path.c_str());
        return false;
    }
    eng.engine = eng.runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (!eng.engine) {
        std::fprintf(stderr, "[HN-TRT] deserializeCudaEngine failed for %s\n", engine_path.c_str());
        eng.release();
        return false;
    }
    eng.context = eng.engine->createExecutionContext();
    if (!eng.context) {
        std::fprintf(stderr, "[HN-TRT] createExecutionContext failed for %s\n", engine_path.c_str());
        eng.release();
        return false;
    }
    if (!cuda_ok(cudaStreamCreate(&eng.stream), "cudaStreamCreate")) {
        eng.release();
        return false;
    }

    const int n = eng.engine->getNbIOTensors();

    // Pin every dynamic input to its profile's MAX shape so downstream output
    // shapes resolve. Engines compiled with --min/opt/maxShapes pinned at
    // batch=16 give us the desired shape via MAX; for engines whose inputs
    // are fully static, the build-time shape has no -1's and setInputShape
    // is a no-op-equivalent.
    auto has_dynamic = [](const nvinfer1::Dims &d) {
        for (int i = 0; i < d.nbDims; ++i) if (d.d[i] < 0) return true;
        return false;
    };
    for (int i = 0; i < n; ++i) {
        const char *name = eng.engine->getIOTensorName(i);
        if (eng.engine->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        auto build_dims = eng.engine->getTensorShape(name);
        if (!has_dynamic(build_dims)) continue;
        auto max_dims = eng.engine->getProfileShape(
            name, 0, nvinfer1::OptProfileSelector::kMAX);
        if (!eng.context->setInputShape(name, max_dims)) {
            std::fprintf(stderr,
                "[HN-TRT] setInputShape failed for %s (engine %s)\n",
                name, engine_path.c_str());
            eng.release();
            return false;
        }
    }
    if (!eng.context->allInputShapesSpecified()) {
        std::fprintf(stderr,
            "[HN-TRT] not all input shapes specified after profile pinning (engine %s)\n",
            engine_path.c_str());
        eng.release();
        return false;
    }

    // Resolve final shapes from the context (handles both static and the
    // dynamic-with-fixed-profile case) and allocate device memory.
    for (int i = 0; i < n; ++i) {
        const char *name = eng.engine->getIOTensorName(i);
        Binding b;
        b.is_input = (eng.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        b.dims     = eng.context->getTensorShape(name);
        b.bytes    = dtype_size(eng.engine->getTensorDataType(name)) * volume(b.dims);
        if (b.bytes == 0) {
            std::fprintf(stderr, "[HN-TRT] tensor %s in %s has zero-volume shape\n",
                         name, engine_path.c_str());
            eng.release();
            return false;
        }
        if (!cuda_ok(cudaMalloc(&b.d_ptr, b.bytes), "cudaMalloc binding")) {
            eng.release();
            return false;
        }
        if (!eng.context->setTensorAddress(name, b.d_ptr)) {
            std::fprintf(stderr, "[HN-TRT] setTensorAddress failed for %s\n", name);
            eng.release();
            return false;
        }
        eng.bindings.emplace(std::string(name), b);
    }

    eng.loaded = true;
    return true;
}

// Diagnostic helper: log the engine's I/O shapes (called once at load).
inline void log_engine_io(const Engine &eng, const char *label) {
    std::fprintf(stderr, "[HN-TRT]   %s bindings:\n", label);
    for (const auto &kv : eng.bindings) {
        const auto &d = kv.second.dims;
        char shape[128] = {0};
        int off = 0;
        for (int i = 0; i < d.nbDims && off < (int)sizeof(shape) - 8; ++i) {
            off += std::snprintf(shape + off, sizeof(shape) - off,
                                 i == 0 ? "%d" : "x%d", static_cast<int>(d.d[i]));
        }
        std::fprintf(stderr, "[HN-TRT]     %-18s %-6s [%s]  %zu B\n",
                     kv.first.c_str(), kv.second.is_input ? "input" : "output",
                     shape, kv.second.bytes);
    }
}

// Single-input convenience: H→D, enqueue, D→H, sync.
// `host_in` is the input tensor's host data (bytes match the binding).
// `host_out` is filled from `out_name`'s device buffer.
inline bool run_single_io(Engine &eng,
                          const std::string &in_name, const float *host_in,
                          const std::string &out_name, float *host_out) {
    Binding *bi = eng.get(in_name);
    Binding *bo = eng.get(out_name);
    if (!bi || !bo) {
        std::fprintf(stderr, "[HN-TRT] missing binding: %s or %s\n",
                     in_name.c_str(), out_name.c_str());
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(bi->d_ptr, host_in, bi->bytes,
                                 cudaMemcpyHostToDevice, eng.stream), "H2D"))
        return false;
    if (!eng.context->enqueueV3(eng.stream)) {
        std::fprintf(stderr, "[HN-TRT] enqueueV3 failed\n");
        return false;
    }
    if (!cuda_ok(cudaMemcpyAsync(host_out, bo->d_ptr, bo->bytes,
                                 cudaMemcpyDeviceToHost, eng.stream), "D2H"))
        return false;
    return cuda_ok(cudaStreamSynchronize(eng.stream), "stream sync");
}

} // namespace jarvis_hn_trt

// ─────────────────────────────────────────────────────────────────────────
// Configuration parsed from manifest.json (written by export_jarvis_onnx.py)
// ─────────────────────────────────────────────────────────────────────────
struct JarvisHybridNetConfig {
    int center_image_size  = 320;   // CenterDetect square input
    int keypoint_bbox_size = 704;   // effTrack crop square input
    int num_joints         = 24;
    int num_cameras        = 16;
    float roi_cube_size_mm = 200.0f;
    float grid_spacing_mm  = 2.0f;
    std::array<float, 3> dataset_mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> dataset_std  = {0.229f, 0.224f, 0.225f};
    std::vector<std::string> keypoint_names;                              // ordered, len=num_joints
    std::vector<std::pair<std::string, std::string>> skeleton_edges;      // pairs of joint names

    // Derived
    int heatmap_hw_padded() const { return keypoint_bbox_size / 2 + 2; }  // 354 for bbox=704
    int voxel_grid_half() const { return int(roi_cube_size_mm / grid_spacing_mm / 2); }  // 50 for 200/2
};

// Parse manifest.json into config. Returns false if file missing or malformed.
inline bool jarvis_hybridnet_load_manifest(JarvisHybridNetConfig &cfg,
                                            const std::string &manifest_path);

// ─────────────────────────────────────────────────────────────────────────
// Runtime state. Holds 3 TRT engines + preallocated host/device scratch.
// Scratch sizes are computed from cfg at load time and never re-allocated.
// ─────────────────────────────────────────────────────────────────────────
struct JarvisHybridNetState {
    bool loaded = false;
    JarvisHybridNetConfig cfg;

    // CUDA device the engines were deserialized on. All inference (and the
    // pre-inference cudaSetDevice in the predict paths) must target this same
    // device — on multi-GPU boxes (e.g. flyrig: A16 sm_86 + RTX 4000 Ada
    // sm_89) hardcoding device 0 could run kernels on a different device than
    // the one that holds the engine's memory. Set in jarvis_hybridnet_load.
    int gpu_device_id = 0;

    jarvis_hn_trt::Logger trt_logger;
    std::unique_ptr<jarvis_hn_trt::Engine> trt_center;     // center_detect.engine
    std::unique_ptr<jarvis_hn_trt::Engine> trt_efftrack;   // hybridnet_efftrack.engine
    std::unique_ptr<jarvis_hn_trt::Engine> trt_hybrid3d;   // hybrid3d.engine

    // Per-frame metadata for GPU preprocessing (predict_frame_device path).
    // Lives on device so the resize/crop kernels can read it directly.
    // Allocated once in jarvis_hybridnet_load, freed in jarvis_hybridnet_unload.
    const uint8_t **d_rgba_ptrs = nullptr;   // N device pointers
    int *d_widths   = nullptr;               // N ints
    int *d_heights  = nullptr;               // N ints
    int *d_cx       = nullptr;               // N ints (crop centers, set per predict)
    int *d_cy       = nullptr;               // N ints

    // Host scratch buffers used by the predict_frame (host-RGB) path —
    // CPU preprocessing fills them, then they're H2D'd into the engines.
    // predict_frame_device skips most of these and writes engine inputs
    // directly from GPU kernels; only the small Hybrid3D aux inputs
    // (camera_matrices, centerHM_input, center3D_input) and the peak-pick
    // intermediate (center_out_high) still flow through host.
    std::vector<float> center_input;       // (N, 3, 320, 320)
    std::vector<float> center_out_high;    // (N, 1, 160, 160) — peak-picked
    std::vector<float> crop_input;         // (N, 3, 704, 704)
    std::vector<float> camera_matrices;    // (1, N, 4, 3)
    std::vector<float> centerHM_input;     // (1, N, 2) float (native pixel coords)
    std::array<float, 3> center3D_input{}; // (1, 3) float (world mm)
    std::vector<float> points3D_out;       // (1, J, 3)
    std::vector<float> confidences_out;    // (1, J)

    // Per-frame timing (last predict call). For UI display.
    double last_center_ms = 0.0;
    double last_efftrack_ms = 0.0;
    double last_hybrid3d_ms = 0.0;
    // Stage-2 diagnostic: number of cameras whose CenterDetect peak passed
    // the threshold and contributed to the center_3D triangulation.
    int last_center_cams_used = 0;
};

// ─────────────────────────────────────────────────────────────────────────
// Lifecycle.
//
// jarvis_hybridnet_load: reads manifest.json, deserializes the three TRT
// engines (center_detect.engine, hybridnet_efftrack.engine, hybrid3d.engine)
// from model_dir, allocates host + device scratch. Returns false on any
// error (engines missing, deserialize failed, cuda malloc failed);
// state.loaded stays false. Safe to call repeatedly (replaces existing).
//
// jarvis_hybridnet_unload: tears down engines and frees memory.
// ─────────────────────────────────────────────────────────────────────────
bool jarvis_hybridnet_load(JarvisHybridNetState &state,
                           const std::string &model_dir,
                           int gpu_device_id = 0);

void jarvis_hybridnet_unload(JarvisHybridNetState &state);

// Detect whether a directory looks like a HybridNet model dir (has
// hybrid3d.onnx and manifest.json). Used by JARVIS Predict Tool to switch UI
// mode when the user points at a model dir.
bool jarvis_hybridnet_dir_is_valid(const std::string &model_dir);

// ─────────────────────────────────────────────────────────────────────────
// Inference.
//
// Predicts a single frame across all cameras and writes 3D keypoints into
// annotations[frame_idx].kp3d. Per-camera 2D projections are also written
// into annotations[frame_idx].cameras[c].keypoints[j] (LabelSource::Predicted)
// so the existing 2D overlay code displays them.
//
// camera_rgb[c]: pointer to uint8 RGB image, row-major (H, W, 3). One per cam.
// widths[c], heights[c]: image dims (must match the calibration's expected
//                        image_width/image_height in CameraParams).
// camera_params[c]: red's calibration. Uses .telecentric flag to dispatch
//                   between projective P=K·[R|t] and telecentric-DLT.
//
// Returns false if pre-conditions fail (< 2 cams detected center, etc.) or
// any TRT call errors. On success, fills annotations and updates state's
// timing fields.
// ─────────────────────────────────────────────────────────────────────────
bool jarvis_hybridnet_predict_frame(
    JarvisHybridNetState &state,
    const std::vector<const uint8_t *> &camera_rgb,
    const std::vector<int> &widths,
    const std::vector<int> &heights,
    const std::vector<CameraParams> &camera_params,
    AnnotationMap &annotations,
    SkeletonContext &skeleton,
    u32 frame_idx);

// Device-input variant. Takes the NVDEC-resident RGBA32 device pointers
// directly, skips host-side cudaMemcpy + alpha-strip + CPU resize/crop.
// Stage 1 and Stage 3 preprocessing run as CUDA kernels that write
// straight into the TRT engines' input device buffers; the rest of the
// pipeline is identical to the host variant. Returns false if state.loaded
// is false or the device scratch buffers weren't allocated, or if the
// supplied frame pointers aren't actually device memory (e.g., red is in
// CPU Buffer mode — the caller should fall back to predict_frame).
//
// camera_rgba_device[c]: device pointer to RGBA32 frame, w*h*4 bytes.
//                        Must be valid for the lifetime of this call.
bool jarvis_hybridnet_predict_frame_device(
    JarvisHybridNetState &state,
    const std::vector<const uint8_t *> &camera_rgba_device,
    const std::vector<int> &widths,
    const std::vector<int> &heights,
    const std::vector<CameraParams> &camera_params,
    AnnotationMap &annotations,
    SkeletonContext &skeleton,
    u32 frame_idx);

// ─────────────────────────────────────────────────────────────────────────
// IMPLEMENTATION
// ─────────────────────────────────────────────────────────────────────────
// All implementations below are stubs (Task 8 fills them in). Public API and
// data layout are settled — Task 8 writes pure body code without changing
// signatures or buffer shapes.
// ─────────────────────────────────────────────────────────────────────────

inline bool jarvis_hybridnet_load_manifest(JarvisHybridNetConfig &cfg,
                                            const std::string &manifest_path) {
    namespace fs = std::filesystem;
    if (!fs::exists(manifest_path)) return false;
    try {
        std::ifstream f(manifest_path);
        nlohmann::json j;
        f >> j;
        if (!j.contains("training_config_summary")) return false;
        const auto &s = j["training_config_summary"];
        cfg.center_image_size  = s.value("center_image_size", 320);
        cfg.keypoint_bbox_size = s.value("keypoint_bbox_size", 704);
        cfg.num_joints         = s.value("num_joints", 24);
        cfg.num_cameras        = s.value("num_cameras", 16);
        cfg.roi_cube_size_mm   = s.value("roi_cube_size_mm", 200.0f);
        cfg.grid_spacing_mm    = s.value("grid_spacing_mm", 2.0f);
        if (s.contains("dataset_mean") && s["dataset_mean"].is_array() && s["dataset_mean"].size() == 3) {
            for (int i = 0; i < 3; ++i) cfg.dataset_mean[i] = s["dataset_mean"][i].get<float>();
        }
        if (s.contains("dataset_std") && s["dataset_std"].is_array() && s["dataset_std"].size() == 3) {
            for (int i = 0; i < 3; ++i) cfg.dataset_std[i] = s["dataset_std"][i].get<float>();
        }
        cfg.keypoint_names.clear();
        if (s.contains("keypoint_names") && s["keypoint_names"].is_array()) {
            for (const auto &name : s["keypoint_names"]) cfg.keypoint_names.push_back(name.get<std::string>());
        }
        cfg.skeleton_edges.clear();
        if (s.contains("skeleton") && s["skeleton"].is_array()) {
            for (const auto &e : s["skeleton"]) {
                if (e.is_array() && e.size() == 2) {
                    cfg.skeleton_edges.emplace_back(e[0].get<std::string>(), e[1].get<std::string>());
                }
            }
        }
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

inline bool jarvis_hybridnet_dir_is_valid(const std::string &model_dir) {
    namespace fs = std::filesystem;
    return fs::exists(fs::path(model_dir) / "hybrid3d.onnx") &&
           fs::exists(fs::path(model_dir) / "hybridnet_efftrack.onnx") &&
           fs::exists(fs::path(model_dir) / "center_detect.onnx") &&
           fs::exists(fs::path(model_dir) / "manifest.json");
}

inline bool jarvis_hybridnet_load(JarvisHybridNetState &state,
                                   const std::string &model_dir,
                                   int gpu_device_id) {
    namespace fs = std::filesystem;
    jarvis_hybridnet_unload(state);  // idempotent reset
    std::fprintf(stderr, "[HybridNet] load: %s\n", model_dir.c_str());

    // Report current GPU memory so the user can see whether red has
    // already saturated the device before TRT engine deserialization.
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            std::fprintf(stderr,
                "[HybridNet]   GPU %d memory: %.2f GiB free / %.2f GiB total\n",
                gpu_device_id, free_b / 1073741824.0, total_b / 1073741824.0);
        }
    }

    fs::path dir(model_dir);
    if (!jarvis_hybridnet_load_manifest(state.cfg, (dir / "manifest.json").string())) {
        std::fprintf(stderr, "[HybridNet] load FAILED: could not parse manifest.json at %s\n",
                     (dir / "manifest.json").string().c_str());
        return false;
    }
    std::fprintf(stderr, "[HybridNet]   manifest: %d joints, %d cams, bbox=%d, roi=%.1fmm grid=%.1fmm\n",
                 state.cfg.num_joints, state.cfg.num_cameras, state.cfg.keypoint_bbox_size,
                 state.cfg.roi_cube_size_mm, state.cfg.grid_spacing_mm);

    // Verify the offline-compiled .engine files are present. They're
    // GPU-architecture + TRT-version specific; users compile them per rig
    // via scripts/compile_tensorrt_engines.sh.
    fs::path cd_eng = dir / "center_detect.engine";
    fs::path et_eng = dir / "hybridnet_efftrack.engine";
    fs::path h3_eng = dir / "hybrid3d.engine";
    if (!fs::exists(cd_eng) || !fs::exists(et_eng) || !fs::exists(h3_eng)) {
        std::fprintf(stderr,
            "[HybridNet] load FAILED: missing one or more .engine files in %s\n"
            "[HybridNet] run: scripts/compile_tensorrt_engines.sh %s\n",
            model_dir.c_str(), model_dir.c_str());
        return false;
    }

    state.gpu_device_id = gpu_device_id;
    cudaSetDevice(gpu_device_id);
    auto try_engine = [&](std::unique_ptr<jarvis_hn_trt::Engine> &out,
                          const fs::path &path, const char *label) -> bool {
        out = std::make_unique<jarvis_hn_trt::Engine>();
        if (!jarvis_hn_trt::load_engine(*out, path.string(), state.trt_logger)) {
            std::fprintf(stderr, "[HybridNet] load FAILED on %s (%s)\n",
                         label, path.string().c_str());
            out.reset();
            return false;
        }
        std::fprintf(stderr, "[HybridNet]   loaded TRT %s (%s)\n",
                     label, path.filename().string().c_str());
        jarvis_hn_trt::log_engine_io(*out, label);
        return true;
    };
    if (!try_engine(state.trt_center,   cd_eng, "CenterDetect") ||
        !try_engine(state.trt_efftrack, et_eng, "HN-effTrack") ||
        !try_engine(state.trt_hybrid3d, h3_eng, "Hybrid3D")) {
        jarvis_hybridnet_unload(state);
        return false;
    }

    // Small device scratch buffers used by predict_frame_device (GPU
    // preprocessing path). Sized for N cams; freed in unload.
    const int N = state.cfg.num_cameras;
    {
        cudaError_t cerr = cudaSuccess;
        cerr = cerr ? cerr : cudaMalloc((void **)&state.d_rgba_ptrs, N * sizeof(uint8_t *));
        cerr = cerr ? cerr : cudaMalloc((void **)&state.d_widths,    N * sizeof(int));
        cerr = cerr ? cerr : cudaMalloc((void **)&state.d_heights,   N * sizeof(int));
        cerr = cerr ? cerr : cudaMalloc((void **)&state.d_cx,        N * sizeof(int));
        cerr = cerr ? cerr : cudaMalloc((void **)&state.d_cy,        N * sizeof(int));
        if (cerr != cudaSuccess) {
            std::fprintf(stderr,
                "[HybridNet] cudaMalloc for device scratch failed: %s — "
                "device entry point will be unavailable\n",
                cudaGetErrorString(cerr));
        }
    }

    // Host scratch — sized from cfg.
    const int J = state.cfg.num_joints;
    const int C = state.cfg.center_image_size;        // 320
    const int B = state.cfg.keypoint_bbox_size;        // 704
    const int Hcen_hi = C / 2;                          // 160 — center high-res heatmap
    state.center_input.assign(N * 3 * C * C, 0.0f);
    state.center_out_high.assign(N * 1 * Hcen_hi * Hcen_hi, 0.0f);
    state.crop_input.assign(N * 3 * B * B, 0.0f);
    state.camera_matrices.assign(1 * N * 4 * 3, 0.0f);
    state.centerHM_input.assign(1 * N * 2, 0.0f);
    state.points3D_out.assign(1 * J * 3, 0.0f);
    state.confidences_out.assign(1 * J, 0.0f);

    state.loaded = true;
    std::fprintf(stderr, "[HybridNet] load SUCCEEDED (TRT direct runtime)\n");
    return true;
}

inline void jarvis_hybridnet_unload(JarvisHybridNetState &state) {
    state.trt_hybrid3d.reset();
    state.trt_efftrack.reset();
    state.trt_center.reset();
    if (state.d_rgba_ptrs) { cudaFree(state.d_rgba_ptrs); state.d_rgba_ptrs = nullptr; }
    if (state.d_widths)    { cudaFree(state.d_widths);    state.d_widths    = nullptr; }
    if (state.d_heights)   { cudaFree(state.d_heights);   state.d_heights   = nullptr; }
    if (state.d_cx)        { cudaFree(state.d_cx);        state.d_cx        = nullptr; }
    if (state.d_cy)        { cudaFree(state.d_cy);        state.d_cy        = nullptr; }
    state.loaded = false;
}

// ─────────────────────────────────────────────────────────────────────────
// Preprocessing helpers
// ─────────────────────────────────────────────────────────────────────────

// Bilinear-resize a uint8 RGB image (row-major HxWx3) to (dst_h x dst_w), then
// ImageNet-normalize and write as float CHW into `out` at the given camera
// slot. `out` is sized (N, 3, dst_h, dst_w) row-major; `slot_index` is which
// of the N batch entries to fill.
inline void jarvis_resize_normalize(
    const uint8_t *src, int src_w, int src_h,
    float *out, int slot_index, int dst_w, int dst_h,
    const std::array<float, 3> &mean,
    const std::array<float, 3> &std_) {
    const float sx = static_cast<float>(src_w) / dst_w;
    const float sy = static_cast<float>(src_h) / dst_h;
    const float inv_std[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};
    const size_t plane = static_cast<size_t>(dst_h) * dst_w;
    float *r_plane = out + slot_index * 3 * plane;
    float *g_plane = r_plane + plane;
    float *b_plane = g_plane + plane;
    for (int y = 0; y < dst_h; ++y) {
        const float src_y = (y + 0.5f) * sy - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        int y1 = y0 + 1;
        const float wy = src_y - y0;
        y0 = std::max(0, std::min(src_h - 1, y0));
        y1 = std::max(0, std::min(src_h - 1, y1));
        for (int x = 0; x < dst_w; ++x) {
            const float src_x = (x + 0.5f) * sx - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            int x1 = x0 + 1;
            const float wx = src_x - x0;
            x0 = std::max(0, std::min(src_w - 1, x0));
            x1 = std::max(0, std::min(src_w - 1, x1));
            const uint8_t *p00 = src + (y0 * src_w + x0) * 3;
            const uint8_t *p01 = src + (y0 * src_w + x1) * 3;
            const uint8_t *p10 = src + (y1 * src_w + x0) * 3;
            const uint8_t *p11 = src + (y1 * src_w + x1) * 3;
            for (int c = 0; c < 3; ++c) {
                const float v = (1 - wx) * (1 - wy) * p00[c] + wx * (1 - wy) * p01[c]
                              + (1 - wx) * wy       * p10[c] + wx * wy       * p11[c];
                const float norm = (v / 255.0f - mean[c]) * inv_std[c];
                float *plane_ptr = (c == 0 ? r_plane : c == 1 ? g_plane : b_plane);
                plane_ptr[y * dst_w + x] = norm;
            }
        }
    }
}

// Crop a square BxB patch from a uint8 RGB image centered at (cx, cy),
// ImageNet-normalize, and write as float CHW into `out[slot]`. The crop
// window is clamped to image bounds — caller is responsible for ensuring
// the center has already been clamped to keep the BxB window in-frame.
inline void jarvis_crop_normalize(
    const uint8_t *src, int src_w, int src_h,
    int cx, int cy, int B,
    float *out, int slot_index,
    const std::array<float, 3> &mean,
    const std::array<float, 3> &std_) {
    const float inv_std[3] = {1.0f / std_[0], 1.0f / std_[1], 1.0f / std_[2]};
    const int x0 = cx - B / 2;
    const int y0 = cy - B / 2;
    const size_t plane = static_cast<size_t>(B) * B;
    float *r_plane = out + slot_index * 3 * plane;
    float *g_plane = r_plane + plane;
    float *b_plane = g_plane + plane;
    for (int y = 0; y < B; ++y) {
        const int sy = std::max(0, std::min(src_h - 1, y0 + y));
        for (int x = 0; x < B; ++x) {
            const int sx = std::max(0, std::min(src_w - 1, x0 + x));
            const uint8_t *p = src + (sy * src_w + sx) * 3;
            r_plane[y * B + x] = (p[0] / 255.0f - mean[0]) * inv_std[0];
            g_plane[y * B + x] = (p[1] / 255.0f - mean[1]) * inv_std[1];
            b_plane[y * B + x] = (p[2] / 255.0f - mean[2]) * inv_std[2];
        }
    }
}

// Find the (x, y) of the max value in a (1, H, W) heatmap slot.
// Returns (peak_x, peak_y, max_val). The 0..1 channel is squeezed away.
inline std::tuple<int, int, float> jarvis_peak_pick(
    const float *heatmap, int W, int H) {
    int best_x = 0, best_y = 0;
    float best_v = heatmap[0];
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float v = heatmap[y * W + x];
            if (v > best_v) { best_v = v; best_x = x; best_y = y; }
        }
    }
    return {best_x, best_y, best_v};
}

// ─────────────────────────────────────────────────────────────────────────
// Shared stage helpers used by both predict_frame and predict_frame_device.
// Stage 2 (peak-pick + DLT triangulate) reads center_out_high which both
// paths populate identically (CenterDetect TRT D2H). Stage 5 (small aux
// inputs) and Stage 7 (write back) operate on host buffers only.
// ─────────────────────────────────────────────────────────────────────────

// Stage 2 — peak-pick CenterDetect heatmaps + DLT-triangulate the animal
// center in world coordinates. Returns false if fewer than 2 cams cleared
// the peak-confidence threshold (need 2 rays to triangulate). Updates
// state.last_center_cams_used.
inline bool jarvis_hn_compute_center_3d(
    JarvisHybridNetState &state,
    const std::vector<int> &widths,
    const std::vector<int> &heights,
    const std::vector<CameraParams> &camera_params,
    Eigen::Vector3d &center_3D)
{
    const int N = state.cfg.num_cameras;
    const int Hcen_hi = state.cfg.center_image_size / 2;
    const size_t cen_plane = static_cast<size_t>(Hcen_hi) * Hcen_hi;
    constexpr float kCenterDetectThreshold = 50.0f;  // matches JARVIS python

    std::vector<Eigen::Vector2d> center_2d_undist;
    std::vector<Eigen::Matrix<double, 3, 4>> center_proj_mats;
    const char *verbose_env = std::getenv("RED_HN_VERBOSE");
    const bool verbose = (verbose_env && verbose_env[0] == '1');
    if (verbose) std::fprintf(stderr, "[HybridNet] stage 2 per-cam:\n");
    for (int c = 0; c < N; ++c) {
        auto [px, py, v] = jarvis_peak_pick(
            state.center_out_high.data() + c * cen_plane, Hcen_hi, Hcen_hi);
        if (verbose) {
            std::fprintf(stderr,
                "  cam %2d: peak=(%d, %d) val=%.2f%s\n",
                c, px, py, v,
                v < kCenterDetectThreshold ? "  [BELOW THRESHOLD]" : "");
        }
        if (v < kCenterDetectThreshold) continue;
        // Project from heatmap (160²) to image (widths[c] × heights[c])
        // assuming the same letterbox-less affine the model trained on.
        const double nx = (px + 0.5) * widths[c]  / static_cast<double>(Hcen_hi);
        const double ny = (py + 0.5) * heights[c] / static_cast<double>(Hcen_hi);
        const auto &cp = camera_params[c];
        // Undistort for projective cams; telecentric calibration usually
        // bakes distortion into the projection matrix already, so we only
        // undistort there if non-zero dist coeffs are present (mirrors
        // red's gui_keypoints.h labeling convention).
        Eigen::Vector2d und = cp.telecentric
            ? red_math::undistortPointTelecentric(Eigen::Vector2d(nx, ny), cp.k, cp.dist_coeffs, cp.dist_center)
            : red_math::undistortPoint(Eigen::Vector2d(nx, ny), cp.k, cp.dist_coeffs);
        center_2d_undist.push_back(und);
        center_proj_mats.push_back(cp.projection_mat);
    }
    state.last_center_cams_used = static_cast<int>(center_2d_undist.size());
    if (center_2d_undist.size() < 2) return false;
    center_3D = red_math::triangulatePoints(center_2d_undist, center_proj_mats);
    return true;
}

// Stage 5 — build the three small host-side inputs Hybrid3D consumes
// (cameraMatrices, centerHM, center3D). The big heatmaps_padded input is
// filled on-device by the pad kernel in Stage 6.
inline void jarvis_hn_assemble_hybrid3d_aux_inputs(
    JarvisHybridNetState &state,
    const std::vector<CameraParams> &camera_params,
    const std::vector<int> &centerHM_x,
    const std::vector<int> &centerHM_y,
    const Eigen::Vector3d &center_3D)
{
    const int N = state.cfg.num_cameras;
    // camera_matrices: (1, N, 4, 3) = transposed projection_mat per cam.
    // P is (3, 4) col-major Eigen; we want P^T as (4, 3) row-major.
    for (int c = 0; c < N; ++c) {
        const auto &P = camera_params[c].projection_mat;
        float *out = state.camera_matrices.data() + c * 12;
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 3; ++col) {
                out[r * 3 + col] = static_cast<float>(P(col, r));
            }
        }
    }
    // centerHM: (1, N, 2) — native pixel coords (matches Python flow,
    // which built centerHM from manual bbox centers on the real images).
    for (int c = 0; c < N; ++c) {
        state.centerHM_input[c * 2 + 0] = static_cast<float>(centerHM_x[c]);
        state.centerHM_input[c * 2 + 1] = static_cast<float>(centerHM_y[c]);
    }
    state.center3D_input[0] = static_cast<float>(center_3D[0]);
    state.center3D_input[1] = static_cast<float>(center_3D[1]);
    state.center3D_input[2] = static_cast<float>(center_3D[2]);
}

// Stage 7 — write 3D keypoints into AnnotationMap and reproject them back
// to 2D for each camera's overlay display. kp.y is stored in ImPlot coords
// (origin bottom-left), so we flip from image coords (origin top-left).
inline void jarvis_hn_write_kp3d_and_2d_overlay(
    JarvisHybridNetState &state,
    AnnotationMap &annotations,
    SkeletonContext &skeleton,
    const std::vector<CameraParams> &camera_params,
    const std::vector<int> &heights,
    u32 frame_idx)
{
    const int N = state.cfg.num_cameras;
    const int J = state.cfg.num_joints;
    FrameAnnotation &fa = get_or_create_frame(
        annotations, frame_idx, skeleton.num_nodes, N);
    for (int j = 0; j < J; ++j) {
        fa.kp3d[j].x = state.points3D_out[j * 3 + 0];
        fa.kp3d[j].y = state.points3D_out[j * 3 + 1];
        fa.kp3d[j].z = state.points3D_out[j * 3 + 2];
        fa.kp3d[j].set_hybridnet(state.confidences_out[j]);
    }
    for (int c = 0; c < N; ++c) {
        const auto &cp = camera_params[c];
        for (int j = 0; j < J; ++j) {
            Eigen::Vector3d p3(fa.kp3d[j].x, fa.kp3d[j].y, fa.kp3d[j].z);
            Eigen::Vector2d uv = cp.telecentric
                ? red_math::projectPointTelecentric(p3, cp.projection_mat, cp.k, cp.dist_coeffs, cp.dist_center)
                : red_math::projectPointR(p3, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
            auto &kp = fa.cameras[c].keypoints[j];
            kp.x = uv[0];
            kp.y = static_cast<double>(heights[c]) - uv[1];
            kp.labeled = true;
            kp.source = LabelSource::Predicted;
            kp.confidence = state.confidences_out[j];
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Main predict entry point
// ─────────────────────────────────────────────────────────────────────────

inline bool jarvis_hybridnet_predict_frame(
    JarvisHybridNetState &state,
    const std::vector<const uint8_t *> &camera_rgb,
    const std::vector<int> &widths,
    const std::vector<int> &heights,
    const std::vector<CameraParams> &camera_params,
    AnnotationMap &annotations,
    SkeletonContext &skeleton,
    u32 frame_idx) {
    if (!state.loaded) return false;
    const int N = state.cfg.num_cameras;
    if ((int)camera_rgb.size() != N || (int)widths.size() != N ||
        (int)heights.size() != N || (int)camera_params.size() != N) {
        std::fprintf(stderr,
            "[HybridNet] aborting: expected %d cams, got rgb=%zu widths=%zu heights=%zu params=%zu\n",
            N, camera_rgb.size(), widths.size(), heights.size(), camera_params.size());
        return false;
    }
    // HN requires all N cams to have valid RGB + dims (model's input shape is
    // fixed). If the caller filtered by visibility (cam_included), nullptr
    // entries will show up here — refuse rather than crashing later in cuDNN.
    int n_missing = 0;
    for (int c = 0; c < N; ++c) {
        if (!camera_rgb[c] || widths[c] <= 0 || heights[c] <= 0) ++n_missing;
    }
    if (n_missing > 0) {
        std::fprintf(stderr,
            "[HybridNet] aborting: %d/%d cameras lack valid RGB at this frame "
            "(check that all camera windows are loaded + visible, or enable "
            "\"Predict from All\" in the JARVIS panel)\n", n_missing, N);
        return false;
    }
    // Clear any stale CUDA error from red's other GPU work (NVDEC, GL interop).
    // Without this the next TRT kernel inherits the sticky error and fails.
    cudaSetDevice(state.gpu_device_id);
    cudaError_t stale = cudaGetLastError();
    if (stale != cudaSuccess) {
        std::fprintf(stderr,
            "[HybridNet] cleared stale CUDA error before inference: %s\n",
            cudaGetErrorString(stale));
    }
    // Memory diagnostic: useful to see whether free GPU memory has shrunk
    // since model load (e.g., NVDEC kept decoding while user navigated).
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            std::fprintf(stderr,
                "[HybridNet] GPU memory at predict time: %.2f GiB free / %.2f GiB total\n",
                free_b / 1073741824.0, total_b / 1073741824.0);
        }
    }
    const int J = state.cfg.num_joints;
    const int C = state.cfg.center_image_size;     // 320
    const int B = state.cfg.keypoint_bbox_size;     // 704
    const int Hcen_hi = C / 2;                       // 160
    const int Heff_hi = B / 2;                       // 352
    const int Hpad   = state.cfg.heatmap_hw_padded();// 354
    const int bbox_hw = B / 2;                        // 352

    using clk = std::chrono::high_resolution_clock;

    // ── STAGE 1: CenterDetect (per-cam 2D center) ─────────────────────
    auto t0 = clk::now();
    for (int c = 0; c < N; ++c) {
        jarvis_resize_normalize(
            camera_rgb[c], widths[c], heights[c],
            state.center_input.data(), c, C, C,
            state.cfg.dataset_mean, state.cfg.dataset_std);
    }
    // Single batched call across all N cams. Engine input shape [N,3,C,C].
    {
        auto &eng = *state.trt_center;
        jarvis_hn_trt::Binding *in = nullptr;
        jarvis_hn_trt::Binding *out_high = nullptr;
        for (auto &kv : eng.bindings) {
            if (kv.second.is_input) in = &kv.second;
            else if (kv.second.dims.nbDims == 4 && kv.second.dims.d[2] == Hcen_hi)
                out_high = &kv.second;
        }
        if (!in || !out_high) {
            std::fprintf(stderr, "[HybridNet] CenterDetect TRT: required bindings missing\n");
            return false;
        }
        const size_t batch_in_bytes  = state.center_input.size()    * sizeof(float);
        const size_t batch_out_bytes = state.center_out_high.size() * sizeof(float);
        if (in->bytes != batch_in_bytes || out_high->bytes != batch_out_bytes) {
            std::fprintf(stderr,
                "[HybridNet] CenterDetect TRT shape mismatch: engine in=%zu out=%zu, "
                "expected batched in=%zu out=%zu — recompile engines with "
                "scripts/compile_tensorrt_engines.sh\n",
                in->bytes, out_high->bytes, batch_in_bytes, batch_out_bytes);
            return false;
        }
        if (!jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(in->d_ptr,
                state.center_input.data(), batch_in_bytes,
                cudaMemcpyHostToDevice, eng.stream), "CD H2D") ||
            !eng.context->enqueueV3(eng.stream) ||
            !jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(state.center_out_high.data(),
                out_high->d_ptr, batch_out_bytes,
                cudaMemcpyDeviceToHost, eng.stream), "CD D2H") ||
            !jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(eng.stream), "CD sync")) {
            return false;
        }
    }
    state.last_center_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    // ── STAGE 2: peak-pick + triangulate center_3D ───────────────────
    Eigen::Vector3d center_3D;
    if (!jarvis_hn_compute_center_3d(state, widths, heights, camera_params, center_3D))
        return false;

    // ── STAGE 3: reproject center_3D → centerHM, crop ────────────────
    // Project with full pinhole+distortion (or telecentric) so centerHM
    // lands on the actual animal in the image — matching how training data
    // constructed centerHM from manual bbox centers on the real (distorted)
    // images. Same projection functions red uses for manual-label workflows
    // (gui_keypoints.h reprojection).
    std::vector<int> centerHM_x(N), centerHM_y(N);
    for (int c = 0; c < N; ++c) {
        const auto &cp = camera_params[c];
        Eigen::Vector2d cHM = cp.telecentric
            ? red_math::projectPointTelecentric(center_3D, cp.projection_mat, cp.k, cp.dist_coeffs, cp.dist_center)
            : red_math::projectPointR(center_3D, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
        int cx = static_cast<int>(std::round(cHM[0]));
        int cy = static_cast<int>(std::round(cHM[1]));
        // Clamp so the BxB crop stays inside the image.
        cx = std::max(bbox_hw, std::min(widths[c]  - bbox_hw, cx));
        cy = std::max(bbox_hw, std::min(heights[c] - bbox_hw, cy));
        centerHM_x[c] = cx;
        centerHM_y[c] = cy;
        jarvis_crop_normalize(camera_rgb[c], widths[c], heights[c],
                              cx, cy, B,
                              state.crop_input.data(), c,
                              state.cfg.dataset_mean, state.cfg.dataset_std);
    }

    // ── STAGE 4: effTrack 2D heatmaps ────────────────────────────────
    // Single batched call across all N cams. Engine input shape [N,3,B,B].
    // Output stays on device — pad kernel later reads it directly into
    // Hybrid3D's input buffer.
    auto t_eff = clk::now();
    {
        auto &eng = *state.trt_efftrack;
        jarvis_hn_trt::Binding *in = nullptr;
        jarvis_hn_trt::Binding *out_high = nullptr;
        for (auto &kv : eng.bindings) {
            if (kv.second.is_input) in = &kv.second;
            else if (kv.second.dims.nbDims == 4 && kv.second.dims.d[2] == Heff_hi)
                out_high = &kv.second;
        }
        if (!in || !out_high) {
            std::fprintf(stderr, "[HybridNet] effTrack TRT: required bindings missing\n");
            return false;
        }
        const size_t batch_in_bytes  = state.crop_input.size() * sizeof(float);
        const size_t batch_out_bytes = (size_t)N * J * Heff_hi * Heff_hi * sizeof(float);
        if (in->bytes != batch_in_bytes || out_high->bytes != batch_out_bytes) {
            std::fprintf(stderr,
                "[HybridNet] effTrack TRT shape mismatch: engine in=%zu out=%zu, "
                "expected batched in=%zu out=%zu — recompile engines with "
                "scripts/compile_tensorrt_engines.sh\n",
                in->bytes, out_high->bytes, batch_in_bytes, batch_out_bytes);
            return false;
        }
        if (!jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(in->d_ptr,
                state.crop_input.data(), batch_in_bytes,
                cudaMemcpyHostToDevice, eng.stream), "ET H2D") ||
            !eng.context->enqueueV3(eng.stream) ||
            !jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(eng.stream), "ET sync")) {
            return false;
        }
    }
    state.last_efftrack_ms = std::chrono::duration<double, std::milli>(clk::now() - t_eff).count();

    // ── STAGE 5: assemble small Hybrid3D inputs (host) ───────────────
    // heatmaps_padded is filled on-device by the pad kernel in Stage 6.
    jarvis_hn_assemble_hybrid3d_aux_inputs(state, camera_params,
                                           centerHM_x, centerHM_y, center_3D);

    // ── STAGE 6: Hybrid3D ─────────────────────────────────────────────
    auto t_h3d = clk::now();
    {
        auto &eng = *state.trt_hybrid3d;
        // Run the device-side pad kernel on Hybrid3D's stream. It reads
        // from effTrack's output device buffer (safe — effTrack stream was
        // synced at end of Stage 4) and writes into Hybrid3D's
        // heatmaps_padded binding directly.
        {
            jarvis_hn_trt::Binding *src_b = nullptr;
            for (auto &kv : state.trt_efftrack->bindings) {
                if (!kv.second.is_input && kv.second.dims.nbDims == 4 &&
                    kv.second.dims.d[2] == Heff_hi) {
                    src_b = &kv.second; break;
                }
            }
            auto *dst_b = eng.get("heatmaps_padded");
            if (!src_b || !dst_b) {
                std::fprintf(stderr,
                    "[HybridNet] pad-on-GPU: missing effTrack output or "
                    "Hybrid3D heatmaps_padded binding\n");
                return false;
            }
            if (!jarvis_hn_trt::cuda_ok(
                    jarvis_hn_pad_heatmaps_device(
                        static_cast<const float *>(src_b->d_ptr),
                        static_cast<float *>(dst_b->d_ptr),
                        N, J, Heff_hi, Heff_hi, eng.stream),
                    "pad heatmaps")) {
                return false;
            }
        }
        auto h2d = [&](const char *name, const void *host, size_t host_bytes) -> bool {
            auto *b = eng.get(name);
            if (!b || !b->is_input) {
                std::fprintf(stderr, "[HybridNet] Hybrid3D TRT: input %s missing\n", name);
                return false;
            }
            if (b->bytes != host_bytes) {
                std::fprintf(stderr,
                    "[HybridNet] Hybrid3D TRT input %s size mismatch: engine=%zu host=%zu\n",
                    name, b->bytes, host_bytes);
                return false;
            }
            return jarvis_hn_trt::cuda_ok(
                cudaMemcpyAsync(b->d_ptr, host, b->bytes,
                                cudaMemcpyHostToDevice, eng.stream),
                "H3D H2D");
        };
        // heatmaps_padded was populated on-device by the pad kernel above;
        // we only H2D the three small CPU-built inputs.
        if (!h2d("centerHM",        state.centerHM_input.data(),
                 state.centerHM_input.size() * sizeof(float)) ||
            !h2d("center3D",        state.center3D_input.data(),
                 state.center3D_input.size() * sizeof(float)) ||
            !h2d("cameraMatrices",  state.camera_matrices.data(),
                 state.camera_matrices.size() * sizeof(float))) {
            return false;
        }
        if (!eng.context->enqueueV3(eng.stream)) {
            std::fprintf(stderr, "[HybridNet] Hybrid3D TRT enqueueV3 failed\n");
            return false;
        }
        auto d2h = [&](const char *name, void *host, size_t host_bytes) -> bool {
            auto *b = eng.get(name);
            if (!b || b->is_input) {
                std::fprintf(stderr, "[HybridNet] Hybrid3D TRT: output %s missing\n", name);
                return false;
            }
            if (b->bytes != host_bytes) {
                std::fprintf(stderr,
                    "[HybridNet] Hybrid3D TRT output %s size mismatch: engine=%zu host=%zu\n",
                    name, b->bytes, host_bytes);
                return false;
            }
            return jarvis_hn_trt::cuda_ok(
                cudaMemcpyAsync(host, b->d_ptr, b->bytes,
                                cudaMemcpyDeviceToHost, eng.stream),
                "H3D D2H");
        };
        if (!d2h("points3D",    state.points3D_out.data(),
                 state.points3D_out.size() * sizeof(float)) ||
            !d2h("confidences", state.confidences_out.data(),
                 state.confidences_out.size() * sizeof(float)) ||
            !jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(eng.stream), "H3D sync")) {
            return false;
        }
    }
    state.last_hybrid3d_ms = std::chrono::duration<double, std::milli>(clk::now() - t_h3d).count();

    // ── STAGE 7: write into AnnotationMap ────────────────────────────
    jarvis_hn_write_kp3d_and_2d_overlay(state, annotations, skeleton,
                                        camera_params, heights, frame_idx);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Device-input predict. Mirrors predict_frame but runs Stages 1 and 3
// preprocessing as CUDA kernels that write directly into the engines'
// input device buffers. Eliminates ~30 ms of host-side cudaMemcpy + RGBA→RGB
// strip, ~70 ms of CPU bilinear resize, and ~25 ms of CPU crop+normalize.
//
// Code duplication note: ~200 lines of Stages 2/5/6/7 are duplicated from
// predict_frame for now. A shared-helper refactor is staged but not yet
// landed; until then keep the two paths in sync by hand.
// ─────────────────────────────────────────────────────────────────────────
inline bool jarvis_hybridnet_predict_frame_device(
    JarvisHybridNetState &state,
    const std::vector<const uint8_t *> &camera_rgba_device,
    const std::vector<int> &widths,
    const std::vector<int> &heights,
    const std::vector<CameraParams> &camera_params,
    AnnotationMap &annotations,
    SkeletonContext &skeleton,
    u32 frame_idx) {
    if (!state.loaded) {
        std::fprintf(stderr,
            "[HybridNet] predict_frame_device: state not loaded\n");
        return false;
    }
    if (!state.d_rgba_ptrs || !state.d_widths || !state.d_heights ||
        !state.d_cx || !state.d_cy) {
        std::fprintf(stderr,
            "[HybridNet] predict_frame_device: device scratch buffers missing\n");
        return false;
    }
    const int N = state.cfg.num_cameras;
    if ((int)camera_rgba_device.size() != N || (int)widths.size() != N ||
        (int)heights.size() != N || (int)camera_params.size() != N) {
        std::fprintf(stderr,
            "[HybridNet] predict_frame_device: expected %d cams, got "
            "rgba=%zu widths=%zu heights=%zu params=%zu\n",
            N, camera_rgba_device.size(), widths.size(), heights.size(),
            camera_params.size());
        return false;
    }
    int n_missing = 0;
    for (int c = 0; c < N; ++c) {
        if (!camera_rgba_device[c] || widths[c] <= 0 || heights[c] <= 0) ++n_missing;
    }
    if (n_missing > 0) {
        std::fprintf(stderr,
            "[HybridNet] predict_frame_device: %d/%d cameras lack device RGBA\n",
            n_missing, N);
        return false;
    }

    cudaSetDevice(state.gpu_device_id);
    cudaError_t stale = cudaGetLastError();
    if (stale != cudaSuccess) {
        std::fprintf(stderr,
            "[HybridNet] cleared stale CUDA error before inference: %s\n",
            cudaGetErrorString(stale));
    }
    // Sync all device work before reading slot.frame from our engine streams.
    cudaDeviceSynchronize();

    // Defensive: this entry point is only valid when the input frames are
    // genuinely on the GPU (red's "GPU Buffer" mode). In CPU Buffer mode
    // slot.frame is plain malloc'd host memory and a device kernel can't
    // dereference it — checking once on cam 0 catches that mismatch and
    // lets the caller fall back to predict_frame's host path cleanly.
    {
        cudaPointerAttributes a{};
        cudaError_t aErr = cudaPointerGetAttributes(&a, camera_rgba_device[0]);
        if (aErr != cudaSuccess || a.type != cudaMemoryTypeDevice) {
            std::fprintf(stderr,
                "[HybridNet] predict_frame_device requires device-resident "
                "RGBA buffers; cam 0 ptr=%p is %s. Caller should use "
                "predict_frame (host RGB) when red is in CPU Buffer mode.\n",
                (const void *)camera_rgba_device[0],
                aErr != cudaSuccess ? "invalid" :
                (a.type == cudaMemoryTypeUnregistered ? "unregistered host memory" :
                 a.type == cudaMemoryTypeHost ? "registered host memory" :
                 a.type == cudaMemoryTypeManaged ? "managed memory" : "unknown"));
            return false;
        }
    }
    {
        size_t free_b = 0, total_b = 0;
        if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            std::fprintf(stderr,
                "[HybridNet] GPU memory at predict time: %.2f GiB free / %.2f GiB total\n",
                free_b / 1073741824.0, total_b / 1073741824.0);
        }
    }

    const int J = state.cfg.num_joints;
    const int C = state.cfg.center_image_size;     // 320
    const int B = state.cfg.keypoint_bbox_size;     // 704
    const int Hcen_hi = C / 2;                       // 160
    const int Heff_hi = B / 2;                       // 352
    const int bbox_hw = B / 2;                        // 352
    const float inv_std[3] = {
        1.0f / state.cfg.dataset_std[0],
        1.0f / state.cfg.dataset_std[1],
        1.0f / state.cfg.dataset_std[2]};

    using clk = std::chrono::high_resolution_clock;

    // ── STAGE 1: GPU resize_normalize → CenterDetect ─────────────────
    auto t0 = clk::now();
    {
        auto &eng = *state.trt_center;
        // Upload per-cam metadata to scratch. Pointer / int packs are small
        // (16x8 + 16x4 + 16x4 = 256 B); H2D is effectively free.
        if (!jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(state.d_rgba_ptrs,
                camera_rgba_device.data(), N * sizeof(uint8_t *),
                cudaMemcpyHostToDevice, eng.stream), "CD ptrs H2D") ||
            !jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(state.d_widths,
                widths.data(), N * sizeof(int),
                cudaMemcpyHostToDevice, eng.stream), "CD widths H2D") ||
            !jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(state.d_heights,
                heights.data(), N * sizeof(int),
                cudaMemcpyHostToDevice, eng.stream), "CD heights H2D")) {
            return false;
        }

        // Find CenterDetect's input + high-res output bindings.
        jarvis_hn_trt::Binding *in = nullptr;
        jarvis_hn_trt::Binding *out_high = nullptr;
        for (auto &kv : eng.bindings) {
            if (kv.second.is_input) in = &kv.second;
            else if (kv.second.dims.nbDims == 4 && kv.second.dims.d[2] == Hcen_hi)
                out_high = &kv.second;
        }
        if (!in || !out_high) {
            std::fprintf(stderr,
                "[HybridNet] CenterDetect TRT bindings missing (device path)\n");
            return false;
        }

        if (!jarvis_hn_trt::cuda_ok(
                jarvis_hn_resize_normalize_device(
                    state.d_rgba_ptrs, state.d_widths, state.d_heights,
                    static_cast<float *>(in->d_ptr),
                    N, C, state.cfg.dataset_mean.data(), inv_std, eng.stream),
                "CD resize+normalize kernel")) {
            return false;
        }
        if (!eng.context->enqueueV3(eng.stream)) {
            std::fprintf(stderr, "[HybridNet] CenterDetect TRT enqueueV3 failed (device path)\n");
            return false;
        }
        // D2H high-res heatmap to host — Stage 2 peak-pick still runs on CPU.
        if (!jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(
                state.center_out_high.data(), out_high->d_ptr,
                out_high->bytes, cudaMemcpyDeviceToHost, eng.stream),
                "CD D2H") ||
            !jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(eng.stream), "CD sync")) {
            return false;
        }
    }
    state.last_center_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    // ── STAGE 2: peak-pick + triangulate center_3D ───────────────────
    Eigen::Vector3d center_3D;
    if (!jarvis_hn_compute_center_3d(state, widths, heights, camera_params, center_3D))
        return false;

    // ── STAGE 3: reproject center_3D → centerHM, GPU crop+normalize ──
    std::vector<int> centerHM_x(N), centerHM_y(N);
    for (int c = 0; c < N; ++c) {
        const auto &cp = camera_params[c];
        Eigen::Vector2d cHM = cp.telecentric
            ? red_math::projectPointTelecentric(center_3D, cp.projection_mat, cp.k, cp.dist_coeffs, cp.dist_center)
            : red_math::projectPointR(center_3D, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
        int cx = static_cast<int>(std::round(cHM[0]));
        int cy = static_cast<int>(std::round(cHM[1]));
        cx = std::max(bbox_hw, std::min(widths[c]  - bbox_hw, cx));
        cy = std::max(bbox_hw, std::min(heights[c] - bbox_hw, cy));
        centerHM_x[c] = cx;
        centerHM_y[c] = cy;
    }

    // ── STAGE 4: effTrack via GPU crop kernel + TRT enqueue ──────────
    auto t_eff = clk::now();
    {
        auto &eng = *state.trt_efftrack;
        if (!jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(state.d_cx,
                centerHM_x.data(), N * sizeof(int),
                cudaMemcpyHostToDevice, eng.stream), "ET cx H2D") ||
            !jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(state.d_cy,
                centerHM_y.data(), N * sizeof(int),
                cudaMemcpyHostToDevice, eng.stream), "ET cy H2D")) {
            return false;
        }
        jarvis_hn_trt::Binding *in = nullptr;
        jarvis_hn_trt::Binding *out_high = nullptr;
        for (auto &kv : eng.bindings) {
            if (kv.second.is_input) in = &kv.second;
            else if (kv.second.dims.nbDims == 4 && kv.second.dims.d[2] == Heff_hi)
                out_high = &kv.second;
        }
        if (!in || !out_high) {
            std::fprintf(stderr,
                "[HybridNet] effTrack TRT bindings missing (device path)\n");
            return false;
        }
        if (!jarvis_hn_trt::cuda_ok(
                jarvis_hn_crop_normalize_device(
                    state.d_rgba_ptrs, state.d_widths, state.d_heights,
                    state.d_cx, state.d_cy,
                    static_cast<float *>(in->d_ptr),
                    N, B, state.cfg.dataset_mean.data(), inv_std, eng.stream),
                "ET crop+normalize kernel")) {
            return false;
        }
        if (!eng.context->enqueueV3(eng.stream) ||
            !jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(eng.stream), "ET sync")) {
            return false;
        }
    }
    state.last_efftrack_ms = std::chrono::duration<double, std::milli>(clk::now() - t_eff).count();

    // ── STAGE 5: assemble Hybrid3D's small CPU-side inputs ───────────
    jarvis_hn_assemble_hybrid3d_aux_inputs(state, camera_params,
                                           centerHM_x, centerHM_y, center_3D);

    // ── STAGE 6: Hybrid3D (pad on GPU + small H2Ds + enqueue) ────────
    auto t_h3d = clk::now();
    {
        auto &eng = *state.trt_hybrid3d;
        // Pad effTrack output device buffer into Hybrid3D's heatmaps_padded
        // device buffer using the device pad kernel. effTrack stream was
        // synced above, so its output is visible to Hybrid3D's stream.
        {
            jarvis_hn_trt::Binding *src_b = nullptr;
            for (auto &kv : state.trt_efftrack->bindings) {
                if (!kv.second.is_input && kv.second.dims.nbDims == 4 &&
                    kv.second.dims.d[2] == Heff_hi) {
                    src_b = &kv.second; break;
                }
            }
            auto *dst_b = eng.get("heatmaps_padded");
            if (!src_b || !dst_b) {
                std::fprintf(stderr,
                    "[HybridNet] pad-on-GPU bindings missing (device path)\n");
                return false;
            }
            if (!jarvis_hn_trt::cuda_ok(
                    jarvis_hn_pad_heatmaps_device(
                        static_cast<const float *>(src_b->d_ptr),
                        static_cast<float *>(dst_b->d_ptr),
                        N, J, Heff_hi, Heff_hi, eng.stream),
                    "pad heatmaps")) {
                return false;
            }
        }
        auto h2d = [&](const char *name, const void *host, size_t host_bytes) -> bool {
            auto *b = eng.get(name);
            if (!b || !b->is_input || b->bytes != host_bytes) return false;
            return jarvis_hn_trt::cuda_ok(
                cudaMemcpyAsync(b->d_ptr, host, b->bytes,
                                cudaMemcpyHostToDevice, eng.stream),
                "H3D H2D");
        };
        if (!h2d("centerHM",       state.centerHM_input.data(),
                 state.centerHM_input.size() * sizeof(float)) ||
            !h2d("center3D",       state.center3D_input.data(),
                 state.center3D_input.size() * sizeof(float)) ||
            !h2d("cameraMatrices", state.camera_matrices.data(),
                 state.camera_matrices.size() * sizeof(float))) {
            std::fprintf(stderr,
                "[HybridNet] Hybrid3D small-input H2D failed (device path)\n");
            return false;
        }
        if (!eng.context->enqueueV3(eng.stream)) {
            std::fprintf(stderr,
                "[HybridNet] Hybrid3D TRT enqueueV3 failed (device path)\n");
            return false;
        }
        auto d2h = [&](const char *name, void *host, size_t host_bytes) -> bool {
            auto *b = eng.get(name);
            if (!b || b->is_input || b->bytes != host_bytes) return false;
            return jarvis_hn_trt::cuda_ok(
                cudaMemcpyAsync(host, b->d_ptr, b->bytes,
                                cudaMemcpyDeviceToHost, eng.stream),
                "H3D D2H");
        };
        if (!d2h("points3D",    state.points3D_out.data(),
                 state.points3D_out.size() * sizeof(float)) ||
            !d2h("confidences", state.confidences_out.data(),
                 state.confidences_out.size() * sizeof(float)) ||
            !jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(eng.stream), "H3D sync")) {
            return false;
        }
    }
    state.last_hybrid3d_ms = std::chrono::duration<double, std::milli>(clk::now() - t_h3d).count();

    // ── STAGE 7: write into AnnotationMap ────────────────────────────
    jarvis_hn_write_kp3d_and_2d_overlay(state, annotations, skeleton,
                                        camera_params, heights, frame_idx);
    return true;
}

#endif // RED_HAS_TENSORRT_HN
#endif // __linux__ || _WIN32
