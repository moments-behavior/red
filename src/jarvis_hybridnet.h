#pragma once
// jarvis_hybridnet.h — Full 3D HybridNet pose estimation via ONNX Runtime
// ─────────────────────────────────────────────────────────────────────────
// Linux/Windows backend for the JARVIS 3-stage pipeline:
//
//   per-cam (16x):  raw RGB → resize 320² → CenterDetect ONNX → 2D peak
//   once:           triangulate via red_math DLT  → center_3D (world mm)
//   per-cam (16x):  reproject center_3D → centerHM (native pixel)
//                   crop 704² at centerHM → HybridNet effTrack ONNX → heatmaps (24, 352, 352)
//   once:           F.pad heatmaps → (1, 16, 24, 354, 354)
//                   build P=K·[R|t] (or telecentric DLT) per cam → cameraMatrices (1, 16, 4, 3)
//                   Hybrid3D ONNX → points3D (1, 24, 3) in world mm, confidences (1, 24)
//
// Mac retains its existing CoreML 2D+triangulate shortcut. This file does
// not compile on Apple.
//
// Designed to coexist with the existing jarvis_inference.h ORT path — the
// JARVIS Predict Tool panel dispatches to whichever backend the loaded model
// provides (presence of hybrid3d.onnx in the model dir == HybridNet mode).
// ─────────────────────────────────────────────────────────────────────────

#if defined(__linux__) || defined(_WIN32)
#ifdef RED_HAS_ONNXRUNTIME

#include "annotation.h"
#include "camera.h"
#include "red_math.h"
#include "skeleton.h"
#include "types.h"
#include "json.hpp"

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

#ifdef RED_HAS_TENSORRT_HN
#include <NvInfer.h>
#include <NvInferPlugin.h>
#endif

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
// TensorRT direct-runtime support (optional, enabled when offline-compiled
// .engine files are present alongside the .onnx files in model_dir).
//
// The engine wrapper is multi-input / multi-output and indexes bindings by
// tensor name — Hybrid3D has 4 inputs (heatmaps_padded, centerHM, center3D,
// cameraMatrices) and ≥2 outputs (points3D, confidences). All bindings get
// device buffers allocated up front from engine shapes; per-predict work is
// just H↔D memcpy + enqueueV3 + stream sync.
// ─────────────────────────────────────────────────────────────────────────
#ifdef RED_HAS_TENSORRT_HN
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
    // shapes resolve. Phase 2.1 compiles the 2D engines with min=opt=max=16,
    // so MAX gives us the batch=16 shape we want; for engines whose inputs
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
#endif // RED_HAS_TENSORRT_HN

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
// Runtime state. Holds 3 ORT sessions + preallocated scratch buffers.
// Scratch sizes are computed from cfg at load time and never re-allocated.
// ─────────────────────────────────────────────────────────────────────────
struct JarvisHybridNetState {
    bool loaded = false;
    JarvisHybridNetConfig cfg;

    // Backend dispatch: when true, the 3 stages run through TRT engines
    // below and the ORT sessions are unused. Set by jarvis_hybridnet_load
    // iff all 3 .engine files were found and deserialized.
    bool use_trt = false;

#ifdef RED_HAS_TENSORRT_HN
    jarvis_hn_trt::Logger trt_logger;
    std::unique_ptr<jarvis_hn_trt::Engine> trt_center;     // center_detect.engine
    std::unique_ptr<jarvis_hn_trt::Engine> trt_efftrack;   // hybridnet_efftrack.engine
    std::unique_ptr<jarvis_hn_trt::Engine> trt_hybrid3d;   // hybrid3d.engine
#endif

    // ORT environment + sessions (fallback path)
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> center_session;     // center_detect.onnx
    std::unique_ptr<Ort::Session> efftrack_session;   // hybridnet_efftrack.onnx
    std::unique_ptr<Ort::Session> hybrid3d_session;   // hybrid3d.onnx

    Ort::MemoryInfo mem_info{nullptr};

    // Per-session input/output tensor names (cached at load time)
    std::vector<std::string> center_input_names, center_output_names;
    std::vector<std::string> efftrack_input_names, efftrack_output_names;
    std::vector<std::string> hybrid3d_input_names, hybrid3d_output_names;

    // Scratch buffers — sized once at load, reused per frame.
    // Layout matches what each ONNX expects (CHW float32, NCHW batched).
    std::vector<float> center_input;       // (N, 3, 320, 320)
    std::vector<float> center_out_low;     // (N, 1, 80, 80)   — discarded
    std::vector<float> center_out_high;    // (N, 1, 160, 160) — peak-picked

    std::vector<float> crop_input;         // (N, 3, 704, 704)
    std::vector<float> eff_out_low;        // (N, 24, 176, 176) — discarded
    std::vector<float> eff_out_high;       // (N, 24, 352, 352) — padded next

    std::vector<float> heatmaps_padded;    // (1, N, 24, 354, 354)
    std::vector<float> camera_matrices;    // (1, N, 4, 3)
    std::vector<float> centerHM_input;     // (1, N, 2) float (native pixel coords)
    std::array<float, 3> center3D_input{}; // (1, 3) float (world mm)

    std::vector<float> heatmap_final_out;  // (1, 24, 50, 50, 50) — usually discarded
    std::vector<float> points3D_out;       // (1, 24, 3)
    std::vector<float> confidences_out;    // (1, 24)

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
// jarvis_hybridnet_load: reads manifest.json, opens 3 ONNX sessions on CUDA
// EP, sizes scratch buffers. Returns false on any error; state.loaded stays
// false. Safe to call repeatedly (replaces existing).
//
// jarvis_hybridnet_unload: tears down sessions and frees memory.
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
// any ORT call errors. On success, fills annotations and updates state's
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

// Caches all input/output names for a session as owned strings.
inline void jarvis_cache_session_io_names(Ort::Session &sess,
                                          std::vector<std::string> &in_names,
                                          std::vector<std::string> &out_names) {
    Ort::AllocatorWithDefaultOptions alloc;
    in_names.clear();
    out_names.clear();
    for (size_t i = 0; i < sess.GetInputCount(); ++i) {
        auto p = sess.GetInputNameAllocated(i, alloc);
        in_names.emplace_back(p.get());
    }
    for (size_t i = 0; i < sess.GetOutputCount(); ++i) {
        auto p = sess.GetOutputNameAllocated(i, alloc);
        out_names.emplace_back(p.get());
    }
}

inline bool jarvis_hybridnet_load(JarvisHybridNetState &state,
                                   const std::string &model_dir,
                                   int gpu_device_id) {
    namespace fs = std::filesystem;
    jarvis_hybridnet_unload(state);  // idempotent reset
    std::fprintf(stderr, "[HybridNet] load: %s\n", model_dir.c_str());

    // Report current GPU memory before any ORT allocations so the user can
    // see whether red has already saturated the device.
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

    // ── TensorRT direct-runtime path ────────────────────────────────────
    // If .engine files are present alongside the .onnx files, deserialize
    // them and skip ORT entirely. Engines are GPU-arch + TRT-version
    // specific — generate via scripts/compile_tensorrt_engines.sh per rig.
    // Set RED_HN_DISABLE_TRT=1 to force the ORT fallback for diagnostics.
#ifdef RED_HAS_TENSORRT_HN
    const char *disable_trt_env = std::getenv("RED_HN_DISABLE_TRT");
    const bool disable_trt = (disable_trt_env && disable_trt_env[0] == '1');
    fs::path cd_eng = dir / "center_detect.engine";
    fs::path et_eng = dir / "hybridnet_efftrack.engine";
    fs::path h3_eng = dir / "hybrid3d.engine";
    const bool all_engines_present =
        fs::exists(cd_eng) && fs::exists(et_eng) && fs::exists(h3_eng);
    if (disable_trt) {
        std::fprintf(stderr, "[HybridNet]   RED_HN_DISABLE_TRT=1 — forcing ORT fallback\n");
    } else if (!all_engines_present) {
        std::fprintf(stderr,
            "[HybridNet]   no .engine files in %s — using ORT CUDA EP "
            "(run scripts/compile_tensorrt_engines.sh to enable TRT)\n",
            model_dir.c_str());
    } else {
        cudaSetDevice(gpu_device_id);
        auto try_engine = [&](std::unique_ptr<jarvis_hn_trt::Engine> &out,
                              const fs::path &path, const char *label) -> bool {
            out = std::make_unique<jarvis_hn_trt::Engine>();
            if (!jarvis_hn_trt::load_engine(*out, path.string(), state.trt_logger)) {
                std::fprintf(stderr, "[HybridNet] TRT load FAILED on %s — will retry with ORT\n", label);
                out.reset();
                return false;
            }
            std::fprintf(stderr, "[HybridNet]   loaded TRT %s (%s)\n",
                         label, path.filename().string().c_str());
            jarvis_hn_trt::log_engine_io(*out, label);
            return true;
        };
        bool ok = try_engine(state.trt_center,   cd_eng, "CenterDetect");
        ok = ok && try_engine(state.trt_efftrack, et_eng, "HN-effTrack");
        ok = ok && try_engine(state.trt_hybrid3d, h3_eng, "Hybrid3D");
        if (ok) {
            state.use_trt = true;
            std::fprintf(stderr,
                "[HybridNet]   TRT path active — ORT CUDA EP will be skipped\n");
        } else {
            // Tear down any partial TRT state and fall through to ORT init.
            state.trt_center.reset();
            state.trt_efftrack.reset();
            state.trt_hybrid3d.reset();
        }
    }
#endif

    // ── ORT fallback path (used when TRT engines are absent or disabled)
    if (!state.use_trt) {
    try {
        state.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "red_jarvis_hybridnet");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[HybridNet] load FAILED: Ort::Env construction threw: %s\n", e.what());
        return false;
    }

    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // CPU-only diagnostic: when RED_HN_CPU_ONLY=1 is set in the environment,
    // skip CUDA EP entirely. Useful for isolating CUDA state-contamination
    // issues between red's NVDEC/GL work and ORT.
    const char *cpu_only_env = std::getenv("RED_HN_CPU_ONLY");
    const bool force_cpu = (cpu_only_env && cpu_only_env[0] == '1');
    // GPU override: RED_HN_GPU=N picks which CUDA device ORT runs on. Default
    // is gpu_device_id (typically 0). On this 9-GPU box, red's NVDEC + GL
    // pipeline saturates the RTX 4000 Ada (device 0); ORT's ~6 GB ReduceMean
    // can't fit alongside. Setting RED_HN_GPU to an idle A16 (e.g. 8) gives
    // ORT its own device with 14+ GiB free.
    const char *gpu_env = std::getenv("RED_HN_GPU");
    if (gpu_env) {
        try {
            gpu_device_id = std::stoi(gpu_env);
            std::fprintf(stderr,
                "[HybridNet]   RED_HN_GPU=%d — ORT will run on this device\n",
                gpu_device_id);
        } catch (...) { /* keep default */ }
    }
    bool cuda_ep_attached = false;
    if (force_cpu) {
        std::fprintf(stderr, "[HybridNet]   RED_HN_CPU_ONLY=1 — CPU EP only (slow, diagnostic)\n");
    } else {
        // Use V2 CUDA provider options so we can constrain memory usage. The
        // Hybrid3D ReduceMean op tries to allocate ~6 GB for the per-voxel
        // mean across 16 cams × 24 joints × 100³ voxels, which fails when
        // red's NVDEC + display buffers have already fragmented the device
        // memory. kSameAsRequested prevents BFC from over-reserving; setting
        // cudnn_conv_use_max_workspace=0 also lowers conv workspace pressure.
        try {
            OrtCUDAProviderOptionsV2 *cuda_options_v2 = nullptr;
            const OrtApi &api = Ort::GetApi();
            Ort::ThrowOnError(api.CreateCUDAProviderOptions(&cuda_options_v2));
            std::string dev_id_str = std::to_string(gpu_device_id);
            // arena_extend_strategy=kSameAsRequested:
            //   Each Op's chunk is allocated when needed and released within
            //   the same Run(). Peak memory per Run() = max of any single
            //   op (~6 GiB for Hybrid3D's ReduceMean), not the sum.
            //
            // kNextPowerOfTwo (the previous setting) cached chunks across
            // Runs but caused unbounded arena growth WITHIN a single Run:
            // reproLayer's 6 GiB ReduceMean reserved an 8 GiB chunk, then
            // V2VNet's first conv tried to extend the arena by another
            // power-of-two chunk, exceeding free memory.
            //
            // kSameAsRequested only works as a sustained strategy when red's
            // NVDEC doesn't grow memory between predicts — which is true
            // once playback buffer is set small enough (24 confirmed safe).
            std::vector<const char *> keys = {
                "device_id",
                "arena_extend_strategy",
                "cudnn_conv_use_max_workspace",
                "do_copy_in_default_stream",
            };
            std::vector<const char *> values = {
                dev_id_str.c_str(),
                "kSameAsRequested",
                "0",
                "1",
            };
            Ort::ThrowOnError(api.UpdateCUDAProviderOptions(
                cuda_options_v2, keys.data(), values.data(), keys.size()));
            Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(
                static_cast<OrtSessionOptions *>(opts), cuda_options_v2));
            api.ReleaseCUDAProviderOptions(cuda_options_v2);
            cuda_ep_attached = true;
            std::fprintf(stderr,
                "[HybridNet]   CUDA EP attached V2 (device %d, "
                "arena=kSameAsRequested, cudnn_conv_workspace=min)\n", gpu_device_id);
        } catch (const std::exception &e) {
            std::fprintf(stderr,
                "[HybridNet]   CUDA EP V2 attach FAILED: %s — falling back to CPU EP\n", e.what());
        }
    }

    auto load_session = [&](const fs::path &path,
                             std::unique_ptr<Ort::Session> &out,
                             const char *label) -> bool {
        try {
            out = std::make_unique<Ort::Session>(*state.env, path.c_str(), opts);
            std::fprintf(stderr, "[HybridNet]   loaded %s (%s)\n", label, path.filename().string().c_str());
            return true;
        } catch (const std::exception &e) {
            std::fprintf(stderr, "[HybridNet] load FAILED on %s (%s): %s\n",
                         label, path.string().c_str(), e.what());
            return false;
        }
    };
    if (!load_session(dir / "center_detect.onnx",      state.center_session,   "CenterDetect")) {
        jarvis_hybridnet_unload(state); return false;
    }
    if (!load_session(dir / "hybridnet_efftrack.onnx", state.efftrack_session, "HN-effTrack")) {
        jarvis_hybridnet_unload(state); return false;
    }
    if (!load_session(dir / "hybrid3d.onnx",           state.hybrid3d_session, "Hybrid3D")) {
        jarvis_hybridnet_unload(state); return false;
    }
    (void)cuda_ep_attached;

    jarvis_cache_session_io_names(*state.center_session,   state.center_input_names,   state.center_output_names);
    jarvis_cache_session_io_names(*state.efftrack_session, state.efftrack_input_names, state.efftrack_output_names);
    jarvis_cache_session_io_names(*state.hybrid3d_session, state.hybrid3d_input_names, state.hybrid3d_output_names);

    state.mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    }  // end if (!state.use_trt) — ORT init block

    // Preallocate scratch — sizes derived from cfg.
    const int N = state.cfg.num_cameras;
    const int J = state.cfg.num_joints;
    const int C = state.cfg.center_image_size;        // 320
    const int B = state.cfg.keypoint_bbox_size;        // 704
    const int Hcen_hi = C / 2;                          // 160 — center high-res heatmap
    const int Hcen_lo = C / 4;                          //  80 — low-res (discarded)
    const int Heff_hi = B / 2;                          // 352 — effTrack high-res
    const int Heff_lo = B / 4;                          // 176 — low-res (discarded)
    const int Hpad   = state.cfg.heatmap_hw_padded();   // 354 — after F.pad
    const int Gh    = state.cfg.voxel_grid_half();      //  50 — voxel side

    state.center_input.assign(N * 3 * C * C, 0.0f);
    state.center_out_low.assign(N * 1 * Hcen_lo * Hcen_lo, 0.0f);
    state.center_out_high.assign(N * 1 * Hcen_hi * Hcen_hi, 0.0f);
    state.crop_input.assign(N * 3 * B * B, 0.0f);
    state.eff_out_low.assign(N * J * Heff_lo * Heff_lo, 0.0f);
    state.eff_out_high.assign(N * J * Heff_hi * Heff_hi, 0.0f);
    state.heatmaps_padded.assign(1 * N * J * Hpad * Hpad, 0.0f);
    state.camera_matrices.assign(1 * N * 4 * 3, 0.0f);
    state.centerHM_input.assign(1 * N * 2, 0.0f);
    state.heatmap_final_out.assign(1 * J * Gh * Gh * Gh, 0.0f);
    state.points3D_out.assign(1 * J * 3, 0.0f);
    state.confidences_out.assign(1 * J, 0.0f);

    state.loaded = true;

    // TRT engines are statically planned — no warmup needed (engine workspace
    // was reserved at deserializeCudaEngine time). Return immediately.
    if (state.use_trt) {
        std::fprintf(stderr, "[HybridNet] load SUCCEEDED (TRT direct runtime)\n");
        return true;
    }

    // Warmup (off by default): if RED_HN_WARMUP=1, run each session once
    // with dummy inputs so the arena allocates its working chunks now.
    // Off because a warmup OOM corrupts the session's BFC arena state and
    // any subsequent real Predict produces nonsense shape errors (we saw
    // an 86 TiB allocation request). Without warmup, the first real
    // Predict allocates fresh with kNextPowerOfTwo, which keeps the
    // chunks cached for repeat predicts.
    const char *warmup_env = std::getenv("RED_HN_WARMUP");
    const bool do_warmup = (warmup_env && warmup_env[0] == '1');
    if (!do_warmup) {
        std::fprintf(stderr, "[HybridNet] load SUCCEEDED (warmup skipped)\n");
        return true;
    }
    try {
        // CenterDetect warmup: (N, 3, C, C)
        {
            std::array<int64_t, 4> in_shape{N, 3, C, C};
            Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                state.mem_info, state.center_input.data(), state.center_input.size(),
                in_shape.data(), in_shape.size());
            std::vector<const char *> in_names_c, out_names_c;
            for (auto &s : state.center_input_names)  in_names_c.push_back(s.c_str());
            for (auto &s : state.center_output_names) out_names_c.push_back(s.c_str());
            (void)state.center_session->Run(Ort::RunOptions{nullptr},
                in_names_c.data(), &in_tensor, 1,
                out_names_c.data(), out_names_c.size());
        }
        // effTrack warmup: (N, 3, B, B)
        {
            std::array<int64_t, 4> in_shape{N, 3, B, B};
            Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
                state.mem_info, state.crop_input.data(), state.crop_input.size(),
                in_shape.data(), in_shape.size());
            std::vector<const char *> in_names_c, out_names_c;
            for (auto &s : state.efftrack_input_names)  in_names_c.push_back(s.c_str());
            for (auto &s : state.efftrack_output_names) out_names_c.push_back(s.c_str());
            (void)state.efftrack_session->Run(Ort::RunOptions{nullptr},
                in_names_c.data(), &in_tensor, 1,
                out_names_c.data(), out_names_c.size());
        }
        // Hybrid3D warmup: 4 inputs at full shape
        {
            std::array<int64_t, 5> hm_shape{1, N, J, Hpad, Hpad};
            std::array<int64_t, 3> chm_shape{1, N, 2};
            std::array<int64_t, 2> c3d_shape{1, 3};
            std::array<int64_t, 4> mat_shape{1, N, 4, 3};
            std::vector<Ort::Value> ins;
            ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
                state.heatmaps_padded.data(), state.heatmaps_padded.size(),
                hm_shape.data(), hm_shape.size()));
            ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
                state.centerHM_input.data(), state.centerHM_input.size(),
                chm_shape.data(), chm_shape.size()));
            ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
                state.center3D_input.data(), state.center3D_input.size(),
                c3d_shape.data(), c3d_shape.size()));
            ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
                state.camera_matrices.data(), state.camera_matrices.size(),
                mat_shape.data(), mat_shape.size()));
            std::vector<const char *> in_names_c, out_names_c;
            for (auto &s : state.hybrid3d_input_names) in_names_c.push_back(s.c_str());
            for (auto &s : state.hybrid3d_output_names) out_names_c.push_back(s.c_str());
            // Reorder to expected input order
            std::vector<Ort::Value> ins_ordered;
            ins_ordered.reserve(in_names_c.size());
            const char *want[] = {"heatmaps_padded", "centerHM", "center3D", "cameraMatrices"};
            for (size_t i = 0; i < in_names_c.size(); ++i) {
                for (size_t w = 0; w < 4; ++w) {
                    if (std::strcmp(in_names_c[i], want[w]) == 0) {
                        ins_ordered.push_back(std::move(ins[w]));
                        break;
                    }
                }
            }
            (void)state.hybrid3d_session->Run(Ort::RunOptions{nullptr},
                in_names_c.data(), ins_ordered.data(), ins_ordered.size(),
                out_names_c.data(), out_names_c.size());
        }
        std::fprintf(stderr,
            "[HybridNet]   warmup OK — ORT arena chunks reserved\n");
    } catch (const std::exception &e) {
        // Warmup OOM corrupts ORT's BFC arena state and any subsequent
        // real predict produces a nonsense shape error (we observed an
        // 86 TiB allocation request after a warmup OOM). The session is
        // unrecoverable; fully unload so the user can re-attempt with
        // less memory pressure.
        std::fprintf(stderr,
            "[HybridNet] warmup FAILED: %s\n"
            "[HybridNet] unloading sessions (warmup OOM leaves arena in a "
            "broken state). Reduce playback buffer size, then re-select "
            "the JARVIS model in the panel to retry.\n",
            e.what());
        jarvis_hybridnet_unload(state);
        return false;
    }

    std::fprintf(stderr, "[HybridNet] load SUCCEEDED\n");
    return true;
}

inline void jarvis_hybridnet_unload(JarvisHybridNetState &state) {
    state.hybrid3d_session.reset();
    state.efftrack_session.reset();
    state.center_session.reset();
    state.env.reset();
#ifdef RED_HAS_TENSORRT_HN
    state.trt_hybrid3d.reset();
    state.trt_efftrack.reset();
    state.trt_center.reset();
#endif
    state.use_trt = false;
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

// Reshape (N, J, H, W) → (1, N, J, H+2, W+2) with zero-pad of 1 on each
// spatial side. Replicates F.pad(heatmaps, [1,1,1,1], 'constant', 0).
inline void jarvis_pad_heatmaps(
    const float *src, int N, int J, int H, int W,
    float *dst, int Hpad) {
    // dst layout: (1, N, J, Hpad, Wpad) row-major. Wpad == Hpad here.
    const int Wpad = Hpad;
    std::fill(dst, dst + static_cast<size_t>(N) * J * Hpad * Wpad, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int j = 0; j < J; ++j) {
            const float *src_jc = src + ((static_cast<size_t>(n) * J + j) * H) * W;
            float *dst_jc = dst + ((static_cast<size_t>(n) * J + j) * Hpad + 1) * Wpad;  // +1 row offset
            for (int y = 0; y < H; ++y) {
                std::memcpy(dst_jc + y * Wpad + 1,  // +1 col offset
                            src_jc + y * W,
                            sizeof(float) * W);
            }
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
    // ORT's CUDA EP otherwise inherits the sticky error and the next kernel
    // (typically the first Conv) fails with cudaErrorInvalidValue.
    cudaSetDevice(0);
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
#ifdef RED_HAS_TENSORRT_HN
    if (state.use_trt) {
        // Single batched call across all N cams. Engines compiled by Phase 2.1
        // of scripts/compile_tensorrt_engines.sh have input shape [N,3,C,C].
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
    } else
#endif
    try {
        std::array<int64_t, 4> in_shape{N, 3, C, C};
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            state.mem_info, state.center_input.data(), state.center_input.size(),
            in_shape.data(), in_shape.size());
        std::vector<const char *> in_names_c, out_names_c;
        for (auto &s : state.center_input_names)  in_names_c.push_back(s.c_str());
        for (auto &s : state.center_output_names) out_names_c.push_back(s.c_str());
        auto outs = state.center_session->Run(
            Ort::RunOptions{nullptr},
            in_names_c.data(), &in_tensor, 1,
            out_names_c.data(), out_names_c.size());
        // CenterDetect exports two outputs: low-res (80x80) and high-res (160x160).
        // We use the high-res one — find it by output count's matching shape.
        for (size_t oi = 0; oi < outs.size(); ++oi) {
            auto info = outs[oi].GetTensorTypeAndShapeInfo();
            auto shape = info.GetShape();
            if (shape.size() == 4 && shape[2] == Hcen_hi) {
                std::memcpy(state.center_out_high.data(),
                            outs[oi].GetTensorData<float>(),
                            state.center_out_high.size() * sizeof(float));
                break;
            }
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[HybridNet] CenterDetect ORT Run failed: %s\n", e.what());
        return false;
    }
    state.last_center_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    // ── STAGE 2: peak-pick + triangulate center_3D ───────────────────
    // For projective cameras: undistort 2D centers, then DLT-triangulate via
    // the projection_mat (=K·[R|t], no distortion). Matches red's standard
    // labeling path AND JARVIS use_dlt=False `reconstructPoint` (which also
    // undistorts first). The model's internal reproLayer always does pure
    // DLT, but centerHM must match where the animal actually is in the
    // image (training built centerHM from manual bbox centers, not DLT
    // reprojections), so we work in real-image coords through Stage 3 and
    // only feed pure-DLT-style matrices to the network.
    //
    // For telecentric: skip undistortion (the telecentric calibration form
    // typically encodes everything in the projection matrix; the user
    // confirmed this varies by DLT-telecentric version, so we mirror red's
    // gui_keypoints.h convention via undistortPointTelecentric only when
    // distortion is present).
    std::vector<Eigen::Vector2d> center_2d_undist;
    std::vector<Eigen::Matrix<double, 3, 4>> center_proj_mats;
    constexpr float kCenterDetectThreshold = 50.0f;  // matches JARVIS python
    const size_t cen_plane = static_cast<size_t>(Hcen_hi) * Hcen_hi;
    // Per-cam diagnostic gated behind RED_HN_VERBOSE=1. When enabled, prints
    // peak value/location and input-buffer mean intensity (sanity check for
    // black/uninitialized frames) for each of the N cameras.
    const char *verbose_env = std::getenv("RED_HN_VERBOSE");
    const bool verbose = (verbose_env && verbose_env[0] == '1');
    if (verbose) std::fprintf(stderr, "[HybridNet] stage 2 per-cam:\n");
    for (int c = 0; c < N; ++c) {
        auto [px, py, v] = jarvis_peak_pick(
            state.center_out_high.data() + c * cen_plane, Hcen_hi, Hcen_hi);
        if (verbose) {
            const size_t plane = static_cast<size_t>(C) * C;
            double accum = 0.0; int n = 0;
            for (size_t i = 0; i < plane; i += 64) {
                accum += state.center_input[c * 3 * plane + i];
                ++n;
            }
            double mean_norm = accum / n;
            std::fprintf(stderr,
                "  cam %2d: peak=(%d, %d) val=%.2f  input_mean(norm)=%.3f%s\n",
                c, px, py, v, mean_norm,
                v < kCenterDetectThreshold ? "  [BELOW THRESHOLD]" : "");
        }
        if (v < kCenterDetectThreshold) continue;
        const double nx = (px + 0.5) * widths[c]  / static_cast<double>(Hcen_hi);
        const double ny = (py + 0.5) * heights[c] / static_cast<double>(Hcen_hi);
        const auto &cp = camera_params[c];
        Eigen::Vector2d und = cp.telecentric
            ? red_math::undistortPointTelecentric(Eigen::Vector2d(nx, ny), cp.k, cp.dist_coeffs)
            : red_math::undistortPoint(Eigen::Vector2d(nx, ny), cp.k, cp.dist_coeffs);
        center_2d_undist.push_back(und);
        center_proj_mats.push_back(cp.projection_mat);
    }
    state.last_center_cams_used = static_cast<int>(center_2d_undist.size());
    if (center_2d_undist.size() < 2) return false;
    Eigen::Vector3d center_3D =
        red_math::triangulatePoints(center_2d_undist, center_proj_mats);

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
            ? red_math::projectPointTelecentric(center_3D, cp.projection_mat, cp.k, cp.dist_coeffs)
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
    auto t_eff = clk::now();
#ifdef RED_HAS_TENSORRT_HN
    if (state.use_trt) {
        // Single batched call across all N cams. Engine input shape [N,3,B,B].
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
        const size_t batch_in_bytes  = state.crop_input.size()   * sizeof(float);
        const size_t batch_out_bytes = state.eff_out_high.size() * sizeof(float);
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
            !jarvis_hn_trt::cuda_ok(cudaMemcpyAsync(state.eff_out_high.data(),
                out_high->d_ptr, batch_out_bytes,
                cudaMemcpyDeviceToHost, eng.stream), "ET D2H") ||
            !jarvis_hn_trt::cuda_ok(cudaStreamSynchronize(eng.stream), "ET sync")) {
            return false;
        }
    } else
#endif
    try {
        std::array<int64_t, 4> in_shape{N, 3, B, B};
        Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
            state.mem_info, state.crop_input.data(), state.crop_input.size(),
            in_shape.data(), in_shape.size());
        std::vector<const char *> in_names_c, out_names_c;
        for (auto &s : state.efftrack_input_names)  in_names_c.push_back(s.c_str());
        for (auto &s : state.efftrack_output_names) out_names_c.push_back(s.c_str());
        auto outs = state.efftrack_session->Run(
            Ort::RunOptions{nullptr},
            in_names_c.data(), &in_tensor, 1,
            out_names_c.data(), out_names_c.size());
        for (size_t oi = 0; oi < outs.size(); ++oi) {
            auto shape = outs[oi].GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 4 && shape[2] == Heff_hi) {
                std::memcpy(state.eff_out_high.data(),
                            outs[oi].GetTensorData<float>(),
                            state.eff_out_high.size() * sizeof(float));
                break;
            }
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[HybridNet] effTrack ORT Run failed: %s\n", e.what());
        return false;
    }
    state.last_efftrack_ms = std::chrono::duration<double, std::milli>(clk::now() - t_eff).count();

    // ── STAGE 5: pad + assemble Hybrid3D inputs ──────────────────────
    jarvis_pad_heatmaps(state.eff_out_high.data(), N, J, Heff_hi, Heff_hi,
                        state.heatmaps_padded.data(), Hpad);
    // camera_matrices: (1, N, 4, 3) = transposed projection_mat per cam.
    for (int c = 0; c < N; ++c) {
        const auto &P = camera_params[c].projection_mat;  // (3, 4) col-major Eigen
        float *out = state.camera_matrices.data() + c * 12;
        // Want P^T as (4, 3) row-major. P[r, c] → P^T[c, r].
        for (int r = 0; r < 4; ++r) {
            for (int col = 0; col < 3; ++col) {
                out[r * 3 + col] = static_cast<float>(P(col, r));
            }
        }
    }
    // centerHM: (1, N, 2) — in NATIVE PIXEL COORDS (matches Python flow)
    for (int c = 0; c < N; ++c) {
        state.centerHM_input[c * 2 + 0] = static_cast<float>(centerHM_x[c]);
        state.centerHM_input[c * 2 + 1] = static_cast<float>(centerHM_y[c]);
    }
    state.center3D_input[0] = static_cast<float>(center_3D[0]);
    state.center3D_input[1] = static_cast<float>(center_3D[1]);
    state.center3D_input[2] = static_cast<float>(center_3D[2]);

    // ── STAGE 6: Hybrid3D ─────────────────────────────────────────────
    auto t_h3d = clk::now();
#ifdef RED_HAS_TENSORRT_HN
    if (state.use_trt) {
        auto &eng = *state.trt_hybrid3d;
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
        if (!h2d("heatmaps_padded", state.heatmaps_padded.data(),
                 state.heatmaps_padded.size() * sizeof(float)) ||
            !h2d("centerHM",        state.centerHM_input.data(),
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
    } else
#endif
    try {
        std::array<int64_t, 5> hm_shape{1, N, J, Hpad, Hpad};
        std::array<int64_t, 3> chm_shape{1, N, 2};
        std::array<int64_t, 2> c3d_shape{1, 3};
        std::array<int64_t, 4> mat_shape{1, N, 4, 3};

        std::vector<Ort::Value> ins;
        ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
            state.heatmaps_padded.data(), state.heatmaps_padded.size(),
            hm_shape.data(), hm_shape.size()));
        ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
            state.centerHM_input.data(), state.centerHM_input.size(),
            chm_shape.data(), chm_shape.size()));
        ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
            state.center3D_input.data(), state.center3D_input.size(),
            c3d_shape.data(), c3d_shape.size()));
        ins.push_back(Ort::Value::CreateTensor<float>(state.mem_info,
            state.camera_matrices.data(), state.camera_matrices.size(),
            mat_shape.data(), mat_shape.size()));

        // Build input-name pointers in the order Ort sees them. The ONNX was
        // exported with names: heatmaps_padded, centerHM, center3D, cameraMatrices.
        std::vector<const char *> in_names_c;
        for (auto &s : state.hybrid3d_input_names) in_names_c.push_back(s.c_str());
        std::vector<const char *> out_names_c;
        for (auto &s : state.hybrid3d_output_names) out_names_c.push_back(s.c_str());

        // ORT requires inputs in the order session declares them. Reorder
        // our four tensors to match. Names in our ONNX export are:
        //   "heatmaps_padded", "centerHM", "center3D", "cameraMatrices"
        std::vector<Ort::Value> ins_ordered;
        ins_ordered.reserve(in_names_c.size());
        const char *want[] = {"heatmaps_padded", "centerHM", "center3D", "cameraMatrices"};
        for (size_t i = 0; i < in_names_c.size(); ++i) {
            for (size_t w = 0; w < 4; ++w) {
                if (std::strcmp(in_names_c[i], want[w]) == 0) {
                    ins_ordered.push_back(std::move(ins[w]));
                    break;
                }
            }
        }

        auto outs = state.hybrid3d_session->Run(
            Ort::RunOptions{nullptr},
            in_names_c.data(), ins_ordered.data(), ins_ordered.size(),
            out_names_c.data(), out_names_c.size());

        // Pull points3D and confidences by name.
        for (size_t oi = 0; oi < outs.size(); ++oi) {
            const std::string &name = state.hybrid3d_output_names[oi];
            const float *data = outs[oi].GetTensorData<float>();
            auto shape = outs[oi].GetTensorTypeAndShapeInfo().GetShape();
            if (name == "points3D") {
                std::memcpy(state.points3D_out.data(), data, J * 3 * sizeof(float));
            } else if (name == "confidences") {
                std::memcpy(state.confidences_out.data(), data, J * sizeof(float));
            }
            (void)shape;
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[HybridNet] Hybrid3D ORT Run failed: %s\n", e.what());
        return false;
    }
    state.last_hybrid3d_ms = std::chrono::duration<double, std::milli>(clk::now() - t_h3d).count();

    // ── STAGE 7: write into AnnotationMap ────────────────────────────
    FrameAnnotation &fa = get_or_create_frame(
        annotations, frame_idx, skeleton.num_nodes, N);
    for (int j = 0; j < J; ++j) {
        fa.kp3d[j].x = state.points3D_out[j * 3 + 0];
        fa.kp3d[j].y = state.points3D_out[j * 3 + 1];
        fa.kp3d[j].z = state.points3D_out[j * 3 + 2];
        fa.kp3d[j].set_hybridnet(state.confidences_out[j]);
    }
    // Also reproject 3D back into each camera's 2D for overlay display.
    // Use the full pinhole-with-distortion (or telecentric) projection so the
    // overlay points land on the actual image features the user sees.
    // NOTE: kp.y is stored in ImPlot coords (origin bottom-left), so we
    // flip from image coords (origin top-left) by subtracting from height.
    // This matches the existing 2D path in jarvis_inference.h:378.
    for (int c = 0; c < N; ++c) {
        const auto &cp = camera_params[c];
        for (int j = 0; j < J; ++j) {
            Eigen::Vector3d p3(fa.kp3d[j].x, fa.kp3d[j].y, fa.kp3d[j].z);
            Eigen::Vector2d uv = cp.telecentric
                ? red_math::projectPointTelecentric(p3, cp.projection_mat, cp.k, cp.dist_coeffs)
                : red_math::projectPointR(p3, cp.r, cp.tvec, cp.k, cp.dist_coeffs);
            auto &kp = fa.cameras[c].keypoints[j];
            kp.x = uv[0];
            kp.y = static_cast<double>(heights[c]) - uv[1];  // image → ImPlot
            kp.labeled = true;
            kp.source = LabelSource::Predicted;
            kp.confidence = state.confidences_out[j];
        }
    }
    return true;
}

#endif // RED_HAS_ONNXRUNTIME
#endif // __linux__ || _WIN32
