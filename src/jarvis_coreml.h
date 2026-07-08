#pragma once
// jarvis_coreml.h — JARVIS pose estimation via native CoreML
//
// Loads .mlpackage/.mlmodelc models and runs CenterDetect + KeypointDetect
// directly on CVPixelBuffers from the VideoToolbox decode pipeline.
// Zero-copy input: CoreML reads IOSurface-backed CVPixelBuffers on GPU/ANE.
//
// macOS only. Preferred over ONNX Runtime on Apple Silicon (~6-20ms/frame).

#ifdef __APPLE__

#include "annotation.h"
#include "camera.h"     // CameraParams (for the 3D HybridNet path)
#include "types.h"
#include <string>
#include <vector>
#include <chrono>
#include <CoreVideo/CoreVideo.h>

#include "jarvis_model_config.h"

// Forward declarations to avoid heavy includes
struct SkeletonContext;

struct JarvisCoreMLState {
    bool loaded = false;
    bool available = false; // set true if macOS 13+
    std::string status;

    // Opaque pointers to MLModel instances
    void *center_model = nullptr;
    void *keypoint_model = nullptr;
    void *v2v_model = nullptr;   // V2VNet 3D CNN — present only for HybridNet models

    // Full model config (shared struct with ONNX backend)
    JarvisModelConfig config;

    // Flat copies for hot-path inference access (avoid indirection)
    int center_input_size = 320;
    int keypoint_input_size = 704;
    int num_joints = 24;

    // Full volumetric 3D HybridNet mode. Enabled when v2vnet.mlpackage is present
    // alongside the 2D models. In this mode jarvis_coreml_predict_frame runs the
    // reprojection + V2VNet + soft-argmax pipeline and writes 3D keypoints
    // directly (the caller must NOT then call reprojection()/triangulate).
    bool hybridnet = false;
    int   hn_num_cameras    = 16;
    float hn_roi_cube_mm    = 200.0f;
    float hn_grid_spacing_mm = 2.0f;
    int   hn_grid_in        = 100;   // reprojected voxel grid (roi/spacing)
    int   hn_grid_out       = 50;    // V2VNet output grid
    // Max cameras used for CenterDetect (crop-ROI localization only; keypoint
    // detection uses all). Mirrored from UserSettings.jarvis_center_cams;
    // clamped to [2, hn_num_cameras] at use.
    int   hn_center_cams    = 8;

    // Persistent HybridNet 3D scratch, allocated once at load (avoids per-frame
    // 385MB heatmap + 96MB h3d/V2V-input churn). Opaque to keep this header ObjC-
    // free: hn_metal is HNMetalReproject* (GPU reprojection, its heatmaps/h3d
    // buffers reused every frame); v2v_input is a retained MLMultiArray*.
    void *hn_metal   = nullptr;   // HNMetalReproject* (nullptr → CPU fallback)
    void *v2v_input  = nullptr;   // MLMultiArray* (1,NJ,gin,gin,gin), reused

    // Timing (per jarvis_coreml_predict_frame call)
    float last_center_ms = 0;
    float last_keypoint_ms = 0;
    float last_hybrid3d_ms = 0;
    float last_total_ms = 0;
};

// Initialize: load .mlpackage (compiles to .mlmodelc on first use, cached after).
// model_dir: directory containing center_detect.mlpackage/ and keypoint_detect.mlpackage/
// Pass pre-parsed config to avoid redundant JSON I/O.
bool jarvis_coreml_init(JarvisCoreMLState &s, const std::string &model_dir,
                         const JarvisModelConfig &cfg);

// Check if CoreML is available on this system
bool jarvis_coreml_available();

// Run full prediction on one frame across all cameras.
// pixel_buffers: CVPixelBufferRef array, one per camera (BGRA, IOSurface-backed)
// camera_params: per-camera calibration. REQUIRED for HybridNet 3D mode
//   (center3D triangulation + reprojection + back-projection); ignored by the
//   2D path. Pass an empty vector when unavailable — HybridNet mode then aborts.
// When s.hybridnet is true this writes 3D keypoints (kp3d) directly; the caller
// must NOT call reprojection()/triangulate afterwards.
bool jarvis_coreml_predict_frame(
    JarvisCoreMLState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<CVPixelBufferRef> &pixel_buffers,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &skeleton,
    int num_cameras,
    const std::vector<CameraParams> &camera_params,
    float confidence_threshold = 0.1f);

void jarvis_coreml_cleanup(JarvisCoreMLState &s);

#endif // __APPLE__
