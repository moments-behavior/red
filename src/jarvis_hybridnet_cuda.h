#pragma once
// jarvis_hybridnet_cuda.h — device-side helpers for the HN TRT path.
//
// Linux/Windows only. Header is safe to include from C++ TUs that don't have
// nvcc; implementations live in jarvis_hybridnet_cuda.cu.

#if defined(__linux__) || defined(_WIN32)

#include <cuda_runtime.h>
#include <cstdint>

// pad_heatmaps_device: zero-pad effTrack's (N, J, H, W) heatmap output into
// Hybrid3D's (1, N, J, H+2, W+2) input layout, all on-device. Eliminates the
// effTrack-D2H + CPU-pad + Hybrid3D-H2D roundtrip (~27 ms for 16 cams × 24
// joints × 352² float32 on the A4000).
//
//   d_src: pointer to N*J*H*W floats (effTrack's "keypoint_heatmaps" binding).
//   d_dst: pointer to N*J*(H+2)*(W+2) floats (Hybrid3D's "heatmaps_padded"
//          binding). Pre-allocated by TRT load_engine; this call overwrites it.
//   stream: launch stream. Caller is responsible for synchronizing any
//          prior writes to d_src on a different stream (e.g. by syncing
//          effTrack's stream) before invoking this.
//
// All dims must be > 0; H and W are pre-pad sizes. Function does not
// validate dtype — the engine bindings are guaranteed float32 by the ONNX
// export. Returns the first cudaError seen during launch / setup.
cudaError_t jarvis_hn_pad_heatmaps_device(
    const float *d_src, float *d_dst,
    int N, int J, int H, int W,
    cudaStream_t stream);

// resize_normalize_device: bilinear-resample N RGBA32 device frames (one per
// camera, sizes given by per-cam widths[]/heights[]) to (N, 3, dst, dst)
// NCHW float32, then ImageNet-normalize with (mean, 1/std). Matches the
// CPU `jarvis_resize_normalize` exactly: pixel-center convention, edge
// clamping after weight calc, alpha channel skipped.
//
//   d_rgba_ptrs:  device array of N device pointers, one RGBA32 frame each.
//   d_widths:     device array of N int (source widths).
//   d_heights:    device array of N int (source heights).
//   d_dst:        N*3*dst*dst floats; CenterDetect's TRT input binding.
//   N:            number of cameras (== HN_BATCH).
//   dst:          output side (320 for Center).
//   mean[3], inv_std[3]: ImageNet constants.
//
// All metadata arrays live on the device; the caller (load_engine) allocates
// them and uploads per-frame.
cudaError_t jarvis_hn_resize_normalize_device(
    const uint8_t *const *d_rgba_ptrs,
    const int *d_widths,
    const int *d_heights,
    float *d_dst,
    int N, int dst,
    const float mean[3], const float inv_std[3],
    cudaStream_t stream);

// crop_normalize_device: extract N square BxB crops (clamped at image edges)
// from N RGBA32 device frames centered at (cx[c], cy[c]), normalize, write
// as (N, 3, B, B) NCHW float32 into effTrack's TRT input binding. Matches
// CPU `jarvis_crop_normalize` exactly.
//
//   d_rgba_ptrs, d_widths, d_heights: as above.
//   d_cx, d_cy: device array of N int (crop centers in image coords).
//   d_dst: N*3*B*B floats; effTrack's TRT input binding.
//   B: crop side (704 for effTrack).
cudaError_t jarvis_hn_crop_normalize_device(
    const uint8_t *const *d_rgba_ptrs,
    const int *d_widths,
    const int *d_heights,
    const int *d_cx,
    const int *d_cy,
    float *d_dst,
    int N, int B,
    const float mean[3], const float inv_std[3],
    cudaStream_t stream);

#endif // __linux__ || _WIN32
