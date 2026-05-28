#pragma once
// jarvis_hybridnet_cuda.h — device-side helpers for the HN TRT path.
//
// Linux/Windows only. Header is safe to include from C++ TUs that don't have
// nvcc; implementations live in jarvis_hybridnet_cuda.cu.

#if defined(__linux__) || defined(_WIN32)

#include <cuda_runtime.h>

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

#endif // __linux__ || _WIN32
