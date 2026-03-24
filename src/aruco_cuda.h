#pragma once
#ifdef _WIN32

#include <cstdint>
#include <cstdio>

// Opaque handle to CUDA compute state for ArUco detection acceleration.
typedef struct ArucoCudaContext *ArucoCudaHandle;

// Create CUDA state (device, streams, kernel resources). Thread-safe, called once.
// Returns nullptr if no CUDA device is available — caller should fall back to CPU.
ArucoCudaHandle aruco_cuda_create();

// Run adaptive threshold + 3x downsample on GPU for multiple window sizes.
// Uses separable box filter (horizontal running sum + vertical sum + threshold),
// then downsamples the binary result 3x on GPU. Only the grayscale image
// is transferred in; only (w/3)*(h/3) per pass is transferred out.
// The full-res binary never leaves the GPU.
//
// Thread-safe: uses internal mutex to serialize GPU access.
//
// This function matches aruco_detect::GpuThresholdFunc signature (ctx as void*).
//   ctx:             ArucoCudaHandle cast to void*
//   gray:            grayscale image data (w*h bytes)
//   w, h:            image dimensions
//   window_sizes:    array of adaptive threshold window sizes (num_passes elements)
//   C:               threshold constant (pixel > local_mean - C -> foreground)
//   num_passes:      number of threshold passes (length of window_sizes array)
//   binary_outputs:  array of num_passes pointers, each pre-allocated to (w/3)*(h/3) bytes
void aruco_cuda_threshold_batch(
    void *ctx,
    const uint8_t *gray, int w, int h,
    const int *window_sizes, int C, int num_passes,
    uint8_t **binary_outputs);

// Destroy CUDA state and free all GPU resources.
void aruco_cuda_destroy(ArucoCudaHandle ctx);

#endif // _WIN32
