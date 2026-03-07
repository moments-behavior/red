#pragma once
#ifdef __APPLE__
#include <cstdint>

// Opaque handle to Metal compute state for ArUco detection acceleration.
typedef struct ArucoMetalContext *ArucoMetalHandle;

// Create shared Metal state (device, queue, pipelines). Thread-safe, called once.
ArucoMetalHandle aruco_metal_create();

// Run adaptive threshold on GPU for multiple window sizes using separable
// box filter. Only the 7MB grayscale image is transferred — box sums are
// computed entirely on GPU (horizontal running sum + vertical sum + threshold).
//
// Thread-safe: uses internal mutex to serialize GPU access. Multiple camera
// threads can call concurrently; they queue behind the mutex while the GPU
// processes ~0.6ms per call.
//
// This function matches aruco_detect::GpuThresholdFunc signature (ctx as void*).
//   ctx:             ArucoMetalHandle cast to void*
//   gray:            grayscale image data (w*h bytes)
//   w, h:            image dimensions
//   window_sizes:    array of adaptive threshold window sizes (num_passes elements)
//   C:               threshold constant (pixel > local_mean - C → foreground)
//   num_passes:      number of threshold passes (length of window_sizes array)
//   binary_outputs:  array of num_passes pointers, each pre-allocated to w*h bytes
void aruco_metal_threshold_batch(
    void *ctx,
    const uint8_t *gray, int w, int h,
    const int *window_sizes, int C, int num_passes,
    uint8_t **binary_outputs);

// Destroy Metal state.
void aruco_metal_destroy(ArucoMetalHandle ctx);

#endif
