#pragma once
#ifdef __APPLE__
#include <cstdint>

// Opaque handle to Metal compute state for ArUco detection acceleration.
typedef struct ArucoMetalContext *ArucoMetalHandle;

// Create shared Metal state (device, queue, pipelines). Thread-safe, called once.
ArucoMetalHandle aruco_metal_create();

// Run adaptive threshold + 3x downsample on GPU for multiple window sizes.
// Uses separable box filter (horizontal running sum + vertical sum + threshold),
// then downsamples the binary result 3x on GPU. Only the 7MB grayscale image
// is transferred in; only (w/3)*(h/3) ~784KB per pass is transferred out.
// The full-res binary never leaves the GPU.
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
//   binary_outputs:  array of num_passes pointers, each pre-allocated to (w/3)*(h/3) bytes
void aruco_metal_threshold_batch(
    void *ctx,
    const uint8_t *gray, int w, int h,
    const int *window_sizes, int C, int num_passes,
    uint8_t **binary_outputs);

// Process video frame on GPU: BGRA→gray + adaptive threshold + 3x downsample.
// ONE command buffer, ONE mutex lock. Writes results to caller-owned buffers:
//   gray_output:    w*h bytes, grayscale image (for cornerSubPix etc.)
//   binary_outputs: num_passes pointers, each (w/3)*(h/3) bytes (downsampled threshold)
// Thread-safe: gray_output is caller-owned, safe to read after return.
#include <CoreVideo/CoreVideo.h>
void aruco_metal_process_video_frame(
    ArucoMetalHandle ctx,
    CVPixelBufferRef pb,
    int w, int h,
    const int *window_sizes, int C, int num_passes,
    uint8_t **binary_outputs,
    uint8_t *gray_output);

// Destroy Metal state.
void aruco_metal_destroy(ArucoMetalHandle ctx);

#endif
