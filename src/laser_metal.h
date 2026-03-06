#pragma once
#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>
#include <cstdint>

struct LaserMetalSpot {
    double cx, cy;
    int pixel_count;
    bool found;
};

// Opaque handle to Metal compute state
typedef struct LaserMetalContext *LaserMetalHandle;

// Create shared Metal state (device, queue, pipelines). Thread-safe, called once.
LaserMetalHandle laser_metal_create();

// Detect green laser spot in a BGRA CVPixelBuffer using Metal compute.
// Thread-safe: multiple threads may call concurrently with the same handle.
// Returns result synchronously (GPU work completes before return).
LaserMetalSpot laser_metal_detect(LaserMetalHandle ctx,
                                   CVPixelBufferRef pixel_buffer,
                                   int green_threshold,
                                   int green_dominance,
                                   int min_blob_pixels,
                                   int max_blob_pixels);

// Detect laser spot + produce RGBA visualization overlay.
// GPU: threshold → erode → dilate → colorize (black/gray/green).
// CPU: BFS connected components on dilated pixels → recolor per blob size.
// rgba_out must be pre-allocated to width*height*4 bytes.
// Returns synchronously after GPU+CPU work completes.
struct LaserMetalVizResult {
    int num_blobs = 0;
    int total_mask_pixels = 0;
};

LaserMetalVizResult laser_metal_detect_viz(
    LaserMetalHandle ctx,
    CVPixelBufferRef pixel_buffer,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels,
    uint8_t *rgba_out);

// Destroy Metal state.
void laser_metal_destroy(LaserMetalHandle ctx);

#endif
