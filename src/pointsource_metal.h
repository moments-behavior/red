#pragma once
#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>
#include <cstdint>

struct PointSourceMetalSpot {
    double cx, cy;
    int pixel_count;
    bool found;
};

// Opaque handle to Metal compute state
typedef struct PointSourceMetalContext *PointSourceMetalHandle;

// Create shared Metal state (device, queue, pipelines). Thread-safe, called once.
PointSourceMetalHandle pointsource_metal_create();

// Detect green light spot in a BGRA CVPixelBuffer using Metal compute.
// Thread-safe: multiple threads may call concurrently with the same handle.
// Returns result synchronously (GPU work completes before return).
PointSourceMetalSpot pointsource_metal_detect(PointSourceMetalHandle ctx,
                                   CVPixelBufferRef pixel_buffer,
                                   int green_threshold,
                                   int green_dominance,
                                   int min_blob_pixels,
                                   int max_blob_pixels,
                                   bool smart_blob = false);

// Detect light spot + produce RGBA visualization overlay.
// GPU: threshold → erode → dilate → colorize (black/gray/green).
// CPU: BFS connected components on dilated pixels → recolor per blob size.
// rgba_out must be pre-allocated to width*height*4 bytes.
// Returns synchronously after GPU+CPU work completes.
struct PointSourceMetalVizResult {
    int num_blobs = 0;
    int total_mask_pixels = 0;
};

PointSourceMetalVizResult pointsource_metal_detect_viz(
    PointSourceMetalHandle ctx,
    CVPixelBufferRef pixel_buffer,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels,
    uint8_t *rgba_out);

// Destroy Metal state.
void pointsource_metal_destroy(PointSourceMetalHandle ctx);

#endif
