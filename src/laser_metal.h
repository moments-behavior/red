#pragma once
#ifdef __APPLE__
#include <CoreVideo/CoreVideo.h>

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

// Destroy Metal state.
void laser_metal_destroy(LaserMetalHandle ctx);

#endif
