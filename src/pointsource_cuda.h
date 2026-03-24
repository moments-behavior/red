#pragma once
#if defined(_WIN32) && defined(USE_CUDA_POINTSOURCE)

#include <cstdint>

struct PointSourceCudaSpot {
    double cx, cy;
    int pixel_count;
    bool found;
};

// Opaque handle to CUDA compute state
typedef struct PointSourceCudaContext *PointSourceCudaHandle;

// Create shared CUDA state (device, streams, allocations). Called once.
// Returns nullptr if CUDA init fails (caller should fall back to CPU).
PointSourceCudaHandle pointsource_cuda_create();

// Detect green light spot in a BGRA frame using CUDA compute.
// bgra_data: pointer to CPU-side BGRA pixel data (width*height*4 bytes).
// stride: bytes per row (may include padding).
// Returns result synchronously (GPU work completes before return).
PointSourceCudaSpot pointsource_cuda_detect(PointSourceCudaHandle ctx,
                                            const uint8_t *bgra_data,
                                            int width, int height, int stride,
                                            int green_threshold,
                                            int green_dominance,
                                            int min_blob_pixels,
                                            int max_blob_pixels,
                                            bool smart_blob = false);

// Detect light spot + produce RGBA visualization overlay.
// GPU: threshold -> erode -> dilate -> colorize (black/gray/green).
// CPU: BFS connected components on dilated pixels -> recolor per blob size.
// rgba_out must be pre-allocated to width*height*4 bytes.
// Returns synchronously after GPU+CPU work completes.
struct PointSourceCudaVizResult {
    int num_blobs = 0;
    int total_mask_pixels = 0;
};

PointSourceCudaVizResult pointsource_cuda_detect_viz(
    PointSourceCudaHandle ctx,
    const uint8_t *bgra_data,
    int width, int height, int stride,
    int green_threshold, int green_dominance,
    int min_blob_pixels, int max_blob_pixels,
    uint8_t *rgba_out);

// Destroy CUDA state and free all GPU allocations.
void pointsource_cuda_destroy(PointSourceCudaHandle ctx);

#endif // _WIN32 && USE_CUDA_POINTSOURCE
