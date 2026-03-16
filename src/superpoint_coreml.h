#pragma once
// superpoint_coreml.h — SuperPoint feature extraction via native CoreML (ANE)
//
// Opaque handle pattern (follows laser_metal.h).
// Input: BGRA CVPixelBuffer from VideoToolbox decode pipeline.
// Output: keypoints (pixel coords), scores, L2-normalized descriptors.
//
// macOS only. Requires macOS 13+ for MLProgram support.

#ifdef __APPLE__

#include <Eigen/Core>
#include <CoreVideo/CoreVideo.h>
#include <string>
#include <vector>

struct SuperPointFeatures {
    std::vector<Eigen::Vector2d> keypoints;  // pixel coords in original image
    std::vector<float> scores;               // detection confidence
    std::vector<float> descriptors_flat;     // row-major [N x 256] for cblas_sgemm
    int num_keypoints = 0;
    int image_width = 0;
    int image_height = 0;
};

// Opaque handle to CoreML model state
typedef struct SuperPointCoreMLContext *SuperPointCoreMLHandle;

// Create: load .mlpackage, compile, allocate buffers.
// model_path: path to superpoint.mlpackage directory.
// input_h, input_w: model input dimensions (must match conversion, e.g. 480x640).
SuperPointCoreMLHandle superpoint_coreml_create(const char *model_path,
                                                  int input_h = 480,
                                                  int input_w = 640);

// Check if CoreML is available (macOS 13+) and model is loaded.
bool superpoint_coreml_available(SuperPointCoreMLHandle handle);

// Extract features from a BGRA CVPixelBuffer.
// Handles resize, grayscale conversion, inference, softmax, NMS, top-K,
// descriptor interpolation, and L2 normalization.
// max_keypoints: maximum number of keypoints to return (sorted by score).
// score_threshold: minimum detection score (default 0.005).
SuperPointFeatures superpoint_coreml_extract(SuperPointCoreMLHandle handle,
                                              CVPixelBufferRef pixel_buffer,
                                              int max_keypoints = 4096,
                                              float score_threshold = 0.005f);

// Destroy: release model and buffers.
void superpoint_coreml_destroy(SuperPointCoreMLHandle handle);

#endif // __APPLE__
