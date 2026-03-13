// jarvis_coreml.mm — CoreML implementation for JARVIS inference
//
// Zero-copy CVPixelBuffer path: VideoToolbox → IOSurface → CoreML → ANE/GPU

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include "jarvis_coreml.h"
#include <filesystem>

// ── Helpers ──

static MLModel *load_mlpackage(const std::string &path, std::string &err) {
    @autoreleasepool {
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:path.c_str()]];

        // Compile .mlpackage to temporary .mlmodelc
        NSError *error = nil;
        NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
        if (!compiled) {
            err = "Compile failed: " +
                  std::string(error.localizedDescription.UTF8String);
            return nil;
        }

        // Load with all compute units (ANE + GPU + CPU)
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;

        MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:config
                                                   error:&error];
        if (!model) {
            err = "Load failed: " +
                  std::string(error.localizedDescription.UTF8String);
            return nil;
        }
        return model;
    }
}

// Create a resized CVPixelBuffer using vImage (Accelerate, NEON-optimized)
// Convert a BGRA CVPixelBuffer to an ImageNet-normalized CHW float MLMultiArray.
// JARVIS training normalizes: (pixel/255 - mean) / std with ImageNet values.
// CoreML's ImageType only does scale=1/255 (no normalization), so we do it manually.
static MLMultiArray *pixelbuf_to_normalized_array(CVPixelBufferRef pb, int w, int h) {
    // ImageNet RGB mean/std
    static const float mean[3] = {0.485f, 0.456f, 0.406f}; // R, G, B
    static const float inv_std[3] = {1.0f/0.229f, 1.0f/0.224f, 1.0f/0.225f};

    NSArray *shape = @[@1, @3, @(h), @(w)];
    NSError *err = nil;
    MLMultiArray *arr = [[MLMultiArray alloc] initWithShape:shape
                                                  dataType:MLMultiArrayDataTypeFloat32
                                                     error:&err];
    if (!arr) return nil;

    CVPixelBufferLockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    uint8_t *src = (uint8_t *)CVPixelBufferGetBaseAddress(pb);
    size_t stride = CVPixelBufferGetBytesPerRow(pb);
    float *dst = (float *)arr.dataPointer;

    // BGRA → normalized RGB CHW
    for (int y = 0; y < h; y++) {
        uint8_t *row = src + y * stride;
        for (int x = 0; x < w; x++) {
            float b = row[x * 4 + 0] / 255.0f;
            float g = row[x * 4 + 1] / 255.0f;
            float r = row[x * 4 + 2] / 255.0f;
            // Channel 0 = R, Channel 1 = G, Channel 2 = B (RGB order for model)
            dst[0 * h * w + y * w + x] = (r - mean[0]) * inv_std[0];
            dst[1 * h * w + y * w + x] = (g - mean[1]) * inv_std[1];
            dst[2 * h * w + y * w + x] = (b - mean[2]) * inv_std[2];
        }
    }

    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);
    return arr;
}

// Resize a CVPixelBuffer to dst_w x dst_h using vImage (squish, matching JARVIS).
static CVPixelBufferRef resize_pixelbuf(CVPixelBufferRef src, int dst_w, int dst_h) {
    int src_w = (int)CVPixelBufferGetWidth(src);
    int src_h = (int)CVPixelBufferGetHeight(src);
    size_t src_stride = CVPixelBufferGetBytesPerRow(src);

    CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    uint8_t *src_data = (uint8_t *)CVPixelBufferGetBaseAddress(src);

    vImage_Buffer src_buf = {src_data, (vImagePixelCount)src_h,
                             (vImagePixelCount)src_w, src_stride};

    CVPixelBufferRef dst = NULL;
    CVPixelBufferCreate(kCFAllocatorDefault, dst_w, dst_h,
                        kCVPixelFormatType_32BGRA, NULL, &dst);
    CVPixelBufferLockBaseAddress(dst, 0);

    uint8_t *dst_data = (uint8_t *)CVPixelBufferGetBaseAddress(dst);
    size_t dst_stride = CVPixelBufferGetBytesPerRow(dst);
    vImage_Buffer dst_buf = {dst_data, (vImagePixelCount)dst_h,
                             (vImagePixelCount)dst_w, dst_stride};

    vImageScale_ARGB8888(&src_buf, &dst_buf, NULL, kvImageNoFlags);

    CVPixelBufferUnlockBaseAddress(dst, 0);
    CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    return dst;
}

// Crop a region from a CVPixelBuffer using vImage
static CVPixelBufferRef crop_pixelbuf(CVPixelBufferRef src,
                                       int cx, int cy, int crop_size) {
    int src_w = (int)CVPixelBufferGetWidth(src);
    int src_h = (int)CVPixelBufferGetHeight(src);
    int half = crop_size / 2;
    int x0 = std::max(0, cx - half);
    int y0 = std::max(0, cy - half);
    int x1 = std::min(src_w, cx + half);
    int y1 = std::min(src_h, cy + half);
    int cw = x1 - x0, ch = y1 - y0;

    CVPixelBufferLockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    uint8_t *src_data = (uint8_t *)CVPixelBufferGetBaseAddress(src);
    size_t src_stride = CVPixelBufferGetBytesPerRow(src);

    // Create crop destination
    CVPixelBufferRef dst = NULL;
    CVPixelBufferCreate(kCFAllocatorDefault, cw, ch,
                        kCVPixelFormatType_32BGRA, NULL, &dst);
    CVPixelBufferLockBaseAddress(dst, 0);
    uint8_t *dst_data = (uint8_t *)CVPixelBufferGetBaseAddress(dst);
    size_t dst_stride = CVPixelBufferGetBytesPerRow(dst);

    // Copy rows with vImage
    vImage_Buffer src_buf = {src_data + y0 * src_stride + x0 * 4,
                             (vImagePixelCount)ch, (vImagePixelCount)cw, src_stride};
    vImage_Buffer dst_buf = {dst_data, (vImagePixelCount)ch,
                             (vImagePixelCount)cw, dst_stride};
    vImageCopyBuffer(&src_buf, &dst_buf, 4, kvImageNoFlags);

    CVPixelBufferUnlockBaseAddress(dst, 0);
    CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
    return dst;
}

// Extract heatmap argmax from MLMultiArray
struct HMPeak { float x, y, confidence; };

static HMPeak heatmap_argmax(MLMultiArray *hm, int channel, int hm_h, int hm_w) {
    // hm shape: [1, C, H, W] — access channel-th plane
    const void *raw = hm.dataPointer;
    vDSP_Length n = (vDSP_Length)(hm_h * hm_w);
    float max_val = -1e9f;
    vDSP_Length max_idx = 0;

    if (hm.dataType == MLMultiArrayDataTypeFloat16) {
        // Convert float16 plane to float32, then use vDSP_maxvi
        const __fp16 *fp16 = (const __fp16 *)raw + channel * n;
        // Use Accelerate vImageConvert for NEON-accelerated fp16→fp32
        vImage_Buffer src_buf = {(void *)fp16, 1, n, n * sizeof(__fp16)};
        std::vector<float> fp32(n);
        vImage_Buffer dst_buf = {fp32.data(), 1, n, n * sizeof(float)};
        vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, kvImageNoFlags);
        vDSP_maxvi(fp32.data(), 1, &max_val, &max_idx, n);
    } else {
        const float *data = (const float *)raw + channel * n;
        vDSP_maxvi(data, 1, &max_val, &max_idx, n);
    }

    HMPeak peak;
    peak.x = (float)(max_idx % hm_w) * 2.0f; // stride-2 heatmap
    peak.y = (float)(max_idx / hm_w) * 2.0f;
    peak.confidence = std::min(max_val, 255.0f) / 255.0f;
    return peak;
}

// ── Public API ──

bool jarvis_coreml_available() {
    if (@available(macOS 13.0, *)) return true;
    return false;
}

bool jarvis_coreml_init(JarvisCoreMLState &s, const std::string &model_dir,
                         const JarvisModelConfig &cfg) {
    s.available = jarvis_coreml_available();
    if (!s.available) {
        s.status = "CoreML requires macOS 13+";
        return false;
    }

    // Release any previously loaded models (prevents pointer-overwrite leak)
    jarvis_coreml_cleanup(s);

    // Store full config for display, and flat copies for inference hot path
    s.config = cfg;
    s.center_input_size = cfg.center_input_size;
    s.keypoint_input_size = cfg.keypoint_input_size;
    s.num_joints = cfg.num_joints;

    // Load models
    std::string cd_path = model_dir + "/center_detect.mlpackage";
    std::string kd_path = model_dir + "/keypoint_detect.mlpackage";

    if (!std::filesystem::exists(cd_path) || !std::filesystem::exists(kd_path)) {
        s.status = "CoreML .mlpackage files not found";
        return false;
    }

    s.status = "Compiling CenterDetect (first time may take ~15s)...";
    std::string err;
    MLModel *cd = load_mlpackage(cd_path, err);
    if (!cd) { s.status = err; return false; }

    s.status = "Compiling KeypointDetect...";
    MLModel *kd = load_mlpackage(kd_path, err);
    if (!kd) { s.status = err; return false; }

    s.center_model = (__bridge_retained void *)cd;
    s.keypoint_model = (__bridge_retained void *)kd;
    s.loaded = true;
    s.status = "CoreML loaded (" + std::to_string(s.num_joints) + " joints, GPU/ANE)";
    return true;
}

bool jarvis_coreml_predict_frame(
    JarvisCoreMLState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<CVPixelBufferRef> &pixel_buffers,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &/*skeleton*/,
    int num_cameras,
    float confidence_threshold) {

    if (!s.loaded) { s.status = "Not loaded"; return false; }

    @autoreleasepool {
        auto t0 = std::chrono::steady_clock::now();
        int num_cams = (int)pixel_buffers.size();
        int num_joints = s.num_joints;

        auto &fa = get_or_create_frame(amap, frame_num,
                        num_joints, num_cams);

        MLModel *cd_model = (__bridge MLModel *)s.center_model;
        MLModel *kd_model = (__bridge MLModel *)s.keypoint_model;

        // Helper: find the largest 4D heatmap output from a CoreML prediction
        auto find_heatmap = [](id<MLFeatureProvider> output) -> MLMultiArray * {
            MLMultiArray *best = nil;
            for (NSString *name in output.featureNames) {
                MLFeatureValue *fv = [output featureValueForName:name];
                if (fv.multiArrayValue && fv.multiArrayValue.shape.count == 4) {
                    MLMultiArray *arr = fv.multiArrayValue;
                    int h = [arr.shape[2] intValue];
                    if (!best || h > [best.shape[2] intValue])
                        best = arr;
                }
            }
            return best;
        };

        // Per-camera pipeline: CenterDetect → crop → KeypointDetect
        // Processing each camera fully before moving to the next improves
        // cache locality and enables early exit when center confidence is low.
        auto t1 = std::chrono::steady_clock::now();
        float center_ms_total = 0, kp_ms_total = 0;

        for (int c = 0; c < num_cams; ++c) {
            if (!pixel_buffers[c]) continue;

            // --- CenterDetect ---
            auto tc0 = std::chrono::steady_clock::now();
            int sz = s.center_input_size;
            CVPixelBufferRef resized = resize_pixelbuf(pixel_buffers[c], sz, sz);
            if (!resized) continue;

            // Convert to ImageNet-normalized float tensor (matching JARVIS training)
            MLMultiArray *cd_tensor = pixelbuf_to_normalized_array(resized, sz, sz);
            CVPixelBufferRelease(resized);
            if (!cd_tensor) continue;

            NSError *error = nil;
            id<MLFeatureProvider> cd_input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                    @{@"image": [MLFeatureValue featureValueWithMultiArray:cd_tensor]}
                    error:&error];
            id<MLFeatureProvider> cd_output = [cd_model predictionFromFeatures:cd_input error:&error];
            if (!cd_output) continue;

            MLMultiArray *cd_hm = find_heatmap(cd_output);
            if (!cd_hm) continue;

            int cd_hm_h = [cd_hm.shape[2] intValue];
            int cd_hm_w = [cd_hm.shape[3] intValue];
            // Non-uniform downsampling scale (squish resize, matching JARVIS)
            float ds_x = (float)cam_widths[c] / sz;
            float ds_y = (float)cam_heights[c] / sz;

            HMPeak center = heatmap_argmax(cd_hm, 0, cd_hm_h, cd_hm_w);
            // Debug: print raw heatmap peak info
            printf("[CenterDetect cam%d] raw_peak=%.4f conf=%.4f pos=(%.1f,%.1f) hm=%dx%d\n",
                   c, center.confidence * 255.0f, center.confidence,
                   center.x, center.y, cd_hm_w, cd_hm_h);
            center.x *= ds_x;
            center.y *= ds_y;
            auto tc1 = std::chrono::steady_clock::now();
            center_ms_total += std::chrono::duration<float, std::milli>(tc1 - tc0).count();

            // Skip KeypointDetect if center confidence is too low
            if (center.confidence < confidence_threshold) continue;

            // --- KeypointDetect ---
            auto tk0 = std::chrono::steady_clock::now();
            int bbox_size = s.keypoint_input_size;
            int half = bbox_size / 2;
            int cx = std::clamp((int)center.x, half, cam_widths[c] - half);
            int cy = std::clamp((int)center.y, half, cam_heights[c] - half);

            CVPixelBufferRef crop = crop_pixelbuf(pixel_buffers[c], cx, cy, bbox_size);
            if (!crop) continue;

            // Normalize crop with ImageNet mean/std (matching JARVIS training)
            MLMultiArray *kd_tensor = pixelbuf_to_normalized_array(crop, bbox_size, bbox_size);
            CVPixelBufferRelease(crop);
            if (!kd_tensor) continue;

            id<MLFeatureProvider> kd_input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                    @{@"image": [MLFeatureValue featureValueWithMultiArray:kd_tensor]}
                    error:&error];
            id<MLFeatureProvider> kd_output = [kd_model predictionFromFeatures:kd_input error:&error];
            if (!kd_output) continue;

            MLMultiArray *kd_hm = find_heatmap(kd_output);
            if (!kd_hm) continue;

            int kd_hm_h = [kd_hm.shape[2] intValue];
            int kd_hm_w = [kd_hm.shape[3] intValue];
            int n_joints = std::min([kd_hm.shape[1] intValue], (int)fa.cameras[c].keypoints.size());

            for (int k = 0; k < n_joints; ++k) {
                HMPeak peak = heatmap_argmax(kd_hm, k, kd_hm_h, kd_hm_w);
                float img_x = peak.x + (float)(cx - half);
                float img_y = peak.y + (float)(cy - half);

                auto &kp = fa.cameras[c].keypoints[k];
                kp.x = img_x;
                kp.y = cam_heights[c] - img_y; // image → ImPlot
                kp.labeled = (peak.confidence >= confidence_threshold);
                kp.confidence = peak.confidence;
                kp.source = LabelSource::Predicted;
            }
            auto tk1 = std::chrono::steady_clock::now();
            kp_ms_total += std::chrono::duration<float, std::milli>(tk1 - tk0).count();
        }
        s.last_center_ms = center_ms_total;
        s.last_keypoint_ms = kp_ms_total;

        auto t4 = std::chrono::steady_clock::now();
        s.last_total_ms = std::chrono::duration<float, std::milli>(t4 - t0).count();
        s.status = "Predicted " + std::to_string(num_joints) + " joints on " +
                   std::to_string(num_cams) + " cameras in " +
                   std::to_string((int)s.last_total_ms) + " ms (CoreML)";
        return true;
    }
}

void jarvis_coreml_cleanup(JarvisCoreMLState &s) {
    if (s.center_model) {
        CFRelease(s.center_model);
        s.center_model = nullptr;
    }
    if (s.keypoint_model) {
        CFRelease(s.keypoint_model);
        s.keypoint_model = nullptr;
    }
    s.loaded = false;
}
