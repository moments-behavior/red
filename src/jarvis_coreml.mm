// jarvis_coreml.mm — CoreML implementation for JARVIS inference
//
// Zero-copy CVPixelBuffer path: VideoToolbox → IOSurface → CoreML → ANE/GPU

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include "jarvis_coreml.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>

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

    vImageScale_ARGB8888(&src_buf, &dst_buf, NULL, kvImageHighQualityResampling);

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
    int offset = channel * hm_h * hm_w;

    int max_idx = 0;
    float max_val = -1e9f;

    if (hm.dataType == MLMultiArrayDataTypeFloat16) {
        const __fp16 *data = (const __fp16 *)raw + offset;
        for (int i = 0; i < hm_h * hm_w; ++i) {
            float v = (float)data[i];
            if (v > max_val) { max_val = v; max_idx = i; }
        }
    } else {
        const float *data = (const float *)raw + offset;
        for (int i = 0; i < hm_h * hm_w; ++i) {
            if (data[i] > max_val) { max_val = data[i]; max_idx = i; }
        }
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
                         const char *model_info_json) {
    s.available = jarvis_coreml_available();
    if (!s.available) {
        s.status = "CoreML requires macOS 13+";
        return false;
    }

    // Parse model_info.json
    if (model_info_json && std::filesystem::exists(model_info_json)) {
        try {
            std::ifstream f(model_info_json);
            nlohmann::json j;
            f >> j;
            if (j.contains("center_detect") && j["center_detect"].contains("input_size"))
                s.center_input_size = j["center_detect"]["input_size"].get<int>();
            if (j.contains("keypoint_detect")) {
                auto &kd = j["keypoint_detect"];
                if (kd.contains("input_size"))
                    s.keypoint_input_size = kd["input_size"].get<int>();
                if (kd.contains("num_joints"))
                    s.num_joints = kd["num_joints"].get<int>();
            }
            if (j.contains("project_name"))
                s.project_name = j["project_name"].get<std::string>();
        } catch (...) {}
    }

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

        // Phase 1: CenterDetect on all cameras
        auto t1 = std::chrono::steady_clock::now();
        struct CenterResult { float x, y, confidence; };
        std::vector<CenterResult> centers(num_cams, {0, 0, 0});

        for (int c = 0; c < num_cams; ++c) {
            if (!pixel_buffers[c]) continue;

            // Resize to CenterDetect input size
            int sz = s.center_input_size;
            CVPixelBufferRef resized = resize_pixelbuf(pixel_buffers[c], sz, sz);
            if (!resized) continue;

            NSError *error = nil;
            MLFeatureValue *imgVal = [MLFeatureValue featureValueWithPixelBuffer:resized];
            NSDictionary *dict = @{@"image": imgVal};
            MLDictionaryFeatureProvider *input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&error];

            id<MLFeatureProvider> output = [cd_model predictionFromFeatures:input error:&error];
            CVPixelBufferRelease(resized);
            if (!output) continue;

            // Find the heatmap output (name varies by conversion)
            MLMultiArray *hm = nil;
            for (NSString *name in output.featureNames) {
                MLFeatureValue *fv = [output featureValueForName:name];
                if (fv.multiArrayValue && fv.multiArrayValue.shape.count == 4) {
                    // Use the larger heatmap (stride-2)
                    MLMultiArray *arr = fv.multiArrayValue;
                    int h = [arr.shape[2] intValue];
                    if (!hm || h > [hm.shape[2] intValue])
                        hm = arr;
                }
            }
            if (!hm) continue;

            int hm_h = [hm.shape[2] intValue];
            int hm_w = [hm.shape[3] intValue];
            float ds_x = (float)cam_widths[c] / sz;
            float ds_y = (float)cam_heights[c] / sz;

            HMPeak peak = heatmap_argmax(hm, 0, hm_h, hm_w);
            centers[c] = {peak.x * ds_x, peak.y * ds_y, peak.confidence};
        }
        auto t2 = std::chrono::steady_clock::now();
        s.last_center_ms = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // Phase 2: KeypointDetect on all cameras
        for (int c = 0; c < num_cams; ++c) {
            if (!pixel_buffers[c] || centers[c].confidence < confidence_threshold)
                continue;

            int bbox_size = s.keypoint_input_size;
            int cx = (int)centers[c].x;
            int cy = (int)centers[c].y;

            // Clamp center
            int half = bbox_size / 2;
            cx = std::clamp(cx, half, cam_widths[c] - half);
            cy = std::clamp(cy, half, cam_heights[c] - half);

            // Crop around center (vImage, ~50μs)
            CVPixelBufferRef crop = crop_pixelbuf(pixel_buffers[c], cx, cy, bbox_size);
            if (!crop) continue;

            NSError *error = nil;
            MLFeatureValue *imgVal = [MLFeatureValue featureValueWithPixelBuffer:crop];
            NSDictionary *dict = @{@"image": imgVal};
            MLDictionaryFeatureProvider *input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&error];

            id<MLFeatureProvider> output = [kd_model predictionFromFeatures:input error:&error];
            CVPixelBufferRelease(crop);
            if (!output) continue;

            // Find the heatmap output
            MLMultiArray *hm = nil;
            for (NSString *name in output.featureNames) {
                MLFeatureValue *fv = [output featureValueForName:name];
                if (fv.multiArrayValue && fv.multiArrayValue.shape.count == 4) {
                    MLMultiArray *arr = fv.multiArrayValue;
                    int h = [arr.shape[2] intValue];
                    if (!hm || h > [hm.shape[2] intValue])
                        hm = arr;
                }
            }
            if (!hm) continue;

            int hm_h = [hm.shape[2] intValue];
            int hm_w = [hm.shape[3] intValue];
            int n_joints = std::min([hm.shape[1] intValue], (int)fa.cameras[c].keypoints.size());

            for (int k = 0; k < n_joints; ++k) {
                HMPeak peak = heatmap_argmax(hm, k, hm_h, hm_w);
                // Convert from crop-local to full-image coordinates
                float img_x = peak.x + (float)(cx - half);
                float img_y = peak.y + (float)(cy - half);

                auto &kp = fa.cameras[c].keypoints[k];
                kp.x = img_x;
                kp.y = cam_heights[c] - img_y; // image → ImPlot
                kp.labeled = (peak.confidence >= confidence_threshold);
                kp.confidence = peak.confidence;
                kp.source = LabelSource::Predicted;
            }
        }
        auto t3 = std::chrono::steady_clock::now();
        s.last_keypoint_ms = std::chrono::duration<float, std::milli>(t3 - t2).count();

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
