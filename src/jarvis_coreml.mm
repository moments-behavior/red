// jarvis_coreml.mm — CoreML implementation for JARVIS inference
//
// Zero-copy CVPixelBuffer path: VideoToolbox → IOSurface → CoreML → ANE/GPU

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include "jarvis_coreml.h"
#include "jarvis_hn_reproject.h"   // validated host reprojection + soft-argmax (CPU fallback)
#include "jarvis_hn_metal.h"       // GPU reprojection (bit-exact port of hn_reproject)
#include "red_math.h"              // undistortPoint, triangulatePoints, projectPointR
#include "skeleton.h"
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

        // Use CPU + ANE only — avoids GPU contention with Metal rendering
        // pipeline which causes bimodal performance (300ms vs 1200ms).
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                           configuration:config
                                                   error:&error];
        if (!model) {
            err = "Load failed: " +
                  std::string(error.localizedDescription.UTF8String);
            [[NSFileManager defaultManager] removeItemAtURL:compiled error:nil];
            return nil;
        }
        // The compiled .mlmodelc is a temp directory owned by the caller; the model
        // is fully loaded into memory above, so remove it (matches learned_ik_coreml).
        // NOTE: this recompiles on every launch — a future improvement is to compile
        // once to a persistent path beside the .mlpackage and reuse it.
        [[NSFileManager defaultManager] removeItemAtURL:compiled error:nil];
        return model;
    }
}

// Create a resized CVPixelBuffer using vImage (Accelerate, NEON-optimized)
// Convert a BGRA CVPixelBuffer to an ImageNet-normalized CHW float MLMultiArray.
// JARVIS training normalizes: (pixel/255 - mean) / std with ImageNet values.
// CoreML's ImageType only does scale=1/255 (no normalization), so we do it manually.
// Uses vDSP for vectorized float conversion + normalization (~10x faster than scalar).
static MLMultiArray *pixelbuf_to_normalized_array(CVPixelBufferRef pb, int w, int h) {
    // ImageNet normalization: scale = 1/(255*std), offset = -mean/std
    // Combined: output = pixel * scale + offset
    static const float scale[3] = {
        1.0f / (255.0f * 0.229f),  // R
        1.0f / (255.0f * 0.224f),  // G
        1.0f / (255.0f * 0.225f),  // B
    };
    static const float offset[3] = {
        -0.485f / 0.229f,  // R
        -0.456f / 0.224f,  // G
        -0.406f / 0.225f,  // B
    };

    int n = w * h;
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

    // Step 1: Deinterleave BGRA rows into planar uint8 B, G, R buffers.
    // Handle stride padding by processing row-by-row.
    std::vector<uint8_t> planar(n * 3); // B, G, R planes
    uint8_t *pB = planar.data();
    uint8_t *pG = pB + n;
    uint8_t *pR = pG + n;

    for (int y = 0; y < h; y++) {
        const uint8_t *row = src + y * stride;
        int off = y * w;
        for (int x = 0; x < w; x++) {
            pB[off + x] = row[x * 4 + 0];
            pG[off + x] = row[x * 4 + 1];
            pR[off + x] = row[x * 4 + 2];
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, kCVPixelBufferLock_ReadOnly);

    // Step 2: Convert uint8 → float and normalize using vDSP.
    // vDSP_vfltu8: uint8 → float,  vDSP_vsmsa: x * scale + offset
    float *dstR = dst;              // Channel 0 = R
    float *dstG = dst + n;          // Channel 1 = G
    float *dstB = dst + 2 * n;      // Channel 2 = B

    vDSP_vfltu8(pR, 1, dstR, 1, (vDSP_Length)n);
    vDSP_vsmsa(dstR, 1, &scale[0], &offset[0], dstR, 1, (vDSP_Length)n);

    vDSP_vfltu8(pG, 1, dstG, 1, (vDSP_Length)n);
    vDSP_vsmsa(dstG, 1, &scale[1], &offset[1], dstG, 1, (vDSP_Length)n);

    vDSP_vfltu8(pB, 1, dstB, 1, (vDSP_Length)n);
    vDSP_vsmsa(dstB, 1, &scale[2], &offset[2], dstB, 1, (vDSP_Length)n);

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
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, dst_w, dst_h,
                        kCVPixelFormatType_32BGRA, NULL, &dst);
    if (status != kCVReturnSuccess || !dst) {
        CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
        return NULL;
    }
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
    CVReturn cr_status = CVPixelBufferCreate(kCFAllocatorDefault, cw, ch,
                        kCVPixelFormatType_32BGRA, NULL, &dst);
    if (cr_status != kCVReturnSuccess || !dst) {
        CVPixelBufferUnlockBaseAddress(src, kCVPixelBufferLock_ReadOnly);
        return NULL;
    }
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
    peak.confidence = std::min(max_val, kHeatmapScale) / kHeatmapScale;
    return peak;
}

// Copy an MLMultiArray to a C-contiguous float32 vector, honoring its strides.
// CRITICAL: CoreML mlprogram outputs can be stride-padded (e.g. V2VNet's
// [1,24,50,50,50] comes back with the last dim padded 50→64 for GPU alignment).
// Reading dataPointer linearly would scramble the volume. Always go via strides.
static void copy_mlarray_contiguous(MLMultiArray *a, std::vector<float> &out) {
    const int nd = (int)a.shape.count;
    std::vector<long> shp(nd), str(nd);
    size_t total = 1;
    for (int i = 0; i < nd; ++i) {
        shp[i] = [a.shape[i] longValue];
        str[i] = [a.strides[i] longValue];
        total *= (size_t)shp[i];
    }
    out.resize(total);
    const bool f16 = (a.dataType == MLMultiArrayDataTypeFloat16);
    const float *pf = f16 ? nullptr : (const float *)a.dataPointer;
    const __fp16 *ph = f16 ? (const __fp16 *)a.dataPointer : nullptr;
    std::vector<long> idx(nd, 0);
    for (size_t o = 0; o < total; ++o) {
        long off = 0;
        for (int d = 0; d < nd; ++d) off += idx[d] * str[d];
        out[o] = f16 ? (float)ph[off] : pf[off];
        for (int d = nd - 1; d >= 0; --d) { if (++idx[d] < shp[d]) break; idx[d] = 0; }
    }
}

// Largest 4D output (the hi-res heatmap head) from a CoreML prediction.
static MLMultiArray *largest_4d_output(id<MLFeatureProvider> output) {
    MLMultiArray *best = nil;
    for (NSString *name in output.featureNames) {
        MLFeatureValue *fv = [output featureValueForName:name];
        if (fv.multiArrayValue && fv.multiArrayValue.shape.count == 4) {
            MLMultiArray *arr = fv.multiArrayValue;
            if (!best || [arr.shape[2] intValue] > [best.shape[2] intValue]) best = arr;
        }
    }
    return best;
}

// Forward decl — full volumetric 3D HybridNet path (defined below).
static bool jarvis_coreml_predict_hybridnet_3d(
    JarvisCoreMLState &s, AnnotationMap &amap, u32 frame_num,
    const std::vector<CVPixelBufferRef> &pbs,
    const std::vector<int> &widths, const std::vector<int> &heights,
    const SkeletonContext &skeleton, int num_cameras,
    const std::vector<CameraParams> &camera_params);

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

    // Full volumetric 3D HybridNet: if v2vnet.mlpackage is present, load it and
    // switch into 3D mode. Reprojection + soft-argmax run in host C++
    // (jarvis_hn_reproject.h); V2VNet (3D CNN) runs on the Apple GPU.
    s.hybridnet = false;
    std::string v2v_path = model_dir + "/v2vnet.mlpackage";
    if (std::filesystem::exists(v2v_path)) {
        MLModel *v2v = load_mlpackage(v2v_path, err);
        if (!v2v) {
            s.status = "V2VNet load failed: " + err;
            return false;
        }
        s.v2v_model = (__bridge_retained void *)v2v;
        s.hybridnet = true;
        // HybridNet grid geometry comes from the already-parsed model_info.json
        // (parse_jarvis_model_info -> cfg), so the file is parsed once. roi_cube/
        // grid_spacing are physical, in the inference calibration's world units;
        // grid_in/out are the V2VNet voxel-grid sides.
        s.hn_num_cameras     = cfg.hn_num_cameras;
        s.hn_roi_cube_mm     = cfg.hn_roi_cube;
        s.hn_grid_spacing_mm = cfg.hn_grid_spacing;
        s.hn_grid_in         = cfg.hn_grid_in;
        s.hn_grid_out        = cfg.hn_grid_out;

        // Allocate the GPU reprojection + persistent scratch once. The heatmap
        // and h3d buffers live in shared memory and are reused every frame; the
        // heatmap buffer is zeroed once here so its 1-px pad borders stay 0
        // (the interior is fully overwritten each frame). If Metal init fails we
        // fall back to the CPU hn_reproject (hn_metal stays null).
        const int hs = s.keypoint_input_size / 2 + 2;   // padded heatmap side (354)
        HNReproParams hp;
        hp.num_cameras = s.hn_num_cameras;
        hp.num_joints  = s.num_joints;
        hp.grid_full   = s.hn_grid_in;
        hp.grid_spacing = s.hn_grid_spacing_mm;
        hp.roi_cube    = s.hn_roi_cube_mm;
        hp.heatmap_size = hs;
        auto *m = new HNMetalReproject();
        std::string merr;
        if (m->init(hp, &merr)) {
            memset(m->heatmaps_ptr(), 0,
                   (size_t)hp.num_cameras * hp.num_joints * hs * hs * sizeof(float));
            s.hn_metal = m;
            // Persistent V2V input (1,NJ,gin,gin,gin) — reused every frame.
            NSError *ve = nil;
            MLMultiArray *vin = [[MLMultiArray alloc]
                initWithShape:@[@1, @(s.num_joints), @(s.hn_grid_in), @(s.hn_grid_in), @(s.hn_grid_in)]
                dataType:MLMultiArrayDataTypeFloat32 error:&ve];
            if (vin) s.v2v_input = (__bridge_retained void *)vin;
        } else {
            delete m;
            fprintf(stderr, "[HybridNet] Metal reprojection unavailable (%s); "
                            "using CPU fallback\n", merr.c_str());
        }
    }

    s.loaded = true;
    s.status = s.hybridnet
        ? ("CoreML HybridNet 3D loaded (" + std::to_string(s.num_joints) +
           " joints, " + std::to_string(s.hn_num_cameras) + " cams, GPU)")
        : ("CoreML loaded (" + std::to_string(s.num_joints) + " joints, GPU/ANE)");
    return true;
}

bool jarvis_coreml_predict_frame(
    JarvisCoreMLState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<CVPixelBufferRef> &pixel_buffers,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &skeleton,
    int num_cameras,
    const std::vector<CameraParams> &camera_params,
    float confidence_threshold) {

    if (!s.loaded) { s.status = "Not loaded"; return false; }

    // Full volumetric 3D HybridNet path — writes kp3d directly.
    if (s.hybridnet) {
        if (camera_params.empty()) {
            s.status = "HybridNet 3D needs camera calibration (none loaded)";
            return false;
        }
        return jarvis_coreml_predict_hybridnet_3d(
            s, amap, frame_num, pixel_buffers, cam_widths, cam_heights,
            skeleton, num_cameras, camera_params);
    }
    (void)skeleton;

    {
        auto t0 = std::chrono::steady_clock::now();
        int num_cams = (int)pixel_buffers.size();
        int num_joints = s.num_joints;

        auto &fa = get_or_create_frame(amap, frame_num,
                        num_joints, num_cams);

        MLModel *cd_model = (__bridge MLModel *)s.center_model;
        MLModel *kd_model = (__bridge MLModel *)s.keypoint_model;

        // (heatmap head selection uses the shared largest_4d_output helper)

        // Per-camera pipeline: CenterDetect → crop → KeypointDetect
        // Processing each camera fully before moving to the next improves
        // cache locality and enables early exit when center confidence is low.
        float center_ms_total = 0, kp_ms_total = 0;

        for (int c = 0; c < num_cams; ++c) {
            @autoreleasepool {
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
            if (!cd_input || error) continue;
            error = nil;
            id<MLFeatureProvider> cd_output = [cd_model predictionFromFeatures:cd_input error:&error];
            if (!cd_output || error) continue;

            MLMultiArray *cd_hm = largest_4d_output(cd_output);
            if (!cd_hm) continue;

            int cd_hm_h = [cd_hm.shape[2] intValue];
            int cd_hm_w = [cd_hm.shape[3] intValue];
            // Non-uniform downsampling scale (squish resize, matching JARVIS)
            float ds_x = (float)cam_widths[c] / sz;
            float ds_y = (float)cam_heights[c] / sz;

            HMPeak center = heatmap_argmax(cd_hm, 0, cd_hm_h, cd_hm_w);
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
            // A camera smaller than the crop makes the clamp bounds cross
            // (hi < lo → std::clamp UB) and the crop under-sized; skip it.
            if (cam_widths[c] < bbox_size || cam_heights[c] < bbox_size) continue;
            int cx = std::clamp((int)center.x, half, cam_widths[c] - half);
            int cy = std::clamp((int)center.y, half, cam_heights[c] - half);

            CVPixelBufferRef crop = crop_pixelbuf(pixel_buffers[c], cx, cy, bbox_size);
            if (!crop) continue;

            // Normalize crop with ImageNet mean/std (matching JARVIS training)
            MLMultiArray *kd_tensor = pixelbuf_to_normalized_array(crop, bbox_size, bbox_size);
            CVPixelBufferRelease(crop);
            if (!kd_tensor) continue;

            error = nil;
            id<MLFeatureProvider> kd_input =
                [[MLDictionaryFeatureProvider alloc] initWithDictionary:
                    @{@"image": [MLFeatureValue featureValueWithMultiArray:kd_tensor]}
                    error:&error];
            if (!kd_input || error) continue;
            error = nil;
            id<MLFeatureProvider> kd_output = [kd_model predictionFromFeatures:kd_input error:&error];
            if (!kd_output || error) continue;

            MLMultiArray *kd_hm = largest_4d_output(kd_output);
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
            } // @autoreleasepool per camera
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
    if (s.v2v_model) {
        CFRelease(s.v2v_model);
        s.v2v_model = nullptr;
    }
    if (s.hn_metal) {
        delete (HNMetalReproject *)s.hn_metal;
        s.hn_metal = nullptr;
    }
    if (s.v2v_input) {
        CFRelease(s.v2v_input);   // balances __bridge_retained
        s.v2v_input = nullptr;
    }
    s.hybridnet = false;
    s.loaded = false;
}

// ─────────────────────────────────────────────────────────────────────────
// Full volumetric 3D HybridNet path (CoreML).
//   per-cam: CenterDetect → 2D peak → undistort
//   once:    DLT triangulate → center3D (world mm)
//   per-cam: reproject center3D → centerHM → crop 704² → KeypointDetect heatmaps
//   once:    host reprojection (hn_reproject) → V2VNet (CoreML/GPU) → soft-argmax
//   write:   kp3d + per-cam 2D overlay
// Mirrors the validated Python reference (hybridnet_reference.py, ~2 mm) and the
// TensorRT host path in jarvis_hybridnet.h. Uses the P^T / K^T conventions.
// ─────────────────────────────────────────────────────────────────────────
static bool jarvis_coreml_predict_hybridnet_3d(
    JarvisCoreMLState &s, AnnotationMap &amap, u32 frame_num,
    const std::vector<CVPixelBufferRef> &pbs,
    const std::vector<int> &widths, const std::vector<int> &heights,
    const SkeletonContext &skeleton, int /*num_cameras*/,
    const std::vector<CameraParams> &cams) {

    const int NC = s.hn_num_cameras;
    const int NJ = s.num_joints;
    const int C  = s.center_input_size;      // 320
    const int B  = s.keypoint_input_size;    // 704
    const int bbox_hw = B / 2;               // 352
    const int Hcen_hi = C / 2;               // 160
    const int hs = B / 2 + 2;                // 354 (padded heatmap side)
    const int gin = s.hn_grid_in;            // 100
    const int gout = s.hn_grid_out;          // 50
    constexpr float kCenterThresh = 50.0f;   // matches JARVIS python / TRT path

    if ((int)pbs.size() < NC || (int)widths.size() < NC ||
        (int)heights.size() < NC || (int)cams.size() < NC) {
        s.status = "HybridNet 3D: fewer cameras/params than model expects";
        return false;
    }
    for (int c = 0; c < NC; ++c) {
        if (!pbs[c]) { s.status = "HybridNet 3D: all " + std::to_string(NC) +
                                  " cameras must have a frame"; return false; }
    }

    MLModel *cd_model = (__bridge MLModel *)s.center_model;
    MLModel *kd_model = (__bridge MLModel *)s.keypoint_model;
    MLModel *v2v_model = (__bridge MLModel *)s.v2v_model;
    auto t0 = std::chrono::steady_clock::now();

    // Env-gated stage profiling (HN_TIMING=1): separates the irreducible ANE
    // predict time from overlappable host preprocessing. now()/dt in ms.
    static const bool hn_timing = getenv("HN_TIMING") != nullptr;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto dt = [](std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double ms_cd_pre = 0, ms_cd_pred = 0, ms_kp_pre = 0, ms_kp_pred = 0, ms_kp_copy = 0;

    // ── STAGE 1+2: CenterDetect → triangulate center3D ──────────────────
    // center3D only locates the 704² crop ROI (keypoint detection still runs on
    // ALL cameras), so a spatially-spread subset triangulates it fine at ~half
    // the center-stage cost. Pick ≤kCenterMaxCams evenly-spaced camera indices.
    // Preprocessing (resize 7MP→320² + normalize) is per-camera independent →
    // run in parallel across cores; the ANE predicts run serially.
    const int kCenterMaxCams = std::max(2, s.hn_center_cams);   // configurable
    const int n_center = std::min(NC, kCenterMaxCams);
    std::vector<int> ccam(n_center);
    for (int i = 0; i < n_center; ++i) ccam[i] = (int)((long)i * NC / n_center);

    std::vector<Eigen::Vector2d> center_2d;
    std::vector<Eigen::Matrix<double, 3, 4>> center_Ps;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
    {
        auto tp = now();
        std::vector<void *> cd_in(NC, nullptr);
        void **cdp = cd_in.data();
        const int *ccp = ccam.data();
        const CVPixelBufferRef *pbp = pbs.data();
        const int Csz = C;
        dispatch_apply(n_center, q, ^(size_t ii) {
            int c = ccp[ii];
            @autoreleasepool {
                if (!pbp[c]) return;
                CVPixelBufferRef rz = resize_pixelbuf(pbp[c], Csz, Csz);
                if (!rz) return;
                MLMultiArray *tensor = pixelbuf_to_normalized_array(rz, Csz, Csz);
                CVPixelBufferRelease(rz);
                if (!tensor) return;
                NSError *e = nil;
                id<MLFeatureProvider> in = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{@"image":[MLFeatureValue featureValueWithMultiArray:tensor]} error:&e];
                if (in) cdp[c] = (__bridge_retained void *)in;
            }
        });
        ms_cd_pre = dt(tp, now());

        auto tq = now();
        for (int ii = 0; ii < n_center; ++ii) {
            int c = ccam[ii];
            @autoreleasepool {
                if (!cd_in[c]) continue;
                id<MLFeatureProvider> in = (__bridge_transfer id<MLFeatureProvider>)cd_in[c];
                cd_in[c] = nullptr;
                NSError *e = nil;
                id<MLFeatureProvider> out = [cd_model predictionFromFeatures:in error:&e];
                if (!out) continue;
                MLMultiArray *hm = largest_4d_output(out);
                if (!hm) continue;
                std::vector<float> plane;
                copy_mlarray_contiguous(hm, plane);           // (1,1,160,160) → 160*160
                int hw = [hm.shape[3] intValue];
                int hh = [hm.shape[2] intValue];
                int best = 0; float bv = plane[0];
                for (int i = 1; i < hh * hw; ++i) if (plane[i] > bv) { bv = plane[i]; best = i; }
                if (bv < kCenterThresh) continue;
                double px = best % hw, py = best / hw;
                double nx = (px + 0.5) * widths[c]  / (double)Hcen_hi;
                double ny = (py + 0.5) * heights[c] / (double)Hcen_hi;
                Eigen::Vector2d und = cams[c].telecentric
                    ? red_math::undistortPointTelecentric({nx, ny}, cams[c].k, cams[c].dist_coeffs, cams[c].dist_center)
                    : red_math::undistortPoint({nx, ny}, cams[c].k, cams[c].dist_coeffs);
                center_2d.push_back(und);
                center_Ps.push_back(cams[c].projection_mat);
            }
        }
        ms_cd_pred = dt(tq, now());
    }
    if (center_2d.size() < 2) {
        s.status = "HybridNet 3D: <2 cameras detected the animal center";
        return false;
    }
    Eigen::Vector3d center3D = red_math::triangulatePoints(center_2d, center_Ps);
    // Degenerate (near-collinear / near-coincident) geometry can yield NaN/Inf,
    // which would propagate through the reprojection clamp into (int) casts (UB)
    // and produce garbage 3D written with a plausible confidence. Fail cleanly.
    if (!center3D.allFinite()) {
        s.status = "HybridNet 3D: degenerate center triangulation (non-finite)";
        return false;
    }
    const float c3[3] = {(float)center3D[0], (float)center3D[1], (float)center3D[2]};

    // ── STAGE 3+4: centerHM → crop → KeypointDetect heatmaps ────────────
    // Write directly into the persistent, cam-major, 1-px-padded (354²) heatmap
    // buffer. With Metal it is the GPU's shared buffer (borders pre-zeroed once
    // at load, interior fully overwritten each frame); otherwise a per-frame
    // zeroed vector. A camera that fails prediction has its slot zeroed so no
    // stale data leaks across frames.
    HNMetalReproject *metal = (HNMetalReproject *)s.hn_metal;
    const size_t cam_slot = (size_t)NJ * hs * hs;
    std::vector<float> heatmaps_cpu;
    float *heatmaps;
    if (metal) {
        heatmaps = metal->heatmaps_ptr();
    } else {
        heatmaps_cpu.assign((size_t)NC * cam_slot, 0.0f);
        heatmaps = heatmaps_cpu.data();
    }
    std::vector<int> cHMx(NC), cHMy(NC);
    // Parallel preprocess: per-cam centerHM → crop (704²) → normalize → provider.
    {
        auto tp = now();
        std::vector<void *> kp_in(NC, nullptr);
        void **kpp = kp_in.data();
        int *cxp = cHMx.data(), *cyp = cHMy.data();
        const CVPixelBufferRef *pbp = pbs.data();
        const CameraParams *camp = cams.data();
        const int *wp = widths.data(), *hp = heights.data();
        const int Bsz = B, bhw = bbox_hw;
        dispatch_apply(NC, q, ^(size_t c) {
            @autoreleasepool {
                // A camera smaller than the crop makes the clamp bounds cross
                // (hi < lo → std::clamp UB). Leave its heatmap slot zeroed (handled
                // below via the null kp_in check) by skipping this camera.
                if (wp[c] < Bsz || hp[c] < Bsz) return;
                Eigen::Vector2d cHM = camp[c].telecentric
                    ? red_math::projectPointTelecentric(center3D, camp[c].projection_mat, camp[c].k, camp[c].dist_coeffs, camp[c].dist_center)
                    : red_math::projectPointR(center3D, camp[c].r, camp[c].tvec, camp[c].k, camp[c].dist_coeffs);
                int cx = std::clamp((int)std::lround(cHM[0]), bhw, wp[c] - bhw);
                int cy = std::clamp((int)std::lround(cHM[1]), bhw, hp[c] - bhw);
                cxp[c] = cx; cyp[c] = cy;
                if (!pbp[c]) return;
                CVPixelBufferRef crop = crop_pixelbuf(pbp[c], cx, cy, Bsz);
                if (!crop) return;
                MLMultiArray *tensor = pixelbuf_to_normalized_array(crop, Bsz, Bsz);
                CVPixelBufferRelease(crop);
                if (!tensor) return;
                NSError *e = nil;
                id<MLFeatureProvider> in = [[MLDictionaryFeatureProvider alloc]
                    initWithDictionary:@{@"image":[MLFeatureValue featureValueWithMultiArray:tensor]} error:&e];
                if (in) kpp[c] = (__bridge_retained void *)in;
            }
        });
        ms_kp_pre = dt(tp, now());

        // Serial predict + copy heatmaps into the (persistent) buffer.
        for (int c = 0; c < NC; ++c) {
            bool ok = false;
            @autoreleasepool {
                if (kp_in[c]) {
                    id<MLFeatureProvider> in = (__bridge_transfer id<MLFeatureProvider>)kp_in[c];
                    kp_in[c] = nullptr;
                    NSError *e = nil;
                    auto tb = now();
                    id<MLFeatureProvider> out = [kd_model predictionFromFeatures:in error:&e];
                    ms_kp_pred += dt(tb, now());
                    MLMultiArray *hm = out ? largest_4d_output(out) : nil;   // (1,24,352,352)
                    if (hm) {
                        // Copy the (stride-padded) CoreML heatmap straight into the
                        // 1-px-padded (NJ,354,354) slot for camera c — a single
                        // stride-aware pass (no contiguous intermediate). Values are
                        // bit-identical to copy_mlarray_contiguous.
                        int kh_h = [hm.shape[2] intValue], kh_w = [hm.shape[3] intValue];
                        long sj = [hm.strides[1] longValue], sy = [hm.strides[2] longValue],
                             sx = [hm.strides[3] longValue];
                        const bool f16 = (hm.dataType == MLMultiArrayDataTypeFloat16);
                        const float  *pf = f16 ? nullptr : (const float *)hm.dataPointer;
                        const __fp16 *ph = f16 ? (const __fp16 *)hm.dataPointer : nullptr;
                        float *dst = heatmaps + (size_t)c * cam_slot;
                        auto td = now();
                        for (int j = 0; j < NJ; ++j) {
                            float *djp = dst + (size_t)j * hs * hs;
                            long jb = (long)j * sj;
                            for (int y = 0; y < kh_h; ++y) {
                                long yb = jb + (long)y * sy;
                                float *row = djp + (size_t)(y + 1) * hs + 1;
                                if (f16) for (int x = 0; x < kh_w; ++x) row[x] = (float)ph[yb + (long)x * sx];
                                else     for (int x = 0; x < kh_w; ++x) row[x] = pf[yb + (long)x * sx];
                            }
                        }
                        ms_kp_copy += dt(td, now());
                        ok = true;
                    }
                }
            }
            // Persistent Metal buffer: clear a failed camera's slot (rare) so the
            // previous frame's heatmaps don't leak. (CPU vector is already zeroed.)
            if (metal && !ok)
                memset(heatmaps + (size_t)c * cam_slot, 0, cam_slot * sizeof(float));
        }
    }

    // ── STAGE 5: assemble reprojection inputs (P^T, K^T, dist) ──────────
    std::vector<float> camMats((size_t)NC * 12), intrMats((size_t)NC * 9),
                       distC((size_t)NC * 5, 0.0f), cHMf((size_t)NC * 2);
    for (int c = 0; c < NC; ++c) {
        const auto &P = cams[c].projection_mat;              // 3x4
        for (int r = 0; r < 4; ++r)
            for (int col = 0; col < 3; ++col)
                camMats[(size_t)c * 12 + r * 3 + col] = (float)P(col, r);   // P^T
        const auto &K = cams[c].k;                           // 3x3
        for (int i = 0; i < 3; ++i)
            for (int jj = 0; jj < 3; ++jj)
                intrMats[(size_t)c * 9 + i * 3 + jj] = (float)K(jj, i);     // K^T
        for (int k = 0; k < 5; ++k) distC[(size_t)c * 5 + k] = (float)cams[c].dist_coeffs(k);
        cHMf[(size_t)c * 2 + 0] = (float)cHMx[c];
        cHMf[(size_t)c * 2 + 1] = (float)cHMy[c];
    }

    // ── STAGE 6a: reprojection → voxel volume (NJ,100³) ─────────────────
    // GPU (Metal) when available — bit-identical to the CPU hn_reproject,
    // ~10ms vs ~450ms. Falls back to the validated host path otherwise.
    HNReproParams P;
    P.num_cameras = NC; P.num_joints = NJ; P.grid_full = gin;
    P.grid_spacing = s.hn_grid_spacing_mm; P.roi_cube = s.hn_roi_cube_mm; P.heatmap_size = hs;
    const size_t vf = (size_t)NJ * gin * gin * gin;
    const float *h3d;
    std::vector<float> h3d_cpu;
    auto tr0 = now();
    if (metal) {
        if (!metal->reproject(camMats.data(), intrMats.data(), distC.data(),
                              c3, cHMf.data())) {
            s.status = "HybridNet 3D: Metal reprojection failed"; return false;
        }
        h3d = metal->h3d_ptr();
    } else {
        h3d_cpu.assign(vf, 0.0f);
        hn_reproject(P, camMats.data(), intrMats.data(), distC.data(),
                     c3, cHMf.data(), heatmaps, h3d_cpu.data());
        h3d = h3d_cpu.data();
    }
    double ms_repro = dt(tr0, now());

    // ── STAGE 6b: V2VNet (CoreML, Apple GPU) ────────────────────────────
    std::vector<float> vout;
    {
        auto th = std::chrono::steady_clock::now();
        NSError *e = nil;
        MLMultiArray *vin = s.v2v_input ? (__bridge MLMultiArray *)s.v2v_input : nil;
        if (!vin) {   // one-off fallback if the persistent input wasn't allocated
            vin = [[MLMultiArray alloc]
                initWithShape:@[@1, @(NJ), @(gin), @(gin), @(gin)]
                dataType:MLMultiArrayDataTypeFloat32 error:&e];
            if (!vin) { s.status = "HybridNet 3D: V2V input alloc failed"; return false; }
        }
        float *dp = (float *)vin.dataPointer;               // (1,NJ,gin³) contiguous
        // Normalize h3d for V2V by the heatmap value range. vsdiv (true divide)
        // matches the original scalar `h3d[i] / kHeatmapScale` bit-for-bit
        // (vsmul by the reciprocal would not).
        float hm_scale = kHeatmapScale;
        vDSP_vsdiv(h3d, 1, &hm_scale, dp, 1, (vDSP_Length)vf);
        id<MLFeatureProvider> in = [[MLDictionaryFeatureProvider alloc]
            initWithDictionary:@{@"vox_in":[MLFeatureValue featureValueWithMultiArray:vin]} error:&e];
        if (!in) { s.status = "HybridNet 3D: V2V input build failed"; return false; }
        id<MLFeatureProvider> out = [v2v_model predictionFromFeatures:in error:&e];
        if (!out) { s.status = "HybridNet 3D: V2V predict failed"; return false; }
        MLMultiArray *vo = [out featureValueForName:@"vox_out"].multiArrayValue;
        if (!vo) for (NSString *n in out.featureNames)         // fallback: first 5D output
            if ([out featureValueForName:n].multiArrayValue.shape.count == 5)
                { vo = [out featureValueForName:n].multiArrayValue; break; }
        if (!vo) { s.status = "HybridNet 3D: V2V output missing"; return false; }
        copy_mlarray_contiguous(vo, vout);                   // stride-aware — CRITICAL (50→64 pad)
        s.last_hybrid3d_ms = std::chrono::duration<float, std::milli>(
            std::chrono::steady_clock::now() - th).count();
    }

    // ── STAGE 6c: soft-argmax → world-mm points ─────────────────────────
    std::vector<float> pts((size_t)NJ * 3), conf((size_t)NJ);
    hn_soft_argmax(P, vout.data(), gout, c3, pts.data(), conf.data());

    // ── STAGE 7: write kp3d + per-cam 2D overlay ────────────────────────
    FrameAnnotation &fa = get_or_create_frame(amap, frame_num, skeleton.num_nodes, NC);
    for (int j = 0; j < NJ && j < (int)fa.kp3d.size(); ++j) {
        fa.kp3d[j].x = pts[j * 3 + 0];
        fa.kp3d[j].y = pts[j * 3 + 1];
        fa.kp3d[j].z = pts[j * 3 + 2];
        fa.kp3d[j].set_hybridnet(conf[j]);
    }
    for (int c = 0; c < NC; ++c) {
        for (int j = 0; j < NJ && j < (int)fa.cameras[c].keypoints.size(); ++j) {
            Eigen::Vector3d p3(pts[j * 3 + 0], pts[j * 3 + 1], pts[j * 3 + 2]);
            Eigen::Vector2d uv = cams[c].telecentric
                ? red_math::projectPointTelecentric(p3, cams[c].projection_mat, cams[c].k, cams[c].dist_coeffs, cams[c].dist_center)
                : red_math::projectPointR(p3, cams[c].r, cams[c].tvec, cams[c].k, cams[c].dist_coeffs);
            auto &kp = fa.cameras[c].keypoints[j];
            kp.x = uv[0];
            kp.y = (double)heights[c] - uv[1];   // image → ImPlot (origin bottom-left)
            kp.labeled = true;
            kp.source = LabelSource::Predicted;
            kp.confidence = conf[j];
        }
    }

    s.last_total_ms = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (hn_timing) {
        fprintf(stderr,
            "[HN_TIMING] total=%.0f | center: pre=%.0f pred=%.0f | "
            "keypoint: pre=%.0f pred=%.0f copy=%.0f | reproj=%.0f v2v=%.0f | "
            "rest=%.0f\n",
            s.last_total_ms, ms_cd_pre, ms_cd_pred, ms_kp_pre, ms_kp_pred, ms_kp_copy,
            ms_repro, s.last_hybrid3d_ms,
            s.last_total_ms - ms_cd_pre - ms_cd_pred - ms_kp_pre - ms_kp_pred -
                ms_kp_copy - ms_repro - s.last_hybrid3d_ms);
    }
    s.status = "HybridNet 3D: " + std::to_string(NJ) + " joints, " +
               std::to_string((int)center_2d.size()) + "/" + std::to_string(NC) +
               " cams, " + std::to_string((int)s.last_total_ms) + " ms";
    return true;
}
