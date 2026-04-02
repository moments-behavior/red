// learned_ik_coreml.mm — CoreML implementation for learned IK inference
//
// Simple float-in, float-out model — no image preprocessing needed.

#import <CoreML/CoreML.h>
#include "learned_ik_coreml.h"
#include <chrono>
#include <iostream>
#include <filesystem>

bool learned_ik_init(LearnedIKState &s, const std::string &mlpackage_path) {
    @autoreleasepool {
        s.loaded = false;
        s.model_path = mlpackage_path;

        if (!std::filesystem::exists(mlpackage_path)) {
            s.status = "File not found: " + mlpackage_path;
            std::cerr << "[LearnedIK] " << s.status << std::endl;
            return false;
        }

        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:mlpackage_path.c_str()]];

        // Compile .mlpackage to .mlmodelc
        NSError *error = nil;
        NSURL *compiled = [MLModel compileModelAtURL:url error:&error];
        if (!compiled) {
            s.status = "Compile failed: " +
                       std::string(error.localizedDescription.UTF8String);
            std::cerr << "[LearnedIK] " << s.status << std::endl;
            return false;
        }

        // Use CPU + ANE (no GPU to avoid Metal contention)
        MLModelConfiguration *config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsCPUAndNeuralEngine;

        MLModel *model = [MLModel modelWithContentsOfURL:compiled
                                            configuration:config
                                                    error:&error];
        if (!model) {
            s.status = "Load failed: " +
                       std::string(error.localizedDescription.UTF8String);
            std::cerr << "[LearnedIK] " << s.status << std::endl;
            return false;
        }

        s.model = (__bridge_retained void *)model;
        s.loaded = true;
        s.status = "Loaded";
        std::cout << "[LearnedIK] Model loaded: " << mlpackage_path << std::endl;
        return true;
    }
}

bool learned_ik_predict(LearnedIKState &s,
                        const float *kp3d, const float *valid,
                        float *qpos_out) {
    if (!s.loaded || !s.model) return false;

    @autoreleasepool {
        auto t0 = std::chrono::high_resolution_clock::now();

        MLModel *model = (__bridge MLModel *)s.model;
        NSError *error = nil;

        // Create input MLMultiArrays
        // kp3d: [1, 24, 3]
        NSArray *kp3d_shape = @[@1, @(s.n_keypoints), @3];
        MLMultiArray *kp3d_arr = [[MLMultiArray alloc]
            initWithShape:kp3d_shape
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&error];
        if (!kp3d_arr) return false;

        // valid_mask: [1, 24]
        NSArray *valid_shape = @[@1, @(s.n_keypoints)];
        MLMultiArray *valid_arr = [[MLMultiArray alloc]
            initWithShape:valid_shape
                 dataType:MLMultiArrayDataTypeFloat32
                    error:&error];
        if (!valid_arr) return false;

        // Copy input data using subscript API (handles Float16/Float32 transparently)
        for (int k = 0; k < s.n_keypoints; k++) {
            for (int c = 0; c < 3; c++) {
                NSArray *idx = @[@0, @(k), @(c)];
                [kp3d_arr setObject:@(kp3d[k * 3 + c]) forKeyedSubscript:idx];
            }
            NSArray *vidx = @[@0, @(k)];
            [valid_arr setObject:@(valid[k]) forKeyedSubscript:vidx];
        }

        // Build feature provider
        MLDictionaryFeatureProvider *input =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
                @"kp3d": kp3d_arr,
                @"valid_mask": valid_arr
            } error:&error];
        if (!input) return false;

        // Run inference
        id<MLFeatureProvider> output = [model predictionFromFeatures:input
                                                               error:&error];
        if (!output) {
            std::cerr << "[LearnedIK] Prediction failed: "
                      << error.localizedDescription.UTF8String << std::endl;
            return false;
        }

        // Extract qpos output.
        // CoreML may return Float16 internally — use objectAtIndexedSubscript
        // which handles type conversion correctly (Float16 → Float64 → float).
        MLMultiArray *qpos_arr = [output featureValueForName:@"qpos"].multiArrayValue;
        if (!qpos_arr) return false;

        for (int j = 0; j < s.n_qpos; j++) {
            qpos_out[j] = [qpos_arr objectAtIndexedSubscript:j].floatValue;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        s.last_inference_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        return true;
    }
}

void learned_ik_cleanup(LearnedIKState &s) {
    if (s.model) {
        MLModel *model = (__bridge_transfer MLModel *)s.model;
        (void)model; // ARC releases
        s.model = nullptr;
    }
    s.loaded = false;
    s.status.clear();
}
