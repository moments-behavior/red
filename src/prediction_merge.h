#pragma once
// prediction_merge.h — Concatenate several per-model prediction stores (.rpred)
// into ONE combined store for the "model set" workflow (jarvis_predict_window).
//
// Each input store came from running one model over the same video, so it holds
// that model's keypoints for the frames where the subject was in view. The
// combined store lays keypoints out end to end: input 0's keypoints occupy
// indices [0, n0), input 1's occupy [n0, n0+n1), and so on. A frame is written
// to the output whenever ANY input has it (sparse union); for an input missing
// that frame, its block is filled with NaN xyz + 0 confidence — the same
// out-of-view sentinel the overlay/promote paths already skip.
//
// The output keypoint count = sum of the inputs' counts, which must equal the
// combined skeleton's num_nodes (see build_combined_skeleton) so every
// `store.num_keypoints() == skeleton.num_nodes` gate keeps holding.

#include "prediction_store.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace predstore {

// Merge `in_paths` (in order) into a single store at `out_path`. Returns false
// with *err set on any I/O error. `in_paths` must be non-empty.
inline bool merge_concat(const std::vector<std::string> &in_paths,
                         const std::string &out_path, std::string *err) {
    if (in_paths.empty()) {
        if (err) *err = "merge_concat: no input stores";
        return false;
    }

    std::vector<PredictionReader> readers(in_paths.size());
    int combined_kp = 0;
    uint32_t total_frames = 0, fps = 0;
    std::vector<int> offset(in_paths.size(), 0);  // node offset per input
    for (size_t i = 0; i < in_paths.size(); ++i) {
        if (!readers[i].open(in_paths[i])) {
            if (err) *err = "merge_concat: cannot open " + in_paths[i];
            return false;
        }
        offset[i] = combined_kp;
        combined_kp += (int)readers[i].num_keypoints();
        total_frames = std::max(total_frames, readers[i].total_frames());
        if (fps == 0) fps = readers[i].fps();
    }
    if (combined_kp <= 0) {
        if (err) *err = "merge_concat: combined keypoint count is zero";
        return false;
    }

    // Union of every stored frame number across all inputs.
    std::set<uint32_t> frames;
    for (auto &r : readers) {
        uint32_t n = r.stored_frames();
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t fn = 0;
            if (r.stored_at(i, &fn)) frames.insert(fn);
        }
    }

    PredictionWriter w;
    if (!w.open(out_path, combined_kp)) {
        if (err) *err = "merge_concat: cannot open output " + out_path;
        return false;
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> row((size_t)combined_kp * 4);
    for (uint32_t fn : frames) {
        // Out-of-view sentinel for every keypoint until an input fills it in.
        for (int k = 0; k < combined_kp; ++k) {
            row[k * 4 + 0] = nan;
            row[k * 4 + 1] = nan;
            row[k * 4 + 2] = nan;
            row[k * 4 + 3] = 0.0f;
        }
        for (size_t i = 0; i < readers.size(); ++i) {
            const float *src = readers[i].frame(fn);
            if (!src) continue;
            int kp = (int)readers[i].num_keypoints();
            std::memcpy(&row[(size_t)offset[i] * 4], src,
                        (size_t)kp * 4 * sizeof(float));
        }
        if (!w.add_frame(fn, row.data())) {
            if (err) *err = "merge_concat: write failed at frame " +
                            std::to_string(fn);
            return false;
        }
    }

    if (!w.finalize(total_frames, fps)) {
        if (err) *err = "merge_concat: finalize failed";
        return false;
    }
    return true;
}

}  // namespace predstore
