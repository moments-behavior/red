#include "cotracker_infer.h"
#include <iostream>
#include <torch/torch.h>

// TorchScript model was traced with these fixed dimensions.
// T and N must match exactly; H/W must be multiples of 8.
static const int T_TRACE = 200;
static const int N_TRACE = 24;

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------
bool CoTrackerInfer::load(const std::string &model_path) {
    loaded_ = false;
    try {
        auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
        model_ = torch::jit::load(model_path, device);
        model_.eval();
        loaded_ = true;
        std::cout << "[CoTrackerInfer] Loaded model on "
                  << (device == torch::kCUDA ? "CUDA" : "CPU")
                  << ": " << model_path << "\n";
    } catch (const c10::Error &e) {
        std::cerr << "[CoTrackerInfer] Failed to load model: " << e.what() << "\n";
    }
    return loaded_;
}

// ---------------------------------------------------------------------------
// track()
//
// query_pts[n] = {frame_idx, pixel_x, pixel_y}  (Y=0 at top, pixel space)
// Returns results[t].tracks[n] = {pixel_x, pixel_y} for t in [0, T_out)
// where T_out = min(frames_rgba.size(), T_TRACE).
// ---------------------------------------------------------------------------
std::vector<CoTrackerInfer::CoTrackerResult>
CoTrackerInfer::track(const std::vector<unsigned char *> &frames_rgba,
                      int width, int height,
                      const std::vector<std::array<float, 3>> &query_pts) {
    int T_real = (int)frames_rgba.size();
    int N_real = (int)query_pts.size();

    // We can only run up to T_TRACE frames and N_TRACE queries (trace dimensions).
    int T_run = std::min(T_real, T_TRACE);
    int N_run = std::min(N_real, N_TRACE);

    std::vector<CoTrackerResult> results(T_run);
    for (auto &r : results) {
        r.tracks.assign(N_real, {0.0f, 0.0f});
        r.vis.assign(N_real, 0.0f);
    }

    if (!loaded_ || T_run == 0 || N_run == 0) return results;

    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Compute downsampled dimensions, aligned to multiple of 8 (CoTracker stride).
    int H_ct = std::max(8, ((int)(height * CT_SCALE + 7) / 8) * 8);
    int W_ct = std::max(8, ((int)(width  * CT_SCALE + 7) / 8) * 8);
    std::cout << "[CoTrackerInfer] image=" << width << "x" << height
              << " H_ct=" << H_ct << " W_ct=" << W_ct
              << " T_run=" << T_run << " N_run=" << N_run << "\n";

    // Build video tensor (1, T_TRACE, 3, H_ct, W_ct) — RGB float [0,255].
    // Frames beyond T_run are zero-padded.
    std::vector<at::Tensor> frame_tensors;
    frame_tensors.reserve(T_TRACE);

    for (int t = 0; t < T_TRACE; ++t) {
        if (t >= T_run || !frames_rgba[t]) {
            frame_tensors.push_back(torch::zeros({3, H_ct, W_ct}, torch::kFloat32));
            continue;
        }
        unsigned char *rgba = frames_rgba[t];
        // RGBA → RGB float (H, W, 3)
        auto rgba_t = torch::from_blob(rgba, {height, width, 4}, torch::kUInt8)
                          .to(torch::kFloat32);
        auto chw = rgba_t.permute({2, 0, 1}); // (4, H, W)
        auto rgb  = chw.slice(0, 0, 3);        // (3, H, W)

        // Resize to (3, H_ct, W_ct)
        auto frame = torch::nn::functional::interpolate(
            rgb.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>{H_ct, W_ct})
                .mode(torch::kBilinear)
                .align_corners(false))
            .squeeze(0); // (3, H_ct, W_ct)

        frame_tensors.push_back(frame.contiguous());
    }

    auto video = torch::stack(frame_tensors, 0)    // (T_TRACE, 3, H_ct, W_ct)
                     .unsqueeze(0)                  // (1, T_TRACE, 3, H_ct, W_ct)
                     .to(device);

    // Build queries tensor (1, N_TRACE, 3).
    // Slots beyond N_run stay at zeros (dummy queries at frame=0, x=0, y=0).
    auto queries = torch::zeros({1, N_TRACE, 3}, torch::kFloat32);
    for (int n = 0; n < N_run; ++n) {
        queries[0][n][0] = query_pts[n][0];                    // frame_idx
        queries[0][n][1] = query_pts[n][1] * CT_SCALE;        // x scaled
        queries[0][n][2] = query_pts[n][2] * CT_SCALE;        // y scaled
    }
    queries = queries.to(device);

    // Run CoTracker3
    at::Tensor tracks, vis;
    try {
        torch::NoGradGuard no_grad;
        auto outputs = model_.forward({video, queries}).toTuple();
        tracks = outputs->elements()[0].toTensor(); // (1, T_TRACE, N_TRACE, 2)
        vis    = outputs->elements()[1].toTensor(); // (1, T_TRACE, N_TRACE)
    } catch (const c10::Error &e) {
        std::cerr << "[CoTrackerInfer] Forward failed: " << e.what() << "\n";
        return results;
    }

    // Extract results for the real frames and queries only.
    tracks = tracks.cpu();
    vis    = vis.sigmoid().cpu();

    float inv_x = 1.0f / CT_SCALE;
    float inv_y = 1.0f / CT_SCALE;

    auto tracks_acc = tracks.accessor<float, 4>(); // [1, T_TRACE, N_TRACE, 2]
    auto vis_acc    = vis.accessor<float, 3>();     // [1, T_TRACE, N_TRACE]

    for (int t = 0; t < T_run; ++t) {
        results[t].tracks.resize(N_real);
        results[t].vis.resize(N_real);
        for (int n = 0; n < N_run; ++n) {
            results[t].tracks[n][0] = tracks_acc[0][t][n][0] * inv_x;
            results[t].tracks[n][1] = tracks_acc[0][t][n][1] * inv_y;
            results[t].vis[n]       = vis_acc[0][t][n];
        }
        // Queries beyond N_run keep default {0,0} / vis=0.
    }

    // Debug: print track positions for query 0 at a few time steps
    {
        static const int dbg_steps[] = {0, 1, 5, 25, 50};
        for (int di = 0; di < 5; ++di) {
            int t = dbg_steps[di];
            if (t >= T_run) break;
            std::cout << "  t=" << t
                      << " vis=" << results[t].vis[0]
                      << " xy=(" << results[t].tracks[0][0]
                      << "," << results[t].tracks[0][1] << ")\n";
        }
    }

    return results;
}
