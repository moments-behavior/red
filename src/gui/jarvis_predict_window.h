#pragma once
// jarvis_predict_window.h — JARVIS Model Selector + Prediction panel
//
// Loads JARVIS CenterDetect + KeypointDetect models (CoreML on macOS,
// ONNX Runtime elsewhere) and provides a UI for running 2D pose predictions.
// Model files can be in <project>/jarvis_models/ or loaded manually.
// If only .pth checkpoints exist, offers Convert to CoreML / ONNX buttons.

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

#include "imgui.h"
#include "app_context.h"
#include "jarvis_inference.h"
#include "prediction_store.h"
#ifdef __APPLE__
#include "jarvis_coreml.h"
#elif defined(_WIN32)
#include "jarvis_tensorrt.h"
#endif
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <functional>

// Thread-safe conversion job (shared between UI thread and worker)
struct ConvertJob {
    std::atomic<bool> running{false};
    std::atomic<bool> finished{false};
    bool success = false;        // only read after finished==true
    std::string message;         // only read after finished==true
    bool force_rescan = false;   // only read after finished==true
    std::string output_path;     // where converted files landed (for auto-redirect)
};

struct JarvisPredictState {
    bool show = false;
    std::string models_folder;  // path to JARVIS project's models/ directory
    float confidence_threshold = 0.1f;

    // Set by "Predict Current Frame" button; consumed by main loop
    bool predict_requested = false;

    // Save 3D predictions to a JARVIS-CLI-compatible Predictions_3D folder
    // (data3D.csv + info.yaml) under <project>/predictions/predictions3D/,
    // in addition to the normal in-red labeled-frame loading. Consumed by the
    // main loop after Predict Current Frame and at Batch Predict completion.
    bool export_predictions3D = false;
    std::string export_status;   // last export result, shown in the panel

    // Where Batch Predict sends its results:
    //   LabelingTool — into the AnnotationMap as labeled frames (legacy; floods
    //                  the Labeling Tool when predicting large sections).
    //   Store        — into a separate mmap'd prediction store (Pose Stats +
    //                  read-only video overlay), leaving the Labeling Tool clean.
    // Single-frame Predict always goes to the Labeling Tool for now.
    enum class PredDest { LabelingTool, Store };
    PredDest batch_prediction_dest = PredDest::Store;
    bool show_prediction_overlay = true;   // draw store predictions on the videos
    std::string store_path;                // current session's .rpred store file
    std::string store_status;              // last store result, shown in the panel

    // --- Saved-predictions picker (load a previous session's store) ---
    // One .rpred file in <project>/predictions/red_store/, plus provenance from
    // its sidecar .json (video/model/frame range) when present.
    struct StoredPrediction {
        std::string path;          // absolute .rpred path
        std::string label;         // "2026-07-11 12:18" (from filename timestamp)
        std::string model;         // model name (sidecar)
        std::string video;         // media_folder (sidecar)
        int frame_start = -1, frame_end = -1;  // sidecar (−1 = unknown)
        uint32_t n_stored = 0, total_frames = 0, fps = 0;
        uint16_t num_keypoints = 0;
    };
    std::vector<StoredPrediction> store_list;  // newest first
    bool store_list_dirty = true;              // rescan red_store/ when set
    std::string active_store_path;             // currently mmap'd store
    std::string load_store_request;            // picker click → main loop opens it
    std::string import_request;                // data3D.csv/folder → main loop imports it
    std::string import_status;                 // last import result, shown in panel

    // Predict from: false = Shown (visible cameras only), true = All cameras.
    // Default to All — HybridNet 3D needs every camera, and "Shown" is rarely useful.
    bool predict_from_all = true;

    // Streaming batch mode: seek once and keep decoders filling the ring ahead
    // of the predict cursor (decode overlaps predict), instead of the chunked
    // seek-fill-predict cycle that re-seeks every buffer and stalls the GPU cold
    // at each boundary. Same frames predicted → same output (verified bit-
    // identical); purely an I/O-scheduling change, ~21% faster on long videos.
    // Default on; uncheck to fall back to the original chunked path.
    bool batch_streaming = true;

    // Batch prediction — non-blocking state machine (one frame per render iteration)
    // STREAM_SEEK/STREAM_RUN are the streaming path (seek once, decoders keep
    // filling the ring ahead of the predict cursor); the others are the chunked path.
    enum class BatchPhase {
        IDLE, SEEK, WAIT_BUFFER, PREDICT, FINISHING, STREAM_SEEK, STREAM_RUN
    };
    bool batch_running = false;
    bool batch_requested = false;
    bool batch_cancel_requested = false;  // Cancel button → routed through FINISHING cleanup
    bool batch_cancelled = false;          // status flag for the FINISHING message
    int batch_start = 0;
    int batch_end = 0;
    int batch_step = 1;
    int batch_current = 0;         // next frame to predict
    int batch_completed = 0;
    int batch_total = 0;
    int batch_skipped = 0;
    std::string batch_status;
    BatchPhase batch_phase = BatchPhase::IDLE;
    int batch_chunk_start = 0;     // first frame of current buffer chunk
    int batch_chunk_last_slot = 0; // last slot to wait for in current chunk
    int batch_wait_frames = 0;     // timeout counter for buffer fill
    int stream_read_head = 0;      // streaming mode: ring slot of the next frame
    std::chrono::steady_clock::time_point batch_t0;
    float batch_predict_ms = 0;
    // I/O overhead profiling (accumulated across all chunks in a batch)
    std::chrono::steady_clock::time_point chunk_seek_t0, chunk_wait_t0;
    float batch_seek_ms = 0;    // total time in blocking seek_all_cameras
    float batch_decode_ms = 0;  // total time waiting for decoders to fill buffers
    int   batch_chunks = 0;     // number of seek/fill cycles

    // Conversion state (thread-safe via shared_ptr)
    std::shared_ptr<ConvertJob> convert_job;
    std::string convert_status;  // UI-side copy, updated from job each frame

    // Filesystem detection cache (avoid scanning every frame)
    std::string cached_models_folder;
    bool cached_has_onnx = false;
    bool cached_has_pth = false;
    bool cached_has_coreml = false;
    bool cached_has_hybridnet = false;
    // Don't auto-retry load every ImGui frame after a failure (spams logs
    // and pegs the GPU with cudaMalloc attempts). User can force a fresh
    // attempt by re-selecting the model in the combo box.
    std::string auto_load_attempted_path;
    bool        auto_load_succeeded = false;
    std::string cached_center_path, cached_keypoint_path, cached_info_path;

    // Relative model path shown in Model Info
    std::string model_dir_display;
};

// Get the active model config from whichever backend is loaded
inline const JarvisModelConfig &jarvis_active_config(
    const JarvisState &jarvis
#ifdef __APPLE__
    , const JarvisCoreMLState &coreml
#elif defined(_WIN32)
    , const JarvisTensorRTState &tensorrt
#endif
) {
    static const JarvisModelConfig empty;
#ifdef __APPLE__
    if (coreml.loaded) return coreml.config;
#elif defined(_WIN32)
    if (tensorrt.loaded) return tensorrt.config;
#endif
    if (jarvis.loaded) return jarvis.config;
    return empty;
}

// Result of loading a model from a directory (CoreML preferred, ONNX fallback)
struct JarvisLoadResult {
    bool loaded = false;
    int num_joints = 0;
    int center_input_size = 0;
    int keypoint_input_size = 0;
    JarvisModelConfig config;
};

// Load JARVIS model from a directory.
// Detection order (each falls through if the dir doesn't have its file set):
//   1. HybridNet 3D (hybrid3d.onnx + manifest.json) — Linux/Win only
//   2. CoreML (.mlpackage) — Mac
//   3. TensorRT (.engine) — Windows
//   4. ONNX Runtime 2-stage (center_detect.onnx + keypoint_detect.onnx)
inline JarvisLoadResult jarvis_load_from_dir(
    const std::string &base_dir,
    JarvisState &jarvis
#ifdef __APPLE__
    , JarvisCoreMLState &jarvis_coreml
#elif defined(_WIN32)
    , JarvisTensorRTState &jarvis_trt
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
    , JarvisHybridNetState &jarvis_hn
#endif
#endif
) {
    namespace fs = std::filesystem;
    JarvisLoadResult r;
    fs::path mi = fs::path(base_dir) / "model_info.json";
    r.config = parse_jarvis_model_info(fs::exists(mi) ? mi.string().c_str() : nullptr);

#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
    // HybridNet 3D mode: presence of hybrid3d.onnx + manifest.json triggers
    // the full 3-stage pipeline. Replaces the 2D+triangulate shortcut.
    if (jarvis_hybridnet_dir_is_valid(base_dir)) {
        jarvis_cleanup(jarvis);
        if (jarvis_hybridnet_load(jarvis_hn, base_dir)) {
            r.loaded = true;
            r.num_joints = jarvis_hn.cfg.num_joints;
            r.center_input_size = jarvis_hn.cfg.center_image_size;
            r.keypoint_input_size = jarvis_hn.cfg.keypoint_bbox_size;
            return r;
        }
        // HN dir was detected but load failed. Do NOT silently fall through
        // to the 2-stage ONNX path — that loads a different model (the same
        // dir has center_detect.onnx + keypoint_detect.onnx). User explicitly
        // chose an HN model dir; surface the failure instead of switching
        // models behind their back.
        std::fprintf(stderr,
            "[JARVIS] HybridNet dir detected at %s but load failed — "
            "refusing to silently fall back to 2-stage ONNX. See "
            "[HybridNet] log lines above for the cause.\n", base_dir.c_str());
        return r;  // r.loaded == false
    } else {
        jarvis_hybridnet_unload(jarvis_hn);
    }
#endif
#endif

#ifdef __APPLE__
    if (fs::exists(fs::path(base_dir) / "center_detect.mlpackage") &&
        fs::exists(fs::path(base_dir) / "keypoint_detect.mlpackage")) {
        jarvis_cleanup(jarvis);
        jarvis_coreml_init(jarvis_coreml, base_dir, r.config);
        if (jarvis_coreml.loaded) {
            r = {true, jarvis_coreml.num_joints, jarvis_coreml.center_input_size,
                 jarvis_coreml.keypoint_input_size, r.config};
            return r;
        }
    }
#elif defined(_WIN32)
    if (fs::exists(fs::path(base_dir) / "center_detect.engine") &&
        fs::exists(fs::path(base_dir) / "keypoint_detect.engine")) {
        jarvis_cleanup(jarvis);
        jarvis_tensorrt_init(jarvis_trt, base_dir, r.config);
        if (jarvis_trt.loaded) {
            r = {true, jarvis_trt.num_joints, jarvis_trt.center_input_size,
                 jarvis_trt.keypoint_input_size, r.config};
            return r;
        }
    }
#endif
    std::string cd = base_dir + "/center_detect.onnx";
    std::string kd = base_dir + "/keypoint_detect.onnx";
    if (fs::exists(cd) && fs::exists(kd)) {
#ifdef __APPLE__
        jarvis_coreml_cleanup(jarvis_coreml);
#elif defined(_WIN32)
        jarvis_tensorrt_cleanup(jarvis_trt);
#endif
        jarvis_init(jarvis, cd.c_str(), kd.c_str(), r.config);
        if (jarvis.loaded) {
            r = {true, jarvis.config.num_joints, jarvis.config.center_input_size,
                 jarvis.config.keypoint_input_size, r.config};
        }
    }
    return r;
}

// Register a model in the project (dedup by name) and save .redproj.
inline void jarvis_register_model(
    ProjectManager &pm,
    const std::string &model_name,
    const std::string &relative_path,
    int nj, int ci_sz, int ki_sz)
{
    ProjectManager::JarvisModelEntry me;
    me.name = model_name;
    me.relative_path = relative_path;
    me.num_joints = nj;
    me.center_input_size = ci_sz;
    me.keypoint_input_size = ki_sz;

    bool dup = false;
    for (int i = 0; i < (int)pm.jarvis_models.size(); ++i) {
        if (pm.jarvis_models[i].name == model_name) {
            pm.jarvis_models[i] = me;
            pm.active_jarvis_model = i;
            dup = true;
            break;
        }
    }
    if (!dup) {
        pm.jarvis_models.push_back(me);
        pm.active_jarvis_model = (int)pm.jarvis_models.size() - 1;
    }
    std::filesystem::path redproj = std::filesystem::path(pm.project_path) /
                                     (pm.project_name + ".redproj");
    save_project_manager_json(pm, redproj);
}

// Scan <project>/predictions/red_store/ for .rpred stores, newest first, and
// populate state.store_list (header counts + provenance from each sidecar .json).
inline void jarvis_scan_prediction_stores(const std::string &project_path,
                                          JarvisPredictState &state) {
    namespace fs = std::filesystem;
    state.store_list.clear();
    if (project_path.empty()) return;
    fs::path dir = fs::path(project_path) / "predictions" / "red_store";
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return;

    for (auto &e : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (e.path().extension() != ".rpred") continue;
        auto hdr = predstore::read_store_header(e.path().string());
        if (!hdr.ok) continue;

        JarvisPredictState::StoredPrediction sp;
        sp.path = e.path().string();
        sp.n_stored = hdr.n_stored;
        sp.total_frames = hdr.total_frames;
        sp.fps = hdr.fps;
        sp.num_keypoints = hdr.num_keypoints;

        // Human label from the filename timestamp: pred_YYYYMMDD-HHMMSS.rpred
        std::string stem = e.path().stem().string();  // pred_20260711-121803
        if (stem.rfind("pred_", 0) == 0 && stem.size() >= 5 + 15) {
            std::string ts = stem.substr(5);           // 20260711-121803
            sp.label = ts.substr(0, 4) + "-" + ts.substr(4, 2) + "-" +
                       ts.substr(6, 2) + " " + ts.substr(9, 2) + ":" +
                       ts.substr(11, 2);
        } else {
            sp.label = stem;
        }

        // Provenance sidecar (optional).
        fs::path side = e.path();
        side.replace_extension(".json");
        std::ifstream sf(side);
        if (sf) {
            try {
                nlohmann::json j; sf >> j;
                sp.model = j.value("model_name", std::string{});
                sp.video = j.value("media_folder", std::string{});
                sp.frame_start = j.value("frame_start", -1);
                sp.frame_end = j.value("frame_end", -1);
            } catch (...) {}
        }
        state.store_list.push_back(std::move(sp));
    }
    // Newest first (path carries the sortable timestamp).
    std::sort(state.store_list.begin(), state.store_list.end(),
              [](const auto &a, const auto &b) { return a.path > b.path; });
}

inline void DrawJarvisPredictWindow(JarvisPredictState &state, JarvisState &jarvis,
#ifdef __APPLE__
                                     JarvisCoreMLState &jarvis_coreml,
#elif defined(_WIN32)
                                     JarvisTensorRTState &jarvis_trt,
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                                     JarvisHybridNetState &jarvis_hn,
#endif
#endif
                                     AppContext &ctx) {
    DrawPanel("JARVIS Predict", state.show,
        [&]() {
        // Availability check. The HybridNet TensorRT direct-runtime path needs
        // no ONNX Runtime, so treat the panel as available when it is compiled
        // in — jarvis.available only reflects the ORT 2-stage path.
        bool ml_available = jarvis.available;
#if defined(RED_HAS_TENSORRT_HN)
        ml_available = true;
#endif
        if (!ml_available) {
            ImGui::TextColored(ImVec4(1, 0.5f, 0, 1),
                               "ONNX Runtime not available");
            ImGui::TextWrapped("Compile with ONNX Runtime in lib/onnxruntime/ "
                               "to enable JARVIS prediction.");
            return;
        }

        // --- Auto-load project model if not yet loaded ---
        auto &pm = ctx.pm;
#ifdef __APPLE__
        bool any_loaded = jarvis.loaded || jarvis_coreml.loaded;
#elif defined(_WIN32)
        bool any_loaded = jarvis.loaded || jarvis_trt.loaded || jarvis_hn.loaded;
#elif defined(__linux__) && (defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN))
        bool any_loaded = jarvis.loaded || jarvis_hn.loaded;
#else
        bool any_loaded = jarvis.loaded;
#endif
        if (!any_loaded && pm.active_jarvis_model >= 0 &&
            pm.active_jarvis_model < (int)pm.jarvis_models.size()) {
            auto &m = pm.jarvis_models[pm.active_jarvis_model];
            std::string base = pm.project_path + "/" + m.relative_path;
            // Skip auto-load if we already tried this exact path and failed:
            // avoids retrying the load (and re-OOMing) every ImGui frame.
            // User can force a retry by re-selecting the model in the combo
            // (which clears auto_load_attempted_path below).
            if (state.auto_load_attempted_path != base ||
                state.auto_load_succeeded) {
                auto lr = jarvis_load_from_dir(base, jarvis
#ifdef __APPLE__
                    , jarvis_coreml
#elif defined(_WIN32)
                    , jarvis_trt
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                    , jarvis_hn
#endif
#endif
                );
                state.auto_load_attempted_path = base;
                state.auto_load_succeeded = lr.loaded;
                state.model_dir_display = m.relative_path;
            }
        }

        // --- Project Models (previously imported) ---
        if (!pm.jarvis_models.empty()) {
            ImGui::SeparatorText("Project Models");
            const char *preview = (pm.active_jarvis_model >= 0 &&
                                   pm.active_jarvis_model < (int)pm.jarvis_models.size())
                ? pm.jarvis_models[pm.active_jarvis_model].name.c_str()
                : "(none)";
            if (ImGui::BeginCombo("##jarvis_model_combo", preview)) {
                for (int i = 0; i < (int)pm.jarvis_models.size(); ++i) {
                    bool selected = (i == pm.active_jarvis_model);
                    if (ImGui::Selectable(pm.jarvis_models[i].name.c_str(), selected)) {
                        pm.active_jarvis_model = i;
                        auto &m = pm.jarvis_models[i];
                        std::string base = pm.project_path + "/" + m.relative_path;
                        jarvis_load_from_dir(base, jarvis
#ifdef __APPLE__
                            , jarvis_coreml
#elif defined(_WIN32)
                            , jarvis_trt
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                            , jarvis_hn
#endif
#endif
                        );
                        state.model_dir_display = m.relative_path;
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (pm.active_jarvis_model >= 0 &&
                pm.active_jarvis_model < (int)pm.jarvis_models.size()) {
                auto &m = pm.jarvis_models[pm.active_jarvis_model];
                ImGui::SameLine();
                ImGui::TextDisabled("(%d joints, %dx%d)", m.num_joints,
                                    m.keypoint_input_size, m.keypoint_input_size);
            }
        }

        // --- Import New Model ---
        ImGui::SeparatorText("Import Model");

        ImGui::Text("Models Folder");
        ImGui::SetNextItemWidth(-60);
        ImGui::InputText("##jarvis_models_folder", &state.models_folder);
        ImGui::SameLine();
        if (ImGui::Button("...##models")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            if (!state.models_folder.empty())
                cfg.path = state.models_folder;
            ImGuiFileDialog::Instance()->OpenDialog(
                "JarvisBrowseModels", "Select Models Folder", nullptr, cfg);
        }

        ImGui::Separator();

        // --- Resolve model paths (cached — only rescan when folder changes) ---
        namespace fs = std::filesystem;

        // Poll conversion job for completion (thread-safe)
        if (state.convert_job && state.convert_job->finished.load()) {
            state.convert_status = state.convert_job->message;
            // On success, auto-import the model into the project
            if (state.convert_job->success && !state.convert_job->output_path.empty()) {
                std::string out_dir = state.convert_job->output_path;
                auto lr = jarvis_load_from_dir(out_dir, jarvis
#ifdef __APPLE__
                    , jarvis_coreml
#elif defined(_WIN32)
                    , jarvis_trt
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                    , jarvis_hn
#endif
#endif
                );
                if (lr.loaded && !pm.project_path.empty()) {
                    std::string model_name = lr.config.project_name;
                    if (model_name.empty())
                        model_name = fs::path(out_dir).filename().string();
                    if (model_name.empty()) model_name = "jarvis_model";

                    std::string rel = fs::relative(
                        fs::path(out_dir), fs::path(pm.project_path)).string();
                    jarvis_register_model(pm, model_name, rel,
                        lr.num_joints, lr.center_input_size, lr.keypoint_input_size);
                    state.model_dir_display = rel;
                    state.convert_status = "Model loaded (" +
                        std::to_string(lr.num_joints) + " joints, " +
                        std::to_string(lr.keypoint_input_size) + "px keypoint)";
                    state.models_folder.clear();
                    state.cached_models_folder.clear();
                }
            }
            if (state.convert_job->force_rescan)
                state.cached_models_folder.clear();
            state.convert_job.reset();
        }

        // Rescan filesystem only when models_folder changes
        if (state.models_folder != state.cached_models_folder) {
            state.cached_models_folder = state.models_folder;
            state.cached_has_onnx = false;
            state.cached_has_pth = false;
            state.cached_has_coreml = false;
            state.cached_has_hybridnet = false;
            state.cached_center_path.clear();
            state.cached_keypoint_path.clear();
            state.cached_info_path.clear();

            bool has_onnx_subdir = false, has_onnx_direct = false;
            bool has_pth = false, has_coreml = false;
            std::string center_path, keypoint_path, info_path;

            if (!state.models_folder.empty() && fs::is_directory(state.models_folder)) {
            auto find_onnx_in = [&](const fs::path &dir) {
                fs::path c = dir / "center_detect.onnx";
                fs::path k = dir / "keypoint_detect.onnx";
                if (fs::exists(c) && fs::exists(k)) {
                    center_path = c.string();
                    keypoint_path = k.string();
                    fs::path mi = dir / "model_info.json";
                    if (fs::exists(mi)) info_path = mi.string();
                    return true;
                }
                return false;
            };

            has_onnx_subdir = find_onnx_in(fs::path(state.models_folder) / "onnx");
            if (!has_onnx_subdir)
                has_onnx_direct = find_onnx_in(fs::path(state.models_folder));

            auto check_coreml = [&](const fs::path &dir) -> bool {
                if (fs::exists(dir / "center_detect.mlpackage") &&
                    fs::exists(dir / "keypoint_detect.mlpackage")) {
                    if (info_path.empty()) {
                        fs::path mi = dir / "model_info.json";
                        if (fs::exists(mi)) info_path = mi.string();
                    }
                    return true;
                }
                return false;
            };
            has_coreml = check_coreml(fs::path(state.models_folder) / "onnx") ||
                         check_coreml(fs::path(state.models_folder));

            if (!has_onnx_subdir && !has_onnx_direct && !has_coreml) {
                auto find_latest_pth = [](const fs::path &module_dir) -> std::string {
                    if (!fs::is_directory(module_dir)) return {};
                    std::vector<fs::path> runs;
                    for (auto &e : fs::directory_iterator(module_dir))
                        if (e.is_directory() && e.path().filename().string().find("Run_") == 0)
                            runs.push_back(e.path());
                    if (runs.empty()) return {};
                    std::sort(runs.begin(), runs.end());
                    for (auto &e : fs::directory_iterator(runs.back()))
                        if (e.path().extension() == ".pth" &&
                            e.path().filename().string().find("final") != std::string::npos)
                            return e.path().string();
                    return {};
                };
                // Check <folder>/CenterDetect/ (direct) then <folder>/models/CenterDetect/ (JARVIS project)
                std::string cd_pth = find_latest_pth(fs::path(state.models_folder) / "CenterDetect");
                std::string kd_pth = find_latest_pth(fs::path(state.models_folder) / "KeypointDetect");
                if (cd_pth.empty() || kd_pth.empty()) {
                    cd_pth = find_latest_pth(fs::path(state.models_folder) / "models" / "CenterDetect");
                    kd_pth = find_latest_pth(fs::path(state.models_folder) / "models" / "KeypointDetect");
                }
                has_pth = !cd_pth.empty() && !kd_pth.empty();
                // Also detect JARVIS project by config.yaml presence
                if (!has_pth && fs::exists(fs::path(state.models_folder) / "config.yaml") &&
                    fs::is_directory(fs::path(state.models_folder) / "models"))
                    has_pth = true;
            }
            state.cached_has_onnx = (has_onnx_subdir || has_onnx_direct);
            state.cached_has_coreml = has_coreml;
            state.cached_has_pth = has_pth;
            state.cached_center_path = center_path;
            state.cached_keypoint_path = keypoint_path;
            state.cached_info_path = info_path;

            // HybridNet 3D: detect presence of hybrid3d.onnx + manifest.json
            // alongside the 2D ONNXes. Same subdir-or-direct logic as above.
            auto check_hybridnet = [&](const fs::path &dir) {
                return fs::exists(dir / "hybrid3d.onnx") &&
                       fs::exists(dir / "manifest.json");
            };
            state.cached_has_hybridnet =
                check_hybridnet(fs::path(state.models_folder) / "onnx") ||
                check_hybridnet(fs::path(state.models_folder));
            }
        }

        // Use cached detection results
        bool has_pth = state.cached_has_pth;
        bool has_coreml = state.cached_has_coreml;
        bool has_hybridnet = state.cached_has_hybridnet;
        std::string center_path = state.cached_center_path;
        std::string keypoint_path = state.cached_keypoint_path;
        std::string info_path = state.cached_info_path;
        bool can_load = !center_path.empty() && !keypoint_path.empty();
        bool can_load_any = can_load || has_coreml || has_hybridnet;

        // Show file detection status
        if (!state.models_folder.empty()) {
            if (has_hybridnet) {
                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1),
                    "Found HybridNet 3D model (CenterDetect + effTrack + Hybrid3D)");
            } else if (can_load && has_coreml) {
                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1),
                    "Found ONNX + CoreML models");
            } else if (has_coreml) {
                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1),
                    "Found CoreML models (.mlpackage)");
            } else if (can_load) {
                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1),
                    "Found ONNX models");
            } else if (has_pth) {
                ImGui::TextColored(ImVec4(1, 0.8f, 0, 1),
                    "Found .pth checkpoints (no ONNX/CoreML files)");
            } else if (fs::is_directory(state.models_folder)) {
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                    "No ONNX, CoreML, or .pth files found");
            }
        }

        // Import to Project button
        if (!can_load_any) ImGui::BeginDisabled();
        if (ImGui::Button("Import to Project")) {
            // Determine source directory
            std::string src_dir;
            if (!center_path.empty()) {
                src_dir = fs::path(center_path).parent_path().string();
            } else {
                fs::path onnx_sub = fs::path(state.models_folder) / "onnx";
                if (fs::exists(onnx_sub / "center_detect.mlpackage"))
                    src_dir = onnx_sub.string();
                else
                    src_dir = state.models_folder;
            }

            // Load model (CoreML preferred, ONNX fallback)
            auto lr = jarvis_load_from_dir(src_dir, jarvis
#ifdef __APPLE__
                , jarvis_coreml
#elif defined(_WIN32)
                , jarvis_trt
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                , jarvis_hn
#endif
#endif
            );

            // Determine model name
            std::string model_name = lr.config.project_name;
            if (model_name.empty())
                model_name = fs::path(src_dir).filename().string();
            if (model_name.empty()) model_name = "jarvis_model";

            // Copy model files into project folder and register
            if (lr.loaded && !pm.project_path.empty()) {
                std::string rel = "jarvis_models/" + model_name;
                fs::path dest = fs::path(pm.project_path) / rel;
                std::error_code ec;
                fs::create_directories(dest, ec);

                // Guard against a self-referential import: when the source folder
                // IS the destination (e.g. re-importing the model already inside the
                // project's jarvis_models/<name>/), the .mlpackage remove_all()+copy
                // below would delete each package and then fail to copy from the
                // just-deleted source, destroying the models (model_info.json, copied
                // via copy_file, would survive — the exact data-loss fingerprint).
                // The files are already in place, so skip the copy and just register.
                std::error_code eq_ec;
                bool same_dir = fs::equivalent(fs::path(src_dir), dest, eq_ec) && !eq_ec;
                if (!same_dir) {
                try {
                for (auto &entry : fs::directory_iterator(src_dir)) {
                    auto fname = entry.path().filename().string();
                    // Include ONNX weights, the legacy model_info.json (old 2-stage
                    // path), and the HybridNet provenance artifacts manifest.json +
                    // training_config.yaml. Without manifest.json the HN dir check
                    // (jarvis_hybridnet_dir_is_valid) fails and the load silently
                    // falls back to the 2-stage path, which then crashes on its
                    // own ORT Run.
                    if (fname.find(".onnx") != std::string::npos ||
                        fname == "model_info.json" ||
                        fname == "manifest.json" ||
                        fname == "training_config.yaml") {
                        fs::copy_file(entry.path(), dest / fname,
                                      fs::copy_options::overwrite_existing, ec);
                    }
                    if (entry.is_directory() && fname.find(".mlpackage") != std::string::npos) {
                        fs::path ml_dest = dest / fname;
                        // Belt-and-braces vs the same_dir guard: never remove_all a
                        // package that is the same filesystem object as the source.
                        std::error_code same_ec;
                        if (fs::equivalent(entry.path(), ml_dest, same_ec) && !same_ec)
                            continue;
                        fs::remove_all(ml_dest, ec);
                        fs::copy(entry.path(), ml_dest,
                                 fs::copy_options::recursive, ec);
                    }
                }
                } catch (...) {}  // tolerate directory iteration errors
                }

                jarvis_register_model(pm, model_name, rel,
                    lr.num_joints, lr.center_input_size, lr.keypoint_input_size);
                state.model_dir_display = rel;
                state.models_folder.clear();
                state.cached_models_folder.clear();
                state.convert_status.clear();
            }
        }
        if (!can_load_any) ImGui::EndDisabled();

        ImGui::SameLine();
        {
            bool show_loaded = jarvis.loaded;
#ifdef __APPLE__
            show_loaded = show_loaded || jarvis_coreml.loaded;
#elif defined(_WIN32)
            show_loaded = show_loaded || jarvis_trt.loaded;
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
            show_loaded = show_loaded || jarvis_hn.loaded;
#endif
#endif
            if (show_loaded) {
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
                if (jarvis_hn.loaded) {
                    ImGui::TextColored(ImVec4(0, 1, 0, 1), "Loaded (HybridNet 3D)");
                } else
#endif
#endif
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Loaded");
            } else if (!jarvis.status.empty()) {
                ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", jarvis.status.c_str());
            }
#ifdef __APPLE__
            else if (!jarvis_coreml.status.empty()) {
                ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", jarvis_coreml.status.c_str());
            }
#elif defined(_WIN32)
            else if (!jarvis_trt.status.empty()) {
                ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", jarvis_trt.status.c_str());
            }
#endif
        }

        // Convert to ONNX button (only if .pth exists and no model loaded)
        {
            bool show_convert = has_pth && !can_load;
#ifdef __APPLE__
            show_convert = show_convert && !jarvis.loaded && !jarvis_coreml.loaded;
#elif defined(_WIN32)
            show_convert = show_convert && !jarvis.loaded && !jarvis_trt.loaded;
#else
            show_convert = show_convert && !jarvis.loaded;
#endif
            bool converting = state.convert_job && state.convert_job->running.load();
            if (show_convert) {
                ImGui::Separator();
                if (!converting) {
                    ImGui::TextWrapped("ONNX files not found. You can convert .pth "
                                       "checkpoints to ONNX using the JARVIS export script.");
                    if (ImGui::Button("Convert to ONNX")) {
                        fs::path project_path = fs::path(state.models_folder).parent_path();
                        fs::path onnx_out = fs::path(state.models_folder) / "onnx";
                        std::string cmd =
                            "conda run -n jarvis python -m jarvis.utils.onnx_export \"" +
                            project_path.string() + "\" --output_dir \"" +
                            onnx_out.string() + "\" 2>&1";

                        auto job = std::make_shared<ConvertJob>();
                        job->running.store(true);
                        job->output_path = onnx_out.string();
                        state.convert_job = job;
                        state.convert_status = "Converting...";

                        std::thread([job, cmd]() {
                            FILE *pipe = popen(cmd.c_str(), "r");
                            if (!pipe) {
                                job->message = "Error: failed to run conversion command";
                                job->success = false;
                                job->running.store(false);
                                job->finished.store(true);
                                return;
                            }
                            char buf[256];
                            std::string output;
                            while (fgets(buf, sizeof(buf), pipe))
                                output += buf;
                            int ret = pclose(pipe);
                            if (ret == 0) {
                                job->message = "ONNX conversion complete.";
                                job->success = true;
                                job->force_rescan = true;
                            } else {
                                job->message = "Conversion failed (exit " +
                                    std::to_string(ret) + "): " + output.substr(0, 200);
                                job->success = false;
                            }
                            job->running.store(false);
                            job->finished.store(true);
                        }).detach();
                    }
                } else {
                    ImGui::BeginDisabled();
                    ImGui::Button("Converting...");
                    ImGui::EndDisabled();
                }
            }
            if (!state.convert_status.empty()) {
                bool is_error = state.convert_status.find("Error") != std::string::npos ||
                                state.convert_status.find("failed") != std::string::npos;
                ImGui::TextColored(
                    is_error ? ImVec4(1, 0.3f, 0.3f, 1) : ImVec4(0.5f, 1, 0.5f, 1),
                    "%s", state.convert_status.c_str());
            }
        }

        // Convert to CoreML button (macOS only — .pth exists, no .mlpackage)
#ifdef __APPLE__
        {
            bool show_coreml_convert = has_pth && !has_coreml &&
                !jarvis.loaded && !jarvis_coreml.loaded;
            bool converting = state.convert_job && state.convert_job->running.load();
            if (show_coreml_convert) {
                ImGui::Separator();
                if (!converting) {
                    ImGui::TextWrapped("CoreML models not found. Convert .pth "
                                       "checkpoints to CoreML for GPU/ANE acceleration. "
                                       "Point Models Folder to the JARVIS project directory "
                                       "(with config.yaml) for best results.");
                    if (ImGui::Button("Convert to CoreML")) {
                        // Find the script: try dev build path, then Homebrew
                        std::string exe_dir = ctx.window->exe_dir;
                        std::string script;
                        for (auto &candidate : {
                            exe_dir + "/../scripts/pth_to_coreml.py",
                            exe_dir + "/../share/red/scripts/pth_to_coreml.py",
                        }) {
                            if (fs::exists(candidate)) {
                                script = fs::canonical(candidate).string();
                                break;
                            }
                        }

                        if (script.empty()) {
                            state.convert_status = "Error: pth_to_coreml.py not found";
                        } else {
                            std::string jarvis_project = state.models_folder;
                            // Output to redproj/jarvis_models/<folder_name>/
                            namespace fs = std::filesystem;
                            std::string folder_name = fs::path(state.models_folder)
                                .filename().string();
                            std::string output_dir =
                                (fs::path(ctx.pm.project_path) /
                                 "jarvis_models" / folder_name).string();
                            fs::create_directories(output_dir);

                            // Pass config overrides from .redproj so the
                            // conversion uses correct input sizes even when
                            // config.yaml is not present alongside the .pth files.
                            std::string overrides;
                            if (ctx.pm.active_jarvis_model >= 0 &&
                                ctx.pm.active_jarvis_model < (int)ctx.pm.jarvis_models.size()) {
                                auto &me = ctx.pm.jarvis_models[ctx.pm.active_jarvis_model];
                                if (me.center_input_size > 0)
                                    overrides += " --center_input_size " + std::to_string(me.center_input_size);
                                if (me.keypoint_input_size > 0)
                                    overrides += " --keypoint_input_size " + std::to_string(me.keypoint_input_size);
                                if (me.num_joints > 0)
                                    overrides += " --num_joints " + std::to_string(me.num_joints);
                            }

                            std::string cmd =
                                "conda run -n coreml python \"" + script +
                                "\" --jarvis_project \"" + jarvis_project +
                                "\" --output_dir \"" + output_dir + "\"" +
                                overrides + " 2>&1";

                            auto job = std::make_shared<ConvertJob>();
                            job->running.store(true);
                            job->output_path = output_dir;
                            state.convert_job = job;
                            state.convert_status = "Converting to CoreML...";

                            std::thread([job, cmd]() {
                                FILE *pipe = popen(cmd.c_str(), "r");
                                if (!pipe) {
                                    job->message = "Error: failed to run conversion command";
                                    job->success = false;
                                    job->running.store(false);
                                    job->finished.store(true);
                                    return;
                                }
                                char buf[256];
                                std::string output;
                                while (fgets(buf, sizeof(buf), pipe))
                                    output += buf;
                                int ret = pclose(pipe);
                                if (ret == 0) {
                                        job->message = "CoreML conversion complete.";
                                    job->success = true;
                                    job->force_rescan = true;
                                } else {
                                    job->message = "CoreML conversion failed (exit " +
                                        std::to_string(ret) + "): " + output.substr(0, 200);
                                    job->success = false;
                                }
                                job->running.store(false);
                                job->finished.store(true);
                            }).detach();
                        }
                    }
                } else {
                    ImGui::BeginDisabled();
                    ImGui::Button("Converting...");
                    ImGui::EndDisabled();
                }
            }
        }
#elif defined(_WIN32)
        // Convert to TensorRT button (Windows only — .pth exists, no .engine)
        {
            bool has_engine = fs::exists(fs::path(state.models_folder) / "center_detect.engine");
            bool show_trt_convert = has_pth && !has_engine &&
                !jarvis.loaded && !jarvis_trt.loaded;
            bool converting = state.convert_job && state.convert_job->running.load();
            if (show_trt_convert) {
                ImGui::Separator();
                if (!converting) {
                    ImGui::TextWrapped("TensorRT engines not found. Convert .pth "
                                       "checkpoints to TensorRT for GPU FP16 acceleration. "
                                       "Point Models Folder to the JARVIS project directory "
                                       "(with config.yaml) for best results.");
                    if (ImGui::Button("Convert to TensorRT")) {
                        std::string exe_dir = ctx.window->exe_dir;
                        std::string script;
                        for (auto &candidate : {
                            exe_dir + "/../scripts/convert_pth_to_trt.py",
                            exe_dir + "/../src/convert_pth_to_trt.py",
                        }) {
                            if (fs::exists(candidate)) {
                                script = fs::canonical(candidate).string();
                                break;
                            }
                        }

                        if (script.empty()) {
                            state.convert_status = "Error: convert_pth_to_trt.py not found";
                        } else {
                            std::string jarvis_project = state.models_folder;
                            std::string folder_name = fs::path(state.models_folder)
                                .filename().string();
                            std::string output_dir =
                                (fs::path(ctx.pm.project_path) /
                                 "jarvis_models" / folder_name).string();
                            fs::create_directories(output_dir);

                            std::string cmd =
                                "python \"" + script +
                                "\" --jarvis_project \"" + jarvis_project +
                                "\" --output_dir \"" + output_dir + "\" 2>&1";

                            auto job = std::make_shared<ConvertJob>();
                            job->running.store(true);
                            job->output_path = output_dir;
                            state.convert_job = job;
                            state.convert_status = "Converting to TensorRT...";

                            std::thread([job, cmd]() {
                                FILE *pipe = popen(cmd.c_str(), "r");
                                if (!pipe) {
                                    job->message = "Error: failed to run conversion command";
                                    job->success = false;
                                    job->running.store(false);
                                    job->finished.store(true);
                                    return;
                                }
                                char buf[256];
                                std::string output;
                                while (fgets(buf, sizeof(buf), pipe))
                                    output += buf;
                                int ret = pclose(pipe);
                                if (ret == 0) {
                                    job->message = "TensorRT conversion complete.";
                                    job->success = true;
                                    job->force_rescan = true;
                                } else {
                                    job->message = "TensorRT conversion failed (exit " +
                                        std::to_string(ret) + "): " + output.substr(0, 200);
                                    job->success = false;
                                }
                                job->running.store(false);
                                job->finished.store(true);
                            }).detach();
                        }
                    }
                } else {
                    ImGui::BeginDisabled();
                    ImGui::Button("Converting...");
                    ImGui::EndDisabled();
                }
            }
        }
#endif

        // --- Model info (shown after loading) ---
        {
            bool onnx_active = jarvis.loaded;
            bool coreml_active = false;
            bool trt_active = false;
#ifdef __APPLE__
            coreml_active = jarvis_coreml.loaded;
#elif defined(_WIN32)
            trt_active = jarvis_trt.loaded;
#endif
            if (onnx_active || coreml_active || trt_active) {
                const auto &cfg = jarvis_active_config(jarvis
#ifdef __APPLE__
                    , jarvis_coreml
#elif defined(_WIN32)
                    , jarvis_trt
#endif
                );

                ImGui::Separator();
                ImGui::SeparatorText("Model Info");
                if (!cfg.project_name.empty())
                    ImGui::Text("Project:        %s", cfg.project_name.c_str());

                if (trt_active)
                    ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "Backend:        TensorRT (GPU FP16)");
                else if (coreml_active)
                    ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1), "Backend:        CoreML (GPU/ANE)");
                else
                    ImGui::Text("Backend:        ONNX Runtime (CPU)");

                ImGui::Text("Joints:         %d", cfg.num_joints);

                if (!cfg.architecture.empty())
                    ImGui::Text("Architecture:   %s", cfg.architecture.c_str());
                ImGui::Text("Center input:   %d x %d", cfg.center_input_size, cfg.center_input_size);
                ImGui::Text("Keypoint input: %d x %d", cfg.keypoint_input_size, cfg.keypoint_input_size);
                if (!cfg.precision.empty())
                    ImGui::Text("Precision:      %s", cfg.precision.c_str());
                if (cfg.imagenet_norm)
                    ImGui::Text("Normalization:  ImageNet (baked)");
                if (!state.model_dir_display.empty())
                    ImGui::Text("Model files:    %s", state.model_dir_display.c_str());
            }
        }

        // --- Prediction section ---
        ImGui::Separator();
        ImGui::SeparatorText("Prediction");

        ImGui::SliderFloat("Confidence Threshold", &state.confidence_threshold,
                           0.0f, 1.0f, "%.2f");

        // Show timing from whichever backend ran last
#ifdef __APPLE__
        if (jarvis_coreml.loaded && jarvis_coreml.last_total_ms > 0) {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                "CoreML: Center %.1f ms + Keypoint %.1f ms = %.1f ms",
                jarvis_coreml.last_center_ms, jarvis_coreml.last_keypoint_ms,
                jarvis_coreml.last_total_ms);
        } else
#elif defined(_WIN32)
        if (jarvis_trt.loaded && jarvis_trt.last_total_ms > 0) {
            ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                "TensorRT: Center %.1f ms + Keypoint %.1f ms = %.1f ms",
                jarvis_trt.last_center_ms, jarvis_trt.last_keypoint_ms,
                jarvis_trt.last_total_ms);
        } else
#endif
        if (jarvis.loaded && jarvis.last_total_ms > 0) {
            ImGui::Text("ONNX: Center %.1f ms + Keypoint %.1f ms = %.1f ms",
                        jarvis.last_center_ms, jarvis.last_keypoint_ms,
                        jarvis.last_total_ms);
        }
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
        if (jarvis_hn.loaded) {
            double total = jarvis_hn.last_center_ms + jarvis_hn.last_efftrack_ms +
                           jarvis_hn.last_hybrid3d_ms;
            if (total > 0) {
                ImGui::TextColored(ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                    "HybridNet 3D: Center %.0f + effTrack %.0f + Hybrid3D %.0f = %.0f ms  [%d/%d cams]",
                    jarvis_hn.last_center_ms, jarvis_hn.last_efftrack_ms,
                    jarvis_hn.last_hybrid3d_ms, total,
                    jarvis_hn.last_center_cams_used, jarvis_hn.cfg.num_cameras);
            }
        }
#endif
#endif

        ImGui::Text("Predict from:");
        ImGui::SameLine();
        if (ImGui::RadioButton("Shown", !state.predict_from_all))
            state.predict_from_all = false;
        ImGui::SameLine();
        if (ImGui::RadioButton("All", state.predict_from_all))
            state.predict_from_all = true;
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Shown: fast, uses visible cameras only\n"
                              "All: seeks all cameras to current frame first\n"
                              "(HybridNet ignores this and always uses all cameras)");

        ImGui::Checkbox("Streaming batch (faster)", &state.batch_streaming);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Batch predict streams frames continuously (seek once,\n"
                              "decode overlaps predict) instead of re-seeking every\n"
                              "buffer. Same output; faster on long videos. Off = the\n"
                              "original chunked path.");

        ImGui::Checkbox("Save 3D to Predictions folder (JARVIS format)",
                        &state.export_predictions3D);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "In addition to loading predictions as labeled frames, write a\n"
                "JARVIS-CLI-compatible Predictions_3D_<timestamp>/ folder\n"
                "(data3D.csv + info.yaml) under the project's\n"
                "predictions/predictions3D/. Batch Predict writes its whole\n"
                "start..end range (frames skipped by Step are NaN rows, so\n"
                "row index == frame). Predict Current Frame writes a 1-frame\n"
                "folder.");

#ifdef __APPLE__
        // HybridNet 3D: how many cameras feed CenterDetect (crop-ROI localization).
        // Keypoint detection always uses all cameras; fewer center cams ≈ halves
        // the center stage at negligible accuracy cost. Persisted in user settings.
        if (jarvis_coreml.hybridnet) {
            int max_cams = std::max(2, jarvis_coreml.hn_num_cameras);
            int cams = std::clamp(ctx.user_settings.jarvis_center_cams, 2, max_cams);
            ImGui::SetNextItemWidth(160);
            if (ImGui::SliderInt("Center cameras", &cams, 2, max_cams)) {
                ctx.user_settings.jarvis_center_cams = cams;
                jarvis_coreml.hn_center_cams = cams;
                save_user_settings(ctx.user_settings);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("Cameras used to locate the crop ROI (CenterDetect).\n"
                                  "Keypoint detection always uses all cameras. Fewer =\n"
                                  "faster center stage; %d = use all. Default 8.", max_cams);
        }
#endif

        bool can_predict = jarvis.loaded;
#ifdef __APPLE__
        can_predict = can_predict || jarvis_coreml.loaded;
#elif defined(_WIN32)
        can_predict = can_predict || jarvis_trt.loaded;
#endif
#if defined(__linux__) || defined(_WIN32)
#if defined(RED_HAS_ONNXRUNTIME) || defined(RED_HAS_TENSORRT_HN)
        can_predict = can_predict || jarvis_hn.loaded;
#endif
#endif
        if (!can_predict) ImGui::BeginDisabled();
        if (ImGui::Button("Predict Current Frame")) {
            state.predict_requested = true;
        }
        if (!can_predict) ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::TextDisabled("Press 6 to predict (hotkey)");

        // --- Batch Prediction ---
        ImGui::Separator();
        ImGui::SeparatorText("Batch Predict");

        if (state.batch_running) {
            // Progress display
            float progress = state.batch_total > 0
                ? (float)(state.batch_completed + state.batch_skipped) / state.batch_total : 0.0f;
            char overlay[64];
            snprintf(overlay, sizeof(overlay), "%d / %d (frame %d)",
                     state.batch_completed, state.batch_total, state.batch_current);
            ImGui::ProgressBar(progress, ImVec2(-FLT_MIN, 0), overlay);
            if (state.batch_skipped > 0)
                ImGui::TextDisabled("(%d skipped — already have manual labels)",
                                    state.batch_skipped);
            if (ImGui::Button("Cancel Batch")) {
                // Route through the state machine so FINISHING runs its cleanup
                // (stop decoders, release state) instead of leaving them running.
                state.batch_cancel_requested = true;
            }
        } else {
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("Start Frame", &state.batch_start);
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("End Frame", &state.batch_end);
            ImGui::SetNextItemWidth(120);
            ImGui::InputInt("Step", &state.batch_step);
            if (state.batch_step < 1) state.batch_step = 1;

            // Show count preview
            if (state.batch_end >= state.batch_start && state.batch_step > 0) {
                int n = (state.batch_end - state.batch_start) / state.batch_step + 1;
                ImGui::TextDisabled("%d frames to predict", n);
            }

            // Destination: separate store (default, keeps the Labeling Tool
            // clean + feeds Pose Stats/overlay) vs. loading as labeled frames.
            using PredDest = JarvisPredictState::PredDest;
            ImGui::Text("Send to:");
            ImGui::SameLine();
            if (ImGui::RadioButton("Predictions store",
                                   state.batch_prediction_dest == PredDest::Store))
                state.batch_prediction_dest = PredDest::Store;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Write to a separate prediction store (Pose Stats + read-only\n"
                    "video overlay). Does NOT add frames to the Labeling Tool, so\n"
                    "predicting large sections stays uncluttered.");
            ImGui::SameLine();
            if (ImGui::RadioButton("Labeling Tool",
                                   state.batch_prediction_dest == PredDest::LabelingTool))
                state.batch_prediction_dest = PredDest::LabelingTool;
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Legacy behavior: load predictions as labeled frames. Floods\n"
                    "the Labeling Tool when predicting large sections.");

            bool can_batch = can_predict &&
                state.batch_end >= state.batch_start && state.batch_step > 0;
            if (!can_batch) ImGui::BeginDisabled();
            if (ImGui::Button("Start Batch Predict")) {
                state.batch_requested = true;
            }
            if (!can_batch) ImGui::EndDisabled();
        }

        if (!state.batch_status.empty()) {
            bool is_done = state.batch_status.find("Complete") != std::string::npos;
            ImGui::TextColored(
                is_done ? ImVec4(0.5f, 1, 0.5f, 1) : ImVec4(1, 0.8f, 0, 1),
                "%s", state.batch_status.c_str());
        }

        if (!state.export_status.empty()) {
            bool is_error = state.export_status.find("Failed") != std::string::npos ||
                            state.export_status.find("Cannot") != std::string::npos ||
                            state.export_status.find("No ") != std::string::npos ||
                            state.export_status.find("Empty") != std::string::npos;
            ImGui::TextColored(
                is_error ? ImVec4(1, 0.3f, 0.3f, 1) : ImVec4(0.5f, 1, 0.5f, 1),
                "%s", state.export_status.c_str());
        }

        if (!state.store_status.empty()) {
            bool is_error = state.store_status.find("Failed") != std::string::npos ||
                            state.store_status.find("Cannot") != std::string::npos ||
                            state.store_status.find("⚠") != std::string::npos;
            ImGui::TextColored(
                is_error ? ImVec4(1, 0.5f, 0.2f, 1) : ImVec4(0.5f, 1, 0.5f, 1),
                "%s", state.store_status.c_str());
        }

        // --- Saved Predictions (load a previous session's store) ---
        ImGui::Separator();
        ImGui::SeparatorText("Saved Predictions");

        if (state.store_list_dirty) {
            jarvis_scan_prediction_stores(pm.project_path, state);
            state.store_list_dirty = false;
        }
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 130);
        if (ImGui::SmallButton("Import JARVIS…")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            if (!pm.media_folder.empty()) cfg.path = pm.media_folder;
            ImGuiFileDialog::Instance()->OpenDialog(
                "JarvisImportPreds", "Select JARVIS data3D.csv", ".csv",
                cfg);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip(
                "Import a JARVIS-CLI 3D prediction folder (data3D.csv + info.yaml,\n"
                "e.g. produced on a cluster) into this project's prediction store,\n"
                "so it gets the same overlay / Pose Stats / Bouts treatment as an\n"
                "in-app Batch Predict.");
        ImGui::SameLine();
        if (ImGui::SmallButton("Refresh")) state.store_list_dirty = true;

        if (!state.import_status.empty()) {
            bool import_err = state.import_status.find("Cannot") != std::string::npos ||
                              state.import_status.find("Failed") != std::string::npos ||
                              state.import_status.find("No ") != std::string::npos ||
                              state.import_status.find("missing") != std::string::npos ||
                              state.import_status.find("no data") != std::string::npos;
            ImGui::TextColored(
                import_err ? ImVec4(1, 0.5f, 0.2f, 1) : ImVec4(0.5f, 1, 0.5f, 1),
                "%s", state.import_status.c_str());
        }

        if (state.store_list.empty()) {
            ImGui::TextDisabled("No saved prediction stores in this project.");
        } else {
            if (ImGui::BeginChild("##store_list", ImVec2(0, 110), true)) {
                for (const auto &sp : state.store_list) {
                    bool is_active = (sp.path == state.active_store_path);
                    char row[256];
                    snprintf(row, sizeof(row), "%s  ·  %u frames%s%s",
                             sp.label.c_str(), sp.n_stored,
                             sp.model.empty() ? "" : "  ·  ",
                             sp.model.c_str());
                    if (ImGui::Selectable(row, is_active))
                        state.load_store_request = sp.path;
                    if (ImGui::IsItemHovered()) {
                        std::string tip = sp.path;
                        if (sp.frame_start >= 0)
                            tip += "\nframes " + std::to_string(sp.frame_start) +
                                   "–" + std::to_string(sp.frame_end);
                        tip += "\n" + std::to_string(sp.num_keypoints) +
                               " keypoints · " + std::to_string(sp.fps) + " fps";
                        if (!sp.video.empty()) tip += "\nvideo: " + sp.video;
                        ImGui::SetTooltip("%s", tip.c_str());
                    }
                }
            }
            ImGui::EndChild();
        }

        // Overlay toggle — shown whenever a store is loaded.
        if (!state.active_store_path.empty()) {
            ImGui::Checkbox("Show predictions overlay on videos",
                            &state.show_prediction_overlay);
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Draw the loaded store's 3D poses (confidence-colored) over\n"
                    "the camera videos as you play. Read-only; independent of the\n"
                    "Labeling Tool.");
        }
        },
        // always_fn: file dialog handlers (run every frame)
        [&]() {
            if (ImGuiFileDialog::Instance()->Display(
                    "JarvisBrowseModels", ImGuiWindowFlags_NoCollapse,
                    ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.models_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                }
                ImGuiFileDialog::Instance()->Close();
            }
            if (ImGuiFileDialog::Instance()->Display(
                    "JarvisImportPreds", ImGuiWindowFlags_NoCollapse,
                    ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.import_request =
                        ImGuiFileDialog::Instance()->GetFilePathName();
                }
                ImGuiFileDialog::Instance()->Close();
            }
        },
        ImVec2(480, 600));
}
