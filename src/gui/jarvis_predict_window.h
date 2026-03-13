#pragma once
// jarvis_predict_window.h — JARVIS Model Selector + Prediction panel
//
// Loads JARVIS CenterDetect + KeypointDetect models (CoreML on macOS,
// ONNX Runtime elsewhere) and provides a UI for running 2D pose predictions.
// Model files can be in <project>/models/onnx/ or loaded manually.
// If only .pth checkpoints exist, offers a "Convert to ONNX" button.

#include "imgui.h"
#include "app_context.h"
#include "jarvis_inference.h"
#ifdef __APPLE__
#include "jarvis_coreml.h"
#endif
#include "gui/panel.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>
#include <atomic>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>

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

    // Conversion state (thread-safe via shared_ptr)
    std::shared_ptr<ConvertJob> convert_job;
    std::string convert_status;  // UI-side copy, updated from job each frame

    // Filesystem detection cache (avoid scanning every frame)
    std::string cached_models_folder;
    bool cached_has_onnx = false;
    bool cached_has_pth = false;
    bool cached_has_coreml = false;
    std::string cached_center_path, cached_keypoint_path, cached_info_path;

    // Relative model path shown in Model Info
    std::string model_dir_display;
};

// Get the active model config from whichever backend is loaded
inline const JarvisModelConfig &jarvis_active_config(
    const JarvisState &jarvis
#ifdef __APPLE__
    , const JarvisCoreMLState &coreml
#endif
) {
    static const JarvisModelConfig empty;
#ifdef __APPLE__
    if (coreml.loaded) return coreml.config;
#endif
    if (jarvis.loaded) return jarvis.config;
    return empty;
}

inline void DrawJarvisPredictWindow(JarvisPredictState &state, JarvisState &jarvis,
#ifdef __APPLE__
                                     JarvisCoreMLState &jarvis_coreml,
#endif
                                     AppContext &ctx) {
    DrawPanel("JARVIS Predict", state.show,
        [&]() {
        // Availability check
        if (!jarvis.available) {
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
#else
        bool any_loaded = jarvis.loaded;
#endif
        if (!any_loaded && pm.active_jarvis_model >= 0 &&
            pm.active_jarvis_model < (int)pm.jarvis_models.size()) {
            auto &m = pm.jarvis_models[pm.active_jarvis_model];
            std::string base = pm.project_path + "/" + m.relative_path;
            std::string mi = base + "/model_info.json";
            auto cfg = parse_jarvis_model_info(
                std::filesystem::exists(mi) ? mi.c_str() : nullptr);
            state.model_dir_display = m.relative_path;
#ifdef __APPLE__
            std::string cd_ml = base + "/center_detect.mlpackage";
            std::string kd_ml = base + "/keypoint_detect.mlpackage";
            if (std::filesystem::exists(cd_ml) && std::filesystem::exists(kd_ml)) {
                jarvis_cleanup(jarvis);
                jarvis_coreml_init(jarvis_coreml, base, cfg);
            } else
#endif
            {
                std::string cd = base + "/center_detect.onnx";
                std::string kd = base + "/keypoint_detect.onnx";
                if (std::filesystem::exists(cd) && std::filesystem::exists(kd)) {
#ifdef __APPLE__
                    jarvis_coreml_cleanup(jarvis_coreml);
#endif
                    jarvis_init(jarvis, cd.c_str(), kd.c_str(), cfg);
                }
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
                        std::string mi = base + "/model_info.json";
                        auto cfg = parse_jarvis_model_info(
                            std::filesystem::exists(mi) ? mi.c_str() : nullptr);
                        state.model_dir_display = m.relative_path;
#ifdef __APPLE__
                        std::string cd_ml = base + "/center_detect.mlpackage";
                        std::string kd_ml = base + "/keypoint_detect.mlpackage";
                        if (std::filesystem::exists(cd_ml) && std::filesystem::exists(kd_ml)) {
                            jarvis_cleanup(jarvis);
                            jarvis_coreml_init(jarvis_coreml, base, cfg);
                        } else
#endif
                        {
                            std::string cd = base + "/center_detect.onnx";
                            std::string kd = base + "/keypoint_detect.onnx";
#ifdef __APPLE__
                            jarvis_coreml_cleanup(jarvis_coreml);
#endif
                            jarvis_init(jarvis, cd.c_str(), kd.c_str(), cfg);
                        }
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
            // On success, redirect models_folder to the output directory
            // so the rescan finds the newly converted .mlpackage files
            if (state.convert_job->success && !state.convert_job->output_path.empty()) {
                state.models_folder = state.convert_job->output_path;
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
            }
        }

        // Use cached detection results
        bool has_pth = state.cached_has_pth;
        bool has_coreml = state.cached_has_coreml;
        std::string center_path = state.cached_center_path;
        std::string keypoint_path = state.cached_keypoint_path;
        std::string info_path = state.cached_info_path;
        bool can_load = !center_path.empty() && !keypoint_path.empty();
        bool can_load_any = can_load || has_coreml;

        // Show file detection status
        if (!state.models_folder.empty()) {
            if (can_load && has_coreml) {
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

            // Parse config once
            auto cfg = parse_jarvis_model_info(
                info_path.empty() ? nullptr : info_path.c_str());

            // Determine model name
            std::string model_name = cfg.project_name;
            if (model_name.empty())
                model_name = fs::path(src_dir).filename().string();
            if (model_name.empty()) model_name = "jarvis_model";

            // Load model — prefer CoreML on macOS
            bool loaded_any = false;
            int nj = 0, ci_sz = 0, ki_sz = 0;
#ifdef __APPLE__
            {
                fs::path cd_ml = fs::path(src_dir) / "center_detect.mlpackage";
                fs::path kd_ml = fs::path(src_dir) / "keypoint_detect.mlpackage";
                if (fs::exists(cd_ml) && fs::exists(kd_ml)) {
                    jarvis_cleanup(jarvis);
                    jarvis_coreml_init(jarvis_coreml, src_dir, cfg);
                    if (jarvis_coreml.loaded) {
                        loaded_any = true;
                        nj = jarvis_coreml.num_joints;
                        ci_sz = jarvis_coreml.center_input_size;
                        ki_sz = jarvis_coreml.keypoint_input_size;
                    }
                }
            }
            if (!loaded_any)
#endif
            if (can_load) {
#ifdef __APPLE__
                jarvis_coreml_cleanup(jarvis_coreml);
#endif
                jarvis_init(jarvis, center_path.c_str(), keypoint_path.c_str(), cfg);
                if (jarvis.loaded) {
                    loaded_any = true;
                    nj = jarvis.config.num_joints;
                    ci_sz = jarvis.config.center_input_size;
                    ki_sz = jarvis.config.keypoint_input_size;
                }
            }

            // Copy model files into project folder
            if (loaded_any && !pm.project_path.empty()) {
                std::string rel = "jarvis_models/" + model_name;
                fs::path dest = fs::path(pm.project_path) / rel;
                std::error_code ec;
                fs::create_directories(dest, ec);

                for (auto &entry : fs::directory_iterator(src_dir)) {
                    auto fname = entry.path().filename().string();
                    if (fname.find(".onnx") != std::string::npos ||
                        fname == "model_info.json") {
                        fs::copy_file(entry.path(), dest / fname,
                                      fs::copy_options::overwrite_existing, ec);
                    }
                    if (entry.is_directory() && fname.find(".mlpackage") != std::string::npos) {
                        fs::path ml_dest = dest / fname;
                        fs::remove_all(ml_dest, ec);
                        fs::copy(entry.path(), ml_dest,
                                 fs::copy_options::recursive, ec);
                    }
                }

                // Register in project (avoid duplicates)
                ProjectManager::JarvisModelEntry me;
                me.name = model_name;
                me.relative_path = rel;
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
                state.model_dir_display = rel;

                fs::path redproj = fs::path(pm.project_path) /
                                   (pm.project_name + ".redproj");
                save_project_manager_json(pm, redproj);

                // Clear import form and stale status
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
#endif
            if (show_loaded) {
                ImGui::TextColored(ImVec4(0, 1, 0, 1), "Loaded");
            } else if (!jarvis.status.empty()) {
                ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", jarvis.status.c_str());
            }
#ifdef __APPLE__
            else if (!jarvis_coreml.status.empty()) {
                ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "%s", jarvis_coreml.status.c_str());
            }
#endif
        }

        // Convert to ONNX button (only if .pth exists and no model loaded)
        {
            bool show_convert = has_pth && !can_load;
#ifdef __APPLE__
            show_convert = show_convert && !jarvis.loaded && !jarvis_coreml.loaded;
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
                                job->message = "Conversion complete. Click Import to Project.";
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

                            std::string cmd =
                                "conda run -n coreml python \"" + script +
                                "\" --jarvis_project \"" + jarvis_project +
                                "\" --output_dir \"" + output_dir + "\" 2>&1";

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
                                    job->message = "CoreML conversion complete. Click Import to Project.";
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
#endif

        // --- Model info (shown after loading) ---
        {
            bool onnx_active = jarvis.loaded;
            bool coreml_active = false;
#ifdef __APPLE__
            coreml_active = jarvis_coreml.loaded;
#endif
            if (onnx_active || coreml_active) {
                const auto &cfg = jarvis_active_config(jarvis
#ifdef __APPLE__
                    , jarvis_coreml
#endif
                );

                ImGui::Separator();
                ImGui::SeparatorText("Model Info");
                if (!cfg.project_name.empty())
                    ImGui::Text("Project:        %s", cfg.project_name.c_str());

                if (coreml_active)
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
#endif
        if (jarvis.loaded && jarvis.last_total_ms > 0) {
            ImGui::Text("ONNX: Center %.1f ms + Keypoint %.1f ms = %.1f ms",
                        jarvis.last_center_ms, jarvis.last_keypoint_ms,
                        jarvis.last_total_ms);
        }

        bool can_predict = jarvis.loaded;
#ifdef __APPLE__
        can_predict = can_predict || jarvis_coreml.loaded;
#endif
        if (!can_predict) ImGui::BeginDisabled();
        if (ImGui::Button("Predict Current Frame")) {
            state.predict_requested = true;
        }
        if (!can_predict) ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::TextDisabled("Press 6 to predict (hotkey)");
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
        },
        ImVec2(480, 600));
}
