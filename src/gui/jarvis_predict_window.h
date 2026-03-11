#pragma once
// jarvis_predict_window.h — JARVIS Model Selector + Prediction panel
//
// Loads JARVIS CenterDetect + KeypointDetect ONNX models and provides
// a UI for running 2D pose predictions on the current frame.
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
#include <filesystem>
#include <string>
#include <thread>

struct JarvisPredictState {
    bool show = false;
    std::string models_folder;  // path to JARVIS project's models/ directory
    std::string config_path;    // path to JARVIS project's config.yaml
    float confidence_threshold = 0.1f;

    // Internal: resolved ONNX paths (found during "Load Model")
    std::string center_onnx_path;
    std::string keypoint_onnx_path;
    std::string model_info_path;

    // Set by "Predict Current Frame" button; consumed by main loop
    // (RGB extraction requires platform-specific pixel buffer access)
    bool predict_requested = false;

    // Conversion state
    bool converting = false;
    std::string convert_status;

    // Filesystem detection cache (avoid scanning every frame)
    std::string cached_models_folder;
    bool cached_has_onnx = false;
    bool cached_has_pth = false;
    std::string cached_center_path, cached_keypoint_path, cached_info_path;
};

inline void DrawJarvisPredictWindow(JarvisPredictState &state, JarvisState &jarvis,
#ifdef __APPLE__
                                     JarvisCoreMLState &jarvis_coreml,
#endif
                                     AppContext &ctx) {
    drawPanel("JARVIS Predict", state.show,
        [&]() {
        // Availability check (same pattern as SAM)
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
            std::string mi_c = std::filesystem::exists(mi) ? mi : "";
#ifdef __APPLE__
            // Prefer CoreML (.mlpackage) for GPU/ANE acceleration
            std::string cd_ml = base + "/center_detect.mlpackage";
            std::string kd_ml = base + "/keypoint_detect.mlpackage";
            if (std::filesystem::exists(cd_ml) && std::filesystem::exists(kd_ml)) {
                jarvis_coreml_init(jarvis_coreml, base,
                                    mi_c.empty() ? nullptr : mi_c.c_str());
            } else
#endif
            {
                // Fallback: ONNX Runtime
                std::string cd = base + "/center_detect.onnx";
                std::string kd = base + "/keypoint_detect.onnx";
                if (std::filesystem::exists(cd) && std::filesystem::exists(kd))
                    jarvis_init(jarvis, cd.c_str(), kd.c_str(),
                                mi_c.empty() ? nullptr : mi_c.c_str());
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
                        std::string cd = base + "/center_detect.onnx";
                        std::string kd = base + "/keypoint_detect.onnx";
                        std::string mi = base + "/model_info.json";
                        jarvis_init(jarvis, cd.c_str(), kd.c_str(),
                                    std::filesystem::exists(mi) ? mi.c_str() : nullptr);
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

        ImGui::Text("config.yaml");
        ImGui::SetNextItemWidth(-60);
        ImGui::InputText("##jarvis_config", &state.config_path);
        ImGui::SameLine();
        if (ImGui::Button("...##config")) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.flags = ImGuiFileDialogFlags_Modal;
            if (!state.config_path.empty()) {
                auto parent = std::filesystem::path(state.config_path).parent_path();
                if (std::filesystem::is_directory(parent))
                    cfg.path = parent.string();
            }
            ImGuiFileDialog::Instance()->OpenDialog(
                "JarvisBrowseConfig", "Select config.yaml", ".yaml,.yml", cfg);
        }

        ImGui::Separator();

        // --- Resolve ONNX paths (cached — only rescan when folder changes) ---
        namespace fs = std::filesystem;

        // Rescan filesystem only when models_folder changes or after conversion
        if (state.models_folder != state.cached_models_folder) {
            state.cached_models_folder = state.models_folder;
            state.cached_has_onnx = false;
            state.cached_has_pth = false;
            state.cached_center_path.clear();
            state.cached_keypoint_path.clear();
            state.cached_info_path.clear();

            bool has_onnx_subdir = false, has_onnx_direct = false, has_pth = false;
            std::string center_path, keypoint_path, info_path;

            if (!state.models_folder.empty() && fs::is_directory(state.models_folder)) {
            // Search order for ONNX files:
            // 1. models/onnx/  (our export convention)
            // 2. models/ directly
            // Then check for .pth files in JARVIS structure:
            // 3. models/CenterDetect/Run_*/  and  models/KeypointDetect/Run_*/

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

            // Check for .pth files in JARVIS structure (CenterDetect/Run_*/, KeypointDetect/Run_*/)
            if (!has_onnx_subdir && !has_onnx_direct) {
                auto find_latest_pth = [](const fs::path &module_dir) -> std::string {
                    if (!fs::is_directory(module_dir)) return {};
                    std::vector<fs::path> runs;
                    for (auto &e : fs::directory_iterator(module_dir))
                        if (e.is_directory() && e.path().filename().string().find("Run_") == 0)
                            runs.push_back(e.path());
                    if (runs.empty()) return {};
                    std::sort(runs.begin(), runs.end());
                    // Find *_final.pth in latest run
                    for (auto &e : fs::directory_iterator(runs.back()))
                        if (e.path().extension() == ".pth" &&
                            e.path().filename().string().find("final") != std::string::npos)
                            return e.path().string();
                    return {};
                };
                std::string cd_pth = find_latest_pth(fs::path(state.models_folder) / "CenterDetect");
                std::string kd_pth = find_latest_pth(fs::path(state.models_folder) / "KeypointDetect");
                has_pth = !cd_pth.empty() && !kd_pth.empty();
            }
            // Cache results
            state.cached_has_onnx = (has_onnx_subdir || has_onnx_direct);
            state.cached_has_pth = has_pth;
            state.cached_center_path = center_path;
            state.cached_keypoint_path = keypoint_path;
            state.cached_info_path = info_path;
            } // end rescan block
        }

        // Use cached detection results
        bool has_pth = state.cached_has_pth;
        std::string center_path = state.cached_center_path;
        std::string keypoint_path = state.cached_keypoint_path;
        std::string info_path = state.cached_info_path;
        bool can_load = !center_path.empty() && !keypoint_path.empty();

        // Show file detection status
        if (!state.models_folder.empty()) {
            if (can_load) {
                std::string loc = state.cached_has_onnx ? "found" : "models/";
                ImGui::TextColored(ImVec4(0.5f, 1, 0.5f, 1),
                    "Found ONNX files in %s", loc.c_str());
            } else if (has_pth) {
                ImGui::TextColored(ImVec4(1, 0.8f, 0, 1),
                    "Found .pth checkpoints (no ONNX files)");
            } else if (fs::is_directory(state.models_folder)) {
                ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                    "No ONNX or .pth files found");
            }
        }

        // Load Model button
        if (!can_load) ImGui::BeginDisabled();
        if (ImGui::Button("Load Model")) {
            state.center_onnx_path = center_path;
            state.keypoint_onnx_path = keypoint_path;
            state.model_info_path = info_path;
            jarvis_init(jarvis, center_path.c_str(), keypoint_path.c_str(),
                        info_path.empty() ? nullptr : info_path.c_str());

            // Import into project: copy ONNX files into project folder
            if (jarvis.loaded && !pm.project_path.empty()) {
                std::string src_dir = fs::path(center_path).parent_path().string();
                // Read model name from model_info.json or derive from folder
                std::string model_name = jarvis.config.project_name;
                if (model_name.empty())
                    model_name = fs::path(src_dir).filename().string();
                if (model_name.empty()) model_name = "jarvis_model";

                std::string rel = "jarvis_models/" + model_name;
                fs::path dest = fs::path(pm.project_path) / rel;
                std::error_code ec;
                fs::create_directories(dest, ec);

                // Copy .onnx, .onnx.data, model_info.json
                for (auto &entry : fs::directory_iterator(src_dir)) {
                    auto fname = entry.path().filename().string();
                    if (fname.find(".onnx") != std::string::npos ||
                        fname == "model_info.json") {
                        fs::copy_file(entry.path(), dest / fname,
                                      fs::copy_options::overwrite_existing, ec);
                    }
                }

                // Add to project (avoid duplicates)
                ProjectManager::JarvisModelEntry me;
                me.name = model_name;
                me.relative_path = rel;
                me.num_joints = jarvis.config.num_joints;
                me.center_input_size = jarvis.config.center_input_size;
                me.keypoint_input_size = jarvis.config.keypoint_input_size;

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

                // Save .redproj
                fs::path redproj = fs::path(pm.project_path) /
                                   (pm.project_name + ".redproj");
                save_project_manager_json(pm, redproj);
            }
        }
        if (!can_load) ImGui::EndDisabled();

        ImGui::SameLine();
        if (jarvis.loaded) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Loaded");
        } else if (!jarvis.status.empty()) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "%s", jarvis.status.c_str());
        }

        // Convert to ONNX button (only if .pth exists but no ONNX)
        if (has_pth && !can_load && !jarvis.loaded) {
            ImGui::Separator();
            if (!state.converting) {
                ImGui::TextWrapped("ONNX files not found. You can convert .pth "
                                   "checkpoints to ONNX using the JARVIS export script.");
                if (ImGui::Button("Convert to ONNX")) {
                    state.converting = true;
                    state.convert_status = "Converting...";

                    // Derive project path: models_folder should be <project>/models
                    fs::path project_path = fs::path(state.models_folder).parent_path();
                    fs::path onnx_out = fs::path(state.models_folder) / "onnx";
                    std::string cmd =
                        "conda run -n jarvis python -m jarvis.utils.onnx_export \"" +
                        project_path.string() + "\" --output_dir \"" +
                        onnx_out.string() + "\" 2>&1";

                    // Run in detached thread to avoid blocking UI
                    std::string cmd_copy = cmd;
                    std::thread([cmd_copy, &state]() {
                        FILE *pipe = popen(cmd_copy.c_str(), "r");
                        if (!pipe) {
                            state.convert_status = "Error: failed to run conversion command";
                            state.converting = false;
                            return;
                        }
                        char buf[256];
                        std::string output;
                        while (fgets(buf, sizeof(buf), pipe))
                            output += buf;
                        int ret = pclose(pipe);
                        if (ret == 0) {
                            state.convert_status = "Conversion complete. Click Load Model.";
                            state.cached_models_folder.clear(); // force rescan
                        } else {
                            state.convert_status = "Conversion failed (exit " +
                                std::to_string(ret) + "): " + output.substr(0, 200);
                        }
                        state.converting = false;
                    }).detach();
                }
            } else {
                ImGui::BeginDisabled();
                ImGui::Button("Converting...");
                ImGui::EndDisabled();
            }
            if (!state.convert_status.empty()) {
                bool is_error = state.convert_status.find("Error") != std::string::npos ||
                                state.convert_status.find("failed") != std::string::npos;
                ImGui::TextColored(
                    is_error ? ImVec4(1, 0.3f, 0.3f, 1) : ImVec4(0.5f, 1, 0.5f, 1),
                    "%s", state.convert_status.c_str());
            }
        }

        // --- Model info (shown after loading) ---
        if (jarvis.loaded) {
            ImGui::Separator();
            ImGui::SeparatorText("Model Info");
            if (!jarvis.config.project_name.empty())
                ImGui::Text("Project:  %s", jarvis.config.project_name.c_str());
            ImGui::Text("Joints:   %d", jarvis.config.num_joints);
            ImGui::Text("Center input:   %d x %d",
                        jarvis.config.center_input_size,
                        jarvis.config.center_input_size);
            ImGui::Text("Keypoint input: %d x %d",
                        jarvis.config.keypoint_input_size,
                        jarvis.config.keypoint_input_size);
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
            // Set flag — main loop handles RGB extraction + prediction
            // (pixel buffer access is platform-specific, same as hotkey 6)
            state.predict_requested = true;
        }
        if (!can_predict) ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::TextDisabled("Press 6 to predict (hotkey)");
        },
        // always_fn: file dialog handlers (run every frame)
        [&]() {
            // Models folder directory picker
            if (ImGuiFileDialog::Instance()->Display(
                    "JarvisBrowseModels", ImGuiWindowFlags_NoCollapse,
                    ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.models_folder =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                }
                ImGuiFileDialog::Instance()->Close();
            }
            // Config file picker
            if (ImGuiFileDialog::Instance()->Display(
                    "JarvisBrowseConfig", ImGuiWindowFlags_NoCollapse,
                    ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    state.config_path =
                        ImGuiFileDialog::Instance()->GetFilePathName();
                }
                ImGuiFileDialog::Instance()->Close();
            }
        },
        ImVec2(420, 520));
}
