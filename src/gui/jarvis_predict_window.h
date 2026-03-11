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
};

inline void DrawJarvisPredictWindow(JarvisPredictState &state, JarvisState &jarvis,
                                     AppContext &ctx) {
    (void)ctx; // AppContext available for future use (e.g. auto-detect project paths)
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

        // --- JARVIS Project section ---
        ImGui::SeparatorText("JARVIS Project");

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

        // --- Resolve ONNX paths and show Load / Convert buttons ---
        namespace fs = std::filesystem;

        // Detect what files are available
        bool has_onnx_subdir = false;
        bool has_onnx_direct = false;
        bool has_pth = false;
        std::string center_path, keypoint_path, info_path;

        if (!state.models_folder.empty() && fs::is_directory(state.models_folder)) {
            // Check models/onnx/ first
            fs::path onnx_dir = fs::path(state.models_folder) / "onnx";
            if (fs::is_directory(onnx_dir)) {
                fs::path c = onnx_dir / "center_detect.onnx";
                fs::path k = onnx_dir / "keypoint_detect.onnx";
                if (fs::exists(c) && fs::exists(k)) {
                    has_onnx_subdir = true;
                    center_path = c.string();
                    keypoint_path = k.string();
                    fs::path mi = onnx_dir / "model_info.json";
                    if (fs::exists(mi)) info_path = mi.string();
                }
            }
            // Check models/ directly
            if (!has_onnx_subdir) {
                fs::path c = fs::path(state.models_folder) / "center_detect.onnx";
                fs::path k = fs::path(state.models_folder) / "keypoint_detect.onnx";
                if (fs::exists(c) && fs::exists(k)) {
                    has_onnx_direct = true;
                    center_path = c.string();
                    keypoint_path = k.string();
                    fs::path mi = fs::path(state.models_folder) / "model_info.json";
                    if (fs::exists(mi)) info_path = mi.string();
                }
            }
            // Check for .pth files (conversion candidate)
            if (!has_onnx_subdir && !has_onnx_direct) {
                for (auto &entry : fs::directory_iterator(state.models_folder)) {
                    if (entry.path().extension() == ".pth") {
                        has_pth = true;
                        break;
                    }
                }
            }
        }

        bool can_load = !center_path.empty() && !keypoint_path.empty();

        // Show file detection status
        if (!state.models_folder.empty()) {
            if (can_load) {
                std::string loc = has_onnx_subdir ? "onnx/" : "models/";
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

        if (jarvis.loaded && jarvis.last_total_ms > 0) {
            ImGui::Text("Center: %.1f ms  Keypoint: %.1f ms  "
                        "Triangulate: %.1f ms  Total: %.1f ms",
                        jarvis.last_center_ms, jarvis.last_keypoint_ms,
                        jarvis.last_triangulate_ms, jarvis.last_total_ms);
        }

        if (!jarvis.loaded) ImGui::BeginDisabled();
        if (ImGui::Button("Predict Current Frame")) {
            // Set flag — main loop handles RGB extraction + prediction
            // (pixel buffer access is platform-specific, same as hotkey 5)
            state.predict_requested = true;
        }
        if (!jarvis.loaded) ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::TextDisabled("Press 5 to predict (hotkey)");
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
