#pragma once
// body_model_window.h — MuJoCo body model viewer + IK controls
//
// Renders a posed body model via Metal, with controls for loading models,
// running inverse kinematics, and adjusting solver parameters.

#include "imgui.h"
#include "annotation.h"
#include "app_context.h"
#include "gui/panel.h"
#include "mujoco_context.h"
#include "mujoco_ik.h"
#include <ImGuiFileDialog.h>
#include <string>

#ifdef RED_HAS_MUJOCO
#include "mujoco_metal_renderer.h"
#endif

struct BodyModelState {
    bool show = false;

    // IK solver state (owns warm-start)
    MujocoIKState ik_state;

    // Display options
    bool auto_solve = false;
    bool show_skin = true;
    bool show_site_markers = true;

    // Deferred unload (processed at start of next frame)
    bool unload_requested = false;

    // Camera for 3D view (good default for looking at rat on arena)
    float cam_lookat[3] = {0.0f, 0.0f, 0.04f};
    float cam_distance  = 0.4f;
    float cam_azimuth   = 160.0f;
    float cam_elevation  = -30.0f;
    bool  show_arena     = true;

    // Track which frame we last solved for
    int last_solved_frame = -1;

    // Renderer handle
#ifdef RED_HAS_MUJOCO
    MujocoRenderer *renderer = nullptr;
#endif

    // Model path for file dialog
    std::string model_path;
};

inline void DrawBodyModelWindow(BodyModelState &state, MujocoContext &mj,
                                AppContext &ctx) {
    DrawPanel("Body Model", state.show,
        [&]() {
#ifndef RED_HAS_MUJOCO
            ImGui::TextColored(ImVec4(1, 0.5, 0, 1),
                "MuJoCo not available. Install mujoco.framework to lib/ and rebuild.");
            return;
#else
            // --- Deferred unload (safe: no ImGui widgets active) ---
            if (state.unload_requested) {
                state.unload_requested = false;
                if (state.renderer) {
                    mujoco_renderer_destroy(state.renderer);
                    state.renderer = nullptr;
                }
                mujoco_ik_reset(state.ik_state);
                mj.unload();
                state.last_solved_frame = -1;
            }

            // --- Model loading ---
            if (!mj.loaded) {
                ImGui::TextWrapped("Load a MuJoCo XML body model to enable "
                                   "inverse kinematics and body model visualization.");
                ImGui::Spacing();

                if (state.model_path.empty()) {
                    // Try default path relative to exe
                    std::string default_path = "models/rodent/rodent_no_collision.xml";
                    if (std::filesystem::exists(default_path))
                        state.model_path = default_path;
                }

                ImGui::InputText("Model XML", &state.model_path);
                ImGui::SameLine();
                if (ImGui::Button("Browse...")) {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 1;
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    if (!state.model_path.empty()) {
                        std::filesystem::path p(state.model_path);
                        if (p.has_parent_path())
                            cfg.path = p.parent_path().string();
                    }
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseMujocoModel", "Select MuJoCo XML", ".xml", cfg);
                }

                if (ImGui::Button("Load Model") && !state.model_path.empty()) {
                    if (mj.load(state.model_path, ctx.skeleton)) {
                        ctx.toasts.push("MuJoCo model loaded: " +
                                        std::to_string(mj.mapped_count) + "/" +
                                        std::to_string(ctx.skeleton.num_nodes) +
                                        " sites matched");
                    } else {
                        ctx.toasts.push("Failed: " + mj.load_error, Toast::Error);
                    }
                }

                if (!mj.load_error.empty())
                    ImGui::TextColored(ImVec4(1, 0.3, 0.3, 1), "%s", mj.load_error.c_str());
                return;
            }

            if (!mj.model) return; // Defensive: loaded but model null

            // --- Model info ---
            ImGui::Text("Model: %s", mj.model_path.c_str());
            ImGui::Text("Sites: %d/%d matched  |  Bodies: %d  |  Joints: %d",
                        mj.mapped_count, ctx.skeleton.num_nodes,
                        (int)mj.model->nbody, (int)mj.model->njnt);
            ImGui::Separator();

            // --- Solver controls ---
            ImGui::Text("IK Solver");
            if (ImGui::SliderFloat("Scale factor", &mj.scale_factor, 0.0f, 3.0f,
                                   mj.scale_factor == 0.0f ? "auto" : "%.4f"))
                mujoco_ik_reset(state.ik_state); // reset warm-start on scale change
            if (mj.scale_factor == 0.0f)
                ImGui::SameLine(), ImGui::TextDisabled("(auto-detect from data)");

            ImGui::SliderInt("Max iterations", &state.ik_state.max_iterations, 1, 2000);
            float reg_log = (state.ik_state.reg_strength > 0)
                                ? log10f((float)state.ik_state.reg_strength) : -6.0f;
            if (ImGui::SliderFloat("Regularization (log10)", &reg_log, -6.0f, 0.0f, "%.1f"))
                state.ik_state.reg_strength = pow(10.0, reg_log);
            ImGui::Checkbox("Auto-solve on frame change", &state.auto_solve);
            ImGui::SameLine();
            if (ImGui::Button("Solve Frame")) {
                auto it = ctx.annotations.find(ctx.current_frame_num);
                if (it != ctx.annotations.end() && !it->second.kp3d.empty()) {
                    mujoco_ik_solve(mj, state.ik_state,
                                    it->second.kp3d.data(),
                                    ctx.skeleton.num_nodes,
                                    ctx.current_frame_num);
                    state.last_solved_frame = ctx.current_frame_num;
                    // Auto-center camera on the torso body after solve
                    int torso_id = mj_name2id(mj.model, mjOBJ_BODY, "torso");
                    if (torso_id >= 0) {
                        state.cam_lookat[0] = (float)mj.data->xpos[3*torso_id+0];
                        state.cam_lookat[1] = (float)mj.data->xpos[3*torso_id+1];
                        state.cam_lookat[2] = (float)mj.data->xpos[3*torso_id+2];
                    }
                }
            }

            // Auto-solve logic
            if (state.auto_solve && ctx.current_frame_num != state.last_solved_frame) {
                state.last_solved_frame = ctx.current_frame_num;
                auto it = ctx.annotations.find(ctx.current_frame_num);
                if (it != ctx.annotations.end() && !it->second.kp3d.empty()) {
                    mujoco_ik_solve(mj, state.ik_state,
                                    it->second.kp3d.data(),
                                    ctx.skeleton.num_nodes,
                                    ctx.current_frame_num);
                    // Follow the model
                    int torso_id = mj_name2id(mj.model, mjOBJ_BODY, "torso");
                    if (torso_id >= 0) {
                        state.cam_lookat[0] = (float)mj.data->xpos[3*torso_id+0];
                        state.cam_lookat[1] = (float)mj.data->xpos[3*torso_id+1];
                        state.cam_lookat[2] = (float)mj.data->xpos[3*torso_id+2];
                    }
                }
            }

            // Solver status
            if (state.ik_state.active_sites > 0) {
                ImVec4 color = state.ik_state.converged
                                   ? ImVec4(0.2f, 0.9f, 0.2f, 1.0f)
                                   : ImVec4(1.0f, 0.7f, 0.2f, 1.0f);
                ImGui::TextColored(color, "%s in %d iters (%.1f ms)  |  "
                                   "Residual: %.1f mm  |  Sites: %d",
                                   state.ik_state.converged ? "Converged" : "Not converged",
                                   state.ik_state.iterations_used,
                                   state.ik_state.solve_time_ms,
                                   state.ik_state.final_residual * 1000.0,
                                   state.ik_state.active_sites);
            }

            ImGui::Separator();

            // --- Display options ---
            ImGui::Checkbox("Skin", &state.show_skin);
            ImGui::SameLine();
            ImGui::Checkbox("Sites", &state.show_site_markers);
            ImGui::SameLine();
            ImGui::Checkbox("Arena", &state.show_arena);

            // --- Camera controls ---
            ImGui::SliderFloat("Distance", &state.cam_distance, 0.05f, 5.0f, "%.2f");

            // --- 3D Viewport ---
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float vp_w = avail.x > 100 ? avail.x : 400;
            float vp_h = avail.y > 100 ? avail.y : 400;
            uint32_t w = (uint32_t)vp_w, h = (uint32_t)vp_h;

            // Create or resize renderer
            if (!state.renderer) {
                state.renderer = mujoco_renderer_create(w, h);
            } else {
                uint32_t cw, ch;
                mujoco_renderer_get_size(state.renderer, &cw, &ch);
                if (cw != w || ch != h)
                    mujoco_renderer_resize(state.renderer, w, h);
            }

            if (state.renderer) {
                mujoco_renderer_render(state.renderer, &mj,
                                       state.cam_lookat, state.cam_distance,
                                       state.cam_azimuth, state.cam_elevation,
                                       state.show_skin, state.show_site_markers,
                                       state.show_arena);
                ImTextureID tex = mujoco_renderer_get_texture(state.renderer);
                if (tex) {
                    ImVec2 cursor = ImGui::GetCursorScreenPos();
                    ImGui::Image(tex, ImVec2(vp_w, vp_h));

                    // Camera interaction on the image
                    if (ImGui::IsItemHovered()) {
                        ImGuiIO &io = ImGui::GetIO();

                        // Scroll to zoom (smooth logarithmic)
                        if (io.MouseWheel != 0.0f)
                            state.cam_distance *= powf(0.9f, io.MouseWheel);
                        state.cam_distance = std::max(0.02f, std::min(5.0f, state.cam_distance));

                        // Left-drag or middle-drag to orbit
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) ||
                            ImGui::IsMouseDragging(ImGuiMouseButton_Middle)) {
                            ImVec2 delta = io.MouseDelta;
                            state.cam_azimuth   -= delta.x * 0.3f;
                            state.cam_elevation += delta.y * 0.3f;
                            state.cam_elevation = std::max(-89.0f, std::min(89.0f, state.cam_elevation));
                        }

                        // Right-drag to pan (in camera-local right/up plane)
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                            ImVec2 delta = io.MouseDelta;
                            float scale = state.cam_distance * 0.001f;
                            float az = state.cam_azimuth * 3.14159f / 180.0f;
                            // Camera right vector projected to XY
                            float rx = cosf(az), ry = -sinf(az);
                            state.cam_lookat[0] += delta.x * scale * rx;
                            state.cam_lookat[1] += delta.x * scale * ry;
                            state.cam_lookat[2] += delta.y * scale;
                        }
                    }
                }
            }

            // --- Unload (deferred to next frame to avoid mid-render teardown) ---
            if (ImGui::Button("Unload Model")) {
                state.unload_requested = true;
            }
#endif // RED_HAS_MUJOCO
        },
        [&]() {
            // File dialog (runs every frame regardless of panel visibility)
            if (ImGuiFileDialog::Instance()->Display("ChooseMujocoModel")) {
                if (ImGuiFileDialog::Instance()->IsOk())
                    state.model_path = ImGuiFileDialog::Instance()->GetFilePathName();
                ImGuiFileDialog::Instance()->Close();
            }
        },
        ImVec2(600, 700));
}
