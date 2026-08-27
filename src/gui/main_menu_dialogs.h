#pragma once
#include "app_context.h"
#include "gui/window_states.h"
#include <ImGuiFileDialog.h>
#include <filesystem>

// Load a .redproj from an explicit path. Shared by the Load Project dialog and
// the Welcome window's Recent Projects list, so both take exactly the same code
// path -- a recent-project click should not have to round-trip through a file
// dialog just to reach the tested loader.
inline void load_project_from_path(
    AppContext &ctx, WindowStates &win, const std::filesystem::path &cfg_path,
    std::function<void()> print_metadata_fn,
    std::function<void(const std::string &)> print_summary_fn,
    std::function<void()> nuke_inference_fn = nullptr) {
    auto &pm = ctx.pm;

    // Legacy calibration projects share the .redproj extension; detect them so
    // the failure is explicit rather than a confusing parse error from the
    // annotation loader.
    if (cfg_path.extension() == ".redproj") {
        try {
            std::ifstream probe(cfg_path, std::ios::binary);
            if (probe) {
                nlohmann::json j;
                probe >> j;
                std::string type = j.value("type", std::string{});
                if (type == "calibration" || type == "laser_calibration") {
                    ctx.popups.pushError(
                        "Calibration projects are no longer supported.\n"
                        "Open an annotation project instead.");
                    return;
                }
            }
        } catch (...) {}
    }

    ProjectManager loaded;
    std::string err;
    if (!load_project_manager_json(&loaded, cfg_path, &err)) {
        ctx.popups.pushError(err);
        return;
    }
    close_project(ctx);
    win.reset();
    if (nuke_inference_fn) nuke_inference_fn();
    pm = loaded;
    if (setup_project(pm, ctx.skeleton, ctx.skeleton_map, &err))
        on_project_loaded(ctx, print_metadata_fn, print_summary_fn);
    else
        ctx.popups.pushError(err);
}

// Handle all main-menu-originated file dialogs. Called once per frame.
inline void HandleMainMenuDialogs(
    AppContext &ctx,
    WindowStates &win,
    const std::string &media_root_dir,
    std::function<void()> print_metadata_fn,
    std::function<void(const std::string &)> print_summary_fn,
    std::function<void()> nuke_inference_fn = nullptr) {
    auto &annot_state = win.annotation;
    auto &pm = ctx.pm;
    auto &user_settings = ctx.user_settings;
    auto &popups = ctx.popups;
    auto &skeleton = ctx.skeleton;
    auto &skeleton_map = ctx.skeleton_map;

    // ChooseProjectDir (Create Project form)
    if (ImGuiFileDialog::Instance()->Display("ChooseProjectDir", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            pm.project_root_path =
                ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseCalibration (Create Project form)
    if (ImGuiFileDialog::Instance()->Display("ChooseCalibration", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            pm.calibration_folder =
                ImGuiFileDialog::Instance()->GetCurrentPath();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseDefaultProjectRoot (Settings menu)
    if (ImGuiFileDialog::Instance()->Display(
            "ChooseDefaultProjectRoot", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string chosen =
                ImGuiFileDialog::Instance()->GetCurrentPath();
            user_settings.default_project_root_path = chosen;
            pm.project_root_path = chosen;
            save_user_settings(user_settings);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseDefaultMediaRoot (Settings menu)
    if (ImGuiFileDialog::Instance()->Display(
            "ChooseDefaultMediaRoot", ImGuiWindowFlags_NoCollapse,
            ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string chosen =
                ImGuiFileDialog::Instance()->GetCurrentPath();
            user_settings.default_media_root_path = chosen;
            pm.media_folder = chosen;
            annot_state.video_folder = chosen;
            save_user_settings(user_settings);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseMedia (Open Video)
    if (ImGuiFileDialog::Instance()->Display("ChooseMedia", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            auto selected_files =
                ImGuiFileDialog::Instance()->GetSelection();
            pm.media_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
            pm.project_name =
                dir_difference(pm.media_folder, media_root_dir);
            load_videos(selected_files, ctx.ps, pm, ctx.window_was_decoding,
                        ctx.demuxers, ctx.dc_context, ctx.scene,
                        ctx.label_buffer_size, ctx.decoder_threads,
                        ctx.is_view_focused);
            if (print_metadata_fn) print_metadata_fn();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseImages (Open Images)
    if (ImGuiFileDialog::Instance()->Display("ChooseImages", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            auto selected_files =
                ImGuiFileDialog::Instance()->GetSelection();
            pm.media_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
            pm.project_name =
                dir_difference(pm.media_folder, media_root_dir);
            load_images(selected_files, ctx.ps, pm, ctx.imgs_names, ctx.scene,
                        ctx.dc_context, ctx.label_buffer_size,
                        ctx.decoder_threads, ctx.is_view_focused,
                        ctx.window_was_decoding);
            ctx.input_is_imgs = true;
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseProject (Load Project)
    if (ImGuiFileDialog::Instance()->Display("ChooseProject", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            const auto sel = ImGuiFileDialog::Instance()->GetSelection();
            std::filesystem::path cfg_path;
            if (!sel.empty()) {
                cfg_path = std::filesystem::path(sel.begin()->second);
            } else {
                std::string full =
                    ImGuiFileDialog::Instance()->GetFilePathName(
                        IGFD_ResultMode_KeepInputFile);
                if (!full.empty())
                    cfg_path = std::filesystem::path(full);
                else
                    cfg_path = std::filesystem::path(
                        ImGuiFileDialog::Instance()->GetCurrentPath());
            }

            load_project_from_path(ctx, win, cfg_path, print_metadata_fn,
                                   print_summary_fn, nuke_inference_fn);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseSkeleton (Create Project form)
    if (ImGuiFileDialog::Instance()->Display("ChooseSkeleton", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            pm.skeleton_file =
                ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ChooseSkeletonSwitch (Switch Skeleton window) — writes into the
    // window's staged field, not pm, since the change isn't applied until
    // the user clicks "Apply" there.
    if (ImGuiFileDialog::Instance()->Display("ChooseSkeletonSwitch", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            win.switch_skeleton.skeleton_file =
                ImGuiFileDialog::Instance()->GetFilePathName();
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // LoadAnnotProject (Annotate > Load)
    if (ImGuiFileDialog::Instance()->Display("LoadAnnotProject", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            const auto sel = ImGuiFileDialog::Instance()->GetSelection();
            std::filesystem::path cfg_path;
            if (!sel.empty()) {
                cfg_path = std::filesystem::path(sel.begin()->second);
            } else {
                std::string full =
                    ImGuiFileDialog::Instance()->GetFilePathName(
                        IGFD_ResultMode_KeepInputFile);
                if (!full.empty())
                    cfg_path = std::filesystem::path(full);
                else
                    cfg_path = std::filesystem::path(
                        ImGuiFileDialog::Instance()->GetCurrentPath());
            }
            ProjectManager loaded;
            std::string err;
            if (!load_project_manager_json(&loaded, cfg_path, &err)) {
                popups.pushError(err);
            } else {
                close_project(ctx);
                win.reset();
                if (nuke_inference_fn) nuke_inference_fn();
                pm = loaded;
                if (setup_project(pm, skeleton, skeleton_map, &err)) {
                    on_project_loaded(ctx, print_metadata_fn, print_summary_fn);
                } else
                    popups.pushError(err);
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}
