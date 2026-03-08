#pragma once
#include "app_context.h"
#include "gui/calibration_tool_window.h"
#include "gui/annotation_dialog.h"
#include <ImGuiFileDialog.h>
#include <filesystem>

// Handle all Navigator-originated file dialogs. Called once per frame.
// Extra state refs that aren't in AppContext are passed explicitly.
inline void HandleNavigatorDialogs(
    AppContext &ctx,
    CalibrationToolState &calib_state,
    AnnotationDialogState &annot_state,
    const std::string &media_root_dir,
    std::function<void()> print_metadata_fn,
    std::function<void(const std::string &)> print_summary_fn) {
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
            calib_state.project.project_root_path = chosen;
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
            std::string old_media_root =
                user_settings.default_media_root_path;
            user_settings.default_media_root_path = chosen;
            pm.media_folder = chosen;
            annot_state.video_folder = chosen;
            if (calib_state.project.config_file.empty() ||
                calib_state.project.config_file == old_media_root)
                calib_state.project.config_file = chosen;
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
            ProjectManager loaded;
            std::string err;
            if (!load_project_manager_json(&loaded, cfg_path, &err)) {
                popups.pushError(err);
            } else {
                pm = loaded;
                if (setup_project(pm, skeleton, skeleton_map, &err))
                    on_project_loaded(ctx, print_metadata_fn, print_summary_fn);
                else
                    popups.pushError(err);
            }
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
                pm = loaded;
                if (setup_project(pm, skeleton, skeleton_map, &err))
                    on_project_loaded(ctx, print_metadata_fn, print_summary_fn);
                else
                    popups.pushError(err);
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }
}
