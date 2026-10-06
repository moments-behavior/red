#pragma once
#include "decode_backend.h"
#include "app_context.h"
#include "gui/tailcycle_open_window.h"
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

    // Opening another project replaces an Untitled one: ask about its
    // unsaved labels first. The prompt calls back here once settled.
    if (untitled_unsaved(ctx)) {
        run_or_confirm_unsaved(ctx, [&ctx, &win, cfg_path, print_metadata_fn,
                                     print_summary_fn, nuke_inference_fn]() {
            load_project_from_path(ctx, win, cfg_path, print_metadata_fn,
                                   print_summary_fn, nuke_inference_fn);
        });
        return;
    }

    // A tailcycle session is a directory holding a session.toml, not a
    // .redproj. Recents carry both, so route by what is actually there rather
    // than asking the caller to know which kind of path it has.
    if (std::filesystem::is_directory(cfg_path)) {
        if (std::filesystem::exists(cfg_path / "session.toml")) {
            std::string status;
            if (!tailcycle_open_session(ctx, cfg_path.string(), std::string(), &status))
                ctx.popups.pushError(status);
            else
                ctx.toasts.pushSuccess(status);
            return;
        }
        // A dataset root: show the browser pointed at it rather than guessing
        // which session was meant. The panel rescans when its root changes.
        std::vector<TailcycleImport::SessionInfo> probe;
        std::string err;
        if (TailcycleImport::scan_dataset(cfg_path.string(), &probe, &err)) {
            // Setting the root is the whole action: tailcycle_pump sees it
            // change, scans, and opens the first session.
            win.tailcycle_open.root = cfg_path.string();
            return;
        }
    }

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
    // Read before win.reset(), which clears it.
    const bool save_labels = !win.load_project_discard_labels;
    win.load_project_discard_labels = false;
    close_project(ctx, save_labels);
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
            annot_state.media_folder = chosen;
            save_user_settings(user_settings);
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // --- Untitled projects -------------------------------------------------
    // Something would replace an Untitled project whose labels are only in
    // memory (run_or_confirm_unsaved): ask what to do with them.
    if (ctx.unsaved_prompt) {
        ImGui::OpenPopup("Unsaved Project##unsaved");
        ctx.unsaved_prompt = false;
    }
    if (ImGui::BeginPopupModal("Unsaved Project##unsaved", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("This Untitled project has labels that have "
                               "not been saved.");
        ImGui::Spacing();
        if (ImGui::Button("Save...")) {
            ctx.after_save_action = std::move(ctx.unsaved_action);
            ctx.unsaved_action = nullptr;
            ctx.save_project_prompt = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Don't Save")) {
            auto action = std::move(ctx.unsaved_action);
            ctx.unsaved_action = nullptr;
            ctx.annotations.clear();   // dropped: the action now goes through
            ImGui::CloseCurrentPopup();
            if (action) action();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
            ctx.unsaved_action = nullptr;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    // Save Project: one file browser -- go to where the project should live,
    // keep or change the folder name. Makes <there>/<name>/ holding
    // <name>.redproj and labeled_data/.
    auto open_save_dialog = [&](const std::string &dir, const std::string &name) {
        IGFD::FileDialogConfig cfg;
        cfg.path = dir;
        cfg.fileName = name;
        cfg.flags = ImGuiFileDialogFlags_Modal | ImGuiFileDialogFlags_HideColumnType;
        // No filter (""): no type list beside the name. Not nullptr, which
        // would make it a folder picker that names the clicked folder.
        ImGuiFileDialog::Instance()->OpenDialog(
            "SaveUntitledProject", "Save Project", "", cfg);
    };
    if (ctx.save_project_prompt) {
        ctx.save_project_prompt = false;
        open_save_dialog(
            default_project_root(ctx.user_settings, ctx.default_dir),
            pm.media_folder.empty()
                ? std::string("Untitled")
                : std::filesystem::path(pm.media_folder).filename().string());
    }
    igfd_file_name_label() = "Folder Name:";
    const bool save_done = ImGuiFileDialog::Instance()->Display(
        "SaveUntitledProject", ImGuiWindowFlags_NoCollapse, ImVec2(680, 440));
    igfd_file_name_label() = "File Name:";
    if (save_done) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            const std::string dir = ImGuiFileDialog::Instance()->GetCurrentPath();
            const std::string name = ImGuiFileDialog::Instance()->GetCurrentFileName(
                IGFD_ResultMode_KeepInputFile);
            ImGuiFileDialog::Instance()->Close();
            std::string err;
            if (save_untitled_project(ctx, name, dir, &err)) {
                ctx.toasts.pushSuccess("Saved project " + pm.project_name);
                auto after = std::move(ctx.after_save_action);
                ctx.after_save_action = nullptr;
                if (after) after();
            } else {
                // Say why and ask again, where the user left off.
                ctx.toasts.pushError(err);
                open_save_dialog(dir, name);
            }
        } else {
            ImGuiFileDialog::Instance()->Close();
            ctx.after_save_action = nullptr;   // whatever waited on it is off
        }
    }

    // Help > About Red
    if (win.show_about) {
        ImGui::OpenPopup("About Red");
        win.show_about = false;
        win.about_open = true;
    }
    ImGui::SetNextWindowSize(ImVec2(440, 0), ImGuiCond_Appearing);
    // Passing &about_open gives the title bar a close x.
    if (ImGui::BeginPopupModal("About Red", &win.about_open,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
#if defined(__APPLE__)
        const char *platform = "macOS, Apple Silicon";
#elif defined(_WIN32)
        const char *platform = "Windows, x64";
#else
        const char *platform = "Linux, x64";
#endif
        ImGui::TextUnformatted("Red -- multi-camera video labeling");
        ImGui::Separator();
        ImGui::Text("Version   %s", RED_VERSION);
        ImGui::Text("Built     %s %s", __DATE__, __TIME__);
        ImGui::Text("Platform  %s", platform);
        ImGui::Text("Decoding  %s", red::decode_backend_name());
        ImGui::TextDisabled("          %s", red::decode_backend_reason());
        ImGui::Separator();
        ImGui::TextLinkOpenURL("github.com/moments-behavior/red",
                               "https://github.com/moments-behavior/red");
        ImGui::TextDisabled("Quote the version above when reporting an issue.");
        ImGui::Spacing();
        if (ImGui::Button("Copy version info")) {
            std::string info = std::string("Red ") + RED_VERSION + " (" +
                               platform + ", built " + __DATE__ + ", " +
                               red::decode_backend_name() + " decoding)";
            ImGui::SetClipboardText(info.c_str());
        }
        if (ImGui::IsKeyPressed(ImGuiKey_Escape))
            ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }

    // Tools > Camera Timestamps: pick a folder, confirm, then reopen the
    // project with it. The decoders take their camera timings when the videos
    // load, so the folder only applies on a reload -- and reopening goes
    // through close_project(), which saves the labels first.
    if (ImGuiFileDialog::Instance()->Display("ChooseProjectTimestamps",
                                             ImGuiWindowFlags_NoCollapse,
                                             ImVec2(680, 440))) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            win.timestamps_pending = ImGuiFileDialog::Instance()->GetCurrentPath();
            win.timestamps_confirm = true;
        }
        ImGuiFileDialog::Instance()->Close();
    }
    if (win.timestamps_confirm) {
        ImGui::OpenPopup("Camera Timestamps##confirm");
        win.timestamps_confirm = false;
    }
    if (ImGui::BeginPopupModal("Camera Timestamps##confirm", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::Text("Use the camera timestamps in:");
        ImGui::TextDisabled("%s", win.timestamps_pending.c_str());
        ImGui::Spacing();
        ImGui::TextUnformatted("The project is reopened to apply it.");
        ImGui::TextDisabled("Saving writes a new labeled_data folder, as "
                            "%s+S does; without saving, edits since the "
                            "last save are lost.", RED_MOD_KEY);
        ImGui::Spacing();
        // Both buttons record the folder in the .redproj and have the main
        // loop reopen the project; they differ only in saving labels first.
        auto apply = [&](bool save_labels) {
            pm.timestamps_folder = win.timestamps_pending;
            const std::string redproj =
                pm.project_path + "/" + pm.project_name + ".redproj";
            std::string save_err;
            if (save_project_manager_json(pm, redproj, &save_err)) {
                win.load_project_request = redproj;  // main loop reopens it
                win.load_project_discard_labels = !save_labels;
            } else {
                ctx.popups.pushError("Could not save the project: " + save_err);
            }
            win.timestamps_pending.clear();
            ImGui::CloseCurrentPopup();
        };
        if (ImGui::Button("Save labels and reload")) apply(true);
        ImGui::SameLine();
        if (ImGui::Button("Reload without saving")) apply(false);
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            win.timestamps_pending.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
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
                        ctx.is_view_focused,
                        ctx.user_settings.default_realtime_playback);
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
                        ctx.window_was_decoding, ImageLayout::Flat, 0.0f,
                        ctx.user_settings.default_realtime_playback);
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
            remember_skeleton_dir(ctx, win.switch_skeleton.skeleton_file);
        }
        ImGuiFileDialog::Instance()->Close();
    }

}
