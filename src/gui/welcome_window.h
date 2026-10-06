#pragma once
#include "imgui.h"
#include "app_context.h"
#include "IconsForkAwesome.h"
#include "video_files.h"
#include "gui/window_states.h"
#include "tailcycle_import.h"
#include "gui/tailcycle_open_window.h"
#include <ImGuiFileDialog.h>
#include <filesystem>

// Blender-style welcome/startup screen shown when no project is loaded.
inline void DrawWelcomeWindow(AppContext &ctx, WindowStates &win) {
    // Skip input on the first frame the welcome screen appears, to prevent
    // click-through from a closing dialog's button registering on a welcome
    // screen button that appears at the same position.
    static int last_drawn_frame = -2;
    int cur_frame = ImGui::GetFrameCount();
    bool just_appeared = (cur_frame - last_drawn_frame > 1);
    last_drawn_frame = cur_frame;
    // Center on viewport
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(520, 0));  // auto height

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoResize |
                             ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoDocking |
                             ImGuiWindowFlags_NoSavedSettings;

    if (!ImGui::Begin("##Welcome", nullptr, flags)) {
        ImGui::End();
        return;
    }

    // Title
    {
        const char *title = "RED";
        const char *subtitle = "Multi-Camera Keypoint Labeling Tool";
        float title_w = ImGui::CalcTextSize(title).x;
        float sub_w = ImGui::CalcTextSize(subtitle).x;
        float avail = ImGui::GetContentRegionAvail().x;

        ImGui::SetCursorPosX((avail - title_w) * 0.5f);
        ImGui::TextColored(ImVec4(0.4f, 0.7f, 1.0f, 1.0f), "%s", title);

        ImGui::SetCursorPosX((avail - sub_w) * 0.5f);
        ImGui::TextDisabled("%s", subtitle);
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
    }

    // Disable all buttons for one frame when the welcome screen first appears,
    // to prevent click-through from a closing dialog's Back button.
    ImGui::BeginDisabled(just_appeared);

    // Open something to look at
    {
        float btn_w = 150.0f;
        float avail = ImGui::GetContentRegionAvail().x;
        float spacing = 10.0f;
        float start_x = (avail - 2 * btn_w - spacing) * 0.5f;

        const ImVec2 row(2 * btn_w + spacing, 30);
        ImGui::SetCursorPosX(start_x);
        if (ImGui::Button("Open Videos", row)) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 0;
            cfg.path = media_browse_dir(ctx);
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseMedia", "Select Video(s)",
                video_ext_filter(), cfg);
        }
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Play videos without a project.");

        // A tailcycle session is not a red project -- it brings its own
        // cameras, calibration and skeleton -- so it sits up here with Open
        // Videos rather than under Annotate.
        if (TailcycleImport::available()) {
            ImGui::Spacing();
            // Line it up under the buttons above.
            ImGui::SetCursorPosX(start_x);
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.5f, 0.5f));
            if (ImGui::Button("Open tailcycle Dataset", row)) {
                run_or_confirm_unsaved(ctx, [&win]() {
                    tailcycle_open_browse(win.tailcycle_open);
                });
            }
            ImGui::PopStyleVar();
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Browse a tailcycle-dataset root and open one of its\n"
                    "sessions. Brings its own cameras and skeleton.");
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Annotate section
    ImGui::TextColored(ImVec4(0.8f, 0.6f, 0.6f, 1.0f), "Annotate");
    ImGui::Spacing();

    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0, 0.5f));
    if (ImGui::Button("Create Annotation Project", ImVec2(-1, 0))) {
        win.annotation.open();
    }
    if (ImGui::Button("Load Annotation Project", ImVec2(-1, 0))) {
        IGFD::FileDialogConfig cfg;
        cfg.countSelectionMax = 1;
        cfg.path = default_project_root(ctx.user_settings, ctx.default_dir);
        cfg.flags = ImGuiFileDialogFlags_Modal;
        ImGuiFileDialog::Instance()->OpenDialog(
            "ChooseProject", "Load Annotation Project",
            "Red Project{.redproj}", cfg);
    }
    ImGui::PopStyleVar();

    // Recent Projects section
    if (!ctx.user_settings.recent_projects.empty()) {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1.0f), "Recent Projects");
        ImGui::Spacing();

        // Each row is one button; while it is hovered an x shows at its
        // right end, and a click there takes the row off the list (the
        // project's files are untouched). Removed after the loop.
        int remove_row = -1;
        for (int ri = 0; ri < (int)ctx.user_settings.recent_projects.size(); ri++) {
            const auto &path = ctx.user_settings.recent_projects[ri];
            std::filesystem::path p(path);
            std::string display = p.parent_path().filename().string() + "/" + p.filename().string();
            std::error_code ec;
            const bool missing = !std::filesystem::exists(path, ec);
            if (missing) display = "[missing] " + display;

            ImGui::PushID(ri);  // unique ID per row
            ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0, 0.5f));
            if (missing)
                ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
            const bool clicked = ImGui::Button(display.c_str(), ImVec2(-1, 0));
            if (missing) ImGui::PopStyleColor();
            ImGui::PopStyleVar();

            const bool row_hovered = ImGui::IsItemHovered();
            const ImVec2 r0 = ImGui::GetItemRectMin(), r1 = ImGui::GetItemRectMax();
            const float h = r1.y - r0.y;
            const ImVec2 x0(r1.x - h, r0.y);   // the x's square, at the right end
            const bool on_x = row_hovered && ImGui::IsMouseHoveringRect(x0, r1);
            if (row_hovered) {
                ImDrawList *dl = ImGui::GetWindowDrawList();
                if (on_x)
                    dl->AddRectFilled(x0, r1, ImGui::GetColorU32(ImGuiCol_ButtonActive),
                                      ImGui::GetStyle().FrameRounding);
                const ImVec2 ts = ImGui::CalcTextSize(ICON_FK_TIMES);
                dl->AddText(ImVec2(x0.x + (h - ts.x) * 0.5f, x0.y + (h - ts.y) * 0.5f),
                            ImGui::GetColorU32(on_x ? ImGuiCol_Text : ImGuiCol_TextDisabled),
                            ICON_FK_TIMES);
                if (on_x)
                    ImGui::SetTooltip("Remove from this list (the project is not deleted)");
                else
                    ImGui::SetTooltip("%s", path.c_str());
            }
            if (clicked) {
                if (on_x)
                    remove_row = ri;
                else if (!missing)
                    // The path is already known, so load it directly rather
                    // than re-asking for it through a file dialog. The main
                    // loop picks this up and runs the same loader the dialog
                    // uses.
                    win.load_project_request = path;
            }
            ImGui::PopID();
        }
        if (remove_row >= 0) {
            auto &list = ctx.user_settings.recent_projects;
            list.erase(list.begin() + remove_row);
            save_user_settings(ctx.user_settings);
        }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Help
    float avail = ImGui::GetContentRegionAvail().x;
    float btn_w = 160.0f;
    ImGui::SetCursorPosX((avail - btn_w) * 0.5f);
    if (ImGui::Button("Help & Tutorials", ImVec2(btn_w, 0))) {
        win.show_help = true;
    }

    ImGui::EndDisabled(); // matches BeginDisabled(just_appeared)

    ImGui::Spacing();
    ImGui::End();
}
