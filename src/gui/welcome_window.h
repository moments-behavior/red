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
#include <vector>

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

    const float avail_w = ImGui::GetContentRegionAvail().x;
    auto centered_text = [&](const char *text, bool disabled, ImVec4 col) {
        ImGui::SetCursorPosX((avail_w - ImGui::CalcTextSize(text).x) * 0.5f +
                             ImGui::GetStyle().WindowPadding.x);
        if (disabled) ImGui::TextDisabled("%s", text);
        else ImGui::TextColored(col, "%s", text);
    };

    // Title, what it is, and the version (what an issue report needs).
    centered_text("Red", false, ImVec4(0.4f, 0.7f, 1.0f, 1.0f));
    centered_text("Multi-Camera Keypoint Labeling Tool", true, ImVec4());
    centered_text(RED_VERSION, true, ImVec4());
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    // Disable all buttons for one frame when the welcome screen first appears,
    // to prevent click-through from a closing dialog's Back button.
    ImGui::BeginDisabled(just_appeared);

    // Projects first: making one, or opening one. Two equal buttons.
    {
        const float gap = ImGui::GetStyle().ItemSpacing.x;
        const ImVec2 sz((avail_w - gap) * 0.5f, 32.0f);
        if (ImGui::Button("New Project", sz))
            win.annotation.open();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Pick the videos or images, the cameras and a "
                              "skeleton, and start labeling.");
        ImGui::SameLine(0, gap);
        if (ImGui::Button("Open Project", sz)) {
            IGFD::FileDialogConfig cfg;
            cfg.countSelectionMax = 1;
            cfg.path = default_project_root(ctx.user_settings, ctx.default_dir);
            cfg.flags = ImGuiFileDialogFlags_Modal;
            ImGuiFileDialog::Instance()->OpenDialog(
                "ChooseProject", "Open Project",
                "Red Project{.redproj}", cfg);
        }
    }

    // Recent Projects section
    if (!ctx.user_settings.recent_projects.empty()) {
        ImGui::Spacing();
        ImGui::Spacing();
        ImGui::TextDisabled("Recent Projects");
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

    // Secondary, as quiet links on one line: just looking at videos, a
    // tailcycle session (it brings its own cameras and skeleton), and help.
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    {
        std::vector<const char *> links = {"Open Videos"};
        if (TailcycleImport::available()) links.push_back("Open tailcycle Dataset");
        links.push_back("Help");
        const char *sep = "  \xC2\xB7  ";   // a middle dot between them
        float w = 0;
        for (size_t i = 0; i < links.size(); ++i)
            w += ImGui::CalcTextSize(links[i]).x + (i ? ImGui::CalcTextSize(sep).x : 0);
        ImGui::SetCursorPosX((avail_w - w) * 0.5f + ImGui::GetStyle().WindowPadding.x);
        for (size_t i = 0; i < links.size(); ++i) {
            if (i) {
                ImGui::SameLine(0, 0);
                ImGui::TextDisabled("%s", sep);
                ImGui::SameLine(0, 0);
            }
            const std::string what = links[i];
            if (ImGui::TextLink(links[i])) {
                if (what == "Open Videos") {
                    IGFD::FileDialogConfig cfg;
                    cfg.countSelectionMax = 0;
                    cfg.path = media_browse_dir(ctx);
                    cfg.flags = ImGuiFileDialogFlags_Modal;
                    ImGuiFileDialog::Instance()->OpenDialog(
                        "ChooseMedia", "Select Video(s)", video_ext_filter(), cfg);
                } else if (what == "Help") {
                    win.show_help = true;
                } else {
                    run_or_confirm_unsaved(ctx, [&win]() {
                        tailcycle_open_browse(win.tailcycle_open);
                    });
                }
            }
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort)) {
                if (what == "Open Videos")
                    ImGui::SetTooltip("Play videos without a project.");
                else if (what == "Help")
                    ImGui::SetTooltip("Red Help: workflows, tools and shortcuts.");
                else
                    ImGui::SetTooltip("Browse a tailcycle-dataset root and open one "
                                      "of its\nsessions. Brings its own cameras and "
                                      "skeleton.");
            }
        }
    }

    ImGui::EndDisabled(); // matches BeginDisabled(just_appeared)

    ImGui::Spacing();
    ImGui::End();
}
