#pragma once
// switch_skeleton_window.h — File -> Switch Skeleton. Lets the user change an
// already-open project's skeleton (built-in preset or custom skeleton.json),
// but ONLY while no frame has any manually-provided label. Manual labels are
// indexed by node position, so re-indexing under a different skeleton would
// silently corrupt them; anything else (predicted/imported/promoted-only
// data, an empty project) is safe to discard and rebuild.

#include "app_context.h"
#include "annotation.h"
#include "annotation_csv.h"
#include "project.h"
#include "gui/panel.h"

#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>

#include <string>
#include <vector>

struct SwitchSkeletonState {
    bool show = false;
    bool initialized = false;   // staged fields seeded from the live project yet?

    // Staged selection (only applied to ProjectManager on "Apply").
    bool  load_from_json = false;
    std::string skeleton_file;
    std::string skeleton_name;
    int   preset_idx = 0;

    std::string status;         // result/error message shown after Apply
    int skeleton_wait = -1;     // waiting on the Skeleton Creator (see "New...")
};

// Applies a staged skeleton selection to an already-open project: reloads
// the SkeletonContext, clears everything indexed by the old node layout
// (annotations), and
// persists the change to the .redproj immediately. Caller must have already
// verified project_has_any_manual_labels(ctx.annotations) == false.
inline bool switch_project_skeleton(AppContext &ctx,
                                    bool new_load_from_json,
                                    const std::string &new_skeleton_file,
                                    const std::string &new_skeleton_name,
                                    std::string *err) {
    auto &pm = ctx.pm;

    // Auto-save whatever's currently loaded under the OLD skeleton before
    // wiping it — cheap insurance, mirrors close_project()'s step 1, even
    // though the caller has already verified nothing manual would be lost.
    if (!pm.keypoints_root_folder.empty() && !ctx.annotations.empty()) {
        std::string save_err;
        AnnotationCSV::save_all(pm.keypoints_root_folder, ctx.skeleton.name,
                                ctx.annotations, ctx.scene->num_cams,
                                ctx.skeleton.num_nodes, pm.camera_names, &save_err,
                                &pm.annotation_config.label_info);
    }

    pm.load_skeleton_from_json = new_load_from_json;
    pm.skeleton_file = new_skeleton_file;
    pm.skeleton_name = new_skeleton_name;
    if (!reload_skeleton(pm, ctx.skeleton, ctx.skeleton_map, err))
        return false;

    // Clear everything indexed by the old skeleton's node layout.
    ctx.annotations.clear();

    // Persist immediately — no explicit "Save Project" step, matching
    // transport_bar.h's sync_fix_enabled toggle.
    if (!pm.project_path.empty() && !pm.project_name.empty()) {
        std::string redproj = pm.project_path + "/" + pm.project_name + ".redproj";
        save_project_manager_json(pm, redproj);
    }
    return true;
}

inline void DrawSwitchSkeletonWindow(SwitchSkeletonState &st, AppContext &ctx) {
    auto &pm = ctx.pm;
    auto &skeleton = ctx.skeleton;
    auto &skeleton_map = ctx.skeleton_map;

    DrawPanel("Switch Skeleton", st.show, [&]() {
        if (pm.project_path.empty()) {
            ImGui::TextDisabled(pm.untitled
                                    ? "Save the project first (" RED_MOD_KEY "+S)."
                                    : "Open or create a project first.");
            st.initialized = false;
            return;
        }

        // Seed staged fields from the live project config the first time the
        // panel opens (and again after a project switch), so edits start
        // from the current setting rather than stale leftovers.
        if (!st.initialized) {
            st.load_from_json = pm.load_skeleton_from_json;
            st.skeleton_file = pm.skeleton_file;
            st.skeleton_name = pm.skeleton_name;
            st.status.clear();
            st.initialized = true;
        }

        bool locked = project_has_any_manual_labels(ctx.annotations);
        if (locked) {
            ImGui::TextColored(ImVec4(1, 0.4f, 0.4f, 1),
                "This project has manually-labeled (or in-progress) frames.");
            ImGui::TextWrapped(
                "Switching skeletons re-indexes every keypoint by node "
                "position, which would silently corrupt existing manual "
                "labels. Finish/export what you need, clear the labels, "
                "then try again.");
            return;
        }

        ImGui::TextWrapped(
            "No manually-labeled frames yet, so it's safe to change the "
            "skeleton. Applying will clear the current annotations, close "
            "any open prediction store, and reload the skeleton.");
        ImGui::Separator();
        ImGui::Text("Current: %s (%d nodes)", skeleton.name.c_str(), skeleton.num_nodes);

        std::vector<const char *> labels_s;
        labels_s.reserve(skeleton_map.size());
        for (auto &kv : skeleton_map) labels_s.push_back(kv.first.c_str());
        if (st.preset_idx >= (int)labels_s.size()) st.preset_idx = 0;
        if (!st.load_from_json) {
            for (int i = 0; i < (int)labels_s.size(); ++i)
                if (st.skeleton_name == labels_s[i]) { st.preset_idx = i; break; }
        }

        // A skeleton saved in the creator after "New..." becomes the choice.
        if (std::string saved = skeleton_saved_since(ctx, st.skeleton_wait);
            !saved.empty()) {
            st.load_from_json = true;
            st.skeleton_file = saved;
            st.skeleton_wait = -1;
        }

        // One row, as in Create Annotation Project: the preset list (or the
        // path and Browse), New..., then File / Preset.
        const ImGuiStyle &style = ImGui::GetStyle();
        const float gap = style.ItemInnerSpacing.x;
        const float new_w = ImGui::CalcTextSize("New...").x + style.FramePadding.x * 2;
        const float mode_w = 90.0f;
        const float avail = ImGui::GetContentRegionAvail().x;
        if (st.load_from_json) {
            const float browse_w = ImGui::CalcTextSize("Browse").x + style.FramePadding.x * 2;
            ImGui::SetNextItemWidth(
                ImMax(50.0f, avail - browse_w - new_w - mode_w - 3 * gap));
            ImGui::InputText("##switch_skel_path", &st.skeleton_file);
            ImGui::SameLine(0.0f, gap);
            if (ImGui::Button("Browse##browse_skel_switch")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = ctx.skeleton_dir;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChooseSkeletonSwitch", "Choose Skeleton", ".json", cfg);
            }
        } else {
            ImGui::BeginDisabled(labels_s.empty());
            ImGui::SetNextItemWidth(ImMax(50.0f, avail - new_w - mode_w - 2 * gap));
            if (ImGui::Combo("##switch_skel_preset", &st.preset_idx,
                             labels_s.data(), (int)labels_s.size())) {
                st.skeleton_name = labels_s[st.preset_idx];
            }
            ImGui::EndDisabled();
            if (!labels_s.empty() && st.skeleton_name.empty())
                st.skeleton_name = labels_s[st.preset_idx];
        }
        // Always there: draw a new skeleton. It is a file, so this goes to
        // File, where the saved skeleton is filled in.
        ImGui::SameLine(0.0f, gap);
        if (ImGui::Button("New...##switch_skel_new")) {
            st.skeleton_wait = open_skeleton_creator_for(ctx);
            st.load_from_json = true;
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort))
            ImGui::SetTooltip("Draw a new skeleton in the Skeleton Creator; "
                              "saving it there picks it here.");
        ImGui::SameLine(0.0f, gap);
        int mode = st.load_from_json ? 0 : 1;
        ImGui::SetNextItemWidth(mode_w);
        if (ImGui::Combo("##switch_skel_mode", &mode, "File\0Preset\0"))
            st.load_from_json = (mode == 0);

        ImGui::Separator();
        bool unchanged = st.load_from_json == pm.load_skeleton_from_json &&
                         st.skeleton_file == pm.skeleton_file &&
                         st.skeleton_name == pm.skeleton_name;
        bool incomplete = (st.load_from_json && st.skeleton_file.empty()) ||
                          (!st.load_from_json && st.skeleton_name.empty());
        ImGui::BeginDisabled(unchanged || incomplete);
        if (ImGui::Button("Apply")) {
            std::string err;
            if (switch_project_skeleton(ctx, st.load_from_json,
                                        st.skeleton_file, st.skeleton_name, &err)) {
                st.status = "Switched to '" + ctx.skeleton.name + "' (" +
                           std::to_string(ctx.skeleton.num_nodes) + " nodes).";
            } else {
                st.status = "Failed: " + err;
            }
        }
        ImGui::EndDisabled();

        if (!st.status.empty()) {
            bool is_error = st.status.rfind("Failed", 0) == 0;
            ImGui::TextColored(is_error ? ImVec4(1, 0.4f, 0.4f, 1) : ImVec4(0.5f, 1, 0.5f, 1),
                               "%s", st.status.c_str());
        }
    });
}
