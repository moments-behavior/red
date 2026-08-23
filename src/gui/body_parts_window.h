#pragma once
// body_parts_window.h — "Body Parts" window: per-group checkboxes that drive
// the shared keypoint-column selection used by the Keypoints window and the
// copy / paste / delete operations.
//
// Each checkbox toggles a whole anatomical group (Head, WingL, T1L, ...) in or
// out of the selection. It reads as checked when ALL of the group's nodes are
// selected, so it stays in sync with manual Shift/Ctrl clicks in the Keypoints
// window and with Ctrl+A. Groups come from SkeletonContext::groups (the Fly50
// built-in preset, or a skeleton JSON's "groups" field).
#include "app_context.h"
#include "gui/keypoint_clipboard.h"
#include <algorithm>
#include <imgui.h>

inline void DrawBodyPartsWindow(AppContext &ctx) {
    auto &skeleton = ctx.skeleton;
    KeypointClipboard &kc = keypoint_clipboard();

    if (ImGui::Begin("Body Parts")) {
        if (!skeleton.has_skeleton || skeleton.num_nodes <= 0) {
            ImGui::TextDisabled("No skeleton loaded.");
        } else if (skeleton.groups.empty()) {
            ImGui::TextWrapped(
                "This skeleton defines no body-part groups. Add a \"groups\" "
                "field to its skeleton JSON to populate this window.");
        } else {
            kc.ensure_size(skeleton.num_nodes);

            ImGui::TextDisabled(
                "Toggle a group to add/remove its keypoints from the selection.");
            ImGui::Spacing();

            for (size_t gi = 0; gi < skeleton.groups.size(); ++gi) {
                const KeypointGroup &grp = skeleton.groups[gi];
                const int total = (int)grp.nodes.size();
                int sel_count = 0;
                for (int n : grp.nodes)
                    if (kc.is_selected(n)) ++sel_count;
                const bool all_sel = (total > 0 && sel_count == total);

                ImGui::PushID((int)gi);
                bool v = all_sel;
                if (ImGui::Checkbox(grp.name.c_str(), &v)) {
                    for (int n : grp.nodes)
                        if (n >= 0 && n < (int)kc.selected.size())
                            kc.selected[(size_t)n] = v ? 1 : 0;
                    if (v && !grp.nodes.empty())
                        kc.anchor = grp.nodes.back();
                }
                // Partial-selection indicator.
                if (sel_count > 0 && sel_count < total) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("(%d/%d)", sel_count, total);
                }
                ImGui::PopID();
            }

            ImGui::Separator();
            if (ImGui::SmallButton("All")) {
                std::fill(kc.selected.begin(), kc.selected.end(), (char)1);
                kc.anchor = 0;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("None"))
                kc.clear_selection();
            ImGui::SameLine();
            ImGui::TextDisabled("%d keypoint(s) selected", kc.count());

            ImGui::Spacing();
            ImGui::TextDisabled(
                "In the Keypoints window: Ctrl+C copy, Ctrl+V paste, Shift+Delete\n"
                "removes the set (or use the Labeling Tool's Copy Sel / Paste).");
        }
    }
    ImGui::End();
}
