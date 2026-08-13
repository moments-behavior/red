#pragma once
#include "app_context.h"
#include "keypoint_colors.h"
#include <imgui.h>

inline void DrawKeypointsWindow(AppContext &ctx) {
    int current_frame_num = ctx.current_frame_num;
    auto &pm = ctx.pm;
    auto *scene = ctx.scene;
    auto &skeleton = ctx.skeleton;
    auto &annotations = ctx.annotations;
    auto &is_view_focused = ctx.is_view_focused;
    const ImVec4 active_kp_color = active_keypoint_color(ctx.user_settings);

    if (ImGui::Begin("Keypoints")) {

        bool keypoints_find =
            annotations.find(current_frame_num) != annotations.end();

        if (skeleton.num_nodes > 0 && skeleton.has_skeleton) {
            const int rows_count = scene->num_cams;
            const int columns_count = skeleton.num_nodes + 1;

            static ImGuiTableFlags table_flags =
                ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                ImGuiTableFlags_SizingFixedFit |
                ImGuiTableFlags_BordersOuter |
                ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_Hideable |
                ImGuiTableFlags_Resizable |
                ImGuiTableFlags_HighlightHoveredColumn;

            float table_height = ImGui::GetContentRegionAvail().y;
            ImVec2 table_size(0.0f, table_height);

            // Top of the table on screen = top of the angled-header band; used
            // below to detect clicks on the (header-row-less) angled headers.
            const float band_top = ImGui::GetCursorScreenPos().y;

            if (ImGui::BeginTable("table_angled_headers", columns_count,
                                  table_flags, table_size)) {
                ImGui::TableSetupColumn(
                    "Name", ImGuiTableColumnFlags_NoHide |
                                ImGuiTableColumnFlags_NoReorder);

                for (int column = 1; column < columns_count && (column - 1) < (int)skeleton.node_names.size(); column++) {
                    ImGui::TableSetupColumn(
                        skeleton.node_names[column - 1].c_str(),
                        ImGuiTableColumnFlags_AngledHeader |
                            ImGuiTableColumnFlags_WidthFixed);
                }

                // Freeze the single angled-header row. We deliberately do NOT
                // call TableHeadersRow(): its horizontal cells duplicated the
                // angled labels. Clicking a keypoint's angled header is handled
                // after the body rows, gated to the header Y-band.
                ImGui::TableSetupScrollFreeze(1, 1);
                ImGui::TableAngledHeadersRow();

                // Lower edge of the angled-header band (top of the first body
                // row), captured while rendering the first row below.
                float first_body_top = -1.0f;

                // Find focused row
                int focused_row = -1;
                for (int row = 0; row < rows_count; row++) {
                    if (row < (int)is_view_focused.size() &&
                        is_view_focused[row]) {
                        focused_row = row;
                        break;
                    }
                }

                auto render_row = [&](int row) {
                    ImGui::PushID(row);
                    ImGui::TableNextRow();

                    if (row < (int)is_view_focused.size() &&
                        is_view_focused[row] && keypoints_find) {
                        ImU32 row_bg_color = ImGui::GetColorU32(
                            ImVec4(0.7f, 0.3f, 0.3f, 0.65f));
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                               row_bg_color);
                    }

                    ImGui::TableSetColumnIndex(0);
                    if (first_body_top < 0.0f)
                        first_body_top = ImGui::GetCursorScreenPos().y;
                    ImGui::AlignTextToFramePadding();
                    ImGui::Text("%s", row < (int)pm.camera_names.size()
                        ? pm.camera_names[row].c_str() : "?");

                    for (int column = 1; column < columns_count; column++) {
                        if (ImGui::TableSetColumnIndex(column)) {
                            if (keypoints_find) {
                                auto &fa = annotations.at(current_frame_num);
                                const int node = column - 1;
                                const bool is_active =
                                    row < (int)fa.cameras.size() &&
                                    fa.cameras[row].active_id == (u32)node;
                                ImVec4 node_color = ImVec4(0, 0, 0, 0);

                                if (is_active) {
                                    node_color = active_kp_color; // user-selected
                                } else if (row < (int)fa.cameras.size() &&
                                           node < (int)fa.cameras[row].keypoints.size() &&
                                           fa.cameras[row].keypoints[node].labeled) {
                                    node_color =
                                        skeleton.node_colors[node];
                                    node_color.w = 0.9f;
                                }

                                const bool triangulated =
                                    node < (int)fa.kp3d.size() &&
                                    fa.kp3d[node].triangulated;

                                // The whole cell is a click target: clicking
                                // sets this keypoint active for this camera view.
                                ImGui::PushID(column);
                                float cell_w = ImGui::GetContentRegionAvail().x;
                                if (cell_w < 1.0f)
                                    cell_w = ImGui::GetFrameHeight();
                                ImVec2 p0 = ImGui::GetCursorScreenPos();
                                if (ImGui::InvisibleButton(
                                        "##kpcell",
                                        ImVec2(cell_w, ImGui::GetFrameHeight()))) {
                                    if (row < (int)fa.cameras.size())
                                        fa.cameras[row].active_id = (u32)node;
                                }
                                if (ImGui::IsItemHovered() &&
                                    node < (int)skeleton.node_names.size() &&
                                    row < (int)pm.camera_names.size()) {
                                    ImGui::SetTooltip(
                                        "Set active: %s / %s",
                                        pm.camera_names[row].c_str(),
                                        skeleton.node_names[node].c_str());
                                }
                                // Triangulated marker, drawn over the cell.
                                if (triangulated)
                                    ImGui::GetWindowDrawList()->AddText(
                                        ImVec2(p0.x + 2.0f, p0.y),
                                        IM_COL32(255, 255, 255, 255), "T");
                                ImGui::PopID();

                                ImU32 cell_bg_color =
                                    ImGui::GetColorU32(node_color);
                                ImGui::TableSetBgColor(
                                    ImGuiTableBgTarget_CellBg,
                                    cell_bg_color);
                            }
                        }
                    }

                    ImGui::PopID();
                };

                // Render focused row first
                if (focused_row != -1) {
                    render_row(focused_row);
                }

                // Render remaining rows
                for (int row = 0; row < rows_count; row++) {
                    if (row == focused_row)
                        continue;
                    render_row(row);
                }

                // Angled-header interaction: the header band is the strip
                // between band_top and the first body row. TableGetHoveredColumn
                // gives the column under the cursor (accounting for horizontal
                // scroll); a click there sets that keypoint active in EVERY
                // camera view at once.
                if (first_body_top > 0.0f) {
                    const int hc = ImGui::TableGetHoveredColumn();
                    const float my = ImGui::GetIO().MousePos.y;
                    const bool in_header =
                        hc >= 1 && hc < columns_count &&
                        my >= band_top && my < first_body_top;
                    if (in_header && (hc - 1) < (int)skeleton.node_names.size())
                        ImGui::SetTooltip("Set active in all cameras: %s",
                                          skeleton.node_names[hc - 1].c_str());
                    if (in_header && keypoints_find &&
                        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        const int node = hc - 1;
                        auto &fa = annotations.at(current_frame_num);
                        for (auto &cam : fa.cameras)
                            cam.active_id = (u32)node;
                    }
                }

                ImGui::EndTable();
            }
        }
    }
    ImGui::End();
}
