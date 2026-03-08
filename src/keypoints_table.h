#pragma once
#include "project.h"
#include "skeleton.h"
#include <imgui.h>

void DrawKeypointsWindow(
    ProjectManager &pm, RenderScene *scene, SkeletonContext &skeleton,
    std::map<u32, KeyPoints *> &keypoints_map, int &current_frame_num,
    std::vector<bool> &is_view_focused) {
    if (ImGui::Begin("Keypoints")) {

        bool keypoints_find =
            keypoints_map.find(current_frame_num) != keypoints_map.end();

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

            if (ImGui::BeginTable("table_angled_headers", columns_count,
                                  table_flags, table_size)) {
                ImGui::TableSetupColumn(
                    "Name", ImGuiTableColumnFlags_NoHide |
                                ImGuiTableColumnFlags_NoReorder);

                for (int column = 1; column < columns_count; column++) {
                    ImGui::TableSetupColumn(
                        skeleton.node_names[column - 1].c_str(),
                        ImGuiTableColumnFlags_AngledHeader |
                            ImGuiTableColumnFlags_WidthFixed);
                }

                ImGui::TableSetupScrollFreeze(1, 2);
                ImGui::TableAngledHeadersRow();
                ImGui::TableHeadersRow();

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
                    ImGui::AlignTextToFramePadding();
                    ImGui::Text("%s", pm.camera_names[row].c_str());

                    for (int column = 1; column < columns_count; column++) {
                        if (ImGui::TableSetColumnIndex(column)) {
                            if (keypoints_find) {
                                ImVec4 node_color = ImVec4(0, 0, 0, 0);

                                if (keypoints_map[current_frame_num]
                                        ->active_id[row] == column - 1) {
                                    node_color = (ImVec4)ImColor::HSV(
                                        0.8f, 1.0f, 1.0f);
                                } else if (keypoints_map[current_frame_num]
                                               ->kp2d[row][column - 1]
                                               .is_labeled) {
                                    node_color =
                                        skeleton.node_colors[column - 1];
                                    node_color.w = 0.9f;
                                }

                                if (keypoints_map[current_frame_num]
                                        ->kp3d[column - 1]
                                        .is_triangulated) {
                                    ImGui::TextColored(
                                        ImVec4(1.0f, 1.0f, 1.0f, 1.0f),
                                        "T");
                                }

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

                ImGui::EndTable();
            }
        }
    }
    ImGui::End();
}
