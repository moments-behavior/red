#pragma once
#include "global.h"
#include "project.h"
#include "skeleton.h"
#include <ImGuiFileDialog.h>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)

void DrawKeypointsWindow(
    ProjectManager &pm, RenderScene *scene, SkeletonContext &skeleton,
    std::map<u32, KeyPoints *> &keypoints_map, int &current_frame_num,
    std::vector<bool> &is_view_focused,
    std::vector<std::string> &bbox_class_names, int &current_bbox_class,
    std::vector<ImVec4> &bbox_class_colors, int &current_bbox_id,
    int &hovered_bbox_cam, int &hovered_bbox_idx, int &hovered_bbox_id,
    float &hovered_bbox_confidence, int &hovered_bbox_class,
    int &hovered_obb_cam, int &hovered_obb_idx, int &hovered_obb_id,
    float &hovered_obb_confidence, int &hovered_obb_class, bool &show_bbox_ids,
    std::string &new_class_name) {
    if (ImGui::Begin("Keypoints")) {
        const float TEXT_BASE_HEIGHT = ImGui::GetTextLineHeightWithSpacing();

        bool keypoints_find =
            keypoints_map.find(current_frame_num) != keypoints_map.end();

        // Only show bounding box class management if skeleton
        // supports bboxes or obbs
        if (skeleton.has_bbox || skeleton.has_obb) {
            // Bounding Box Class Management Section
            ImGui::SeparatorText("Bounding Box Classes");

            // Current class selection combo
            if (ImGui::BeginCombo(
                    "Current Class",
                    bbox_class_names[current_bbox_class].c_str())) {
                for (int i = 0; i < bbox_class_names.size(); i++) {
                    bool is_selected = (current_bbox_class == i);

                    // Show color indicator next to class name
                    ImGui::ColorButton("##color", bbox_class_colors[i],
                                       ImGuiColorEditFlags_NoTooltip |
                                           ImGuiColorEditFlags_NoBorder,
                                       ImVec2(15, 15));
                    ImGui::SameLine();

                    if (ImGui::Selectable(bbox_class_names[i].c_str(),
                                          is_selected)) {
                        current_bbox_class = i;
                        current_bbox_id = 0; // Reset bbox ID when
                                             // switching classes
                    }
                    if (is_selected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            // Display current bbox ID
            ImGui::Text("Current Bounding Box ID: %d", current_bbox_id);

            // Checkbox to toggle bbox ID display on frame
            ImGui::Checkbox("Show Bbox IDs on Frame", &show_bbox_ids);

            // Display hovered bbox ID if any bbox is being hovered
            if (hovered_bbox_cam >= 0 && hovered_bbox_idx >= 0) {
                ImGui::Text("Hovered Bbox ID: %d", hovered_bbox_id);
            } else if (hovered_obb_cam >= 0 && hovered_obb_idx >= 0) {
                ImGui::Text("Hovered OBB ID: %d", hovered_obb_id);
            }

            // Add new class
            ImGui::SetNextItemWidth(200);
            ImGui::InputTextWithHint("##new_class", "Enter new class name...",
                                     &new_class_name);
            ImGui::SameLine();
            if (ImGui::Button("Add Class") && new_class_name.size() > 0) {
                bbox_class_names.push_back(std::string(new_class_name));
                // Generate a unique color for the new class (HSV
                // with different hues)
                float hue = (bbox_class_colors.size() *
                             0.618034f); // Golden ratio for nice
                                         // color distribution
                while (hue > 1.0f)
                    hue -= 1.0f;
                ImVec4 new_color = (ImVec4)ImColor::HSV(hue, 0.8f, 1.0f);
                bbox_class_colors.push_back(new_color);
                current_bbox_class = bbox_class_names.size() - 1;
                new_class_name.clear();
            }

            // Edit current class color
            if (current_bbox_class >= 0 &&
                current_bbox_class < bbox_class_colors.size()) {
                ImGui::SetNextItemWidth(200);
                ImGui::ColorEdit3(
                    "Class Color",
                    (float *)&bbox_class_colors[current_bbox_class],
                    ImGuiColorEditFlags_NoInputs);
            }

            // Delete class (only if not the last one)
            if (bbox_class_names.size() > 1) {
                ImGui::SameLine();
                if (ImGui::Button("Delete Class")) {
                    bbox_class_names.erase(bbox_class_names.begin() +
                                           current_bbox_class);
                    bbox_class_colors.erase(bbox_class_colors.begin() +
                                            current_bbox_class);
                    if (current_bbox_class >= bbox_class_names.size()) {
                        current_bbox_class = bbox_class_names.size() - 1;
                    }
                }
            }

            ImGui::Separator();
        }

        // Check if skeleton is valid and has nodes before creating
        // the table
        if (skeleton.num_nodes > 0) {
            // Different table behavior based on skeleton
            // configuration
            if (skeleton.has_skeleton && skeleton.has_bbox && keypoints_find) {
                // Show bounding box keypoints table
                int bbox_count = 0;

                // Count total bounding boxes across all cameras
                for (int cam = 0; cam < scene->num_cams; cam++) {
                    if (keypoints_map[current_frame_num]
                            ->bbox2d_list[cam]
                            .size() > 0) {
                        bbox_count += keypoints_map[current_frame_num]
                                          ->bbox2d_list[cam]
                                          .size();
                    }
                }

                if (bbox_count > 0) {
                    const int columns_count = skeleton.num_nodes + 1;
                    static ImGuiTableFlags table_flags =
                        ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                        ImGuiTableFlags_SizingFixedFit |
                        ImGuiTableFlags_BordersOuter |
                        ImGuiTableFlags_BordersInnerH |
                        ImGuiTableFlags_Hideable | ImGuiTableFlags_Resizable |
                        ImGuiTableFlags_HighlightHoveredColumn;

                    if (ImGui::BeginTable(
                            "table_bbox_keypoints", columns_count, table_flags,
                            ImVec2(0.0f, TEXT_BASE_HEIGHT * 12))) {
                        ImGui::TableSetupColumn(
                            "Bbox", ImGuiTableColumnFlags_NoHide |
                                        ImGuiTableColumnFlags_NoReorder);
                        for (int column = 1; column < columns_count; column++)
                            ImGui::TableSetupColumn(
                                skeleton.node_names[column - 1].c_str(),
                                ImGuiTableColumnFlags_AngledHeader |
                                    ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupScrollFreeze(1, 2);

                        ImGui::TableAngledHeadersRow();
                        ImGui::TableHeadersRow();

                        // Iterate through all bounding boxes
                        int bbox_row = 0;

                        // Use stored active bbox information from
                        // plot context
                        int active_cam = -1;
                        int active_bbox_global = -1;

                        // Find the focused camera with an active
                        // bbox
                        for (int cam = 0; cam < scene->num_cams &&
                                          cam < user_active_bbox_idx.size();
                             cam++) {
                            if (is_view_focused[cam] &&
                                user_active_bbox_idx[cam] != -1) {
                                // Verify the active bbox still
                                // exists and has keypoints
                                if (user_active_bbox_idx[cam] <
                                    keypoints_map[current_frame_num]
                                        ->bbox2d_list[cam]
                                        .size()) {
                                    BoundingBox &bbox =
                                        keypoints_map[current_frame_num]
                                            ->bbox2d_list
                                                [cam]
                                                [user_active_bbox_idx[cam]];
                                    if (bbox.state == RectTwoPoints &&
                                        bbox.has_bbox_keypoints &&
                                        bbox.bbox_keypoints2d) {
                                        active_cam = cam;
                                        active_bbox_global =
                                            user_active_bbox_idx[cam];
                                        break;
                                    }
                                }
                            }
                        }

                        // If no active bbox found from tracking,
                        // fall back to first bbox in focused camera
                        if (active_cam == -1) {
                            for (int cam = 0; cam < scene->num_cams; cam++) {
                                if (is_view_focused[cam] &&
                                    keypoints_map[current_frame_num]
                                            ->bbox2d_list[cam]
                                            .size() > 0) {
                                    for (int bbox_idx = 0;
                                         bbox_idx <
                                         keypoints_map[current_frame_num]
                                             ->bbox2d_list[cam]
                                             .size();
                                         bbox_idx++) {
                                        BoundingBox &bbox =
                                            keypoints_map[current_frame_num]
                                                ->bbox2d_list[cam][bbox_idx];
                                        if (bbox.state == RectTwoPoints &&
                                            bbox.has_bbox_keypoints &&
                                            bbox.bbox_keypoints2d) {
                                            active_cam = cam;
                                            active_bbox_global = bbox_idx;
                                            break;
                                        }
                                    }
                                    break;
                                }
                            }
                        }

                        // Render the table with highlighting
                        bbox_row = 0;
                        for (int cam = 0; cam < scene->num_cams; cam++) {
                            for (int bbox_idx = 0;
                                 bbox_idx < keypoints_map[current_frame_num]
                                                ->bbox2d_list[cam]
                                                .size();
                                 bbox_idx++) {
                                BoundingBox &bbox =
                                    keypoints_map[current_frame_num]
                                        ->bbox2d_list[cam][bbox_idx];

                                // Only show completed bboxes with
                                // keypoints
                                if (bbox.state == RectTwoPoints &&
                                    bbox.has_bbox_keypoints &&
                                    bbox.bbox_keypoints2d) {
                                    ImGui::PushID(bbox_row);
                                    ImGui::TableNextRow();

                                    // Highlight row for active
                                    // bounding box
                                    bool is_active_bbox_row =
                                        (cam == active_cam &&
                                         bbox_idx == active_bbox_global);
                                    if (is_active_bbox_row) {
                                        // Bright blue background
                                        // for active bbox row
                                        ImU32 row_bg_color = ImGui::GetColorU32(
                                            ImVec4(0.2f, 0.4f, 0.8f,
                                                   0.7f)); // Vibrant
                                                           // blue
                                        ImGui::TableSetBgColor(
                                            ImGuiTableBgTarget_RowBg0,
                                            row_bg_color);
                                    } else if (is_view_focused[cam]) {
                                        // Red background for
                                        // focused camera (when not
                                        // active bbox)
                                        ImU32 row_bg_color = ImGui::GetColorU32(
                                            ImVec4(0.7f, 0.3f, 0.3f, 0.65f));
                                        ImGui::TableSetBgColor(
                                            ImGuiTableBgTarget_RowBg0,
                                            row_bg_color);
                                    }

                                    ImGui::TableSetColumnIndex(0);
                                    ImGui::AlignTextToFramePadding();
                                    // Show bbox number with camera
                                    // identifier
                                    ImGui::Text("C%d-B%d", cam, bbox_idx);

                                    for (int column = 1; column < columns_count;
                                         column++) {
                                        if (ImGui::TableSetColumnIndex(
                                                column)) {
                                            ImVec4 node_color =
                                                ImVec4(0, 0, 0,
                                                       0); // Transparent
                                                           // by
                                                           // default

                                            // Check if this is the
                                            // active keypoint for
                                            // this bbox
                                            if (bbox.active_kp_id &&
                                                bbox.active_kp_id[cam] ==
                                                    column - 1) {
                                                // Vibrant blue
                                                // color for active
                                                // keypoint
                                                node_color =
                                                    (ImVec4)ImColor::HSV(
                                                        0.6f, 1.0f,
                                                        1.0f); // Vibrant
                                                               // blue
                                                               // (HSV:
                                                               // 216°,
                                                               // 100%,
                                                               // 100%)
                                            } else {
                                                // Check if keypoint
                                                // is labeled
                                                if (bbox.bbox_keypoints2d
                                                        [cam][column - 1]
                                                            .is_labeled) {
                                                    node_color =
                                                        skeleton.node_colors
                                                            [column - 1];
                                                    node_color.w = 0.9;
                                                }
                                            }

                                            ImU32 cell_bg_color =
                                                ImGui::GetColorU32(node_color);
                                            ImGui::TableSetBgColor(
                                                ImGuiTableBgTarget_CellBg,
                                                cell_bg_color);
                                        }
                                    }
                                    ImGui::PopID();
                                    bbox_row++;
                                }
                            }
                        }
                        ImGui::EndTable();
                    }
                } else {
                    ImGui::Text("No bounding boxes with keypoints found");
                }
            } else if (skeleton.has_skeleton && !skeleton.has_bbox) {

                // Show regular camera keypoints table
                const int rows_count = scene->num_cams;
                const int columns_count = skeleton.num_nodes + 1;

                static ImGuiTableFlags table_flags =
                    ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY |
                    ImGuiTableFlags_SizingFixedFit |
                    ImGuiTableFlags_BordersOuter |
                    ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_Hideable |
                    ImGuiTableFlags_Resizable |
                    ImGuiTableFlags_HighlightHoveredColumn;

                // Table height grows with window
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

                    // ---------------------------------------------------------------------
                    // Find focused row (first one with is_view_focused[row] ==
                    // true)
                    // ---------------------------------------------------------------------
                    int focused_row = -1;
                    for (int row = 0; row < rows_count; row++) {
                        if (row < (int)is_view_focused.size() &&
                            is_view_focused[row]) {
                            focused_row = row;
                            break;
                        }
                    }

                    // ---------------------------------------------------------------------
                    // Helper lambda to render a single row (so we can call it
                    // twice)
                    // ---------------------------------------------------------------------
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

                        // Column 0: camera name
                        ImGui::TableSetColumnIndex(0);
                        ImGui::AlignTextToFramePadding();
                        ImGui::Text("%s", pm.camera_names[row].c_str());

                        // Other columns: keypoints
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

                    // ---------------------------------------------------------------------
                    // 1) Render the focused row first (if any)
                    // ---------------------------------------------------------------------
                    if (focused_row != -1) {
                        render_row(focused_row);
                    }

                    // ---------------------------------------------------------------------
                    // 2) Render the rest of the rows, skipping the focused one
                    // ---------------------------------------------------------------------
                    for (int row = 0; row < rows_count; row++) {
                        if (row == focused_row)
                            continue;
                        render_row(row);
                    }

                    ImGui::EndTable();
                }

            } else {
                ImGui::Text("Bounding box mode: No keypoints to display");
            }
        } else {
            ImGui::Text("Bounding box mode: No keypoints to display");
        }

        if (keypoints_find) {
            if (skeleton.has_bbox) {
                ImGui::Separator();
                ImGui::Text("Bounding Box Info:");

                if (hovered_bbox_cam >= 0 && hovered_bbox_idx >= 0) {
                    ImGui::Text("Camera: %d, Box: %d", hovered_bbox_cam,
                                hovered_bbox_idx);
                    ImGui::Text("Class: %d, Confidence: %.1f%%",
                                hovered_bbox_class,
                                hovered_bbox_confidence * 100.0f);
                } else {
                    ImGui::Text("Hover over a bounding box to see details");
                }
            }

            if (skeleton.has_obb) {
                ImGui::Separator();
                ImGui::Text("Oriented Bounding Box Info:");

                if (hovered_obb_class >= 0) {
                    ImGui::Text("Class: %d, Confidence: %.1f%%",
                                hovered_obb_class,
                                hovered_obb_confidence * 100.0f);
                } else {
                    ImGui::Text("Hover over an OBB to see details");
                }
            }
        }
    }
    ImGui::End();
}
