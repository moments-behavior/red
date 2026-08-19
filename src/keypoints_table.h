#pragma once
#include "app_context.h"
#include "keypoint_colors.h"
#include "gui/keypoint_clipboard.h"
#include "gui/shortcuts.h"
#include <imgui.h>
#include <algorithm>
#include <string>

inline void DrawKeypointsWindow(AppContext &ctx) {
    int current_frame_num = ctx.current_frame_num;
    auto &pm = ctx.pm;
    auto *scene = ctx.scene;
    auto &skeleton = ctx.skeleton;
    auto &annotations = ctx.annotations;
    auto &is_view_focused = ctx.is_view_focused;
    const ImVec4 active_kp_color = active_keypoint_color(ctx.user_settings);
    KeypointClipboard &kc = keypoint_clipboard();

    if (ImGui::Begin("Keypoints")) {

        bool keypoints_find =
            annotations.find(current_frame_num) != annotations.end();

        if (skeleton.num_nodes > 0 && skeleton.has_skeleton) {
            const int rows_count = scene->num_cams;
            const int columns_count = skeleton.num_nodes + 1;

            // Keep the multi-selection sized to the current skeleton.
            kc.ensure_size(skeleton.num_nodes);

            // Selection colors: a selected keypoint's angled NAME is recolored
            // (bright blue) in the header, and its empty body cells get a faint
            // tint so the selection also reads down the column.
            const ImVec4 sel_text = ImVec4(0.35f, 0.72f, 1.0f, 1.0f);
            ImVec4 sel_fill = ImGui::GetStyleColorVec4(ImGuiCol_Header);
            sel_fill.w = 0.28f;

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

            // Hover targets for the Delete key (resolved with precedence after
            // the table): a specific cell, a column header, or neither.
            int hover_row = -1, hover_node = -1, hover_header_node = -1;

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

                // Angled header row, but with per-column TEXT color so a
                // selected keypoint's slanted NAME itself changes color. This
                // mirrors ImGui::TableAngledHeadersRow() (imgui_tables.cpp),
                // overriding only TextColor for selected columns; the hover
                // background highlight (BgColor1) is left as stock.
                {
                    ImGuiContext &g = *GImGui;
                    ImGuiTable *table = g.CurrentTable;
                    ImGuiTableTempData *temp_data = table->TempData;
                    temp_data->AngledHeadersRequests.resize(0);
                    temp_data->AngledHeadersRequests.reserve(
                        table->ColumnsEnabledCount);

                    const ImGuiID row_id = ImGui::GetID("##AngledHeaders");
                    ImGuiTableInstanceData *table_instance =
                        ImGui::TableGetInstanceData(table,
                                                    table->InstanceCurrent);
                    int highlight_column_n = table->HighlightColumnHeader;
                    if (highlight_column_n == -1 &&
                        table->HoveredColumnBody != -1)
                        if (table_instance->HoveredRowLast == 0 &&
                            table->HoveredColumnBorder == -1 &&
                            (g.ActiveId == 0 || g.ActiveId == row_id ||
                             (table->IsActiveIdInTable || g.DragDropActive)))
                            highlight_column_n = table->HoveredColumnBody;

                    const ImU32 col_header_bg =
                        ImGui::GetColorU32(ImGuiCol_TableHeaderBg);
                    const ImU32 col_text = ImGui::GetColorU32(ImGuiCol_Text);
                    const ImU32 col_text_sel = ImGui::GetColorU32(sel_text);
                    const ImU32 col_hover = ImGui::GetColorU32(ImGuiCol_Header);
                    for (int order_n = 0; order_n < table->ColumnsCount;
                         order_n++)
                        if (IM_BITARRAY_TESTBIT(table->EnabledMaskByDisplayOrder,
                                                order_n)) {
                            const int column_n =
                                table->DisplayOrderToIndex[order_n];
                            ImGuiTableColumn *column = &table->Columns[column_n];
                            if ((column->Flags &
                                 ImGuiTableColumnFlags_AngledHeader) == 0)
                                continue;
                            const bool sel = kc.is_selected(column_n - 1);
                            ImGuiTableHeaderData request = {
                                (ImGuiTableColumnIdx)column_n,
                                sel ? col_text_sel : col_text,
                                col_header_bg,
                                (column_n == highlight_column_n) ? col_hover
                                                                 : 0u};
                            temp_data->AngledHeadersRequests.push_back(request);
                        }

                    ImGui::TableAngledHeadersRowEx(
                        row_id, g.Style.TableAngledHeadersAngle, 0.0f,
                        temp_data->AngledHeadersRequests.Data,
                        temp_data->AngledHeadersRequests.Size);
                }

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
                            const int node = column - 1;

                            float cell_w = ImGui::GetContentRegionAvail().x;
                            if (cell_w < 1.0f)
                                cell_w = ImGui::GetFrameHeight();
                            ImVec2 p0 = ImGui::GetCursorScreenPos();

                            if (keypoints_find) {
                                auto &fa = annotations.at(current_frame_num);
                                const bool is_active =
                                    row < (int)fa.cameras.size() &&
                                    fa.cameras[row].active_id == (u32)node;
                                const bool labeled =
                                    row < (int)fa.cameras.size() &&
                                    node < (int)fa.cameras[row].keypoints.size() &&
                                    fa.cameras[row].keypoints[node].labeled;
                                ImVec4 node_color = ImVec4(0, 0, 0, 0);

                                // Fill shows placement status regardless of
                                // active state: the node color once the keypoint
                                // is labeled, else transparent. The active
                                // keypoint is drawn as an outline below, so
                                // whether it has been placed stays visible.
                                if (labeled) {
                                    node_color =
                                        skeleton.node_colors[node];
                                    node_color.w = 0.9f;
                                } else if (kc.is_selected(node)) {
                                    // Selected but empty: tint so the selection
                                    // is visible in the body too.
                                    node_color = sel_fill;
                                }

                                const bool triangulated =
                                    node < (int)fa.kp3d.size() &&
                                    fa.kp3d[node].triangulated;

                                // The whole cell is a click target: clicking
                                // sets this keypoint active for this camera view.
                                ImGui::PushID(column);
                                if (ImGui::InvisibleButton(
                                        "##kpcell",
                                        ImVec2(cell_w, ImGui::GetFrameHeight()))) {
                                    if (row < (int)fa.cameras.size())
                                        fa.cameras[row].active_id = (u32)node;
                                }
                                // Active keypoint: outline the cell (in the
                                // user's "Active Keypoint" color) rather than
                                // filling it, so its placement color stays
                                // visible. Expand to the cell-bg edges by the
                                // cell padding. Matches the Labeling Tool's
                                // highlight-outline style.
                                if (is_active) {
                                    const ImVec2 cp = ImGui::GetStyle().CellPadding;
                                    const ImVec2 rmin = ImGui::GetItemRectMin();
                                    const ImVec2 rmax = ImGui::GetItemRectMax();
                                    ImGui::GetWindowDrawList()->AddRect(
                                        ImVec2(rmin.x - cp.x, rmin.y - cp.y),
                                        ImVec2(rmax.x + cp.x, rmax.y + cp.y),
                                        ImGui::GetColorU32(active_kp_color),
                                        0.0f, 0, 3.0f);
                                }
                                if (ImGui::IsItemHovered()) {
                                    hover_row = row;
                                    hover_node = node;
                                    if (node < (int)skeleton.node_names.size() &&
                                        row < (int)pm.camera_names.size()) {
                                        if (kc.count() >= 2)
                                            ImGui::SetTooltip(
                                                "%s / %s\n"
                                                "Click: set active   Delete: remove selected set (%d)",
                                                pm.camera_names[row].c_str(),
                                                skeleton.node_names[node].c_str(),
                                                kc.count());
                                        else
                                            ImGui::SetTooltip(
                                                "%s / %s\n"
                                                "Click: set active   Delete: remove from this camera",
                                                pm.camera_names[row].c_str(),
                                                skeleton.node_names[node].c_str());
                                    }
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
                // scroll). The header is where a set of keypoint columns is
                // multi-selected, File-Explorer style:
                //   plain click  -> select just this one + set it active in all
                //                   cameras (preserves the old gesture),
                //   Shift+click  -> range-select from the anchor,
                //   Ctrl+click   -> toggle this one in/out of the selection.
                if (first_body_top > 0.0f) {
                    const int hc = ImGui::TableGetHoveredColumn();
                    const float my = ImGui::GetIO().MousePos.y;
                    const bool in_header =
                        hc >= 1 && hc < columns_count &&
                        my >= band_top && my < first_body_top;
                    if (in_header && (hc - 1) < (int)skeleton.node_names.size()) {
                        hover_header_node = hc - 1;
                        if (kc.count() >= 2)
                            ImGui::SetTooltip(
                                "%s\n"
                                "Click: select   Shift/Ctrl: multi-select   "
                                "Delete: remove selected set (%d)",
                                skeleton.node_names[hc - 1].c_str(),
                                kc.count());
                        else
                            ImGui::SetTooltip(
                                "%s\n"
                                "Click: select   Shift/Ctrl: multi-select   "
                                "Delete: remove from all cameras",
                                skeleton.node_names[hc - 1].c_str());
                    }
                    if (in_header &&
                        ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        const int node = hc - 1;
                        if (node >= 0 && node < (int)kc.selected.size()) {
                            ImGuiIO &io = ImGui::GetIO();
                            if (io.KeyShift && kc.anchor >= 0 &&
                                kc.anchor < (int)kc.selected.size()) {
                                int lo = std::min(kc.anchor, node);
                                int hi = std::max(kc.anchor, node);
                                if (!io.KeyCtrl)
                                    std::fill(kc.selected.begin(),
                                              kc.selected.end(), (char)0);
                                for (int n = lo; n <= hi; ++n)
                                    kc.selected[(size_t)n] = 1;
                                // anchor unchanged (range pivot)
                            } else if (io.KeyCtrl) {
                                kc.selected[(size_t)node] =
                                    kc.selected[(size_t)node] ? 0 : 1;
                                kc.anchor = node;
                            } else {
                                std::fill(kc.selected.begin(),
                                          kc.selected.end(), (char)0);
                                kc.selected[(size_t)node] = 1;
                                kc.anchor = node;
                                if (keypoints_find) {
                                    auto &fa =
                                        annotations.at(current_frame_num);
                                    for (auto &cam : fa.cameras)
                                        cam.active_id = (u32)node;
                                }
                            }
                        }
                    }
                }

                ImGui::EndTable();
            }

            // ── Select All (Ctrl+A): toggle every keypoint column. Scoped to
            //    the Keypoints window so it never clashes with the image-view
            //    'A' (previous active keypoint), which only fires over a plot. ──
            // Copy / Paste below share the same scope so a stray Ctrl+V over
            // an image view or another panel never overwrites labels.
            const bool kp_win_active =
                ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) ||
                ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows);
            if (kp_win_active && keys::pressed(keys::Sc::SelectAllKeypoints)) {
                if (kc.count() >= skeleton.num_nodes) {
                    kc.clear_selection();
                } else {
                    std::fill(kc.selected.begin(), kc.selected.end(), (char)1);
                    kc.anchor = 0;
                }
            }

            // ── Delete key (precedence: hovered cell -> hovered header ->
            //    selection over the window) ──
            if (keypoints_find &&
                keys::pressed(keys::Sc::DeleteKeypoint)) {
                auto &fa = annotations.at(current_frame_num);
                const bool win_hovered = ImGui::IsWindowHovered(
                    ImGuiHoveredFlags_RootAndChildWindows);
                auto delete_selection = [&]() {
                    int n = delete_selected_all_cameras(
                        kc, fa, skeleton.num_nodes, scene->num_cams);
                    if (n)
                        ctx.toasts.pushSuccess(
                            "Deleted " + std::to_string(n) +
                            " keypoint(s) from all cameras");
                };
                if (win_hovered && kc.count() >= 2) {
                    // A built-up multi-selection takes priority over whatever is
                    // hovered, so Delete works right where you finished
                    // selecting (over a name or a cell) — no need to move the
                    // cursor to an empty spot first.
                    delete_selection();
                } else if (hover_row >= 0 && hover_node >= 0) {
                    delete_node_from_camera(fa, hover_node, hover_row);
                } else if (hover_header_node >= 0) {
                    delete_node_all_cameras(fa, hover_header_node,
                                            scene->num_cams);
                } else if (win_hovered && kc.any()) {
                    delete_selection();
                }
            }

            // ── Copy (Ctrl+C): snapshot the selected node set ──
            if (kp_win_active && keys::pressed(keys::Sc::CopyKeypoints)) {
                if (!kc.any()) {
                    ctx.toasts.push(
                        "Select keypoint columns first (click a name above)",
                        Toast::Warning, 4.0f);
                } else if (!keypoints_find) {
                    ctx.toasts.push("Nothing to copy on this frame",
                                    Toast::Warning, 4.0f);
                } else {
                    int sel = kc.count();
                    int n = copy_selected_keypoints(
                        kc, annotations.at(current_frame_num),
                        skeleton.num_nodes, scene->num_cams, skeleton.name);
                    if (n == 0)
                        ctx.toasts.push(
                            "None of the " + std::to_string(sel) +
                            " selected keypoints are labeled here",
                            Toast::Warning, 4.0f);
                    else if (n == sel)
                        ctx.toasts.pushSuccess(
                            "Copied " + std::to_string(n) + " keypoint(s)");
                    else
                        ctx.toasts.pushSuccess(
                            "Copied " + std::to_string(n) + " of " +
                            std::to_string(sel) + " selected keypoint(s)");
                }
            }

            // ── Paste (Ctrl+V): overwrite the copied node set onto this frame ──
            if (kp_win_active && keys::pressed(keys::Sc::PasteKeypoints)) {
                if (!kc.has_clip()) {
                    ctx.toasts.push(
                        "Clipboard is empty (copy with Ctrl+C first)",
                        Toast::Warning, 4.0f);
                } else if (!paste_identity_ok(kc, skeleton.num_nodes,
                                              scene->num_cams, skeleton.name)) {
                    ctx.toasts.push(
                        "Clipboard is from a different skeleton \xE2\x80\x94 "
                        "cannot paste",
                        Toast::Warning, 5.0f);
                } else {
                    FrameAnnotation &fa = get_or_create_frame(
                        annotations, (u32)current_frame_num,
                        skeleton.num_nodes, scene->num_cams);
                    int n = paste_keypoints(kc, fa, skeleton.num_nodes,
                                            scene->num_cams);
                    ctx.toasts.pushSuccess(
                        "Pasted " + std::to_string(n) + " keypoint(s)");
                }
            }
        }
    }
    ImGui::End();
}
