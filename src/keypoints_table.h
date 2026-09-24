#pragma once
#include "app_context.h"
#include "keypoint_colors.h"
#include "gui/keypoint_clipboard.h"
#include "gui/shortcuts.h"
#include <imgui.h>
#include <algorithm>
#include <string>

// The per-camera x per-keypoint grid. Drawn inline inside the Labeling Tool
// rather than in a window of its own: it is the labeling surface for whatever
// the tool's controls act on, and splitting them across two dockable windows
// meant the animal selector, Triangulate and the table it reports on could sit
// on opposite sides of the screen. `height` is what the caller can spare.
inline void DrawKeypointsTable(AppContext &ctx, float height) {
    int current_frame_num = ctx.current_frame_num;
    // The table shows the animal being edited, not the first one -- otherwise
    // dragging animal 2's keypoint clears animal 2's 3D while the table keeps
    // reporting animal 0's, which reads as the state failing to follow.
    auto &pm = ctx.pm;
    auto *scene = ctx.scene;
    auto &skeleton = ctx.skeleton;
    auto &annotations = ctx.annotations;
    auto &is_view_focused = ctx.is_view_focused;
    const ImVec4 active_kp_color = active_keypoint_color(ctx.user_settings);
    KeypointClipboard &kc = keypoint_clipboard();

    {

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

            float table_height = height;
            ImVec2 table_size(0.0f, table_height);

            // Top of the table on screen = top of the angled-header band; used
            // below to detect clicks on the (header-row-less) angled headers.
            const float band_top = ImGui::GetCursorScreenPos().y;

            // Hover targets for the Delete key (resolved with precedence after
            // the table): a specific cell, a column header, or neither.
            int hover_row = -1, hover_node = -1, hover_header_node = -1;

            // The table ID carries the skeleton name. ImGui saves column
            // settings -- widths, visibility, display order -- per table ID in
            // the ini, so a fixed ID makes a 24-keypoint skeleton inherit the
            // layout of a 25-keypoint one opened earlier: same columns, wrong
            // arrangement, and no way to tell from the data.
            const std::string table_id =
                "table_angled_headers##" + skeleton.name;
            if (ImGui::BeginTable(table_id.c_str(), columns_count,
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
                // 1 column + 2 rows: the angled header and the 3D row
                // submitted right below it, so a sixteen-camera rig cannot
                // scroll the solve out of sight.
                ImGui::TableSetupScrollFreeze(1, 2);

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

                    const bool row_focused =
                        row < (int)is_view_focused.size() &&
                        is_view_focused[row] && keypoints_find;

                    ImGui::TableSetColumnIndex(0);
                    if (first_body_top < 0.0f)
                        first_body_top = ImGui::GetCursorScreenPos().y;
                    ImGui::AlignTextToFramePadding();
                    // Clicking the name brings that camera's view to the
                    // front. Only SetWindowFocus is needed: red.cpp notices
                    // the focus change next frame and moves the highlight
                    // here, so the two cannot disagree about which view is
                    // current.
                    if (row < (int)pm.camera_names.size()) {
                        const std::string &cam_name = pm.camera_names[row];
                        ImGui::PushID(row);
                        // The Selectable IS the focus highlight, restyled to
                        // the red the cell tint used, so there is one thing
                        // drawing this state rather than two.
                        ImGui::PushStyleColor(ImGuiCol_Header,
                                              ImVec4(0.7f, 0.3f, 0.3f, 0.65f));
                        const bool clicked =
                            ImGui::Selectable(cam_name.c_str(), row_focused,
                                              ImGuiSelectableFlags_None);
                        ImGui::PopStyleColor();
                        if (clicked)
                            ImGui::SetWindowFocus(cam_name.c_str());
                        if (ImGui::IsItemHovered())
                            ImGui::SetTooltip("Bring %s to the front",
                                              cam_name.c_str());
                        ImGui::PopID();
                    } else {
                        ImGui::Text("?");
                    }

                    for (int column = 1; column < columns_count; column++) {
                        if (ImGui::TableSetColumnIndex(column)) {
                            const int node = column - 1;

                            float cell_w = ImGui::GetContentRegionAvail().x;
                            if (cell_w < 1.0f)
                                cell_w = ImGui::GetFrameHeight();
                            ImVec2 p0 = ImGui::GetCursorScreenPos();

                            if (keypoints_find) {
                                auto &fa = instance_or_first(annotations.at(current_frame_num), ctx.active_instance);
                                const bool is_active =
                                    row < (int)fa.cameras.size() &&
                                    fa.cameras[row].active_id == (u32)node;
                                const bool labeled =
                                    row < (int)fa.cameras.size() &&
                                    node < (int)fa.cameras[row].keypoints.size() &&
                                    fa.cameras[row].keypoints[node].usable();
                                const bool occluded =
                                    row < (int)fa.cameras.size() &&
                                    node < (int)fa.cameras[row].keypoints.size() &&
                                    fa.cameras[row].keypoints[node].is_occluded();
                                const bool user_annotated =
                                    labeled && fa.cameras[row].keypoints[node].is_manual();
                                const bool projected =
                                    labeled && !fa.cameras[row].keypoints[node].is_manual();
                                // Coordinate origin, independent of the above:
                                // a point you placed and then refreshed with T
                                // is both manual and reprojected.
                                const bool reproj =
                                    labeled &&
                                    fa.cameras[row].keypoints[node].reprojected;
                                const bool model_made =
                                    labeled &&
                                    fa.cameras[row].keypoints[node].is_predicted();

                                const bool observed =
                                    labeled &&
                                    fa.cameras[row].keypoints[node].is_observed();

                                // All THREE axes, spelled out. The tooltip is
                                // the one place the full state is legible, so
                                // it reports the combination rather than
                                // picking whichever part a marker can show.
                                // Visibility used to be missing from it
                                // entirely: a reprojection someone had
                                // confirmed and one nobody had looked at both
                                // read "reprojected from 3D".
                                const char *state_text =
                                    occluded     ? "occluded / outside frame"
                                    : !labeled   ? "not placed"
                                    : user_annotated
                                        ? (reproj ? "user annotated, reprojected"
                                                  : "user annotated")
                                    : model_made
                                        ? (reproj ? "model predicted, reprojected"
                                                  : "model predicted")
                                    : reproj     ? "reprojected from 3D"
                                                 : "placed";
                                const char *vis_text =
                                    occluded   ? nullptr   // the state IS the visibility
                                    : !labeled ? nullptr
                                    : observed ? "confirmed visible here"
                                               : "visibility not judged";
                                ImVec4 node_color = ImVec4(0, 0, 0, 0);

                                // Fill shows placement status regardless of
                                // active state: the node color once the keypoint
                                // is labeled, else transparent. The active
                                // keypoint is drawn as an outline below, so
                                // whether it has been placed stays visible.
                                if (labeled) {
                                    node_color =
                                        skeleton.node_colors[node];
                                    node_color.w = projected ? 0.5f : 0.9f;
                                } else if (occluded) {
                                    node_color = ImVec4(0.85f, 0.25f, 0.25f, 0.9f);
                                } else if (kc.is_selected(node)) {
                                    // Selected but empty: tint so the selection
                                    // is visible in the body too.
                                    node_color = sel_fill;
                                }

                                // The whole cell is a click target, and it
                                // does both halves of "work on this one":
                                // makes the keypoint active in that camera,
                                // and brings that camera's view to the front.
                                //
                                // Picking a cell is already a statement about
                                // which view you mean -- the row IS a camera.
                                // Leaving the view behind meant reading the
                                // table to find the cell worth fixing and then
                                // hunting for the tab it belonged to.
                                //
                                // SetWindowFocus alone, as with the camera
                                // names above: red.cpp sees the focus change
                                // next frame and moves the highlight, so the
                                // two cannot disagree about the current view.
                                ImGui::PushID(column);
                                if (ImGui::InvisibleButton(
                                        "##kpcell",
                                        ImVec2(cell_w, ImGui::GetFrameHeight()))) {
                                    if (row < (int)fa.cameras.size())
                                        fa.cameras[row].active_id = (u32)node;
                                    if (row < (int)pm.camera_names.size())
                                        ImGui::SetWindowFocus(
                                            pm.camera_names[row].c_str());
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
                                                "%s / %s\n%s%s%s\n"
                                                "Click: set active + show view   "
                                                "Delete: remove selected set (%d)",
                                                pm.camera_names[row].c_str(),
                                                skeleton.node_names[node].c_str(),
                                                state_text,
                                                vis_text ? "\n" : "",
                                                vis_text ? vis_text : "",
                                                kc.count());
                                        else
                                            ImGui::SetTooltip(
                                                "%s / %s\n%s%s%s\n"
                                                "Click: set active + show view   "
                                                "Delete: remove from this camera",
                                                pm.camera_names[row].c_str(),
                                                skeleton.node_names[node].c_str(),
                                                state_text,
                                                vis_text ? "\n" : "",
                                                vis_text ? vis_text : "");
                                    }
                                }
                                // T marks coordinates that came from the 3D,
                                // which a hand-placed point acquires the
                                // moment you press T -- so it shows on those
                                // too. Alpha carries the other axis: solid
                                // where nobody placed the point, faint where
                                // the position is yours and was merely
                                // refreshed.
                                if (reproj)
                                    ImGui::GetWindowDrawList()->AddText(
                                        ImVec2(p0.x + 2.0f, p0.y),
                                        user_annotated
                                            ? IM_COL32(255, 255, 255, 130)
                                            : IM_COL32(255, 255, 255, 255),
                                        "T");
                                if (occluded)
                                    ImGui::GetWindowDrawList()->AddText(
                                        ImVec2(p0.x + 2.0f, p0.y),
                                        IM_COL32(255, 120, 120, 255), "X");

                                // A confirmed-visible point that you did NOT
                                // place: a reprojection someone looked at and
                                // vouched for, or an imported one a model
                                // vouched for. Marked because it is the state
                                // you cannot otherwise tell from a
                                // reprojection nobody has judged.
                                //
                                // Not marked on your own points: placing one
                                // already means you saw it, so a tick on every
                                // cell you filled in would say nothing and
                                // clutter the common case. Bottom-right, clear
                                // of the T and X at the top-left.
                                if (observed && !user_annotated) {
                                    const ImVec2 tsz = ImGui::CalcTextSize("v");
                                    ImGui::GetWindowDrawList()->AddText(
                                        ImVec2(p0.x + cell_w - tsz.x - 2.0f,
                                               p0.y + ImGui::GetFrameHeight() -
                                                   tsz.y),
                                        IM_COL32(210, 255, 210, 220), "v");
                                }
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

                // === 3D row ===
                // The solve, per keypoint, pinned under the header and above
                // the 2D rows it came from. Its state used to be reachable
                // only by hovering each point in a camera view one at a time,
                // and the Triangulated/Predicted split was not visible at all
                // -- which matters most on a tailcycle session, where every
                // node arrives Predicted and you need to see which ones you
                // have actually re-solved.
                // Shown whenever the project HAS a 3D layer, not only once a
                // frame does. The 2D rows below already work this way -- they
                // draw and leave their cells empty -- and the row going
                // missing until the first B made the table look like it had
                // no 3D row at all, on exactly the frames where you are
                // deciding whether to start one.
                if (!project_is_2d(ctx.pm) && scene->num_cams > 1) {
                    const FrameAnnotation *fa3d =
                        keypoints_find
                            ? &instance_or_first(
                                  annotations.at(current_frame_num),
                                  ctx.active_instance)
                            : nullptr;
                    ImGui::PushID("row3d");
                    ImGui::TableNextRow();
                    ImGui::TableSetColumnIndex(0);
                    // First body row now, so it marks where the angled-header
                    // band ends -- otherwise clicks here would be taken for
                    // header clicks and would select keypoint columns.
                    if (first_body_top < 0.0f)
                        first_body_top = ImGui::GetCursorScreenPos().y;
                    ImGui::AlignTextToFramePadding();
                    ImGui::TextUnformatted("3D");
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Triangulated position per keypoint, for the animal "
                            "being edited.\nStrong = solved here, faint = "
                            "predicted by a model.");

                    for (int column = 1; column < columns_count; column++) {
                        if (!ImGui::TableSetColumnIndex(column)) continue;
                        const int node = column - 1;
                        // No frame yet: the cell is drawn, just empty, so the
                        // row keeps its shape and its column alignment.
                        static const Keypoint3D kNo3D{};
                        const Keypoint3D &k3 =
                            (fa3d && node < (int)fa3d->kp3d.size())
                                ? fa3d->kp3d[node]
                                : kNo3D;

                        float cell_w = ImGui::GetContentRegionAvail().x;
                        if (cell_w < 1.0f) cell_w = 1.0f;
                        ImGui::PushID(column);
                        ImGui::InvisibleButton(
                            "##kp3dcell",
                            ImVec2(cell_w, ImGui::GetFrameHeight()));
                        if (k3.exist && ImGui::IsItemHovered())
                            ImGui::SetTooltip(
                                "%s\n%s\n(%.2f, %.2f, %.2f)",
                                node < (int)skeleton.node_names.size()
                                    ? skeleton.node_names[node].c_str() : "",
                                k3.is_predicted()
                                    ? "predicted" : "triangulated",
                                k3.x, k3.y, k3.z);
                        ImGui::PopID();

                        // Coloured the way the 2D cells above are: the node's
                        // own colour, with alpha carrying how the position was
                        // arrived at. There it is placed 0.9 against
                        // reprojected 0.5; the 3D counterpart is solved here
                        // against predicted by a model. One reading for the
                        // whole column -- the hue says which keypoint, the
                        // strength says how much of it is yours.
                        if (k3.exist && node < (int)skeleton.node_colors.size()) {
                            ImVec4 c = skeleton.node_colors[node];
                            c.w = k3.is_predicted() ? 0.5f : 0.9f;
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg,
                                                   ImGui::GetColorU32(c));
                        }
                    }
                    ImGui::PopID();
                }

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
                                        instance_or_first(annotations.at(current_frame_num), ctx.active_instance);
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
            //    the keypoints table so it never clashes with the image-view
            //    'A' (previous active keypoint), which only fires over a plot. ──
            if ((ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows) ||
                 ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) &&
                keys::pressed(keys::Sc::SelectAllKeypoints)) {
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
                auto &fa = instance_or_first(annotations.at(current_frame_num), ctx.active_instance);
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
            if (keys::pressed(keys::Sc::CopyKeypoints)) {
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
                        kc, instance_or_first(annotations.at(current_frame_num), ctx.active_instance),
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
            if (keys::pressed(keys::Sc::PasteKeypoints)) {
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
}
