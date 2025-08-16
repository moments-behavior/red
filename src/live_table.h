#pragma once
#include <algorithm>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)
#include <string>
#include <vector>

struct LiveTable {
    std::vector<std::string> col_names = {"A", "B"};
    std::vector<std::vector<std::string>> rows = {std::vector<std::string>(2)};
    bool is_open = false;
};

inline void EnsureShape(LiveTable &t) {
    if (t.col_names.empty())
        t.col_names.push_back("Col 1");
    const int C = (int)t.col_names.size();
    for (auto &r : t.rows)
        r.resize(C);
    if (t.rows.empty())
        t.rows.emplace_back(C);
}

inline void InsertColumn(LiveTable &t, int at,
                         const std::string &name = "New") {
    at = std::clamp(at, 0, (int)t.col_names.size());
    t.col_names.insert(t.col_names.begin() + at, name);
    for (auto &r : t.rows)
        r.insert(r.begin() + at, std::string{});
}

inline void RemoveColumn(LiveTable &t, int at) {
    if (t.col_names.size() <= 1)
        return;
    t.col_names.erase(t.col_names.begin() + at);
    for (auto &r : t.rows)
        r.erase(r.begin() + at);
}

inline void InsertRow(LiveTable &t, int at) {
    at = std::clamp(at, 0, (int)t.rows.size());
    t.rows.insert(t.rows.begin() + at,
                  std::vector<std::string>(t.col_names.size()));
}

inline void RemoveRow(LiveTable &t, int at) {
    if (t.rows.empty())
        return;
    t.rows.erase(t.rows.begin() + at);
    if (t.rows.empty())
        t.rows.emplace_back(t.col_names.size());
}

inline void DrawLiveTable(LiveTable &t, const char *window_id = "Live Table") {
    if (!t.is_open)
        return;

    if (!ImGui::Begin(window_id, &t.is_open, ImGuiWindowFlags_None)) {
        ImGui::End();
        return;
    }

    EnsureShape(t);

    if (ImGui::Button("+ Row"))
        InsertRow(t, (int)t.rows.size());
    ImGui::SameLine();
    if (ImGui::Button("+ Col"))
        InsertColumn(t, (int)t.col_names.size());

    // ---- pending structural edits (applied safely after drawing) ----
    enum class RowOp { None, InsertAbove, InsertBelow, Delete };
    enum class ColOp { None, InsertLeft, InsertRight, Delete };
    RowOp pending_row = RowOp::None;
    int pending_row_idx = -1;

    ColOp pending_col = ColOp::None;
    int pending_col_idx = -1;

    const int C = (int)t.col_names.size();
    ImGuiTableFlags flags =
        ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
        ImGuiTableFlags_Hideable | ImGuiTableFlags_Borders |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp |
        ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;

    // Flat header background
    ImVec4 row_bg = ImGui::GetStyleColorVec4(ImGuiCol_TableRowBg);
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, row_bg);

    // Frameless InputText helpers
    auto PushFrameless = []() {
        ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_FrameBgActive, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
    };
    auto PopFrameless = []() {
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(3);
    };

    // Stronger hover/active highlight on the last item (cell)
    auto HighlightCellFromLastItem = []() {
        const bool hovered = ImGui::IsItemHovered(
            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
        const bool active = ImGui::IsItemActive();
        if (!hovered && !active)
            return;
        ImVec4 hov = ImGui::GetStyleColorVec4(ImGuiCol_HeaderHovered);
        ImVec4 act = ImGui::GetStyleColorVec4(ImGuiCol_HeaderActive);
        hov.x *= 1.1f;
        hov.y *= 1.1f;
        hov.z *= 1.1f;
        hov.w = 0.35f;
        act.x *= 1.15f;
        act.y *= 1.15f;
        act.z *= 1.15f;
        act.w = 0.45f;
        ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg,
                               ImGui::GetColorU32(active ? act : hov));
    };

    bool need_apply_and_return = false;

    if (ImGui::BeginTable(
            "##tbl", C + 1, flags,
            ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 20.0f))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 48.0f);
        for (int c = 0; c < C; ++c)
            ImGui::TableSetupColumn(t.col_names[c].c_str());

        // ---- Header ----
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("#");

        for (int c = 0; c < C; ++c) {
            ImGui::TableSetColumnIndex(c + 1);
            ImGui::PushID(c);

            // measure cell region
            ImGuiStyle &style = ImGui::GetStyle();
            const float cell_x0 = ImGui::GetCursorPosX();
            const float total_w = ImGui::GetContentRegionAvail().x;
            const float spacing = style.ItemInnerSpacing.x;

            // size for SmallButton("⋮")
            ImVec2 btn_sz = ImGui::CalcTextSize("⋮");
            btn_sz.x += style.FramePadding.x * 2.0f;
            btn_sz.y = ImGui::GetFrameHeight();

            const float input_w = ImMax(1.0f, total_w - btn_sz.x - spacing);

            // frameless editable header text
            PushFrameless();
            ImGui::SetNextItemWidth(input_w);
            ImGui::InputText("##colname", &t.col_names[c]);
            PopFrameless();
            HighlightCellFromLastItem();

            // right-click on header text opens popup
            if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
                ImGui::OpenPopup("col_menu");

            // right-align the ⋮ button in the header cell
            const float x_right = cell_x0 + ImMax(0.0f, total_w - btn_sz.x);
            ImGui::SameLine(0.0f, spacing);
            ImGui::SetCursorPosX(x_right);
            if (ImGui::SmallButton("⋮"))
                ImGui::OpenPopup("col_menu");

            // per-column popup (unique via PushID)
            if (ImGui::BeginPopup("col_menu")) {
                ImGui::InputText("Name", &t.col_names[c]);
                if (ImGui::MenuItem("Insert left")) {
                    pending_col = ColOp::InsertLeft;
                    pending_col_idx = c;
                    need_apply_and_return = true;
                }
                if (ImGui::MenuItem("Insert right")) {
                    pending_col = ColOp::InsertRight;
                    pending_col_idx = c;
                    need_apply_and_return = true;
                }
                if (ImGui::MenuItem("Delete", nullptr, false, C > 1)) {
                    pending_col = ColOp::Delete;
                    pending_col_idx = c;
                    need_apply_and_return = true;
                }
                ImGui::EndPopup();
            }

            ImGui::PopID();

            // if a column structure change was requested, stop drawing now
            if (need_apply_and_return)
                break;
        }

        // If column op requested, skip rendering rows this frame
        if (!need_apply_and_return) {
            // ---- Body ----
            for (int r = 0; r < (int)t.rows.size(); ++r) {
                ImGui::TableNextRow();

                // Row index: invisible button over text (no highlight)
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(r);

                ImGui::Text("%d", r);

                // overlay an invisible button exactly over the text rect
                ImVec2 rect_min = ImGui::GetItemRectMin();
                ImVec2 rect_max = ImGui::GetItemRectMax();
                ImVec2 rect_size =
                    ImVec2(rect_max.x - rect_min.x, rect_max.y - rect_min.y);

                ImVec2 saved_cursor = ImGui::GetCursorScreenPos();
                ImGui::SetCursorScreenPos(rect_min);
                if (ImGui::InvisibleButton("##row_btn", rect_size)) {
                    ImGui::OpenPopup("row_menu");
                }
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Click for row menu");
                ImGui::SetCursorScreenPos(saved_cursor);

                if (ImGui::BeginPopup("row_menu")) {
                    if (ImGui::MenuItem("Insert above")) {
                        pending_row = RowOp::InsertAbove;
                        pending_row_idx = r;
                        need_apply_and_return = true;
                    }
                    if (ImGui::MenuItem("Insert below")) {
                        pending_row = RowOp::InsertBelow;
                        pending_row_idx = r;
                        need_apply_and_return = true;
                    }
                    if (ImGui::MenuItem("Delete", nullptr, false,
                                        t.rows.size() > 1)) {
                        pending_row = RowOp::Delete;
                        pending_row_idx = r;
                        need_apply_and_return = true;
                    }
                    ImGui::EndPopup();
                }
                ImGui::PopID();

                // if a row structure change was requested, stop drawing now
                if (need_apply_and_return)
                    break;

                // Row cells
                for (int c = 0; c < C; ++c) {
                    ImGui::TableSetColumnIndex(c + 1);
                    ImGui::PushID(r * C + c);

                    PushFrameless();
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::InputText("##cell", &t.rows[r][c]);
                    PopFrameless();
                    HighlightCellFromLastItem();

                    ImGui::PopID();
                }
            }
        }

        ImGui::EndTable();
    }
    ImGui::PopStyleColor(); // TableHeaderBg

    // ---- Apply pending ops safely AFTER finishing the table draw ----
    if (need_apply_and_return) {
        // Close any open popup to avoid dangling ImGui state
        ImGui::CloseCurrentPopup();

        // Column ops
        if (pending_col != ColOp::None && pending_col_idx >= 0) {
            switch (pending_col) {
            case ColOp::InsertLeft:
                InsertColumn(t, pending_col_idx, "New");
                break;
            case ColOp::InsertRight:
                InsertColumn(t, pending_col_idx + 1, "New");
                break;
            case ColOp::Delete:
                RemoveColumn(t, pending_col_idx);
                break;
            default:
                break;
            }
        }

        // Row ops
        if (pending_row != RowOp::None && pending_row_idx >= 0) {
            switch (pending_row) {
            case RowOp::InsertAbove:
                InsertRow(t, pending_row_idx);
                break;
            case RowOp::InsertBelow:
                InsertRow(t, pending_row_idx + 1);
                break;
            case RowOp::Delete:
                RemoveRow(t, pending_row_idx);
                break;
            default:
                break;
            }
        }

        // We changed the table shape; end the window now.
        ImGui::End();
        return; // next frame will redraw with the new shape
    }

    ImGui::End();
}
