#pragma once
#include "render.h"
#include "utils.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <fstream>
#include <imgui.h>
#include <misc/cpp/imgui_stdlib.h> // for InputText(std::string&)
#include <string>
#include <vector>

struct LiveTable {
    std::vector<std::string> col_names = {"A", "B"};
    std::vector<std::vector<std::string>> rows = {std::vector<std::string>(2)};
    bool is_open = false;
    bool auto_width = false;
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

inline void OnCellShiftClick(int row, int col, const std::string &value,
                             render_scene *scene, double video_fps,
                             PlaybackState &ps, bool *video_loaded) {

    if (!(*video_loaded)) {
        return;
    }

    if (value.empty()) {
        printf("Shift+Click on cell (%d, %d): <empty>\n", row, col);
        return;
    }

    char *endptr = nullptr;
    errno = 0;
    long val = std::strtol(value.c_str(), &endptr, 10);

    if (endptr == value.c_str() || *endptr != '\0' || errno == ERANGE) {
        // Conversion failed or out of range
        printf("Shift+Click on cell (%d, %d): \"%s\" (not an integer)\n", row,
               col, value.c_str());
    } else {
        printf("Shift+Click on cell (%d, %d): int=%ld\n", row, col, val);
        seek_all_cameras(scene, val, video_fps, ps, true);
    }
}

inline void DrawLiveTable(LiveTable &t, const char *window_id,
                          render_scene *scene, double video_fps,
                          PlaybackState &ps, bool *video_loaded,
                          std::string &media_dir) {
    if (!t.is_open)
        return;
    if (!ImGui::Begin(window_id, &t.is_open, ImGuiWindowFlags_None)) {
        ImGui::End();
        return;
    }

    EnsureShape(t);

    // ---------------- Toolbar ----------------
    if (ImGui::Button("+ Row"))
        InsertRow(t, (int)t.rows.size());
    ImGui::SameLine();
    if (ImGui::Button("+ Col"))
        InsertColumn(t, (int)t.col_names.size());
    ImGui::SameLine();

    if (ImGui::Button("Load CSV")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = media_dir;
        ImGuiFileDialog::Instance()->OpenDialog("LoadCSVDlg", "Load CSV/TSV",
                                                ".csv,.tsv", cfg);
    }
    ImGui::SameLine();
    if (ImGui::Button("Save CSV")) {
        IGFD::FileDialogConfig cfg;
        cfg.path = media_dir;
        cfg.fileName = "spreadsheet.csv";
        cfg.flags = ImGuiFileDialogFlags_ConfirmOverwrite;
        ImGuiFileDialog::Instance()->OpenDialog("SaveCSVDlg", "Save CSV/TSV",
                                                ".csv,.tsv", cfg);
    }
    ImGui::SameLine();
    ImGui::Checkbox("Auto width", &t.auto_width);
    ImGui::Text(
        "Hold [Shift] and click a frame number to jump when video is loaded.");

    // -------- Deferred ops (avoid mutate-while-drawing) --------
    enum class RowOp { None, InsertAbove, InsertBelow, Delete };
    enum class ColOp { None, InsertLeft, InsertRight, Delete };
    RowOp pending_row = RowOp::None;
    int pending_row_idx = -1;
    ColOp pending_col = ColOp::None;
    int pending_col_idx = -1;
    bool need_apply_and_return = false;

    // -------- CSV LOAD dialog + parse (apply after draw) --------
    bool csv_apply = false;
    std::vector<std::string> csv_cols;
    std::vector<std::vector<std::string>> csv_rows;

    if (ImGuiFileDialog::Instance()->Display("LoadCSVDlg")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path = ImGuiFileDialog::Instance()->GetFilePathName();
            auto detect_delim = [](const std::string &s) -> char {
                size_t c = std::count(s.begin(), s.end(), ',');
                size_t t = std::count(s.begin(), s.end(), '\t');
                return (t > c) ? '\t' : ',';
            };
            auto split_csv_line = [](const std::string &line, char delim) {
                std::vector<std::string> out;
                out.reserve(16);
                std::string cur;
                bool inq = false;
                for (size_t i = 0; i < line.size(); ++i) {
                    char ch = line[i];
                    if (ch == '"') {
                        if (inq && i + 1 < line.size() && line[i + 1] == '"') {
                            cur.push_back('"');
                            ++i;
                        } else
                            inq = !inq;
                    } else if (ch == delim && !inq) {
                        out.push_back(cur);
                        cur.clear();
                    } else
                        cur.push_back(ch);
                }
                out.push_back(cur);
                return out;
            };
            auto trim_crlf = [](std::string &s) {
                while (!s.empty() && (s.back() == '\r' || s.back() == '\n'))
                    s.pop_back();
            };

            std::ifstream fin(path);
            if (fin) {
                std::string line;
                std::vector<std::vector<std::string>> tmp;
                char delim = ',';
                bool first = true;
                while (std::getline(fin, line)) {
                    trim_crlf(line);
                    if (first) {
                        delim = detect_delim(line);
                        first = false;
                    }
                    tmp.push_back(split_csv_line(line, delim));
                }
                fin.close();
                if (!tmp.empty()) {
                    csv_cols = tmp.front();
                    tmp.erase(tmp.begin());
                    const int C2 = (int)csv_cols.size();
                    for (auto &r : tmp)
                        r.resize(C2);
                    csv_rows = std::move(tmp);
                    csv_apply = true;
                }
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // -------- CSV SAVE dialog + write --------
    if (ImGuiFileDialog::Instance()->Display("SaveCSVDlg")) {
        if (ImGuiFileDialog::Instance()->IsOk()) {
            std::string path = ImGuiFileDialog::Instance()->GetFilePathName();
            auto ends_with = [](const std::string &s, const char *suf) {
                size_t n = std::strlen(suf);
                return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
            };
            char delim = (ends_with(path, ".tsv") || ends_with(path, ".TSV"))
                             ? '\t'
                             : ',';
            auto quote_field = [delim](const std::string &s) {
                bool need =
                    s.find('"') != std::string::npos ||
                    s.find('\n') != std::string::npos ||
                    s.find('\r') != std::string::npos ||
                    s.find(delim) != std::string::npos ||
                    (!s.empty() && (s.front() == ' ' || s.back() == ' '));
                if (!need)
                    return s;
                std::string out;
                out.reserve(s.size() + 2);
                out.push_back('"');
                for (char ch : s)
                    out += (ch == '"') ? "\"\"" : std::string(1, ch);
                out.push_back('"');
                return out;
            };
            std::ofstream fout(path, std::ios::binary);
            if (fout) {
                for (int c = 0; c < (int)t.col_names.size(); ++c) {
                    if (c)
                        fout.put(delim);
                    fout << quote_field(t.col_names[c]);
                }
                fout << "\n";
                const int Cw = (int)t.col_names.size();
                for (const auto &row : t.rows) {
                    for (int c = 0; c < Cw; ++c) {
                        if (c)
                            fout.put(delim);
                        const std::string v =
                            (c < (int)row.size()) ? row[c] : std::string{};
                        fout << quote_field(v);
                    }
                    fout << "\n";
                }
                fout.flush();
            }
        }
        ImGuiFileDialog::Instance()->Close();
    }

    // ---------------- Visual helpers ----------------
    ImGuiStyle &style = ImGui::GetStyle();
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

    // ---------------- Auto column sizing ----------------
    const int C = (int)t.col_names.size();

    // width of the header "⋮" button
    ImVec2 btn_sz = ImGui::CalcTextSize("⋮");
    btn_sz.x += style.FramePadding.x * 2.0f;
    btn_sz.y = ImGui::GetFrameHeight();

    std::vector<float> col_widths;
    if (!t.auto_width) {
        const float COL_MIN_W = 80.0f, COL_MAX_W = 600.0f;
        const float cell_pad = style.CellPadding.x * 2.0f;
        const float frame_pad = style.FramePadding.x * 2.0f;
        const float inner_sp = style.ItemInnerSpacing.x;
        col_widths.assign(C, COL_MIN_W);
        for (int c = 0; c < C; ++c) {
            float w = ImGui::CalcTextSize(t.col_names[c].c_str()).x;
            for (const auto &row : t.rows)
                if (c < (int)row.size())
                    w = std::max(w, ImGui::CalcTextSize(row[c].c_str()).x);
            w += cell_pad + frame_pad + inner_sp + btn_sz.x;
            col_widths[c] = ImClamp(w, COL_MIN_W, COL_MAX_W);
        }
    }

    // Flat header background
    ImVec4 row_bg = ImGui::GetStyleColorVec4(ImGuiCol_TableRowBg);
    ImGui::PushStyleColor(ImGuiCol_TableHeaderBg, row_bg);

    // ---------------- Table (fills remaining window height) ----------------
    ImGuiTableFlags flags =
        ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable |
        ImGuiTableFlags_Hideable | ImGuiTableFlags_Borders |
        ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp |
        ImGuiTableFlags_ScrollX | ImGuiTableFlags_ScrollY;

    // Use ImVec2(0,0) to consume all remaining content region
    if (ImGui::BeginTable("##tbl", C + 1, flags, ImVec2(0, 0))) {
        ImGui::TableSetupScrollFreeze(1, 1);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 48.0f);

        if (t.auto_width) {
            for (int c = 0; c < C; ++c)
                ImGui::TableSetupColumn(t.col_names[c].c_str(),
                                        ImGuiTableColumnFlags_WidthStretch,
                                        1.0f);
        } else {
            for (int c = 0; c < C; ++c)
                ImGui::TableSetupColumn(t.col_names[c].c_str(),
                                        ImGuiTableColumnFlags_WidthFixed,
                                        col_widths[c]);
        }

        // Header
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        ImGui::TableSetColumnIndex(0);
        ImGui::TextUnformatted("#");

        for (int c = 0; c < C; ++c) {
            ImGui::TableSetColumnIndex(c + 1);
            ImGui::PushID(c);

            const float cell_x0 = ImGui::GetCursorPosX();
            const float total_w = ImGui::GetContentRegionAvail().x;
            const float inner_sp = style.ItemInnerSpacing.x;
            const float input_w = ImMax(1.0f, total_w - btn_sz.x - inner_sp);

            PushFrameless();
            ImGui::SetNextItemWidth(input_w);
            ImGui::InputText("##colname", &t.col_names[c]);
            PopFrameless();
            HighlightCellFromLastItem();

            if (ImGui::IsItemClicked(ImGuiMouseButton_Right))
                ImGui::OpenPopup("col_menu");

            const float x_right = cell_x0 + ImMax(0.0f, total_w - btn_sz.x);
            ImGui::SameLine(0.0f, inner_sp);
            ImGui::SetCursorPosX(x_right);
            if (ImGui::SmallButton("⋮"))
                ImGui::OpenPopup("col_menu");

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
            if (need_apply_and_return)
                break;
        }

        // Body
        if (!need_apply_and_return) {
            for (int r = 0; r < (int)t.rows.size(); ++r) {
                ImGui::TableNextRow();

                // Index cell: invisible button + tooltip (no highlight)
                ImGui::TableSetColumnIndex(0);
                ImGui::PushID(r);
                ImGui::Text("%d", r);

                ImVec2 min = ImGui::GetItemRectMin(),
                       max = ImGui::GetItemRectMax();
                ImVec2 size(max.x - min.x, max.y - min.y);
                ImVec2 saved = ImGui::GetCursorScreenPos();
                ImGui::SetCursorScreenPos(min);
                if (ImGui::InvisibleButton("##row_btn", size))
                    ImGui::OpenPopup("row_menu");
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Click for row menu");
                ImGui::SetCursorScreenPos(saved);

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

                if (need_apply_and_return)
                    break;

                for (int c = 0; c < C; ++c) {
                    ImGui::TableSetColumnIndex(c + 1);
                    ImGui::PushID(r * C + c);

                    ImGuiTable *table = ImGui::GetCurrentTable();
                    ImRect cell_rect = ImGui::TableGetCellBgRect(table, c + 1);
                    ImGuiIO &io = ImGui::GetIO();

                    // Pre-hit test for Shift+click (no extra widget)
                    bool suppress_activation = false;
                    const bool hovered_cell = ImGui::IsMouseHoveringRect(
                        cell_rect.Min, cell_rect.Max, false);
                    if (io.KeyShift && hovered_cell &&
                        (ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                         ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))) {
                        OnCellShiftClick(r, c, t.rows[r][c], scene, video_fps,
                                         ps, video_loaded);
                        suppress_activation = true;
                        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    }

                    if (suppress_activation) {
                        ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
                        ImGui::PushStyleVar(ImGuiStyleVar_Alpha,
                                            1.0f); // keep visuals
                    }

                    PushFrameless();
                    ImGui::SetNextItemWidth(-FLT_MIN);
                    ImGui::InputText("##cell", &t.rows[r][c]); // stable height
                    PopFrameless();

                    if (suppress_activation) {
                        ImGui::PopStyleVar();
                        ImGui::PopItemFlag();
                    } else {
                        if (io.KeyShift && ImGui::IsItemHovered())
                            ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
                    }

                    HighlightCellFromLastItem();
                    ImGui::PopID();
                }
            }
        }

        ImGui::EndTable();
    }
    ImGui::PopStyleColor(); // TableHeaderBg

    // -------- Apply deferred ops / CSV replace --------
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
        ImGui::End();
        return;
    }
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
        ImGui::End();
        return;
    }
    if (csv_apply) {
        t.col_names = std::move(csv_cols);
        t.rows = std::move(csv_rows);
        EnsureShape(t);
        ImGui::End();
        return;
    }

    ImGui::End();
}
