#pragma once

// bout_table.h — Bout Inspector table panel (Green-style trial table)
//
// Sortable, filterable, keyboard-navigable table of walking bouts.
// Selecting a bout seeks RED's video to the bout's frame range.

#include "../bout_inspector.h"
#include "../app_context.h"
#include "../utils.h"
#include "imgui.h"

enum BoutTableColumn {
    BCol_ID, BCol_Start, BCol_End, BCol_Duration, BCol_Speed,
    BCol_MaxSpeed, BCol_Confidence, BCol_Height, BCol_IK, BCol_Status,
    BCol_COUNT
};

inline void DrawBoutTable(AppContext &ctx) {
    auto &state = ctx.bout_state;
    auto &db = ctx.bout_db;
    if (!state.active || !db.is_open()) return;

    // Seek video to a bout's start frame
    auto select_bout = [&](const BoutRow &b, int row) {
        state.selected_bout_id = b.id;
        state.selected_row = row;
        state.clamp_start = b.start_frame;
        state.clamp_end = b.end_frame;
        ctx.current_frame_num = b.start_frame;
        if (ctx.scene && ctx.dc_context)
            seek_all_cameras(ctx.scene, b.start_frame,
                             ctx.dc_context->video_fps, ctx.ps, false);
        // Prefetch prediction pages for this bout
        ctx.bout_preds.prefetch_range(b.start_frame, b.end_frame);
    };

    // Refresh data if filters changed
    if (state.filters.dirty) {
        state.filtered_bouts = db.query_bouts(state.filters);
        state.filters.dirty = false;
        // Preserve selection if possible
        state.selected_row = -1;
        for (int i = 0; i < (int)state.filtered_bouts.size(); ++i) {
            if (state.filtered_bouts[i].id == state.selected_bout_id) {
                state.selected_row = i;
                break;
            }
        }
    }

    auto &bouts = state.filtered_bouts;

    ImGui::Begin("Bout Inspector");

    // ── Filter bar ────────────────────────────────────────────────
    {
        ImGui::Text("Bouts: %d", (int)bouts.size());
        ImGui::SameLine();

        // Status filter
        const char *status_labels[] = {"All", "Pending", "Accepted", "Rejected"};
        ImGui::SetNextItemWidth(100);
        if (ImGui::Combo("##status_filter", &state.filters.status_filter,
                          status_labels, 4)) {
            state.filters.dirty = true;
        }

        ImGui::SameLine();
        ImGui::SetNextItemWidth(80);
        if (ImGui::DragFloat("min dur", &state.filters.min_duration_s,
                              0.01f, 0.0f, 5.0f, "%.2fs")) {
            state.filters.dirty = true;
        }

        ImGui::SameLine();
        ImGui::Checkbox("Clamp", &state.clamp_playback);

        // Status summary
        int n_pending = 0, n_accepted = 0, n_rejected = 0;
        for (auto &b : bouts) {
            if (b.status == BoutPending) n_pending++;
            else if (b.status == BoutAccepted) n_accepted++;
            else if (b.status == BoutRejected) n_rejected++;
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.6f, 0.6f, 0.6f, 1), "P:%d", n_pending);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1), "A:%d", n_accepted);
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1), "R:%d", n_rejected);
    }

    // ── Table ─────────────────────────────────────────────────────
    ImGuiTableFlags flags = ImGuiTableFlags_Sortable |
                            ImGuiTableFlags_RowBg |
                            ImGuiTableFlags_ScrollY |
                            ImGuiTableFlags_Resizable |
                            ImGuiTableFlags_Reorderable |
                            ImGuiTableFlags_Hideable |
                            ImGuiTableFlags_BordersOuter;

    if (ImGui::BeginTable("bout_table", BCol_COUNT, flags,
                           ImVec2(0, ImGui::GetContentRegionAvail().y - 30))) {
        ImGui::TableSetupScrollFreeze(0, 1);  // freeze header
        ImGui::TableSetupColumn("ID",     ImGuiTableColumnFlags_DefaultSort);
        ImGui::TableSetupColumn("Start");
        ImGui::TableSetupColumn("End");
        ImGui::TableSetupColumn("Dur(s)", ImGuiTableColumnFlags_DefaultSort);
        ImGui::TableSetupColumn("Speed");
        ImGui::TableSetupColumn("MaxSpd", ImGuiTableColumnFlags_DefaultHide);
        ImGui::TableSetupColumn("Conf");
        ImGui::TableSetupColumn("Height", ImGuiTableColumnFlags_DefaultHide);
        ImGui::TableSetupColumn("IK(mm)");
        ImGui::TableSetupColumn("Status");
        ImGui::TableHeadersRow();

        // Handle sorting
        if (ImGuiTableSortSpecs *sorts = ImGui::TableGetSortSpecs()) {
            if (sorts->SpecsDirty && sorts->SpecsCount > 0) {
                auto spec = sorts->Specs[0];
                bool asc = (spec.SortDirection == ImGuiSortDirection_Ascending);
                std::sort(bouts.begin(), bouts.end(),
                    [&](const BoutRow &a, const BoutRow &b) -> bool {
                        float va = 0, vb = 0;
                        switch (spec.ColumnIndex) {
                        case BCol_ID:         va = (float)a.id; vb = (float)b.id; break;
                        case BCol_Start:      va = (float)a.start_frame; vb = (float)b.start_frame; break;
                        case BCol_End:        va = (float)a.end_frame; vb = (float)b.end_frame; break;
                        case BCol_Duration:   va = a.duration_s; vb = b.duration_s; break;
                        case BCol_Speed:      va = a.mean_speed; vb = b.mean_speed; break;
                        case BCol_MaxSpeed:   va = a.max_speed; vb = b.max_speed; break;
                        case BCol_Confidence: va = a.mean_confidence; vb = b.mean_confidence; break;
                        case BCol_Height:     va = a.scut_z_mean; vb = b.scut_z_mean; break;
                        case BCol_IK:         va = a.ik_mean_mm; vb = b.ik_mean_mm; break;
                        case BCol_Status:     va = (float)a.status; vb = (float)b.status; break;
                        }
                        // NaN-safe
                        if (std::isnan(va) && std::isnan(vb)) return false;
                        if (std::isnan(va)) return false;  // NaN last
                        if (std::isnan(vb)) return true;
                        return asc ? (va < vb) : (va > vb);
                    });
                sorts->SpecsDirty = false;
                // Update selected_row after sort
                state.selected_row = -1;
                for (int i = 0; i < (int)bouts.size(); ++i) {
                    if (bouts[i].id == state.selected_bout_id) {
                        state.selected_row = i;
                        break;
                    }
                }
            }
        }

        // Rows with virtual scrolling
        ImGuiListClipper clipper;
        clipper.Begin((int)bouts.size());
        while (clipper.Step()) {
            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                auto &b = bouts[row];
                bool is_selected = (row == state.selected_row);

                ImGui::TableNextRow();

                // Row color by status
                if (b.status == BoutAccepted)
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                        ImGui::GetColorU32(ImVec4(0.15f, 0.4f, 0.15f, 0.3f)));
                else if (b.status == BoutRejected)
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                        ImGui::GetColorU32(ImVec4(0.4f, 0.15f, 0.15f, 0.3f)));

                if (is_selected) {
                    ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                        ImGui::GetColorU32(ImVec4(0.9f, 0.6f, 0.2f, 0.4f)));
                }

                // ID column with Selectable
                ImGui::TableSetColumnIndex(BCol_ID);
                char label[16];
                snprintf(label, sizeof(label), "%d", b.id);

                if (is_selected) {
                    ImGui::PushStyleColor(ImGuiCol_Header,        ImVec4(0.9f, 0.6f, 0.2f, 0.4f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderHovered,  ImVec4(0.9f, 0.6f, 0.2f, 0.5f));
                    ImGui::PushStyleColor(ImGuiCol_HeaderActive,   ImVec4(0.9f, 0.6f, 0.2f, 0.55f));
                }

                if (ImGui::Selectable(label, is_selected,
                        ImGuiSelectableFlags_SpanAllColumns)) {
                    select_bout(b, row);
                }

                if (is_selected) ImGui::PopStyleColor(3);

                // Data columns
                ImGui::TableSetColumnIndex(BCol_Start);
                ImGui::Text("%d", b.start_frame);

                ImGui::TableSetColumnIndex(BCol_End);
                ImGui::Text("%d", b.end_frame);

                ImGui::TableSetColumnIndex(BCol_Duration);
                ImGui::Text("%.2f", b.duration_s);

                ImGui::TableSetColumnIndex(BCol_Speed);
                ImGui::Text("%.1f", b.mean_speed);

                ImGui::TableSetColumnIndex(BCol_MaxSpeed);
                ImGui::Text("%.1f", b.max_speed);

                ImGui::TableSetColumnIndex(BCol_Confidence);
                if (b.mean_confidence > 0)
                    ImGui::Text("%.2f", b.mean_confidence);
                else
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "-");

                ImGui::TableSetColumnIndex(BCol_Height);
                ImGui::Text("%.2f", b.scut_z_mean);

                ImGui::TableSetColumnIndex(BCol_IK);
                if (std::isfinite(b.ik_mean_mm)) {
                    ImVec4 col = b.ik_mean_mm < 0.5f
                        ? ImVec4(0.4f, 0.9f, 0.4f, 1) : ImVec4(0.9f, 0.7f, 0.3f, 1);
                    ImGui::TextColored(col, "%.2f", b.ik_mean_mm);
                } else {
                    ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1), "-");
                }

                ImGui::TableSetColumnIndex(BCol_Status);
                if (b.status == BoutAccepted)
                    ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.4f, 1), "OK");
                else if (b.status == BoutRejected)
                    ImGui::TextColored(ImVec4(0.9f, 0.4f, 0.4f, 1), "REJ");
                else
                    ImGui::TextColored(ImVec4(0.7f, 0.7f, 0.7f, 1), "...");
            }
        }

        // ── Keyboard navigation ───────────────────────────────────
        if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
            !ImGui::GetIO().WantTextInput) {
            int new_row = state.selected_row;

            if (ImGui::IsKeyPressed(ImGuiKey_DownArrow) || ImGui::IsKeyPressed(ImGuiKey_J))
                new_row = std::min(state.selected_row + 1, (int)bouts.size() - 1);
            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow) || ImGui::IsKeyPressed(ImGuiKey_K))
                new_row = std::max(state.selected_row - 1, 0);
            if (ImGui::IsKeyPressed(ImGuiKey_PageDown))
                new_row = std::min(state.selected_row + 20, (int)bouts.size() - 1);
            if (ImGui::IsKeyPressed(ImGuiKey_PageUp))
                new_row = std::max(state.selected_row - 20, 0);
            if (ImGui::IsKeyPressed(ImGuiKey_Home))
                new_row = 0;
            if (ImGui::IsKeyPressed(ImGuiKey_End))
                new_row = (int)bouts.size() - 1;

            if (new_row != state.selected_row && new_row >= 0 &&
                new_row < (int)bouts.size()) {
                select_bout(bouts[new_row], new_row);

                // Auto-scroll table to show selected row
                float row_height = ImGui::GetTextLineHeightWithSpacing();
                ImGui::SetScrollY(new_row * row_height -
                                  ImGui::GetWindowHeight() * 0.4f);
            }

            // Accept/Reject shortcuts
            if (state.selected_row >= 0 && state.selected_row < (int)bouts.size()) {
                auto &b = bouts[state.selected_row];
                if (ImGui::IsKeyPressed(ImGuiKey_Enter) ||
                    ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
                    b.status = BoutAccepted;
                    db.update_status(b.id, BoutAccepted);
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Delete) ||
                    ImGui::IsKeyPressed(ImGuiKey_Backspace)) {
                    b.status = BoutRejected;
                    db.update_status(b.id, BoutRejected);
                }
            }
        }

        ImGui::EndTable();
    }

    // ── Bottom bar: selected bout info ────────────────────────────
    if (state.selected_row >= 0 && state.selected_row < (int)bouts.size()) {
        auto &b = bouts[state.selected_row];
        ImGui::Text("Bout %d | %d-%d | %.2fs | %.1f mm/s | Enter=Accept Delete=Reject",
                     b.id, b.start_frame, b.end_frame, b.duration_s, b.mean_speed);
    }

    ImGui::End();
}

