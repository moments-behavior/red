#pragma once
// pump_events_window.h — "Pump Events" window: pumpctl dispense timings placed
// on the video timeline, with click-to-seek and next/previous jumps.
//
// The parsing and the timestamp -> frame mapping live in pump_events_core.h;
// this file is UI plus the glue that pulls the reference camera's frame
// timestamps out of the recording folder.
//
// The list is loaded automatically whenever the media folder changes (pumpctl
// can be told to write its log straight into orange's recording folder), and a
// log elsewhere can be pointed at explicitly with Browse.
//
// Seeking follows the house convention: the table only raises seek_requested /
// seek_frame, and the main loop performs the seek after panels.drawAll().

#include "app_context.h"
#include "global.h"
#include "gui/panel.h"
#include "imgui.h"
#include "pump_events_core.h"
#include "sync_plan.h"
#include <ImGuiFileDialog.h>
#include <misc/cpp/imgui_stdlib.h>

#include <algorithm>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

struct PumpEventsState {
    bool show = false;

    // Mirrored to/from ProjectManager so they persist in the .redproj; see
    // pump_events_sync_project(). Held here too so the widgets have a stable
    // address and the panel works before a project exists.
    std::string log_path;      // explicit override; empty = auto-discovered
    float offset_ms = 0.0f;
    bool show_pulls = false;   // a "pull" is a draw, not a dispense
    std::string synced_project;  // project_path these were adopted from

    std::vector<pump_events::PumpEvent> events;
    pump_events::LoadReport report;
    pump_events::ClockAxis axis = pump_events::ClockAxis::None;

    // Reference camera's frame timestamps, kept because sync_fix_load_plan
    // discards its copy (media_loader.h) and we re-resolve on every nudge.
    std::vector<int64_t> ref_ns;
    std::vector<int64_t> ref_sys_ns;
    int64_t period_ns = 0;
    std::string ref_cam;

    // Guards against reloading every frame; also records what the current
    // `frame` values mean, since the sync fix changes that.
    std::string loaded_for_folder;
    bool loaded_fix_on = false;
    bool reload_requested = false;

    std::set<std::string> hidden_pumps;
    std::vector<std::string> sources;  // log files actually read
    std::string status;
    int placed = 0;

    // Consumed by the main loop.
    bool seek_requested = false;
    int  seek_frame = 0;
};

// Row visibility. Kept as a free function so the menu-bar jumps and the table
// agree on exactly which events are navigable.
inline bool pump_event_visible(const PumpEventsState &st,
                               const pump_events::PumpEvent &e) {
    if (!st.show_pulls && !e.is_dispense()) return false;
    if (st.hidden_pumps.count(e.pump)) return false;
    return true;
}

// Map an mp4 index from the reference camera into whatever coordinate system
// the timeline is currently in. With the sync fix ON, frame numbers are
// canonical trigger slots, not mp4 indices (see transport_bar.h).
inline int pump_events_to_timeline(const AppContext &ctx, int mp4_index) {
    if (mp4_index < 0) return -1;
    const bool fix_on = ctx.dc_context && ctx.dc_context->sync_fix_active.load();
    if (!fix_on) return mp4_index;
    if (!g_sync_fix.plan.usable() || ctx.pm.camera_names.empty()) return mp4_index;
    const sync_plan::SyncCam *cam0 = g_sync_fix.plan.cam(ctx.pm.camera_names[0]);
    if (!cam0) return mp4_index;
    return (int)cam0->slot_of_pos(mp4_index);
}

// Re-run the mapping only (cheap: a binary search per event). Used when the
// offset nudge moves or the sync fix toggles.
inline void pump_events_resolve(PumpEventsState &st, const AppContext &ctx) {
    pump_events::ResolveInputs in;
    in.ref_ns = &st.ref_ns;
    in.ref_sys_ns = &st.ref_sys_ns;
    in.axis = st.axis;
    in.offset_ns = (int64_t)((double)st.offset_ms * 1e6);
    in.tol_ns = st.period_ns > 0 ? st.period_ns : 0;

    st.placed = pump_events::resolve(st.events, in);

    // Events resolve to reference-camera mp4 indices; translate to the active
    // timeline afterwards so the core stays free of playback concerns.
    for (auto &e : st.events) {
        e.frame = pump_events_to_timeline(ctx, e.frame);
        e.end_frame = pump_events_to_timeline(ctx, e.end_frame);
    }
    st.loaded_fix_on = ctx.dc_context && ctx.dc_context->sync_fix_active.load();
}

// Full load: find the logs, read them, pull the reference camera's timestamps
// out of the recording folder, then resolve.
inline void pump_events_load(PumpEventsState &st, const AppContext &ctx) {
    namespace pe = pump_events;
    namespace ct = camera_timestamps;

    st.events.clear();
    st.sources.clear();
    st.ref_ns.clear();
    st.ref_sys_ns.clear();
    st.report = pe::LoadReport{};
    st.axis = pe::ClockAxis::None;
    st.period_ns = 0;
    st.placed = 0;
    st.status.clear();
    st.loaded_for_folder = ctx.pm.media_folder;

    // 1) Which log(s)?
    if (!st.log_path.empty())
        st.sources.push_back(st.log_path);
    else
        st.sources = pe::discover_logs_with_parent(ctx.pm.media_folder);

    if (st.sources.empty()) {
        st.status = "No pumpctl log found in the recording folder.";
        return;
    }
    st.events = pe::load_files(st.sources, st.report);
    if (st.events.empty()) {
        st.status = st.report.error.empty()
                        ? "Log contains no dispenses."
                        : st.report.error;
        return;
    }

    // 2) Frame timestamps for the reference camera. Mirrors media_loader.h:
    // look next to the videos, then one level up.
    if (ctx.pm.camera_names.empty()) {
        st.status = "No cameras loaded — cannot place dispenses on the timeline.";
        return;
    }
    std::vector<std::string> tokens;
    for (const auto &name : ctx.pm.camera_names)
        tokens.push_back(sync_plan::detail::cam_token(name));

    const std::string lab_pattern = "cam{cam}_timestamps_*.csv";
    ct::CameraTimestamps ts = ct::load(ctx.pm.media_folder, tokens, lab_pattern);
    if (ts.format == ct::Format::None) {
        std::filesystem::path parent =
            std::filesystem::path(ctx.pm.media_folder).parent_path();
        if (!parent.empty())
            ts = ct::load(parent.string(), tokens, lab_pattern);
    }
    if (ts.format == ct::Format::None) {
        st.status = "No frame timestamps for this recording — dispenses are "
                    "listed by clock time, but cannot be seeked to.";
        return;
    }

    const std::string ref_tok = sync_plan::detail::cam_token(ctx.pm.camera_names[0]);
    auto it = ts.frame_ns.find(ref_tok);
    if (it == ts.frame_ns.end() || it->second.empty()) {
        st.status = "No timestamps for reference camera " + ctx.pm.camera_names[0];
        return;
    }
    st.ref_cam = ctx.pm.camera_names[0];
    st.ref_ns = it->second;
    auto sit = ts.frame_sys_ns.find(ref_tok);
    if (sit != ts.frame_sys_ns.end()) st.ref_sys_ns = sit->second;
    st.period_ns = ct::frame_period_ns(st.ref_ns);
    st.axis = pe::pick_axis(ts.format);

    pump_events_resolve(st, ctx);

    char buf[256];
    snprintf(buf, sizeof(buf), "%d of %zu dispenses fall inside this recording.",
             st.placed, st.events.size());
    st.status = buf;
}

// Keep the panel's settings and the project's copy in step. On a project
// change we adopt the project's values (it is the thing that was just loaded);
// otherwise the panel is authoritative and we mirror into the project so any
// save picks the settings up. Returns true if adopting changed anything.
inline bool pump_events_sync_project(PumpEventsState &st, AppContext &ctx) {
    if (st.synced_project != ctx.pm.project_path) {
        st.synced_project = ctx.pm.project_path;
        const bool changed = st.log_path != ctx.pm.pump_log_path ||
                             st.offset_ms != ctx.pm.pump_offset_ms ||
                             st.show_pulls != ctx.pm.pump_show_pulls;
        st.log_path = ctx.pm.pump_log_path;
        st.offset_ms = ctx.pm.pump_offset_ms;
        st.show_pulls = ctx.pm.pump_show_pulls;
        return changed;
    }
    ctx.pm.pump_log_path = st.log_path;
    ctx.pm.pump_offset_ms = st.offset_ms;
    ctx.pm.pump_show_pulls = st.show_pulls;
    return false;
}

// Persist the settings now. Nothing else in the app writes the .redproj when
// these change, so the panel saves on edit — the same thing sync_fix_toggle
// does for its project-scoped flag (transport_bar.h).
inline void pump_events_save_project(AppContext &ctx) {
    if (ctx.pm.project_path.empty() || ctx.pm.project_name.empty()) return;
    std::string redproj =
        ctx.pm.project_path + "/" + ctx.pm.project_name + ".redproj";
    save_project_manager_json(ctx.pm, redproj);
}

// Auto-load when the media folder changes. Called from the main loop.
inline void pump_events_auto_load(PumpEventsState &st, AppContext &ctx) {
    if (pump_events_sync_project(st, ctx)) st.reload_requested = true;
    if (ctx.pm.media_folder.empty()) return;
    const bool fix_on = ctx.dc_context && ctx.dc_context->sync_fix_active.load();

    if (st.reload_requested || st.loaded_for_folder != ctx.pm.media_folder) {
        st.reload_requested = false;
        const bool had_events = !st.events.empty();
        pump_events_load(st, ctx);
        if (!st.events.empty())
            ctx.toasts.pushSuccess("Loaded " + std::to_string(st.events.size()) +
                                   " pump dispenses (" +
                                   std::to_string(st.placed) + " in range)");
        else if (had_events || !st.log_path.empty())
            ctx.toasts.pushError(st.status);
        return;
    }
    // The sync fix redefines what a frame number means; re-map without re-reading.
    if (!st.events.empty() && fix_on != st.loaded_fix_on)
        pump_events_resolve(st, ctx);
}

// Seek to the next/previous visible dispense relative to the current frame.
// Returns false when there is none, so the caller can beep/ignore.
inline bool pump_events_jump(PumpEventsState &st, int current_frame, bool forward) {
    auto keep = [&st](const pump_events::PumpEvent &e) {
        return pump_event_visible(st, e);
    };
    const int idx = forward
                        ? pump_events::next_after(st.events, current_frame, keep)
                        : pump_events::prev_before(st.events, current_frame, keep);
    if (idx < 0) return false;
    st.seek_requested = true;
    st.seek_frame = st.events[idx].frame;
    return true;
}

inline void DrawPumpEventsWindow(PumpEventsState &st, AppContext &ctx) {
    DrawPanel(
        "Pump Events",
        st.show,
        [&]() {
            // --- source ---
            ImGui::TextDisabled("pumpctl dispense log");
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "pump_dispense_*.jsonl, written by pumpctl. Found "
                    "automatically in the recording folder when pumpctl is set "
                    "to log into it.");

            ImGui::SetNextItemWidth(-160);
            ImGui::InputTextWithHint("##pump_log", "(auto-discovered)",
                                     &st.log_path);
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                st.reload_requested = true;
                pump_events_sync_project(st, ctx);
                pump_events_save_project(ctx);
            }
            ImGui::SameLine();
            if (ImGui::Button("Browse##pump_log")) {
                IGFD::FileDialogConfig cfg;
                cfg.countSelectionMax = 1;
                cfg.path = st.log_path.empty() ? ctx.pm.media_folder : st.log_path;
                cfg.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog(
                    "ChoosePumpLog", "Choose pumpctl dispense log",
                    "Pump log{.jsonl},.jsonl", cfg);
            }
            ImGui::SameLine();
            if (ImGui::Button("Reload##pump_log")) st.reload_requested = true;

            if (st.sources.size() > 1) {
                ImGui::TextDisabled("%zu log files merged", st.sources.size());
                if (ImGui::IsItemHovered()) {
                    std::string tip;
                    for (const auto &s : st.sources)
                        tip += std::filesystem::path(s).filename().string() + "\n";
                    ImGui::SetTooltip("%s", tip.c_str());
                }
            }

            // --- alignment ---
            ImGui::Separator();
            ImGui::SetNextItemWidth(160);
            if (ImGui::DragFloat("Offset (ms)", &st.offset_ms, 1.0f, -60000.0f,
                                 60000.0f, "%.0f"))
                pump_events_resolve(st, ctx);
            // Save once the drag ends, not on every frame of it.
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                pump_events_sync_project(st, ctx);
                pump_events_save_project(ctx);
            }
            ImGui::SameLine();
            if (ImGui::SmallButton("Reset##pump_off") && st.offset_ms != 0.0f) {
                st.offset_ms = 0.0f;
                pump_events_resolve(st, ctx);
                pump_events_sync_project(st, ctx);
                pump_events_save_project(ctx);
            }
            ImGui::SameLine();
            ImGui::TextDisabled("(%s)", pump_events::axis_name(st.axis));
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    "Dispenses are matched to frames on this clock.\n"
                    "PTP: pumpctl's ptp_ns against the camera timestamps — the "
                    "same hardware clock, so no offset should be needed.\n"
                    "Nudge only if the two were genuinely not synchronised.");

            // --- filters ---
            if (ImGui::Checkbox("Show pulls", &st.show_pulls)) {
                pump_events_sync_project(st, ctx);
                pump_events_save_project(ctx);
            }
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip("A \"pull\" draws liquid back into the "
                                  "syringe; it is not a dispense.");

            std::vector<std::string> pumps;
            for (const auto &e : st.events)
                if (std::find(pumps.begin(), pumps.end(), e.pump) == pumps.end())
                    pumps.push_back(e.pump);
            std::sort(pumps.begin(), pumps.end());
            for (const auto &p : pumps) {
                ImGui::SameLine();
                bool on = !st.hidden_pumps.count(p);
                if (ImGui::Checkbox((" " + p + "##pump_f").c_str(), &on)) {
                    if (on) st.hidden_pumps.erase(p);
                    else    st.hidden_pumps.insert(p);
                }
            }

            // --- status ---
            if (!st.status.empty()) {
                const bool warn = st.placed == 0;
                ImGui::TextColored(warn ? ImVec4(1.0f, 0.75f, 0.35f, 1.0f)
                                        : ImVec4(0.5f, 1.0f, 0.5f, 1.0f),
                                   "%s", st.status.c_str());
            }
            if (st.report.skipped_lines > 0)
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.35f, 1.0f),
                                   "%d unreadable line(s) skipped",
                                   st.report.skipped_lines);

            ImGui::Separator();

            // --- table ---
            ImGuiTableFlags tf = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                 ImGuiTableFlags_ScrollY |
                                 ImGuiTableFlags_SizingStretchProp;
            if (ImGui::BeginTable("##pump_tbl", 7, tf, ImVec2(0, 0))) {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn("#");
                ImGui::TableSetupColumn("Frame");
                ImGui::TableSetupColumn("Clock (UTC)");
                ImGui::TableSetupColumn("Pump");
                ImGui::TableSetupColumn("Volume");
                ImGui::TableSetupColumn("Est. dur");
                ImGui::TableSetupColumn("Source");
                ImGui::TableHeadersRow();

                int shown = 0;
                for (size_t i = 0; i < st.events.size(); ++i) {
                    const pump_events::PumpEvent &e = st.events[i];
                    if (!pump_event_visible(st, e)) continue;
                    ++shown;

                    ImGui::TableNextRow();
                    if (e.dry)
                        ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg0,
                                               IM_COL32(90, 50, 30, 120));

                    ImGui::TableSetColumnIndex(0);
                    char lbl[48];
                    snprintf(lbl, sizeof(lbl), "%d##pump%zu", shown, i);
                    const bool seekable = e.frame >= 0;
                    if (!seekable) ImGui::BeginDisabled();
                    if (ImGui::Selectable(lbl, false,
                                          ImGuiSelectableFlags_SpanAllColumns) &&
                        seekable) {
                        st.seek_requested = true;
                        st.seek_frame = e.frame;
                    }
                    if (!seekable) ImGui::EndDisabled();
                    if (ImGui::IsItemHovered() && !e.wire.empty())
                        ImGui::SetTooltip(
                            "%s%s%s\nsteps %d @ %d us\nwire: %s%s",
                            e.source.c_str(),
                            e.experiment.empty() ? "" : " · ",
                            e.experiment.c_str(), e.steps, e.delay_us,
                            e.wire.c_str(),
                            e.dry ? "\nDRY — syringe had nothing to give" : "");

                    ImGui::TableSetColumnIndex(1);
                    if (seekable) ImGui::Text("%d", e.frame);
                    else ImGui::TextDisabled("%s", e.has_ptp ? "outside" : "no time");

                    ImGui::TableSetColumnIndex(2);
                    ImGui::Text("%s", pump_events::format_clock(e).c_str());

                    ImGui::TableSetColumnIndex(3);
                    if (e.is_dispense()) ImGui::Text("%s", e.pump.c_str());
                    else ImGui::TextDisabled("%s pull", e.pump.c_str());

                    ImGui::TableSetColumnIndex(4);
                    ImGui::Text("%s", pump_events::format_volume(e).c_str());

                    ImGui::TableSetColumnIndex(5);
                    ImGui::Text("%.0f ms", e.estimated_actual_ms);
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            "Estimated from steps and step delay — pumpctl "
                            "records no completion event, so an interrupted "
                            "dispense still shows its full intended length.");

                    ImGui::TableSetColumnIndex(6);
                    ImGui::TextDisabled("%s", e.source.c_str());
                }
                ImGui::EndTable();
            }
        },
        // always_fn: the dialog must keep drawing even while the panel is shut.
        [&]() {
            if (ImGuiFileDialog::Instance()->Display("ChoosePumpLog",
                                                     ImGuiWindowFlags_NoCollapse,
                                                     ImVec2(680, 440))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    auto sel = ImGuiFileDialog::Instance()->GetSelection();
                    if (!sel.empty()) {
                        st.log_path = sel.begin()->second;
                        st.reload_requested = true;
                        pump_events_sync_project(st, ctx);
                        pump_events_save_project(ctx);
                    }
                }
                ImGuiFileDialog::Instance()->Close();
            }
        },
        ImVec2(620, 520));
}
