#pragma once
// collab_window.h -- the Collaboration panel.
//
// UI and glue only; the engine lives in src/collab_sync.h and the merge rules
// in src/collab_ops.h, both free of ImGui so they can be unit-tested. Same
// split as pump_events_core.h / pump_events_window.h.

#include <cstdio>
#include <string>
#include <vector>

#include "app_context.h"
#include "collab_sync.h"
#include "gui/gui_helpers.h"
#include "gui/panel.h"
#include "imgui.h"
#include <ImGuiFileDialog.h>

using CollabState = collab::CollabState;

namespace collab_ui {

inline std::string ago(int64_t then_ms) {
    if (then_ms <= 0) return "never";
    const int64_t d = (collab::now_ms() - then_ms) / 1000;
    if (d < 2) return "just now";
    if (d < 60) return std::to_string(d) + "s ago";
    if (d < 3600) return std::to_string(d / 60) + "m ago";
    if (d < 86400) return std::to_string(d / 3600) + "h ago";
    return std::to_string(d / 86400) + "d ago";
}

inline std::string short_peer(const std::string &p) {
    return p.size() > 8 ? p.substr(0, 8) : p;
}

// ── Setup / Sync ──

inline void tab_sync(CollabState &st, AppContext &ctx) {
    ImGui::TextDisabled("You are %s (%s)", st.display_name.c_str(),
                        short_peer(st.peer_id).c_str());
    ImGui::Separator();

    bool changed = false;

    char host[256];
    std::snprintf(host, sizeof(host), "%s", st.relay_host.c_str());
    ImGui::SetNextItemWidth(240);
    if (ImGui::InputText("Relay host", host, sizeof(host))) {
        st.relay_host = host;
        changed = true;
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    if (ImGui::InputInt("Port", &st.relay_port, 0, 0)) changed = true;

    char room[128];
    std::snprintf(room, sizeof(room), "%s", st.room.c_str());
    ImGui::SetNextItemWidth(240);
    if (ImGui::InputText("Room", room, sizeof(room))) {
        st.room = room;
        const auto it = ctx.user_settings.collab_room_secrets.find(st.room);
        st.psk = (it != ctx.user_settings.collab_room_secrets.end())
                     ? it->second
                     : std::string{};
        collab::save_room_settings(st, ctx.pm.project_path);
    }

    char psk[256];
    std::snprintf(psk, sizeof(psk), "%s", st.psk.c_str());
    ImGui::SetNextItemWidth(240);
    if (ImGui::InputText("Room secret", psk, sizeof(psk),
                         ImGuiInputTextFlags_Password)) {
        st.psk = psk;
        if (!st.room.empty()) {
            ctx.user_settings.collab_room_secrets[st.room] = st.psk;
            save_user_settings(ctx.user_settings);
        }
    }
    ImGui::SameLine();
    HelpMarker(
        "The shared secret for this room, as configured on the relay. Stored "
        "on this machine only (~/.config/red/user_settings.json, owner-read "
        "only) and never written into the project or sent to collaborators.");

    if (ImGui::Checkbox("Enable collaboration for this project", &st.enabled)) {
        st.status = st.enabled ? "Ready" : "Collaboration off for this project";
        changed = true;
    }

    if (ImGui::Checkbox("Auto-sync", &st.auto_sync)) {
        ctx.user_settings.collab_auto_sync = st.auto_sync;
        save_user_settings(ctx.user_settings);
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90);
    if (ImGui::InputInt("seconds", &st.auto_sync_seconds, 0, 0)) {
        if (st.auto_sync_seconds < 10) st.auto_sync_seconds = 10;
        ctx.user_settings.collab_auto_sync_seconds = st.auto_sync_seconds;
        save_user_settings(ctx.user_settings);
    }

    if (changed) {
        // Relay address doubles as the default for the next project, so it is
        // stored both machine-wide and per project.
        ctx.user_settings.collab_relay_host = st.relay_host;
        ctx.user_settings.collab_relay_port = st.relay_port;
        save_user_settings(ctx.user_settings);
        collab::save_room_settings(st, ctx.pm.project_path);
    }

    ImGui::Separator();

    const bool busy = st.syncing.load();
    ImGui::BeginDisabled(busy || !collab::collab_configured(st));
    if (ImGui::Button(busy ? "Syncing..." : "Sync now",
                      ImVec2(120, 0)))
        collab::collab_sync_now(st, ctx);
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Text("%d out / %d in", st.pending_out, st.last_merged_in);
    ImGui::SameLine();
    ImGui::TextDisabled("last sync %s", ago(st.last_sync_ms).c_str());

    ImGui::TextWrapped("%s", st.status.c_str());

    if (st.dirty_since_save) {
        ImGui::TextColored(ImVec4(0.95f, 0.75f, 0.2f, 1.0f),
                           "Merged edits are not saved yet -- press Ctrl+S to "
                           "write a label snapshot.");
    }

    ImGui::Separator();
    ImGui::TextDisabled(
        "Traffic is authenticated but NOT encrypted. If the annotations are\n"
        "sensitive, tunnel the relay:\n"
        "  ssh -N -L %d:localhost:%d user@relay-host",
        st.relay_port, st.relay_port);
}

// ── Peers ──

inline void tab_peers(CollabState &st) {
    if (st.peers.empty()) {
        ImGui::TextDisabled(
            "No peers seen yet. Presence updates on each sync.");
        return;
    }
    if (ImGui::BeginTable("##peers", 3,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Collaborator");
        ImGui::TableSetupColumn("Frame");
        ImGui::TableSetupColumn("Last seen");
        ImGui::TableHeadersRow();
        for (const auto &p : st.peers) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            const bool me = (p.peer == st.peer_id);
            ImGui::Text("%s%s", p.display_name.c_str(), me ? " (you)" : "");
            ImGui::TableNextColumn();
            if (p.current_frame >= 0) ImGui::Text("%lld",
                                                  (long long)p.current_frame);
            else ImGui::TextDisabled("-");
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", ago(p.last_seen_ms).c_str());
        }
        ImGui::EndTable();
    }
}

// ── Comments ──

inline void tab_comments(CollabState &st, AppContext &ctx) {
    ImGui::InputTextMultiline("##comment", st.comment_buf,
                              sizeof(st.comment_buf), ImVec2(-1, 60));
    ImGui::Checkbox("Pin to the current keypoint", &st.comment_pin_to_keypoint);
    ImGui::SameLine();

    const uint32_t frame = (uint32_t)ctx.current_frame_num;
    int camera = -1, node = -1;
    if (st.comment_pin_to_keypoint) {
        const auto it = ctx.annotations.find(frame);
        if (it != ctx.annotations.end() && !it->second.cameras.empty()) {
            camera = 0;
            node = (int)it->second.cameras[0].active_id;
        }
    }

    ImGui::BeginDisabled(st.comment_buf[0] == '\0' || !st.opened);
    if (ImGui::Button("Post")) {
        collab::collab_post_comment(st, ctx, st.comment_buf, frame, camera,
                                    node);
        st.comment_buf[0] = '\0';
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("on frame %u%s", frame,
                        (camera >= 0) ? " (pinned)" : "");

    ImGui::Separator();

    if (st.comments.empty()) {
        ImGui::TextDisabled("No comments yet.");
        return;
    }

    if (ImGui::BeginChild("##comments", ImVec2(0, 0))) {
        for (auto &kv : st.comments) {
            collab::Comment &c = kv.second;
            if (c.text.empty()) continue;   // resolve arrived before its post

            ImGui::PushID(c.id.c_str());
            if (c.resolved)
                ImGui::TextDisabled("%s - frame %u  (resolved)",
                                    c.author_name.c_str(), c.frame);
            else
                ImGui::Text("%s - frame %u", c.author_name.c_str(), c.frame);

            ImGui::SameLine();
            ImGui::TextDisabled("%s", ago(c.created_ms).c_str());

            ImGui::TextWrapped("%s", c.text.c_str());

            if (ImGui::SmallButton("Go to frame")) {
                st.seek_requested = true;
                st.seek_frame = (int)c.frame;
            }
            ImGui::SameLine();
            if (ImGui::SmallButton(c.resolved ? "Reopen" : "Resolve"))
                collab::collab_resolve_comment(st, ctx, c.id, !c.resolved);

            ImGui::Separator();
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
}

// ── History ──

inline void tab_history(CollabState &st, AppContext &ctx) {
    ImGui::TextDisabled(
        "Every edit is kept. When two people change the same thing, the newer\n"
        "one wins -- but the other is still here and can be restored.");
    ImGui::Separator();

    if (ImGui::Button("Use current frame")) st.hist_frame = ctx.current_frame_num;
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("Frame", &st.hist_frame, 0, 0);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(100);
    ImGui::Combo("Kind", &st.hist_kind, "2D keypoint\0" "3D keypoint\0");

    if (st.hist_kind == 0) {
        ImGui::SetNextItemWidth(100);
        ImGui::InputInt("Camera", &st.hist_camera, 0, 0);
        ImGui::SameLine();
    }
    ImGui::SetNextItemWidth(100);
    ImGui::InputInt("Node", &st.hist_node, 0, 0);

    if (st.hist_frame < 0) st.hist_frame = 0;
    if (st.hist_camera < 0) st.hist_camera = 0;
    if (st.hist_node < 0) st.hist_node = 0;

    collab::ObjKey key;
    key.cls = collab::obj_class_of(st.hist_kind == 0 ? collab::OpKind::Kp2dSet
                                                     : collab::OpKind::Kp3dSet);
    key.frame = (uint32_t)st.hist_frame;
    key.camera = (int16_t)(st.hist_kind == 0 ? st.hist_camera : -1);
    key.node = (int16_t)st.hist_node;

    const std::vector<collab::HistoryEntry> entries =
        collab::history_for(st, key);

    ImGui::Separator();
    if (entries.empty()) {
        ImGui::TextDisabled("No recorded edits for this keypoint.");
        return;
    }

    if (ImGui::BeginTable("##hist", 5,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
        ImGui::TableSetupColumn("Author");
        ImGui::TableSetupColumn("Value");
        ImGui::TableSetupColumn("When");
        ImGui::TableSetupColumn("Clock");
        ImGui::TableSetupColumn("");
        ImGui::TableHeadersRow();

        for (size_t i = 0; i < entries.size(); ++i) {
            const collab::Op &op = *entries[i].op;
            ImGui::TableNextRow();
            ImGui::PushID((int)i);

            ImGui::TableNextColumn();
            if (entries[i].winner)
                ImGui::Text("%s  [current]",
                            op.author.empty() ? short_peer(op.peer).c_str()
                                              : op.author.c_str());
            else
                ImGui::TextDisabled("%s",
                                    op.author.empty()
                                        ? short_peer(op.peer).c_str()
                                        : op.author.c_str());

            ImGui::TableNextColumn();
            const bool labeled = op.payload.value("labeled", false);
            const bool has3d = op.payload.contains("z");
            if (!labeled && !has3d) {
                ImGui::TextDisabled("cleared");
            } else if (has3d) {
                ImGui::Text("%.2f, %.2f, %.2f", op.payload.value("x", 0.0),
                            op.payload.value("y", 0.0),
                            op.payload.value("z", 0.0));
            } else {
                ImGui::Text("%.2f, %.2f", op.payload.value("x", 0.0),
                            op.payload.value("y", 0.0));
            }

            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", ago(op.wall_ms).c_str());

            ImGui::TableNextColumn();
            ImGui::TextDisabled("%llu", (unsigned long long)op.lamport);

            ImGui::TableNextColumn();
            if (!entries[i].winner) {
                if (ImGui::SmallButton("Restore"))
                    collab::collab_restore(st, ctx, op);
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
}

// ── Share / Clone ──

inline void tab_share(CollabState &st, AppContext &ctx) {
    ImGui::TextWrapped(
        "Share moves this project -- calibration, the most recent labels, the "
        "skeleton, and optionally the media -- into the room so another "
        "machine can clone it.");
    ImGui::Separator();

    ImGui::Checkbox("Include media (videos/images)", &st.plan_include_media);
    ImGui::SameLine();
    HelpMarker(
        "Media is matched by content hash, so a file the other machine already "
        "has transfers zero bytes. For a large rig the fastest path is to copy "
        "the videos once by USB or rsync and leave this on -- the clone will "
        "verify them and send nothing.\n\n"
        "Turn it off to move only the project, calibration, and labels.");

    const bool busy = st.transfer.running.load();

    ImGui::BeginDisabled(busy || !collab::collab_configured(st));
    if (ImGui::Button("Publish this project", ImVec2(180, 0)))
        collab::collab_share_project(st, ctx, st.plan_include_media);
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(busy || !collab::collab_configured(st));
    if (ImGui::Button("Check what's shared", ImVec2(180, 0))) {
        IGFD::FileDialogConfig cfg;
        cfg.countSelectionMax = 1;
        cfg.path = st.clone_dest.empty()
                       ? ctx.user_settings.default_project_root_path
                       : st.clone_dest;
        cfg.flags = ImGuiFileDialogFlags_Modal;
        // nullptr filter == directory picker: a clone lands in a folder.
        ImGuiFileDialog::Instance()->OpenDialog(
            "CollabCloneDest", "Folder to clone the shared project into",
            nullptr, cfg);
    }
    ImGui::EndDisabled();

    if (busy) {
        const uint64_t done = st.transfer.bytes_done.load();
        const uint64_t total = st.transfer.bytes_total.load();
        const float frac = total ? (float)((double)done / (double)total) : 0.0f;
        char overlay[128];
        std::snprintf(overlay, sizeof(overlay), "%s / %s  (%d/%d files)",
                      collab::format_bytes(done).c_str(),
                      collab::format_bytes(total).c_str(),
                      st.transfer.files_done.load(), st.transfer.files_total);
        ImGui::ProgressBar(frac, ImVec2(-1, 0), overlay);
        if (ImGui::Button("Cancel")) st.transfer.cancel.store(true);
        ImGui::TextDisabled(
            "Cancelling keeps what has already transferred; resuming picks up "
            "where it stopped.");
    }

    if (st.plan_valid) {
        ImGui::Separator();
        ImGui::Text("Shared by %s",
                    st.remote_manifest.created_by_name.empty()
                        ? "(unknown)"
                        : st.remote_manifest.created_by_name.c_str());
        ImGui::Text("%zu file(s): %zu already here, %zu to download (%s)",
                    st.plan.file_count(), st.plan.already_present.size(),
                    st.plan.needed.size(),
                    collab::format_bytes(st.plan.bytes_needed).c_str());
        if (st.plan.bytes_skipped > 0)
            ImGui::TextDisabled("%s of media skipped by the checkbox above",
                                collab::format_bytes(st.plan.bytes_skipped).c_str());

        if (!st.remote_manifest.binding.compatible_with(st.binding, nullptr)) {
            std::string why;
            st.remote_manifest.binding.compatible_with(st.binding, &why);
            ImGui::TextColored(ImVec4(0.95f, 0.4f, 0.3f, 1.0f),
                               "Cannot merge with this project: %s",
                               why.c_str());
            ImGui::TextDisabled(
                "Cloning into a NEW folder is still fine -- that gives you the "
                "shared project as its own project.");
        }

        ImGui::BeginDisabled(busy || st.clone_dest.empty());
        if (ImGui::Button("Download into that folder", ImVec2(220, 0)))
            collab::collab_clone(st, ctx, st.clone_dest);
        ImGui::EndDisabled();
        if (!st.clone_dest.empty())
            ImGui::TextDisabled("-> %s", st.clone_dest.c_str());
    }
}

}  // namespace collab_ui

// The file dialog handler must run even when the panel is hidden, so it lives
// in DrawPanel's always_fn -- the pattern pump_events_window.h established.
inline void DrawCollabWindow(CollabState &st, AppContext &ctx) {
    DrawPanel(
        "Collaboration", st.show,
        [&]() {
            if (!st.opened) {
                ImGui::TextDisabled("Open a project to collaborate on it.");
                return;
            }
            if (ImGui::BeginTabBar("##collab_tabs")) {
                if (ImGui::BeginTabItem("Sync")) {
                    collab_ui::tab_sync(st, ctx);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Peers")) {
                    collab_ui::tab_peers(st);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Comments")) {
                    collab_ui::tab_comments(st, ctx);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("History")) {
                    collab_ui::tab_history(st, ctx);
                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Share")) {
                    collab_ui::tab_share(st, ctx);
                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();
            }
        },
        [&]() {
            if (ImGuiFileDialog::Instance()->Display("CollabCloneDest",
                                                     ImGuiWindowFlags_NoCollapse,
                                                     ImVec2(700, 400))) {
                if (ImGuiFileDialog::Instance()->IsOk()) {
                    st.clone_dest =
                        ImGuiFileDialog::Instance()->GetCurrentPath();
                    collab::collab_fetch_manifest(st, ctx, st.clone_dest);
                }
                ImGuiFileDialog::Instance()->Close();
            }
        },
        ImVec2(760, 580));
}
