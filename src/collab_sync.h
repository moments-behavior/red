#pragma once
// collab_sync.h -- the collaboration engine that sits between the annotation
// model and the relay.
//
// Threading follows the house patterns exactly:
//   * Long work runs on a detached std::thread with an atomic "busy" flag and
//     a shared_ptr result handoff, like gui/export_window.h.
//   * Every GUI-visible effect crosses back through ctx.deferred
//     (src/deferred_queue.h) and lands on the main thread. ImGui state and
//     ctx.annotations are never touched off the main thread.
//
// Capturing edits. Annotations are mutated by direct field writes at ~40 call
// sites, and ImPlot::DragPoint writes straight into the model's doubles
// through a pointer, so there is no single choke point to instrument. Instead
// a shadow copy is diffed on a timer: whatever changed becomes ops. The main
// thread does the (cheap) copy, the worker does the (expensive) diff.
//
// The echo trap. Applying a remote op mutates ctx.annotations, which the next
// diff would otherwise see as a fresh LOCAL edit and re-publish forever. The
// merge path closes that loop by diffing out any pending local edits FIRST,
// then applying the remote ops, then re-basing the shadow. See merge_ops().

#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "app_context.h"
#include "collab_bundle.h"
#include "collab_client.h"
#include "collab_io.h"
#include "collab_log.h"
#include "collab_ops.h"

namespace collab {

// How often the shadow diff runs. Fast enough that a crash loses at most a
// second of edits, slow enough that the copy is invisible.
constexpr double kCaptureIntervalSec = 1.0;

// ── Share / clone job status ──

enum class TransferKind { None, Share, Clone };

struct TransferJob {
    TransferKind kind = TransferKind::None;
    std::atomic<bool> running{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> cancel{false};
    std::atomic<uint64_t> bytes_done{0};
    std::atomic<uint64_t> bytes_total{0};
    std::atomic<int> files_done{0};
    int files_total = 0;
    // Written by the worker, read by the main thread once `finished` is set
    // with acquire ordering -- the export_window.h handoff.
    std::shared_ptr<std::string> result;
    bool ok = false;
};

struct CollabState {
    // ── UI ──
    bool show = false;

    // ── Per-project config (persisted through ProjectHandlerRegistry) ──
    std::string room;
    std::string relay_host;
    int         relay_port = 7373;
    bool        enabled = false;

    // ── Machine-local config (from UserSettings) ──
    std::string peer_id;
    std::string display_name;
    std::string psk;
    bool        auto_sync = true;
    int         auto_sync_seconds = 60;

    // ── Engine state (main thread unless noted) ──
    OpLog          log;
    OpFactory      factory;
    LwwState       lww;
    CommentStore   comments;
    std::vector<Op> history;      // every op seen, for the History tab
    RoomBinding    binding;
    bool           opened = false;

    // The shadow copy the diff compares against. Guarded because the capture
    // worker reads it while the main thread may be merging into it.
    AnnotationMap shadow;
    std::mutex    shadow_mu;

    // ── Status shown in the panel ──
    std::string status = "Not configured";
    std::string last_error;
    bool     last_sync_ok = false;
    int64_t  last_sync_ms = 0;
    int      pending_out = 0;
    int      last_merged_in = 0;
    int      merged_total = 0;
    bool     dirty_since_save = false;
    std::vector<PeerPresence> peers;

    std::atomic<bool> capturing{false};
    std::atomic<bool> syncing{false};

    // Bumped whenever the project changes. A worker captures the generation it
    // started under and its deferred callback checks it before touching
    // anything: without this, a sync still in flight when the user switches
    // projects would merge the OLD project's ops into the NEW project's
    // annotations.
    std::atomic<uint64_t> generation{1};

    TransferJob transfer;

    // Manifest fetched for the Share/Clone tab.
    bool     have_remote_manifest = false;
    Manifest remote_manifest;
    TransferPlan plan;
    bool     plan_include_media = true;
    bool     plan_valid = false;

    double last_capture_time = 0.0;
    double last_auto_sync_time = 0.0;

    // Comment composer
    char comment_buf[1024] = {0};
    bool comment_pin_to_keypoint = true;

    // Cross-panel actions go through request flags consumed by the main loop,
    // never by calling into playback directly -- the house rule stated in
    // gui/pump_events_window.h.
    bool seek_requested = false;
    int  seek_frame = 0;

    // Where a clone would land, chosen through the file dialog.
    std::string clone_dest;

    // History tab selection
    int hist_frame = 0;
    int hist_camera = 0;
    int hist_node = 0;
    int hist_kind = 0;   // 0 = 2D keypoint, 1 = 3D keypoint

    ~CollabState() { shutdown(); }

    // Waits for in-flight workers so the state can be destroyed or the project
    // switched. Detached threads capture `this`, so this must not return while
    // one is still running.
    void shutdown() {
        transfer.cancel.store(true);
        for (int i = 0; i < 6000 && (capturing.load() || syncing.load() ||
                                     transfer.running.load()); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

// =========================================================================
// Identity
// =========================================================================

// Ensures this machine has a stable peer id and a display name. The peer id is
// the LWW tiebreak, so it has to survive restarts -- regenerating it would
// make old and new ops from the same person resolve inconsistently.
inline void ensure_identity(AppContext &ctx) {
    bool changed = false;
    if (ctx.user_settings.collab_peer_id.empty()) {
        ctx.user_settings.collab_peer_id = random_hex(16);
        changed = true;
    }
    if (ctx.user_settings.collab_display_name.empty()) {
        ctx.user_settings.collab_display_name = "user-" +
            ctx.user_settings.collab_peer_id.substr(0, 6);
        changed = true;
    }
    if (changed) save_user_settings(ctx.user_settings);
}

// =========================================================================
// Per-project settings
// =========================================================================
//
// These live in <project>/.collab/room.json rather than in the .redproj.
//
// ProjectHandlerRegistry (project_handler.h) is the seam designed for this,
// but it is threaded through save_project_manager_json as an optional
// argument that no call site passes, so using it would mean touching all six
// existing save sites across unrelated panels. The .collab directory already
// exists for this feature, keeps sync bookkeeping out of the shared project
// file, and sidesteps the known gap where close_project() does not reset every
// serialized pm field. The room SECRET is never written here -- it stays in
// UserSettings, on this machine only.
struct RoomSettings {
    std::string room;
    std::string relay_host;
    int         relay_port = 7373;
    bool        enabled = false;
};

inline fs::path room_settings_path(const std::string &project_path) {
    return fs::path(project_path) / ".collab" / "room.json";
}

inline void load_room_settings(CollabState &st, const std::string &project_path) {
    std::string content;
    if (!read_file(room_settings_path(project_path), content, nullptr)) return;
    const nlohmann::json j =
        nlohmann::json::parse(content.begin(), content.end(), nullptr, false);
    if (j.is_discarded()) return;

    st.room = j.value("room", st.room);
    st.enabled = j.value("enabled", false);
    const std::string host = j.value("relay_host", std::string{});
    if (!host.empty()) st.relay_host = host;
    const int port = j.value("relay_port", 0);
    if (port > 0) st.relay_port = port;
}

// Follows the codebase convention of persisting immediately on change rather
// than tracking a dirty flag and offering an explicit save.
inline bool save_room_settings(const CollabState &st,
                               const std::string &project_path) {
    if (project_path.empty()) return false;
    nlohmann::json j;
    j["room"] = st.room;
    j["relay_host"] = st.relay_host;
    j["relay_port"] = st.relay_port;
    j["enabled"] = st.enabled;
    std::string err;
    return write_file_atomic(room_settings_path(project_path), j.dump(2), &err);
}

// =========================================================================
// Project lifecycle
// =========================================================================

// The shape this project imposes on a room. Cameras are compared by name AND
// order because annotations index them positionally.
inline RoomBinding binding_of(const AppContext &ctx) {
    RoomBinding b;
    b.skeleton_name = ctx.skeleton.name;
    b.num_nodes = ctx.skeleton.num_nodes;
    b.camera_names = ctx.pm.camera_names;
    return b;
}

// Replays the on-disk log so the History and Comments tabs have data, and
// bases the shadow on the annotations as loaded.
//
// The replay deliberately does NOT rebuild ctx.annotations: the CSV snapshot
// remains authoritative, exactly as chosen when this was designed as a
// sidecar. The log's job is attribution, history, and what to push.
inline void collab_open_project(CollabState &st, AppContext &ctx) {
    st.opened = false;
    st.history.clear();
    st.comments.clear();
    st.lww = LwwState{};
    st.peers.clear();
    st.have_remote_manifest = false;
    st.plan_valid = false;

    if (ctx.pm.project_path.empty()) {
        st.status = "No project open";
        return;
    }

    ensure_identity(ctx);
    st.peer_id = ctx.user_settings.collab_peer_id;
    st.display_name = ctx.user_settings.collab_display_name;
    st.relay_host = ctx.user_settings.collab_relay_host;
    st.relay_port = ctx.user_settings.collab_relay_port;
    st.auto_sync = ctx.user_settings.collab_auto_sync;
    st.auto_sync_seconds = ctx.user_settings.collab_auto_sync_seconds;
    st.binding = binding_of(ctx);

    std::string err;
    if (!st.log.open(ctx.pm.project_path, st.peer_id, &err)) {
        st.status = "Could not open .collab: " + err;
        st.last_error = err;
        return;
    }
    st.log.seed(st.factory);
    st.factory.author = st.display_name;

    LogStats ls;
    if (st.log.read_all(st.history, ls, &err)) {
        // Rebuild comments and the LWW winner table from history so the
        // panel opens with the full picture rather than only what arrives
        // after startup.
        AnnotationMap scratch;
        apply_ops(scratch, st.history, ctx.skeleton.num_nodes,
                  (int)ctx.pm.camera_names.size(), st.lww, &st.comments,
                  &st.factory.clock);
        if (ls.lines_skipped > 0)
            ctx.toasts.push("Collab log: skipped " +
                                std::to_string(ls.lines_skipped) +
                                " unreadable line(s)",
                            Toast::Warning);
    }

    {
        std::lock_guard<std::mutex> lock(st.shadow_mu);
        st.shadow = ctx.annotations;
    }

    std::vector<Op> unsent;
    if (st.log.read_local_since(st.log.cursor().sent_seq, unsent, &err))
        st.pending_out = (int)unsent.size();

    load_room_settings(st, ctx.pm.project_path);

    // The room secret is machine-local: it lives in UserSettings, keyed by
    // room, and is never written into the project or shipped to a peer.
    if (!st.room.empty()) {
        const auto it = ctx.user_settings.collab_room_secrets.find(st.room);
        st.psk = (it != ctx.user_settings.collab_room_secrets.end())
                     ? it->second
                     : std::string{};
    }

    st.opened = true;
    st.status = st.enabled ? "Ready" : "Collaboration off for this project";
}

inline void collab_close_project(CollabState &st) {
    // Invalidate in-flight workers BEFORE waiting on them, so anything that
    // lands late is discarded rather than applied to whatever comes next.
    st.generation.fetch_add(1);
    st.shutdown();
    st.opened = false;
    st.enabled = false;
    st.history.clear();
    st.comments.clear();
    st.peers.clear();
    st.lww = LwwState{};
    st.factory = OpFactory{};
    st.log = OpLog{};
    st.pending_out = 0;
    st.last_merged_in = 0;
    st.merged_total = 0;
    st.have_remote_manifest = false;
    st.plan_valid = false;
    st.status = "No project open";
    {
        std::lock_guard<std::mutex> lock(st.shadow_mu);
        st.shadow.clear();
    }
}

// =========================================================================
// Capture: turn local edits into ops
// =========================================================================

// Diffs `snapshot` against the shadow and appends whatever changed. Runs on a
// worker; the caller has already copied the annotations on the main thread.
inline int capture_into_log(CollabState &st,
                            const std::shared_ptr<AnnotationMap> &snapshot) {
    std::lock_guard<std::mutex> lock(st.shadow_mu);

    std::vector<Op> ops = diff(st.shadow, *snapshot, st.factory);
    if (ops.empty()) return 0;

    std::string err;
    if (!st.log.append_local(ops, &err)) {
        st.last_error = "Could not record edits: " + err;
        // Roll the factory back so the failed ops' sequence numbers are
        // reused rather than leaving a permanent gap.
        st.factory.next_seq -= ops.size();
        return -1;
    }

    st.shadow = *snapshot;
    for (Op &op : ops) st.history.push_back(std::move(op));
    return (int)st.history.size();
}

// Called every frame. Copies the annotations at most once a second and hands
// the copy to a worker.
inline void collab_tick_capture(CollabState &st, AppContext &ctx, double now) {
    if (!st.opened || !st.enabled) return;
    if (st.capturing.load()) return;
    if (now - st.last_capture_time < kCaptureIntervalSec) return;
    st.last_capture_time = now;

    // The copy is the only part that must happen on the main thread.
    auto snapshot = std::make_shared<AnnotationMap>(ctx.annotations);
    const uint64_t gen = st.generation.load();
    st.capturing.store(true);

    std::thread([&st, snapshot, gen]() {
        if (st.generation.load() == gen) capture_into_log(st, snapshot);
        st.capturing.store(false, std::memory_order_release);
    }).detach();
}

// =========================================================================
// Merge: apply remote ops without echoing them back
// =========================================================================

// Main thread only.
inline void merge_ops(CollabState &st, AppContext &ctx,
                      const std::vector<Op> &ops) {
    if (ops.empty()) return;
    std::lock_guard<std::mutex> lock(st.shadow_mu);

    // 1. Capture any local edits made since the last diff. If this were
    //    skipped, step 3 would fold them into the shadow and they would never
    //    become ops -- silent data loss.
    std::vector<Op> local = diff(st.shadow, ctx.annotations, st.factory);
    if (!local.empty()) {
        std::string err;
        if (st.log.append_local(local, &err)) {
            for (const Op &op : local) st.history.push_back(op);
        } else {
            st.factory.next_seq -= local.size();
            st.last_error = "Could not record edits before merging: " + err;
        }
    }

    // 2. Apply the remote ops to the live model under LWW.
    const ApplyStats s =
        apply_ops(ctx.annotations, ops, ctx.skeleton.num_nodes,
                  (int)ctx.pm.camera_names.size(), st.lww, &st.comments,
                  &st.factory.clock);

    // 3. Re-base the shadow. Now identical to the live model, so the next
    //    diff sees the merged edits as already-known rather than as new local
    //    work -- this is what stops the echo loop.
    st.shadow = ctx.annotations;

    for (const Op &op : ops) st.history.push_back(op);

    st.last_merged_in = s.applied;
    st.merged_total += s.applied;
    if (s.applied > 0) st.dirty_since_save = true;

    if (s.rejected > 0)
        ctx.toasts.push("Collab: dropped " + std::to_string(s.rejected) +
                            " op(s) that do not fit this skeleton",
                        Toast::Warning);
}

// =========================================================================
// Sync
// =========================================================================

// Everything one sync round trip produced. Filled on the worker, consumed on
// the main thread.
struct SyncResult {
    bool ok = false;
    std::string error;
    std::vector<Op> incoming;
    std::vector<PeerPresence> peers;
    int pushed = 0;
    int rejected = 0;
    uint64_t new_sent_seq = 0;
    uint64_t new_recv_seq = 0;
    int malformed = 0;
};

inline bool collab_configured(const CollabState &st) {
    return st.opened && st.enabled && !st.room.empty() &&
           !st.relay_host.empty() && !st.psk.empty();
}

// Kicks off one sync. Returns false if it could not be started.
inline bool collab_sync_now(CollabState &st, AppContext &ctx) {
    if (!collab_configured(st)) {
        st.status = "Set a relay host, room, and secret first";
        return false;
    }
    if (st.syncing.load()) return false;

    // Fold in anything edited since the last capture so a manual "Sync now"
    // never leaves recent work behind.
    {
        std::lock_guard<std::mutex> lock(st.shadow_mu);
        std::vector<Op> local = diff(st.shadow, ctx.annotations, st.factory);
        if (!local.empty()) {
            std::string err;
            if (st.log.append_local(local, &err)) {
                st.shadow = ctx.annotations;
                for (const Op &op : local) st.history.push_back(op);
            } else {
                st.factory.next_seq -= local.size();
            }
        }
    }

    RelayConfig cfg;
    cfg.host = st.relay_host;
    cfg.port = (uint16_t)st.relay_port;
    cfg.room = st.room;
    cfg.psk = st.psk;

    const RoomBinding binding = st.binding;
    const std::string peer = st.peer_id;
    const std::string name = st.display_name;
    const int64_t current_frame = ctx.current_frame_num;

    std::vector<Op> to_push;
    std::string err;
    st.log.read_local_since(st.log.cursor().sent_seq, to_push, &err);

    const uint64_t since = st.log.cursor().recv_seq;
    auto result = std::make_shared<SyncResult>();
    const uint64_t gen = st.generation.load();

    st.syncing.store(true);
    st.status = "Syncing...";

    std::thread([&st, &ctx, cfg, binding, peer, name, current_frame, to_push,
                 since, result, gen]() {
        RelaySession s;
        std::string e;

        do {
            if (!s.connect(cfg, peer, name, binding, &e)) break;

            uint64_t relay_seq = 0;
            if (!to_push.empty()) {
                if (!s.push_ops(to_push, result->pushed, result->rejected,
                                relay_seq, &e))
                    break;
                result->new_sent_seq = to_push.back().seq;
            }

            // Pull in batches until the relay says there is no more.
            uint64_t cursor = since;
            bool more = true;
            while (more) {
                std::vector<Op> batch;
                uint64_t high = cursor;
                int malformed = 0;
                if (!s.pull_ops(cursor, batch, high, more, malformed, &e))
                    break;
                result->malformed += malformed;
                for (Op &op : batch) result->incoming.push_back(std::move(op));
                if (high == cursor) break;   // no progress; stop rather than spin
                cursor = high;
            }
            if (!e.empty()) break;
            result->new_recv_seq = cursor;

            s.presence(current_frame, result->peers, &e);
            e.clear();  // presence is best-effort; never fail a sync over it

            s.bye();
            result->ok = true;
        } while (false);

        if (!result->ok) result->error = e;

        // Cleared here, not in the callback: shutdown() waits on this flag from
        // the main thread, which is the same thread that would have to run the
        // callback -- clearing it there would deadlock the wait.
        st.syncing.store(false, std::memory_order_release);

        // Land every effect on the main thread.
        ctx.deferred.enqueue([&st, &ctx, result, gen]() {
            if (st.generation.load() != gen) return;  // project changed underneath
            st.last_sync_ms = now_ms();
            st.last_sync_ok = result->ok;

            if (!result->ok) {
                st.last_error = result->error;
                st.status = "Sync failed: " + result->error;
                ctx.toasts.push("Collab sync failed: " + result->error,
                                Toast::Error);
                return;
            }

            // Only the ops other peers authored need merging; our own come
            // back from the relay and would be no-ops anyway.
            std::vector<Op> remote;
            for (const Op &op : result->incoming)
                if (op.peer != st.peer_id) remote.push_back(op);

            std::string err;
            st.log.append_remote(remote, &err);
            merge_ops(st, ctx, remote);

            st.log.cursor().recv_seq = result->new_recv_seq;
            if (result->new_sent_seq > st.log.cursor().sent_seq)
                st.log.cursor().sent_seq = result->new_sent_seq;
            st.log.cursor().last_sync_ms = st.last_sync_ms;
            st.log.save_cursor(&err);

            st.peers = result->peers;
            {
                // Recomputed rather than zeroed: the capture worker may have
                // appended more ops while this sync was in flight, and showing
                // "0 out" when work is still queued is misleading. Under the
                // engine lock because that worker writes the same log.
                std::lock_guard<std::mutex> lock(st.shadow_mu);
                std::vector<Op> still_unsent;
                std::string e2;
                if (st.log.read_local_since(st.log.cursor().sent_seq,
                                            still_unsent, &e2))
                    st.pending_out = (int)still_unsent.size();
            }
            st.last_error.clear();

            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "Synced: %d out, %d in%s", result->pushed,
                          st.last_merged_in,
                          result->malformed
                              ? ", some incoming ops were unreadable"
                              : "");
            st.status = buf;
            if (result->pushed > 0 || st.last_merged_in > 0)
                ctx.toasts.pushSuccess(buf);
        });
    }).detach();

    return true;
}

// Per-frame driver: capture on a timer, and auto-sync if enabled.
inline void collab_tick(CollabState &st, AppContext &ctx, double now) {
    if (!st.opened) return;
    collab_tick_capture(st, ctx, now);

    if (!st.auto_sync || !collab_configured(st)) return;
    if (st.syncing.load()) return;
    if (now - st.last_auto_sync_time < (double)st.auto_sync_seconds) return;
    st.last_auto_sync_time = now;
    collab_sync_now(st, ctx);
}

// =========================================================================
// Comments
// =========================================================================

inline void collab_post_comment(CollabState &st, AppContext &ctx,
                                const std::string &text, uint32_t frame,
                                int camera, int node) {
    if (!st.opened || text.empty()) return;

    // shadow_mu doubles as the engine lock: it is what serializes access to
    // st.factory, whose sequence counter and Lamport clock the capture worker
    // also advances. Without it two ops could be minted with the same seq.
    std::lock_guard<std::mutex> lock(st.shadow_mu);

    nlohmann::json body;
    body["text"] = text;
    const Op op = st.factory.make(OpKind::CommentPost, frame, camera, node,
                                  body, random_hex(8));

    std::string err;
    if (!st.log.append_local({op}, &err)) {
        st.factory.next_seq -= 1;
        ctx.toasts.pushError("Could not save comment: " + err);
        return;
    }
    st.history.push_back(op);

    // Reflect it locally straight away; the relay round trip only publishes
    // it to others.
    Comment &c = st.comments[op.obj_id];
    c.id = op.obj_id;
    c.author = op.peer;
    c.author_name = op.author;
    c.text = text;
    c.created_ms = op.wall_ms;
    c.frame = frame;
    c.camera = (int16_t)camera;
    c.node = (int16_t)node;
    st.lww.record(key_of(op), op);
    ++st.pending_out;
}

inline void collab_resolve_comment(CollabState &st, AppContext &ctx,
                                   const std::string &id, bool resolved) {
    if (!st.opened) return;
    std::lock_guard<std::mutex> lock(st.shadow_mu);

    const auto it = st.comments.find(id);
    if (it == st.comments.end()) return;

    nlohmann::json body;
    body["resolved"] = resolved;
    const Op op = st.factory.make(OpKind::CommentResolve, it->second.frame,
                                  it->second.camera, it->second.node, body, id);
    std::string err;
    if (!st.log.append_local({op}, &err)) {
        st.factory.next_seq -= 1;
        ctx.toasts.pushError("Could not save: " + err);
        return;
    }
    st.history.push_back(op);
    it->second.resolved = resolved;
    st.lww.record(key_of(op), op);
    ++st.pending_out;
}

// =========================================================================
// History: restore a superseded value
// =========================================================================

// Re-applies an old op's value as a NEW op at the current clock. History is
// never rewritten -- the restore is itself an attributable edit, so the
// timeline stays a true record of what happened.
inline void collab_restore(CollabState &st, AppContext &ctx, const Op &old) {
    if (!st.opened) return;
    std::lock_guard<std::mutex> lock(st.shadow_mu);

    const Op op = st.factory.make(old.kind, old.frame, old.camera, old.node,
                                  old.payload, old.obj_id);
    std::string err;
    if (!st.log.append_local({op}, &err)) {
        st.factory.next_seq -= 1;
        ctx.toasts.pushError("Could not record restore: " + err);
        return;
    }
    st.history.push_back(op);

    apply_ops(ctx.annotations, {op}, ctx.skeleton.num_nodes,
              (int)ctx.pm.camera_names.size(), st.lww, &st.comments,
              &st.factory.clock);
    st.shadow = ctx.annotations;
    st.dirty_since_save = true;
    ++st.pending_out;

    ctx.toasts.pushSuccess("Restored the value from " + op.author);
}

// Which op currently owns an object, and everything that lost to it.
struct HistoryEntry {
    const Op *op = nullptr;
    bool winner = false;
};

inline std::vector<HistoryEntry> history_for(const CollabState &st,
                                             const ObjKey &key) {
    std::vector<HistoryEntry> out;
    const Op *best = nullptr;
    for (const Op &op : st.history) {
        if (!(key_of(op) == key)) continue;
        out.push_back({&op, false});
        if (!best || op_newer(op, *best)) best = &op;
    }
    for (auto &e : out) e.winner = (e.op == best);
    std::sort(out.begin(), out.end(),
              [](const HistoryEntry &a, const HistoryEntry &b) {
                  return op_newer(*a.op, *b.op);
              });
    return out;
}

// =========================================================================
// Project sharing
// =========================================================================

// Maps a manifest entry back to the local file it came from.
inline std::string find_abs(const std::vector<FileRef> &files,
                            const std::string &rel) {
    for (const FileRef &f : files)
        if (f.rel_path == rel) return f.abs_path;
    return std::string{};
}

// Everything worth shipping, re-homed under a flat, self-contained layout.
//
// The .redproj is deliberately NOT included as a file: it holds absolute
// machine-local paths, so it travels as JSON inside the manifest and is
// rewritten on arrival. Only the most recent label snapshot is shipped --
// labeled_data/ accumulates one folder per save and never prunes, so sending
// all of them would multiply the transfer for no benefit.
inline std::vector<FileRef> collect_project_files(AppContext &ctx,
                                                  bool include_media) {
    std::vector<FileRef> files;

    if (!ctx.pm.calibration_folder.empty())
        scan_dir_prefixed(ctx.pm.calibration_folder, "calibration",
                          category::kCalibration, files);

    if (!ctx.pm.keypoints_root_folder.empty()) {
        std::string recent, err;
        if (AnnotationCSV::find_most_recent_labels(ctx.pm.keypoints_root_folder,
                                                   recent, err) > 0 &&
            !recent.empty()) {
            const std::string leaf =
                std::filesystem::path(recent).filename().string();
            scan_dir_prefixed(recent, "labeled_data/" + leaf, category::kLabels,
                              files);
        }
    }

    if (ctx.pm.load_skeleton_from_json && !ctx.pm.skeleton_file.empty() &&
        file_exists(ctx.pm.skeleton_file)) {
        FileRef r;
        r.abs_path = ctx.pm.skeleton_file;
        r.rel_path = "skeleton.json";
        r.category = category::kSkeleton;
        files.push_back(r);
    }

    if (include_media && !ctx.pm.media_folder.empty())
        scan_dir_prefixed(ctx.pm.media_folder, "media", category::kMedia,
                          files);

    return files;
}

// Publishes this project to the room: manifest first, then only the blobs the
// relay does not already hold.
inline bool collab_share_project(CollabState &st, AppContext &ctx,
                                 bool include_media) {
    if (!collab_configured(st)) {
        st.status = "Set a relay host, room, and secret first";
        return false;
    }
    if (st.transfer.running.load()) return false;

    auto files = std::make_shared<std::vector<FileRef>>(
        collect_project_files(ctx, include_media));
    if (files->empty()) {
        ctx.toasts.push("Nothing to share -- this project has no calibration, "
                        "labels, or media on disk",
                        Toast::Warning);
        return false;
    }

    auto meta = std::make_shared<Manifest>();
    meta->project_name = ctx.pm.project_name;
    meta->binding = st.binding;
    meta->created_by = st.peer_id;
    meta->created_by_name = st.display_name;
    meta->created_ms = now_ms();
    {
        nlohmann::json pj;
        to_json(pj, ctx.pm);
        meta->project_json = pj;
    }

    RelayConfig cfg;
    cfg.host = st.relay_host;
    cfg.port = (uint16_t)st.relay_port;
    cfg.room = st.room;
    cfg.psk = st.psk;

    const RoomBinding binding = st.binding;
    const std::string peer = st.peer_id;
    const std::string name = st.display_name;

    st.transfer.kind = TransferKind::Share;
    st.transfer.running.store(true);
    st.transfer.finished.store(false);
    st.transfer.cancel.store(false);
    st.transfer.bytes_done.store(0);
    st.transfer.bytes_total.store(0);
    st.transfer.files_done.store(0);
    st.transfer.files_total = (int)files->size();
    st.transfer.result = std::make_shared<std::string>();
    st.status = "Hashing project files...";

    const uint64_t gen = st.generation.load();
    std::thread([&st, &ctx, cfg, binding, peer, name, files, meta, gen]() {
        std::string err;
        bool ok = false;

        do {
            int done = 0;
            const bool cancelled = st.transfer.cancel.load();
            if (!build_manifest(*files, *meta, &err, &done, &cancelled)) break;

            uint64_t total = 0;
            for (const auto &e : meta->entries) total += e.size;
            st.transfer.bytes_total.store(total);

            RelaySession s;
            if (!s.connect(cfg, peer, name, binding, &err)) break;
            if (!s.put_manifest(*meta, &err)) break;

            std::vector<std::string> hashes;
            for (const auto &e : meta->entries) hashes.push_back(e.sha256);
            std::vector<std::string> needed;
            if (!s.blobs_needed(hashes, needed, &err)) break;

            // Anything the relay already holds costs nothing -- this is what
            // makes re-sharing, or sharing a project that overlaps another,
            // nearly free.
            uint64_t skipped = 0;
            for (const auto &e : meta->entries)
                if (std::find(needed.begin(), needed.end(), e.sha256) ==
                    needed.end())
                    skipped += e.size;
            st.transfer.bytes_done.store(skipped);

            bool aborted = false;
            uint64_t base = skipped;
            for (const auto &e : meta->entries) {
                if (std::find(needed.begin(), needed.end(), e.sha256) ==
                    needed.end()) {
                    st.transfer.files_done.fetch_add(1);
                    continue;
                }
                const std::string abs = find_abs(*files, e.rel_path);
                if (abs.empty()) continue;

                const uint64_t entry_base = base;
                if (!s.put_blob(e.sha256, abs,
                                [&](uint64_t d, uint64_t t) {
                                    (void)t;
                                    st.transfer.bytes_done.store(entry_base + d);
                                    return !st.transfer.cancel.load();
                                },
                                &err)) {
                    aborted = true;
                    break;
                }
                base += e.size;
                st.transfer.bytes_done.store(base);
                st.transfer.files_done.fetch_add(1);
            }
            if (aborted) break;

            s.bye();
            ok = true;
        } while (false);

        *st.transfer.result =
            ok ? ("Shared " + std::to_string(meta->entries.size()) +
                  " file(s), " + format_bytes(meta->total_bytes()))
               : ("Share failed: " + err);
        st.transfer.ok = ok;
        st.transfer.running.store(false);
        st.transfer.finished.store(true, std::memory_order_release);

        ctx.deferred.enqueue([&st, &ctx, gen]() {
            if (st.generation.load() != gen) return;
            const std::string msg = *st.transfer.result;
            st.status = msg;
            if (st.transfer.ok) ctx.toasts.pushSuccess(msg);
            else ctx.toasts.pushError(msg);
        });
    }).detach();

    return true;
}

// Fetches the room's manifest and works out what would actually move.
inline bool collab_fetch_manifest(CollabState &st, AppContext &ctx,
                                  const std::string &dest_root) {
    if (!collab_configured(st)) {
        st.status = "Set a relay host, room, and secret first";
        return false;
    }
    if (st.transfer.running.load()) return false;

    RelayConfig cfg;
    cfg.host = st.relay_host;
    cfg.port = (uint16_t)st.relay_port;
    cfg.room = st.room;
    cfg.psk = st.psk;

    const RoomBinding binding = st.binding;
    const std::string peer = st.peer_id;
    const std::string name = st.display_name;
    const bool include_media = st.plan_include_media;
    const std::string dest = dest_root;

    auto man = std::make_shared<Manifest>();
    auto found = std::make_shared<bool>(false);

    st.transfer.kind = TransferKind::None;
    st.transfer.running.store(true);
    st.transfer.finished.store(false);
    st.transfer.result = std::make_shared<std::string>();
    st.status = "Fetching manifest...";

    const uint64_t gen = st.generation.load();
    std::thread([&st, &ctx, cfg, binding, peer, name, man, found, dest,
                 include_media, gen]() {
        std::string err;
        bool ok = false;
        auto plan = std::make_shared<TransferPlan>();

        do {
            RelaySession s;
            if (!s.connect(cfg, peer, name, binding, &err)) break;
            if (!s.get_manifest(*man, *found, &err)) break;
            s.bye();
            // Hashing the destination is the slow part of planning, and it is
            // what makes "already have the videos" a verified claim rather
            // than a guess. Off the main thread on purpose.
            if (*found && !dest.empty())
                *plan = plan_transfer(*man, dest, include_media);
            ok = true;
        } while (false);

        *st.transfer.result = ok ? std::string{} : err;
        st.transfer.ok = ok;
        st.transfer.running.store(false);
        st.transfer.finished.store(true, std::memory_order_release);

        ctx.deferred.enqueue([&st, &ctx, man, found, plan, ok, gen]() {
            if (st.generation.load() != gen) return;
            if (!ok) {
                st.status = "Could not fetch manifest: " + *st.transfer.result;
                ctx.toasts.pushError(st.status);
                return;
            }
            if (!*found) {
                st.have_remote_manifest = false;
                st.plan_valid = false;
                st.status = "No project has been shared into this room yet";
                return;
            }
            st.remote_manifest = *man;
            st.plan = *plan;
            st.have_remote_manifest = true;
            st.plan_valid = true;
            st.status = "Manifest: " +
                        std::to_string(st.plan.file_count()) + " file(s), " +
                        format_bytes(st.plan.bytes_needed) + " to download";
        });
    }).detach();

    return true;
}

// Downloads everything missing, then writes the .redproj with paths rewritten
// for this machine.
inline bool collab_clone(CollabState &st, AppContext &ctx,
                         const std::string &dest_root) {
    if (!collab_configured(st) || !st.have_remote_manifest) return false;
    if (st.transfer.running.load()) return false;

    RelayConfig cfg;
    cfg.host = st.relay_host;
    cfg.port = (uint16_t)st.relay_port;
    cfg.room = st.room;
    cfg.psk = st.psk;

    const RoomBinding binding = st.binding;
    const std::string peer = st.peer_id;
    const std::string name = st.display_name;
    const bool include_media = st.plan_include_media;
    const std::string dest = dest_root;
    auto man = std::make_shared<Manifest>(st.remote_manifest);

    st.transfer.kind = TransferKind::Clone;
    st.transfer.running.store(true);
    st.transfer.finished.store(false);
    st.transfer.cancel.store(false);
    st.transfer.bytes_done.store(0);
    st.transfer.bytes_total.store(st.plan.bytes_needed);
    st.transfer.files_done.store(0);
    st.transfer.files_total = (int)st.plan.needed.size();
    st.transfer.result = std::make_shared<std::string>();
    st.status = "Cloning...";

    const uint64_t gen = st.generation.load();
    std::thread([&st, &ctx, cfg, binding, peer, name, man, dest,
                 include_media, gen]() {
        std::string err;
        bool ok = false;
        auto opened_path = std::make_shared<std::string>();

        do {
            if (!ensure_dir(dest, &err)) break;

            const TransferPlan plan = plan_transfer(*man, dest, include_media);
            st.transfer.bytes_total.store(plan.bytes_needed);
            st.transfer.files_total = (int)plan.needed.size();

            RelaySession s;
            if (!s.connect(cfg, peer, name, binding, &err)) break;

            uint64_t base = 0;
            bool aborted = false;
            for (const auto &e : plan.needed) {
                const uint64_t entry_base = base;
                if (!s.get_blob(e.sha256, dest,
                                std::filesystem::path(dest) / e.rel_path,
                                [&](uint64_t d, uint64_t t) {
                                    (void)t;
                                    st.transfer.bytes_done.store(entry_base + d);
                                    return !st.transfer.cancel.load();
                                },
                                &err)) {
                    aborted = true;
                    break;
                }
                base += e.size;
                st.transfer.bytes_done.store(base);
                st.transfer.files_done.fetch_add(1);
            }
            if (aborted) break;
            s.bye();

            // The .redproj is written LAST and only after every blob has
            // landed and verified, so an interrupted clone leaves a directory
            // that simply will not open rather than a project that opens and
            // then fails on missing media.
            const std::string proj_name =
                man->project_name.empty() ? std::string("shared")
                                          : man->project_name;
            const nlohmann::json rewritten =
                rewrite_project_paths(man->project_json, dest, proj_name);
            const std::filesystem::path redproj =
                std::filesystem::path(dest) / (proj_name + ".redproj");
            if (!write_file_atomic(redproj, rewritten.dump(2), &err)) break;

            *opened_path = redproj.string();
            ok = true;
        } while (false);

        *st.transfer.result = ok ? *opened_path : err;
        st.transfer.ok = ok;
        st.transfer.running.store(false);
        st.transfer.finished.store(true, std::memory_order_release);

        ctx.deferred.enqueue([&st, &ctx, ok, gen]() {
            if (st.generation.load() != gen) return;
            if (!ok) {
                st.status = "Clone failed: " + *st.transfer.result;
                ctx.toasts.pushError(st.status);
                return;
            }
            st.status = "Cloned. Open it with File > Load Project: " +
                        *st.transfer.result;
            ctx.toasts.pushSuccess("Clone complete -- open it with "
                                   "File > Load Project");
        });
    }).detach();

    return true;
}

}  // namespace collab
