#pragma once
// collab_log.h -- the on-disk op log.
//
// Layout, all inside <project>/.collab/ so a project stays self-contained and
// the existing labeled_data/ snapshot flow is untouched:
//
//   .collab/
//     ops/local.log            ops authored on this machine
//     ops/remote-<peer>.log    one file per peer we have heard from
//     cursor.json              how far we have pushed to / pulled from the relay
//     room.json                room id, relay address, binding (see collab_client.h)
//
// Logs are JSONL and strictly append-only. Nothing is ever rewritten, so a
// losing edit stays recoverable forever and the History view has real data to
// show. Sequence numbers and the Lamport high-water mark are RE-DERIVED from
// the logs on open rather than trusted from cursor.json -- the logs are the
// authority, so a lost or stale cursor file can never cause a peer to reuse a
// sequence number.
//
// Torn tails are expected, not exceptional: a crash mid-append leaves a
// partial final line. Readers drop unparsable lines and report the count
// instead of failing the whole log, because losing one edit is recoverable and
// refusing to open the project is not.

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "collab_io.h"
#include "collab_ops.h"
#include "json.hpp"

namespace collab {

namespace fs = std::filesystem;

// How far this peer has synced with the relay.
struct SyncCursor {
    uint64_t sent_seq = 0;   // highest local seq the relay has acknowledged
    uint64_t recv_seq = 0;   // highest relay sequence we have ingested
    int64_t  last_sync_ms = 0;
};

inline void to_json(nlohmann::json &j, const SyncCursor &c) {
    j = nlohmann::json{{"sent_seq", c.sent_seq},
                       {"recv_seq", c.recv_seq},
                       {"last_sync_ms", c.last_sync_ms}};
}

inline void from_json(const nlohmann::json &j, SyncCursor &c) {
    c.sent_seq = j.value("sent_seq", (uint64_t)0);
    c.recv_seq = j.value("recv_seq", (uint64_t)0);
    c.last_sync_ms = j.value("last_sync_ms", (int64_t)0);
}

struct LogStats {
    size_t ops_read = 0;
    size_t lines_skipped = 0;   // torn tail or malformed -- surfaced, not hidden
    size_t files_read = 0;
};

class OpLog {
  public:
    // Opens (creating if needed) the .collab tree under `project_dir` and
    // scans every log to recover the sequence and Lamport high-water marks.
    bool open(const fs::path &project_dir, const std::string &peer_id,
              std::string *err = nullptr) {
        root_ = project_dir / ".collab";
        ops_dir_ = root_ / "ops";
        peer_ = peer_id;
        if (!ensure_dir(ops_dir_, err)) return false;

        load_cursor();

        // Derive next_seq and the Lamport mark from what is actually on disk.
        std::vector<Op> all;
        LogStats st;
        if (!read_all(all, st, err)) return false;

        next_seq_ = 1;
        lamport_ = 0;
        for (const Op &op : all) {
            if (op.lamport > lamport_) lamport_ = op.lamport;
            if (op.peer == peer_ && op.seq >= next_seq_) next_seq_ = op.seq + 1;
        }
        opened_ = true;
        return true;
    }

    bool is_open() const { return opened_; }
    const fs::path &root() const { return root_; }
    const fs::path &ops_dir() const { return ops_dir_; }

    // Seeds a factory so locally authored ops continue the sequence and the
    // Lamport clock rather than restarting at 1 (which would make every op
    // after a restart lose LWW against pre-restart edits).
    void seed(OpFactory &f) const {
        f.peer = peer_;
        f.next_seq = next_seq_;
        f.clock.value = lamport_;
    }

    uint64_t next_seq() const { return next_seq_; }
    uint64_t lamport() const { return lamport_; }

    // ── Append ──

    bool append_local(const std::vector<Op> &ops, std::string *err = nullptr) {
        if (ops.empty()) return true;
        std::string blob;
        for (const Op &op : ops) {
            nlohmann::json j;
            to_json(j, op);
            blob += j.dump();
            blob += '\n';
            if (op.seq >= next_seq_) next_seq_ = op.seq + 1;
            if (op.lamport > lamport_) lamport_ = op.lamport;
        }
        return append_raw(local_path(), blob, err);
    }

    // Remote ops are filed per author so a peer's history stays contiguous and
    // one corrupt file cannot take the others down with it.
    bool append_remote(const std::vector<Op> &ops, std::string *err = nullptr) {
        std::map<std::string, std::string> by_peer;
        for (const Op &op : ops) {
            if (op.peer == peer_) continue;  // our own ops echoed back
            nlohmann::json j;
            to_json(j, op);
            by_peer[op.peer] += j.dump();
            by_peer[op.peer] += '\n';
            if (op.lamport > lamport_) lamport_ = op.lamport;
        }
        for (const auto &kv : by_peer)
            if (!append_raw(remote_path(kv.first), kv.second, err)) return false;
        return true;
    }

    // ── Read ──

    // Every op from every log. Order is irrelevant -- apply_ops resolves.
    bool read_all(std::vector<Op> &out, LogStats &stats,
                  std::string *err = nullptr) const {
        out.clear();
        stats = LogStats{};

        std::error_code ec;
        if (!fs::exists(ops_dir_, ec)) return true;

        std::vector<fs::path> files;
        for (const auto &entry : fs::directory_iterator(ops_dir_, ec)) {
            if (ec) break;
            if (!entry.is_regular_file()) continue;
            if (entry.path().extension() != ".log") continue;
            files.push_back(entry.path());
        }
        // Deterministic order so a replay is reproducible run to run.
        std::sort(files.begin(), files.end());

        for (const auto &p : files) {
            if (!read_log_file(p, out, stats, err)) return false;
            ++stats.files_read;
        }
        return true;
    }

    // Local ops the relay has not acknowledged yet -- the push set.
    bool read_local_since(uint64_t seq, std::vector<Op> &out,
                          std::string *err = nullptr) const {
        out.clear();
        LogStats st;
        std::vector<Op> all;
        if (!read_log_file(local_path(), all, st, err)) return false;
        for (Op &op : all)
            if (op.seq > seq) out.push_back(std::move(op));
        std::sort(out.begin(), out.end(),
                  [](const Op &a, const Op &b) { return a.seq < b.seq; });
        return true;
    }

    // ── Cursor ──

    const SyncCursor &cursor() const { return cursor_; }
    SyncCursor &cursor() { return cursor_; }

    bool save_cursor(std::string *err = nullptr) const {
        nlohmann::json j;
        to_json(j, cursor_);
        return write_file_atomic(root_ / "cursor.json", j.dump(2), err);
    }

    // ── Paths ──

    fs::path local_path() const { return ops_dir_ / "local.log"; }

    fs::path remote_path(const std::string &peer) const {
        return ops_dir_ / ("remote-" + sanitize(peer) + ".log");
    }

  private:
    // Peer ids are our own generated hex, but they arrive over the network, so
    // never let one steer a path traversal.
    static std::string sanitize(const std::string &s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '-' || c == '_';
            out.push_back(ok ? c : '_');
        }
        if (out.empty()) out = "unknown";
        if (out.size() > 64) out.resize(64);
        return out;
    }

    static bool append_raw(const fs::path &p, const std::string &blob,
                           std::string *err) {
        if (blob.empty()) return true;
        // append_line adds the trailing newline, so hand it the blob minus its
        // own final one.
        std::string body = blob;
        if (!body.empty() && body.back() == '\n') body.pop_back();
        return append_line(p, body, err);
    }

    static bool read_log_file(const fs::path &p, std::vector<Op> &out,
                              LogStats &stats, std::string *err) {
        if (!file_exists(p)) return true;

        std::string content;
        if (!read_file(p, content, err)) return false;

        std::istringstream iss(content);
        std::string line;
        while (std::getline(iss, line)) {
            if (line.empty()) continue;
            if (!line.empty() && line.back() == '\r') line.pop_back();

            const nlohmann::json j =
                nlohmann::json::parse(line.begin(), line.end(), nullptr,
                                      /*allow_exceptions=*/false);
            Op op;
            if (j.is_discarded() || !op_from_json(j, op)) {
                // A torn final line is the normal case after a crash; a
                // malformed middle line means real corruption. Both are
                // counted and reported rather than silently dropped.
                ++stats.lines_skipped;
                continue;
            }
            out.push_back(std::move(op));
            ++stats.ops_read;
        }
        return true;
    }

    void load_cursor() {
        std::string content;
        if (!read_file(root_ / "cursor.json", content, nullptr)) return;
        const nlohmann::json j = nlohmann::json::parse(
            content.begin(), content.end(), nullptr, false);
        if (j.is_discarded()) return;
        from_json(j, cursor_);
    }

    fs::path    root_;
    fs::path    ops_dir_;
    std::string peer_;
    SyncCursor  cursor_;
    uint64_t    next_seq_ = 1;
    uint64_t    lamport_ = 0;
    bool        opened_ = false;
};

}  // namespace collab
