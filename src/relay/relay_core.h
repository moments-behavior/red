#pragma once
// relay_core.h -- the RED collaboration relay, minus its command line.
//
// Split out of red_relay.cpp so the loopback test can run a real relay
// in-process on an ephemeral port instead of shelling out to a binary -- the
// same core-header / thin-driver split pump_events_core.h uses.
//
// One collaborator runs this on a host the others can reach (a VPS, or any
// port-forwarded machine). Every RED client dials OUT to it, so no client
// needs an open port and machines behind unrelated NATs can collaborate.
//
// The relay is deliberately dumb. It authenticates, orders, persists, and
// fans out; it does not understand annotations. Ops are validated only far
// enough to reject garbage, then stored opaquely. That keeps the merge
// semantics entirely on the clients, where they are unit-tested, and means an
// older relay can serve newer clients.
//
// Single-threaded poll() loop, no thread per client. A relay serves a handful
// of collaborators, and one loop is far easier to reason about than a thread
// pool sharing an append-only log.
//
//   red_relay --port 7373 --data ./relay-data --secrets rooms.json
//
// rooms.json:
//   { "rooms": { "rig-a": { "psk": "a-long-random-shared-secret" } } }
//
// SECURITY: connections are authenticated with HMAC-SHA256 but the traffic is
// NOT encrypted. Anyone who can observe the network sees annotation
// coordinates, comments, and file contents. Run this behind an SSH tunnel or
// on a WireGuard/Tailscale network if that matters:
//   ssh -N -L 7373:localhost:7373 user@relay-host

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <cstdarg>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "collab_bundle.h"
#include "collab_hash.h"
#include "collab_io.h"
#include "collab_ops.h"
#include "collab_proto.h"
#include "collab_socket.h"
#include "json.hpp"

namespace fs = std::filesystem;
using namespace collab;
using namespace collab::proto;

namespace collab {
namespace relay {

// ── Tunables ──
static constexpr int      kPollTimeoutMs   = 1000;
static constexpr int64_t  kIdleTimeoutMs   = 300000;   // 5 min
static constexpr size_t   kMaxOpsPerBatch  = 2000;
static constexpr size_t   kMaxBatchBytes   = 4u * 1024 * 1024;
static constexpr uint64_t kDefaultQuotaGb  = 200;

// Set by the caller -- a signal handler in the binary, a test harness
// otherwise -- to unwind the poll loop at the next tick.
using StopFlag = std::atomic<bool>;

// Tests run a relay in-process and do not want its chatter interleaved with
// assertion output.
inline bool g_quiet = false;

inline void logf(const char *fmt, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    if (g_quiet) return;
    std::fprintf(stderr, "[relay] %s\n", buf);
    std::fflush(stderr);
}

// =========================================================================
// Room
// =========================================================================

struct Presence {
    std::string peer;
    std::string display_name;
    int64_t     last_seen_ms = 0;
    int64_t     current_frame = -1;
};

// One collaborative project. Ops live in an append-only file; only a byte
// offset per op is held in memory, so a room with a million ops costs ~8 MB of
// RAM rather than a few hundred.
struct Room {
    std::string name;
    std::string psk;
    fs::path    dir;

    nlohmann::json binding;          // RoomBinding, adopted from the first joiner
    bool           binding_set = false;

    std::vector<uint64_t> offsets;   // offsets[i] -> byte offset of relay_seq i+1
    uint64_t              file_size = 0;

    std::map<std::string, Presence> presence;

    // Highest op sequence accepted from each peer. A client advances its
    // "sent" cursor only after a whole sync succeeds, so a push that lands but
    // is followed by a failed pull gets re-pushed next time. Ops are
    // idempotent for clients, but the relay log would grow a duplicate copy
    // every retry -- this drops them at the door.
    std::map<std::string, uint64_t> high_seq;

    uint64_t next_seq() const { return offsets.size() + 1; }
    fs::path ops_path() const { return dir / "ops.log"; }
    fs::path meta_path() const { return dir / "meta.json"; }
    fs::path manifest_path() const { return dir / "manifest.json"; }

    // Rebuilds the offset index from the log. Torn tails are truncated rather
    // than tolerated: unlike the client, the relay is the ordering authority,
    // so a half-written final line must not occupy a sequence number.
    bool load(std::string *err) {
        if (!ensure_dir(dir, err)) return false;

        offsets.clear();
        file_size = 0;

        std::string content;
        if (file_exists(ops_path()) && !read_file(ops_path(), content, err))
            return false;

        uint64_t off = 0;
        uint64_t good_end = 0;
        size_t start = 0;
        while (start < content.size()) {
            const size_t nl = content.find('\n', start);
            if (nl == std::string::npos) break;   // torn tail
            offsets.push_back(off);
            off = nl + 1;
            good_end = off;
            start = nl + 1;
        }
        file_size = good_end;

        if (good_end < content.size()) {
            logf("room '%s': truncating %zu byte torn tail from ops.log",
                 name.c_str(), content.size() - (size_t)good_end);
            std::string trimmed = content.substr(0, good_end);
            if (!write_file_atomic(ops_path(), trimmed, err)) return false;
        }

        // Rebuild the per-peer high-water marks from the log so dedupe
        // survives a relay restart.
        high_seq.clear();
        {
            size_t pos = 0;
            while (pos < good_end) {
                const size_t nl = content.find('\n', pos);
                if (nl == std::string::npos) break;
                const nlohmann::json j = nlohmann::json::parse(
                    content.begin() + pos, content.begin() + nl, nullptr, false);
                pos = nl + 1;
                if (j.is_discarded()) continue;
                const std::string pr = j.value("peer", std::string{});
                const uint64_t sq = j.value("seq", (uint64_t)0);
                if (!pr.empty() && sq > high_seq[pr]) high_seq[pr] = sq;
            }
        }

        std::string meta;
        if (read_file(meta_path(), meta, nullptr)) {
            const nlohmann::json j = nlohmann::json::parse(
                meta.begin(), meta.end(), nullptr, false);
            if (!j.is_discarded() && j.contains("binding")) {
                binding = j["binding"];
                binding_set = true;
            }
        }
        return true;
    }

    bool save_meta(std::string *err) const {
        nlohmann::json j;
        j["binding"] = binding;
        j["ops"] = offsets.size();
        return write_file_atomic(meta_path(), j.dump(2), err);
    }

    // Appends one op line and assigns it the next relay sequence.
    bool append(const std::string &line, uint64_t &assigned, std::string *err) {
        std::FILE *f = std::fopen(ops_path().string().c_str(), "ab");
        if (!f) {
            if (err) *err = "cannot append to " + ops_path().string();
            return false;
        }
        const std::string rec = line + "\n";
        const bool ok = std::fwrite(rec.data(), 1, rec.size(), f) == rec.size();
        if (ok) flush_to_disk(f);
        std::fclose(f);
        if (!ok) {
            if (err) *err = "short write to " + ops_path().string();
            return false;
        }
        offsets.push_back(file_size);
        file_size += rec.size();
        assigned = offsets.size();
        return true;
    }

    // Reads ops with relay_seq > `since`, bounded by count and byte budget.
    bool read_since(uint64_t since, std::vector<std::string> &out,
                    uint64_t &high_seq, bool &more) const {
        out.clear();
        high_seq = since;
        more = false;
        if (since >= offsets.size()) return true;

        std::FILE *f = std::fopen(ops_path().string().c_str(), "rb");
        if (!f) return false;

        size_t bytes = 0;
        for (uint64_t i = since; i < offsets.size(); ++i) {
            if (out.size() >= kMaxOpsPerBatch || bytes >= kMaxBatchBytes) {
                more = true;
                break;
            }
            const uint64_t start = offsets[i];
            const uint64_t end =
                (i + 1 < offsets.size()) ? offsets[i + 1] : file_size;
            if (end <= start) continue;

            std::string line;
            line.resize(static_cast<size_t>(end - start));
            if (std::fseek(f, static_cast<long>(start), SEEK_SET) != 0) break;
            if (std::fread(&line[0], 1, line.size(), f) != line.size()) break;
            while (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
                line.pop_back();

            bytes += line.size();
            out.push_back(std::move(line));
            high_seq = i + 1;
        }
        std::fclose(f);
        return true;
    }
};

// =========================================================================
// Blob store
// =========================================================================

// Content-addressed and shared across rooms: two projects referencing the same
// video file store it once.
struct BlobStore {
    fs::path dir;
    fs::path staging;
    uint64_t quota_bytes = kDefaultQuotaGb * 1024ull * 1024 * 1024;

    bool init(const fs::path &data_dir, std::string *err) {
        dir = data_dir / "blobs";
        staging = data_dir / "blobs" / "incoming";
        return ensure_dir(dir, err) && ensure_dir(staging, err);
    }

    fs::path path_of(const std::string &hash) const { return dir / hash; }

    bool have(const std::string &hash) const {
        return valid_blob_id(hash) && file_exists(path_of(hash));
    }

    uint64_t total_bytes() const {
        uint64_t n = 0;
        std::error_code ec;
        for (const auto &e : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (e.is_regular_file(ec)) n += size_of_file(e.path());
        }
        return n;
    }

    // Evicts least-recently-modified blobs until back under quota, never
    // touching one a live manifest still references. Losing a referenced blob
    // would turn a clone into a silent partial failure.
    void enforce_quota(const std::vector<std::string> &referenced) {
        uint64_t total = total_bytes();
        if (total <= quota_bytes) return;

        struct Cand {
            fs::path p;
            uint64_t size;
            fs::file_time_type mtime;
        };
        std::vector<Cand> cands;
        std::error_code ec;
        for (const auto &e : fs::directory_iterator(dir, ec)) {
            if (ec) break;
            if (!e.is_regular_file(ec)) continue;
            const std::string name = e.path().filename().string();
            if (std::find(referenced.begin(), referenced.end(), name) !=
                referenced.end())
                continue;
            cands.push_back({e.path(), size_of_file(e.path()),
                             fs::last_write_time(e.path(), ec)});
        }
        std::sort(cands.begin(), cands.end(),
                  [](const Cand &a, const Cand &b) { return a.mtime < b.mtime; });

        for (const Cand &c : cands) {
            if (total <= quota_bytes) break;
            fs::remove(c.p, ec);
            if (!ec) {
                total -= c.size;
                logf("evicted blob %s (%s)", c.p.filename().string().c_str(),
                     format_bytes(c.size).c_str());
            }
        }
    }
};

// =========================================================================
// Connection
// =========================================================================

enum class ConnState { WaitHello, WaitAuth, Ready, Closing };

struct Conn {
    net::Socket sock;
    ConnState   state = ConnState::WaitHello;

    std::string room;
    std::string peer;
    std::string display_name;
    std::string nonce;

    std::vector<uint8_t> in_buf;
    std::vector<uint8_t> out_buf;
    size_t out_sent = 0;

    int64_t last_activity_ms = 0;

    void send(const std::vector<uint8_t> &frame) {
        out_buf.insert(out_buf.end(), frame.begin(), frame.end());
    }

    void send_error(const std::string &msg) {
        nlohmann::json j;
        j["message"] = msg;
        send(encode(Msg::Error, j));
    }

    void deny(const std::string &reason) {
        nlohmann::json j;
        j["reason"] = reason;
        send(encode(Msg::Deny, j));
        state = ConnState::Closing;
    }
};

// =========================================================================
// Relay
// =========================================================================

class Relay {
  public:
    bool init(uint16_t port, const fs::path &data_dir,
              const fs::path &secrets_path, uint64_t quota_gb,
              std::string *err) {
        data_dir_ = data_dir;
        if (!ensure_dir(data_dir_, err)) return false;
        if (!ensure_dir(data_dir_ / "rooms", err)) return false;
        if (!blobs_.init(data_dir_, err)) return false;
        blobs_.quota_bytes = quota_gb * 1024ull * 1024 * 1024;

        if (!load_secrets(secrets_path, err)) return false;
        if (!listener_.listen_on(port, err)) return false;

        logf("listening on port %u, data in %s, %zu room(s) configured",
             (unsigned)port, data_dir_.string().c_str(), rooms_.size());
        logf("traffic is authenticated but NOT encrypted -- tunnel it if the "
             "annotations are sensitive");
        return true;
    }

    void run(StopFlag &stop) {
        while (!stop.load()) {
            std::vector<net::PollItem> items;
            items.push_back({listener_.fd(), true, false, false, false, false});
            for (auto &c : conns_) {
                net::PollItem it;
                it.fd = c->sock.fd();
                it.want_read = true;
                it.want_write = c->out_sent < c->out_buf.size();
                items.push_back(it);
            }

            const int rc = net::poll_wait(items, kPollTimeoutMs);
            if (rc < 0 && !stop.load()) {
                logf("poll failed; stopping");
                break;
            }

            // accept_new() appends to conns_, so the polled count must be
            // captured first -- otherwise the loop below indexes `items` past
            // its end for any connection accepted on this tick. Newly accepted
            // connections are simply serviced on the next pass.
            const size_t polled = conns_.size();
            if (items[0].can_read) accept_new();

            for (size_t i = 0; i < polled; ++i) {
                Conn &c = *conns_[i];
                const net::PollItem &it = items[i + 1];
                if (it.hung_up) {
                    c.state = ConnState::Closing;
                    continue;
                }
                if (it.can_read) on_readable(c);
                if (c.state != ConnState::Closing && it.can_write) flush(c);
            }

            reap();
        }
        logf("shutting down");
    }

    // The port actually bound -- tests listen on 0 and read it back.
    uint16_t bound_port() const { return listener_.port(); }

  private:
    // ── secrets ──

    bool load_secrets(const fs::path &p, std::string *err) {
        std::string content;
        if (!read_file(p, content, err)) {
            if (err)
                *err = "cannot read secrets file " + p.string() +
                       " -- expected {\"rooms\": {\"name\": {\"psk\": \"...\"}}}";
            return false;
        }
        const nlohmann::json j =
            nlohmann::json::parse(content.begin(), content.end(), nullptr, false);
        if (j.is_discarded() || !j.contains("rooms") || !j["rooms"].is_object()) {
            if (err) *err = "secrets file has no \"rooms\" object";
            return false;
        }

        for (auto it = j["rooms"].begin(); it != j["rooms"].end(); ++it) {
            const std::string name = it.key();
            std::string psk;
            if (it.value().is_string()) psk = it.value().get<std::string>();
            else psk = it.value().value("psk", std::string{});

            if (psk.size() < 16) {
                if (err)
                    *err = "room '" + name +
                           "' has a shared secret shorter than 16 characters; "
                           "use a long random string";
                return false;
            }
            if (!valid_room_name(name)) {
                if (err)
                    *err = "room name '" + name +
                           "' may only contain letters, digits, '-' and '_'";
                return false;
            }

            auto room = std::make_unique<Room>();
            room->name = name;
            room->psk = psk;
            room->dir = data_dir_ / "rooms" / name;
            if (!room->load(err)) return false;
            logf("room '%s': %zu ops on disk", name.c_str(),
                 room->offsets.size());
            rooms_[name] = std::move(room);
        }
        return !rooms_.empty();
    }

    // Room names become directory names, so they are constrained at load time
    // rather than sanitized at use time.
    static bool valid_room_name(const std::string &s) {
        if (s.empty() || s.size() > 64) return false;
        for (char c : s) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                            (c >= '0' && c <= '9') || c == '-' || c == '_';
            if (!ok) return false;
        }
        return true;
    }

    // ── connection lifecycle ──

    void accept_new() {
        for (;;) {
            net::Socket s;
            bool again = false;
            std::string err;
            if (!listener_.accept_one(s, again, &err)) {
                if (!again) logf("accept failed: %s", err.c_str());
                return;
            }
            auto c = std::make_unique<Conn>();
            c->sock = std::move(s);
            c->last_activity_ms = now_ms();
            conns_.push_back(std::move(c));
        }
    }

    void reap() {
        const int64_t now = now_ms();
        for (auto &c : conns_) {
            if (c->state == ConnState::Closing && c->out_sent >= c->out_buf.size())
                c->sock.close();
            else if (now - c->last_activity_ms > kIdleTimeoutMs) {
                logf("dropping idle connection (peer '%s')", c->peer.c_str());
                c->sock.close();
            }
        }
        conns_.erase(std::remove_if(conns_.begin(), conns_.end(),
                                    [](const std::unique_ptr<Conn> &c) {
                                        return !c->sock.valid();
                                    }),
                     conns_.end());
    }

    void flush(Conn &c) {
        while (c.out_sent < c.out_buf.size()) {
            const size_t remain = c.out_buf.size() - c.out_sent;
            const auto n = ::send(c.sock.fd(),
                                  reinterpret_cast<const char *>(c.out_buf.data() +
                                                                 c.out_sent),
                                  static_cast<int>(remain),
#if defined(MSG_NOSIGNAL)
                                  MSG_NOSIGNAL
#else
                                  0
#endif
            );
            if (n > 0) {
                c.out_sent += static_cast<size_t>(n);
                continue;
            }
            if (n < 0 && net::would_block(net::last_error())) return;
            c.state = ConnState::Closing;
            c.sock.close();
            return;
        }
        c.out_buf.clear();
        c.out_sent = 0;
    }

    void on_readable(Conn &c) {
        uint8_t buf[65536];
        for (;;) {
            const auto n = ::recv(c.sock.fd(), reinterpret_cast<char *>(buf),
                                  sizeof(buf), 0);
            if (n > 0) {
                if (c.in_buf.size() + static_cast<size_t>(n) > kMaxPendingBytes) {
                    logf("peer '%s' exceeded the pending-byte limit; dropping",
                         c.peer.c_str());
                    c.state = ConnState::Closing;
                    c.sock.close();
                    return;
                }
                c.in_buf.insert(c.in_buf.end(), buf, buf + n);
                c.last_activity_ms = now_ms();
                continue;
            }
            if (n == 0) {
                c.state = ConnState::Closing;
                c.sock.close();
                return;
            }
            if (net::would_block(net::last_error())) break;
            c.state = ConnState::Closing;
            c.sock.close();
            return;
        }

        // Drain every complete frame the read produced.
        for (;;) {
            Frame f;
            size_t consumed = 0;
            std::string err;
            const Decode r = decode(c.in_buf, f, consumed, &err);
            if (r == Decode::NeedMore) return;
            if (r == Decode::Bad) {
                logf("protocol error from peer '%s': %s", c.peer.c_str(),
                     err.c_str());
                c.send_error(err);
                c.state = ConnState::Closing;
                flush(c);
                c.sock.close();
                return;
            }
            c.in_buf.erase(c.in_buf.begin(), c.in_buf.begin() + consumed);
            handle(c, f);
            if (c.state == ConnState::Closing) {
                flush(c);
                return;
            }
        }
    }

    // ── message dispatch ──

    void handle(Conn &c, const Frame &f) {
        // Nothing but the handshake is honored before authentication.
        if (c.state == ConnState::WaitHello && f.type != Msg::Hello) {
            c.deny("expected Hello");
            return;
        }
        if (c.state == ConnState::WaitAuth && f.type != Msg::Auth) {
            c.deny("expected Auth");
            return;
        }

        switch (f.type) {
            case Msg::Hello:        on_hello(c, f); break;
            case Msg::Auth:         on_auth(c, f); break;
            case Msg::PushOps:      on_push(c, f); break;
            case Msg::PullOps:      on_pull(c, f); break;
            case Msg::Presence:     on_presence(c, f); break;
            case Msg::PutManifest:  on_put_manifest(c, f); break;
            case Msg::GetManifest:  on_get_manifest(c, f); break;
            case Msg::HaveBlobs:    on_have_blobs(c, f); break;
            case Msg::BlobPut:      on_blob_put(c, f); break;
            case Msg::BlobGet:      on_blob_get(c, f); break;
            case Msg::Bye:          c.state = ConnState::Closing; break;
            default:
                c.send_error(std::string("unexpected message ") +
                             msg_name(f.type));
                break;
        }
    }

    void on_hello(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        if (j.is_discarded()) {
            c.deny("malformed Hello");
            return;
        }
        const uint32_t ver = j.value("version", 0u);
        if (ver != kVersion) {
            c.deny("protocol version " + std::to_string(ver) +
                   " is not supported by this relay (expected " +
                   std::to_string(kVersion) + ")");
            return;
        }

        c.room = j.value("room", std::string{});
        c.peer = j.value("peer", std::string{});
        c.display_name = j.value("display_name", std::string{});

        if (c.peer.empty() || c.peer.size() > 64) {
            c.deny("missing or oversized peer id");
            return;
        }
        // A room the relay does not know and a wrong secret give the same
        // shaped rejection path; the reason string still says which, because
        // this is a lab tool and a confusing setup error costs more than the
        // marginal disclosure that a room name exists.
        if (rooms_.find(c.room) == rooms_.end()) {
            c.deny("no such room '" + c.room + "' on this relay");
            return;
        }

        c.nonce = random_hex(32);
        nlohmann::json out;
        out["nonce"] = c.nonce;
        c.send(encode(Msg::Challenge, out));
        c.state = ConnState::WaitAuth;
    }

    void on_auth(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        if (j.is_discarded()) {
            c.deny("malformed Auth");
            return;
        }
        Room &room = *rooms_[c.room];

        const std::string want = hmac_sha256_hex(
            room.psk, auth_message(c.nonce, c.room, c.peer));
        const std::string got = j.value("tag", std::string{});

        if (!constant_time_equal(want, got)) {
            logf("auth failed for peer '%s' in room '%s'", c.peer.c_str(),
                 c.room.c_str());
            c.deny("authentication failed -- check the room's shared secret");
            return;
        }

        // The room's shape is set by whoever joins first; everyone after must
        // match. Annotations are indexed positionally by camera and node, so
        // merging across skeletons would scramble labels irrecoverably.
        nlohmann::json client_binding = j.value("binding", nlohmann::json::object());
        if (!room.binding_set) {
            room.binding = client_binding;
            room.binding_set = true;
            std::string err;
            if (!room.save_meta(&err)) logf("meta save failed: %s", err.c_str());
            logf("room '%s': binding adopted from peer '%s'", c.room.c_str(),
                 c.peer.c_str());
        } else {
            RoomBinding a, b;
            from_json(room.binding, a);
            from_json(client_binding, b);
            std::string why;
            if (!a.compatible_with(b, &why)) {
                logf("peer '%s' rejected from room '%s': %s", c.peer.c_str(),
                     c.room.c_str(), why.c_str());
                c.deny(why);
                return;
            }
        }

        nlohmann::json out;
        out["relay_seq"] = room.offsets.size();
        out["binding"] = room.binding;
        c.send(encode(Msg::Welcome, out));
        c.state = ConnState::Ready;
        logf("peer '%s' (%s) joined room '%s'", c.peer.c_str(),
             c.display_name.c_str(), c.room.c_str());
    }

    void on_push(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        Room &room = *rooms_[c.room];

        int accepted = 0, rejected = 0, duplicate = 0;
        if (j.contains("ops") && j["ops"].is_array()) {
            for (const auto &jop : j["ops"]) {
                // Validate only enough to keep garbage out of the log. The
                // relay never interprets what an op means.
                Op probe;
                if (!op_from_json(jop, probe)) {
                    ++rejected;
                    continue;
                }
                if (probe.peer != c.peer) {
                    // A peer may only push ops it authored; otherwise one
                    // client could forge another's history.
                    ++rejected;
                    continue;
                }
                const auto hs = room.high_seq.find(probe.peer);
                if (hs != room.high_seq.end() && probe.seq <= hs->second) {
                    ++duplicate;
                    continue;
                }
                uint64_t assigned = 0;
                std::string err;
                if (!room.append(jop.dump(), assigned, &err)) {
                    logf("append failed in room '%s': %s", c.room.c_str(),
                         err.c_str());
                    c.send_error("relay could not persist ops: " + err);
                    return;
                }
                room.high_seq[probe.peer] = probe.seq;
                ++accepted;
            }
        }
        if (rejected)
            logf("room '%s': rejected %d malformed/misattributed op(s) from '%s'",
                 c.room.c_str(), rejected, c.peer.c_str());

        nlohmann::json out;
        out["accepted"] = accepted;
        out["rejected"] = rejected;
        out["duplicate"] = duplicate;
        out["relay_seq"] = room.offsets.size();
        c.send(encode(Msg::PushAck, out));
    }

    void on_pull(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        Room &room = *rooms_[c.room];
        const uint64_t since = j.value("since_seq", (uint64_t)0);

        std::vector<std::string> lines;
        uint64_t high = since;
        bool more = false;
        if (!room.read_since(since, lines, high, more)) {
            c.send_error("relay could not read its op log");
            return;
        }

        nlohmann::json ops = nlohmann::json::array();
        for (const std::string &line : lines) {
            const nlohmann::json op =
                nlohmann::json::parse(line.begin(), line.end(), nullptr, false);
            if (!op.is_discarded()) ops.push_back(op);
        }

        nlohmann::json out;
        out["ops"] = ops;
        out["relay_seq"] = high;
        out["more"] = more;
        c.send(encode(Msg::OpsBatch, out));
    }

    void on_presence(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        Room &room = *rooms_[c.room];

        Presence &p = room.presence[c.peer];
        p.peer = c.peer;
        p.display_name = c.display_name;
        p.last_seen_ms = now_ms();
        p.current_frame = j.value("current_frame", (int64_t)-1);

        nlohmann::json peers = nlohmann::json::array();
        for (const auto &kv : room.presence) {
            nlohmann::json e;
            e["peer"] = kv.second.peer;
            e["display_name"] = kv.second.display_name;
            e["last_seen_ms"] = kv.second.last_seen_ms;
            e["current_frame"] = kv.second.current_frame;
            peers.push_back(e);
        }
        nlohmann::json out;
        out["peers"] = peers;
        c.send(encode(Msg::PresenceList, out));
    }

    void on_put_manifest(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        Room &room = *rooms_[c.room];

        if (!j.contains("manifest")) {
            c.send_error("PutManifest without a manifest");
            return;
        }
        Manifest m;
        std::string err;
        if (!manifest_from_json(j["manifest"], m, &err)) {
            c.send_error("rejected manifest: " + err);
            return;
        }
        if (!write_file_atomic(room.manifest_path(), j["manifest"].dump(2), &err)) {
            c.send_error("relay could not store the manifest: " + err);
            return;
        }
        logf("room '%s': manifest updated by '%s' (%zu files, %s)",
             c.room.c_str(), c.peer.c_str(), m.entries.size(),
             format_bytes(m.total_bytes()).c_str());

        nlohmann::json out;
        out["present"] = true;
        c.send(encode(Msg::Manifest, out));
    }

    void on_get_manifest(Conn &c, const Frame &) {
        Room &room = *rooms_[c.room];
        std::string content;
        nlohmann::json out;
        if (!read_file(room.manifest_path(), content, nullptr)) {
            out["present"] = false;
        } else {
            const nlohmann::json m = nlohmann::json::parse(
                content.begin(), content.end(), nullptr, false);
            if (m.is_discarded()) {
                out["present"] = false;
            } else {
                out["present"] = true;
                out["manifest"] = m;
            }
        }
        c.send(encode(Msg::Manifest, out));
    }

    void on_have_blobs(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        nlohmann::json needed = nlohmann::json::array();
        if (j.contains("hashes") && j["hashes"].is_array()) {
            for (const auto &h : j["hashes"]) {
                if (!h.is_string()) continue;
                const std::string hash = h.get<std::string>();
                if (!valid_blob_id(hash)) continue;
                if (!blobs_.have(hash)) needed.push_back(hash);
            }
        }
        nlohmann::json out;
        out["hashes"] = needed;
        c.send(encode(Msg::BlobsNeeded, out));
    }

    void on_blob_put(Conn &c, const Frame &f) {
        nlohmann::json hdr;
        const uint8_t *data = nullptr;
        size_t len = 0;
        if (!decode_blob(f, hdr, data, len)) {
            c.send_error("malformed BlobPut");
            return;
        }
        const std::string hash = hdr.value("hash", std::string{});
        const uint64_t offset = hdr.value("offset", (uint64_t)0);
        const uint64_t total = hdr.value("total", (uint64_t)0);

        if (!valid_blob_id(hash)) {
            c.send_error("malformed blob id");
            return;
        }
        if (blobs_.have(hash)) {
            // Already stored; the sender can stop.
            nlohmann::json out;
            out["hash"] = hash;
            out["received"] = total;
            out["complete"] = true;
            c.send(encode(Msg::BlobPutAck, out));
            return;
        }

        // A mismatched offset means the sender and the relay disagree about
        // how much has landed -- after a dropped connection, say. Answer with
        // the true count so the sender can seek and resume, rather than
        // failing the whole transfer.
        const uint64_t staged = resume_offset_at(blobs_.staging, hash);
        if (offset != staged) {
            nlohmann::json out;
            out["hash"] = hash;
            out["received"] = staged;
            out["complete"] = false;
            out["resync"] = true;
            c.send(encode(Msg::BlobPutAck, out));
            return;
        }

        std::string err;
        if (!append_chunk_at(blobs_.staging, hash, offset, data, len, &err)) {
            c.send_error("chunk rejected: " + err);
            return;
        }

        const uint64_t have = resume_offset_at(blobs_.staging, hash);
        bool complete = false;
        if (total > 0 && have >= total) {
            // finalize verifies the content hash and discards on mismatch, so
            // a truncated or tampered upload never enters the store.
            if (!finalize_blob_at(blobs_.staging, hash, blobs_.path_of(hash),
                                  &err)) {
                logf("blob %s failed verification: %s", hash.c_str(),
                     err.c_str());
                c.send_error("blob failed verification: " + err);
                return;
            }
            complete = true;
            blobs_.enforce_quota(referenced_blobs());
        }

        nlohmann::json out;
        out["hash"] = hash;
        out["received"] = complete ? total : have;
        out["complete"] = complete;
        c.send(encode(Msg::BlobPutAck, out));
    }

    void on_blob_get(Conn &c, const Frame &f) {
        const nlohmann::json j = f.json();
        const std::string hash = j.value("hash", std::string{});
        const uint64_t offset = j.value("offset", (uint64_t)0);
        uint64_t want = j.value("len", (uint64_t)kChunkSize);
        if (want > kChunkSize) want = kChunkSize;

        if (!valid_blob_id(hash) || !blobs_.have(hash)) {
            c.send_error("no such blob " + hash);
            return;
        }

        const fs::path p = blobs_.path_of(hash);
        const uint64_t total = size_of_file(p);
        if (offset > total) {
            c.send_error("blob offset past end of file");
            return;
        }
        const uint64_t n = std::min<uint64_t>(want, total - offset);

        std::vector<uint8_t> buf(static_cast<size_t>(n));
        std::FILE *fp = std::fopen(p.string().c_str(), "rb");
        if (!fp) {
            c.send_error("cannot read blob " + hash);
            return;
        }
        bool ok = std::fseek(fp, static_cast<long>(offset), SEEK_SET) == 0;
        if (ok && n > 0)
            ok = std::fread(buf.data(), 1, buf.size(), fp) == buf.size();
        std::fclose(fp);
        if (!ok) {
            c.send_error("short read on blob " + hash);
            return;
        }

        nlohmann::json hdr;
        hdr["hash"] = hash;
        hdr["offset"] = offset;
        hdr["total"] = total;
        hdr["eof"] = (offset + n >= total);
        c.send(encode_blob(Msg::BlobChunk, hdr, buf.data(), buf.size()));
    }

    // Every blob any room's manifest still points at -- these survive eviction.
    std::vector<std::string> referenced_blobs() const {
        std::vector<std::string> out;
        for (const auto &kv : rooms_) {
            std::string content;
            if (!read_file(kv.second->manifest_path(), content, nullptr))
                continue;
            const nlohmann::json j = nlohmann::json::parse(
                content.begin(), content.end(), nullptr, false);
            if (j.is_discarded()) continue;
            Manifest m;
            if (!manifest_from_json(j, m, nullptr)) continue;
            for (const auto &e : m.entries) out.push_back(e.sha256);
        }
        return out;
    }

    fs::path        data_dir_;
    net::Listener   listener_;
    BlobStore       blobs_;
    std::map<std::string, std::unique_ptr<Room>> rooms_;
    std::vector<std::unique_ptr<Conn>> conns_;
};


}  // namespace relay
}  // namespace collab
