#pragma once
// collab_client.h -- the protocol client: one blocking request/response
// session against a relay.
//
// No threads and no GUI here. This is the piece the loopback test drives
// directly; collab_sync.h wraps it in a background thread and marshals results
// back to the main thread through DeferredQueue.
//
// Every call is a round trip: send a frame, wait for the matching reply. That
// is a deliberate simplification -- sync is eventual, runs on a timer, and
// moves a batch at a time, so pipelining would add concurrency bugs to save
// milliseconds nobody is waiting on. The one exception is blob transfer, which
// chunks in a loop but still acknowledges every chunk so it can resume.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "collab_bundle.h"
#include "collab_hash.h"
#include "collab_io.h"
#include "collab_ops.h"
#include "collab_proto.h"
#include "collab_socket.h"
#include "json.hpp"

namespace collab {

struct RelayConfig {
    std::string host;
    uint16_t    port = 7373;
    std::string room;
    std::string psk;
    int         timeout_ms = 15000;
};

struct PeerPresence {
    std::string peer;
    std::string display_name;
    int64_t     last_seen_ms = 0;
    int64_t     current_frame = -1;
};

// Reports bytes moved so the UI can show a progress bar on a multi-gigabyte
// transfer. Returning false cancels.
using TransferProgress =
    std::function<bool(uint64_t done, uint64_t total)>;

class RelaySession {
  public:
    ~RelaySession() { close(); }

    bool connected() const { return sock_.valid(); }
    const RoomBinding &room_binding() const { return room_binding_; }
    uint64_t relay_seq() const { return relay_seq_; }

    void close() {
        if (sock_.valid()) {
            sock_.shutdown_write();
            sock_.close();
        }
        in_buf_.clear();
    }

    // Connects and completes the HMAC challenge-response.
    //
    // `binding` is this project's shape. The relay adopts it if the room is
    // new and enforces it otherwise, so a peer with a different skeleton or
    // camera list is refused here rather than corrupting annotations later.
    bool connect(const RelayConfig &cfg, const std::string &peer_id,
                 const std::string &display_name, const RoomBinding &binding,
                 std::string *err) {
        close();
        cfg_ = cfg;
        peer_ = peer_id;

        if (!net::connect_to(cfg.host, cfg.port, cfg.timeout_ms, sock_, err))
            return false;

        nlohmann::json hello;
        hello["version"] = proto::kVersion;
        hello["room"] = cfg.room;
        hello["peer"] = peer_id;
        hello["display_name"] = display_name;
        if (!send(proto::encode(proto::Msg::Hello, hello), err)) return false;

        proto::Frame f;
        if (!recv(f, err)) return false;
        if (f.type == proto::Msg::Deny) return denied(f, err);
        if (f.type != proto::Msg::Challenge) {
            if (err) *err = unexpected(f, "Challenge");
            return false;
        }

        const std::string nonce = f.json().value("nonce", std::string{});
        if (nonce.empty()) {
            if (err) *err = "relay sent an empty challenge";
            return false;
        }

        nlohmann::json jb;
        to_json(jb, binding);
        nlohmann::json auth;
        auth["tag"] = hmac_sha256_hex(
            cfg.psk, proto::auth_message(nonce, cfg.room, peer_id));
        auth["binding"] = jb;
        if (!send(proto::encode(proto::Msg::Auth, auth), err)) return false;

        if (!recv(f, err)) return false;
        if (f.type == proto::Msg::Deny) return denied(f, err);
        if (f.type != proto::Msg::Welcome) {
            if (err) *err = unexpected(f, "Welcome");
            return false;
        }

        const nlohmann::json w = f.json();
        relay_seq_ = w.value("relay_seq", (uint64_t)0);
        if (w.contains("binding")) from_json(w["binding"], room_binding_);
        return true;
    }

    void bye() {
        if (!sock_.valid()) return;
        std::string ignored;
        send(proto::encode(proto::Msg::Bye), &ignored);
        close();
    }

    // ── ops ──

    // `duplicate` (optional) counts ops the relay already had. A client
    // re-pushes after a partial sync failure, so a non-zero count here is
    // normal recovery, not an error.
    bool push_ops(const std::vector<Op> &ops, int &accepted, int &rejected,
                  uint64_t &relay_seq, std::string *err,
                  int *duplicate = nullptr) {
        nlohmann::json jops = nlohmann::json::array();
        for (const Op &op : ops) {
            nlohmann::json j;
            to_json(j, op);
            jops.push_back(std::move(j));
        }
        nlohmann::json body;
        body["ops"] = jops;
        if (!send(proto::encode(proto::Msg::PushOps, body), err)) return false;

        proto::Frame f;
        if (!recv(f, err)) return false;
        if (!expect(f, proto::Msg::PushAck, "PushAck", err)) return false;

        const nlohmann::json j = f.json();
        accepted = j.value("accepted", 0);
        rejected = j.value("rejected", 0);
        if (duplicate) *duplicate = j.value("duplicate", 0);
        relay_seq = j.value("relay_seq", (uint64_t)0);
        relay_seq_ = relay_seq;
        return true;
    }

    bool pull_ops(uint64_t since, std::vector<Op> &out, uint64_t &high_seq,
                  bool &more, int &malformed, std::string *err) {
        out.clear();
        malformed = 0;

        nlohmann::json body;
        body["since_seq"] = since;
        if (!send(proto::encode(proto::Msg::PullOps, body), err)) return false;

        proto::Frame f;
        if (!recv(f, err)) return false;
        if (!expect(f, proto::Msg::OpsBatch, "OpsBatch", err)) return false;

        const nlohmann::json j = f.json();
        if (j.contains("ops") && j["ops"].is_array()) {
            for (const auto &jop : j["ops"]) {
                Op op;
                // Ops come from other peers via a relay that stores them
                // opaquely, so this is the first place they are really
                // validated. A bad one is counted and dropped, never applied.
                if (!op_from_json(jop, op)) {
                    ++malformed;
                    continue;
                }
                out.push_back(std::move(op));
            }
        }
        high_seq = j.value("relay_seq", since);
        more = j.value("more", false);
        relay_seq_ = high_seq;
        return true;
    }

    // ── presence ──

    bool presence(int64_t current_frame, std::vector<PeerPresence> &out,
                  std::string *err) {
        out.clear();
        nlohmann::json body;
        body["current_frame"] = current_frame;
        if (!send(proto::encode(proto::Msg::Presence, body), err)) return false;

        proto::Frame f;
        if (!recv(f, err)) return false;
        if (!expect(f, proto::Msg::PresenceList, "PresenceList", err))
            return false;

        const nlohmann::json j = f.json();
        if (j.contains("peers") && j["peers"].is_array()) {
            for (const auto &jp : j["peers"]) {
                PeerPresence p;
                p.peer = jp.value("peer", std::string{});
                p.display_name = jp.value("display_name", std::string{});
                p.last_seen_ms = jp.value("last_seen_ms", (int64_t)0);
                p.current_frame = jp.value("current_frame", (int64_t)-1);
                out.push_back(std::move(p));
            }
        }
        return true;
    }

    // ── manifests ──

    bool put_manifest(const Manifest &m, std::string *err) {
        nlohmann::json jm;
        to_json(jm, m);
        nlohmann::json body;
        body["manifest"] = jm;
        if (!send(proto::encode(proto::Msg::PutManifest, body), err))
            return false;

        proto::Frame f;
        if (!recv(f, err)) return false;
        return expect(f, proto::Msg::Manifest, "Manifest", err);
    }

    bool get_manifest(Manifest &out, bool &present, std::string *err) {
        present = false;
        if (!send(proto::encode(proto::Msg::GetManifest), err)) return false;

        proto::Frame f;
        if (!recv(f, err)) return false;
        if (!expect(f, proto::Msg::Manifest, "Manifest", err)) return false;

        const nlohmann::json j = f.json();
        if (!j.value("present", false)) return true;
        if (!j.contains("manifest")) return true;
        if (!manifest_from_json(j["manifest"], out, err)) return false;
        present = true;
        return true;
    }

    // ── blobs ──

    // Asks which of `hashes` the relay does NOT have. This is what makes
    // "everyone already has the videos" cost nothing.
    bool blobs_needed(const std::vector<std::string> &hashes,
                      std::vector<std::string> &needed, std::string *err) {
        needed.clear();
        nlohmann::json body;
        body["hashes"] = hashes;
        if (!send(proto::encode(proto::Msg::HaveBlobs, body), err)) return false;

        proto::Frame f;
        if (!recv(f, err)) return false;
        if (!expect(f, proto::Msg::BlobsNeeded, "BlobsNeeded", err))
            return false;

        const nlohmann::json j = f.json();
        if (j.contains("hashes") && j["hashes"].is_array())
            for (const auto &h : j["hashes"])
                if (h.is_string()) needed.push_back(h.get<std::string>());
        return true;
    }

    // Uploads a local file, resuming from whatever the relay already holds.
    bool put_blob(const std::string &hash, const fs::path &file,
                  const TransferProgress &progress, std::string *err) {
        if (!valid_blob_id(hash)) {
            if (err) *err = "malformed blob id";
            return false;
        }
        const uint64_t total = size_of_file(file);

        std::FILE *fp = std::fopen(file.string().c_str(), "rb");
        if (!fp) {
            if (err) *err = "cannot open " + file.string();
            return false;
        }

        uint64_t offset = 0;
        std::vector<uint8_t> buf(proto::kChunkSize);
        bool ok = true;

        for (;;) {
            if (progress && !progress(offset, total)) {
                if (err) *err = "cancelled";
                ok = false;
                break;
            }

            const uint64_t remain = (total > offset) ? total - offset : 0;
            const size_t want =
                static_cast<size_t>(std::min<uint64_t>(remain, proto::kChunkSize));

            if (std::fseek(fp, static_cast<long>(offset), SEEK_SET) != 0) {
                if (err) *err = "seek failed on " + file.string();
                ok = false;
                break;
            }
            if (want > 0 && std::fread(buf.data(), 1, want, fp) != want) {
                if (err) *err = "short read on " + file.string();
                ok = false;
                break;
            }

            nlohmann::json hdr;
            hdr["hash"] = hash;
            hdr["offset"] = offset;
            hdr["total"] = total;
            if (!send(proto::encode_blob(proto::Msg::BlobPut, hdr, buf.data(),
                                         want),
                      err)) {
                ok = false;
                break;
            }

            proto::Frame f;
            if (!recv(f, err)) { ok = false; break; }
            if (!expect(f, proto::Msg::BlobPutAck, "BlobPutAck", err)) {
                ok = false;
                break;
            }

            const nlohmann::json j = f.json();
            if (j.value("complete", false)) {
                if (progress) progress(total, total);
                break;
            }
            // The relay reports what it actually holds; on a resync this
            // rewinds or fast-forwards us to the truth.
            offset = j.value("received", offset + want);
            if (offset >= total && total > 0) {
                // Sent everything but the relay has not verified it yet; one
                // more empty chunk lets it finalize.
                continue;
            }
        }

        std::fclose(fp);
        return ok;
    }

    // Downloads into the project's staging area, resuming from any .part left
    // by an earlier attempt, then verifies and moves it into place.
    bool get_blob(const std::string &hash, const fs::path &project_root,
                  const fs::path &dest, const TransferProgress &progress,
                  std::string *err) {
        if (!valid_blob_id(hash)) {
            if (err) *err = "malformed blob id";
            return false;
        }

        uint64_t offset = resume_offset(project_root, hash);
        uint64_t total = 0;

        for (;;) {
            nlohmann::json body;
            body["hash"] = hash;
            body["offset"] = offset;
            body["len"] = proto::kChunkSize;
            if (!send(proto::encode(proto::Msg::BlobGet, body), err))
                return false;

            proto::Frame f;
            if (!recv(f, err)) return false;
            if (!expect(f, proto::Msg::BlobChunk, "BlobChunk", err))
                return false;

            nlohmann::json hdr;
            const uint8_t *data = nullptr;
            size_t len = 0;
            if (!decode_blob(f, hdr, data, len)) {
                if (err) *err = "malformed BlobChunk";
                return false;
            }
            total = hdr.value("total", (uint64_t)0);

            if (len > 0 &&
                !append_chunk(project_root, hash, offset, data, len, err))
                return false;
            offset += len;

            if (progress && !progress(offset, total)) {
                // Cancelling leaves the .part in place on purpose, so the next
                // attempt resumes instead of starting over.
                if (err) *err = "cancelled";
                return false;
            }

            if (hdr.value("eof", false) || (total > 0 && offset >= total)) break;
            if (len == 0) {
                if (err) *err = "relay stopped sending before end of blob";
                return false;
            }
        }

        // finalize verifies the digest and discards on mismatch, so a
        // truncated or tampered transfer can never land as a valid file.
        return finalize_blob(project_root, hash, dest, err);
    }

  private:
    bool send(const std::vector<uint8_t> &frame, std::string *err) {
        if (!sock_.valid()) {
            if (err) *err = "not connected";
            return false;
        }
        return sock_.send_all(frame, err);
    }

    // Blocks until one whole frame is available. Socket timeouts mean this
    // returns rather than hanging forever if the relay goes silent.
    bool recv(proto::Frame &out, std::string *err) {
        for (;;) {
            size_t consumed = 0;
            std::string derr;
            const proto::Decode r = proto::decode(in_buf_, out, consumed, &derr);
            if (r == proto::Decode::Ok) {
                in_buf_.erase(in_buf_.begin(), in_buf_.begin() + consumed);
                if (out.type == proto::Msg::Error) {
                    if (err)
                        *err = "relay: " +
                               out.json().value("message", std::string("error"));
                    return false;
                }
                return true;
            }
            if (r == proto::Decode::Bad) {
                if (err) *err = "protocol error: " + derr;
                close();
                return false;
            }

            uint8_t buf[65536];
            size_t got = 0;
            const auto res = sock_.recv_some(buf, sizeof(buf), got, err);
            if (res == net::Socket::RecvResult::Data) {
                if (in_buf_.size() + got > proto::kMaxPendingBytes) {
                    if (err) *err = "relay sent more than the frame limit allows";
                    close();
                    return false;
                }
                in_buf_.insert(in_buf_.end(), buf, buf + got);
                continue;
            }
            if (res == net::Socket::RecvResult::Timeout) {
                if (err) *err = "timed out waiting for the relay";
                return false;
            }
            if (res == net::Socket::RecvResult::Closed) {
                if (err) *err = "relay closed the connection";
                close();
                return false;
            }
            close();
            return false;
        }
    }

    bool expect(const proto::Frame &f, proto::Msg want, const char *name,
                std::string *err) {
        if (f.type == want) return true;
        if (f.type == proto::Msg::Deny) return denied(f, err);
        if (err) *err = unexpected(f, name);
        return false;
    }

    bool denied(const proto::Frame &f, std::string *err) {
        if (err)
            *err = f.json().value("reason", std::string("relay denied the "
                                                        "connection"));
        close();
        return false;
    }

    static std::string unexpected(const proto::Frame &f, const char *want) {
        return std::string("expected ") + want + " from the relay, got " +
               proto::msg_name(f.type);
    }

    RelayConfig          cfg_;
    std::string          peer_;
    net::Socket          sock_;
    std::vector<uint8_t> in_buf_;
    RoomBinding          room_binding_;
    uint64_t             relay_seq_ = 0;
};

}  // namespace collab
