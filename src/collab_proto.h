#pragma once
// collab_proto.h -- wire format for the RED collaboration relay.
//
// Deliberately free of any socket call: everything here is a pure function
// over byte buffers, so the framing can be unit-tested against truncation,
// oversize, and garbage input without opening a port. collab_socket.h is the
// only file that talks to the network.
//
// Framing
//   [u32 big-endian length][u8 type][payload...]
//   `length` counts the type byte plus the payload, so a minimal frame is 1.
//   Big-endian is written by hand rather than via htonl to keep this header
//   free of platform includes.
//
// Payloads are UTF-8 JSON, except the blob-transfer messages which carry
//   [u32 header length][header JSON][raw bytes]
// so a 4 MiB chunk moves without base64 inflating it by a third.
//
// SECURITY: frames are AUTHENTICATED but NOT ENCRYPTED. A network observer
// sees annotation coordinates, comments, and file contents in the clear. This
// is a deliberate, documented choice -- shipping a hand-rolled cipher would be
// worse. Operators who need confidentiality run the relay behind an SSH tunnel
// or a WireGuard/Tailscale network; see the README section this points to.

#include <cstdint>
#include <string>
#include <vector>

#include "json.hpp"

namespace collab {
namespace proto {

// Bumped only on an incompatible framing or handshake change. The relay
// refuses a client whose version it does not recognize, with a clear reason,
// rather than half-speaking an older dialect.
constexpr uint32_t kVersion = 1;

// A frame must fit comfortably in memory on both ends; the largest legitimate
// one is a blob chunk plus its header.
constexpr uint32_t kChunkSize    = 4u * 1024 * 1024;
constexpr uint32_t kMaxFrameSize = 8u * 1024 * 1024;

// Frames are read into a growable buffer; this caps how much unparsed input we
// will hold before declaring the peer hostile or broken.
constexpr size_t kMaxPendingBytes = 2u * kMaxFrameSize;

// ── Message types ──
enum class Msg : uint8_t {
    // Handshake
    Hello        = 1,   // C->R {version, room, peer, display_name, skeleton{...}}
    Challenge    = 2,   // R->C {nonce}
    Auth         = 3,   // C->R {tag}
    Welcome      = 4,   // R->C {relay_seq, room{...}}
    Deny         = 5,   // R->C {reason}

    // Op sync
    PushOps      = 10,  // C->R {ops:[...]}
    PushAck      = 11,  // R->C {accepted, relay_seq}
    PullOps      = 12,  // C->R {since_seq, max}
    OpsBatch     = 13,  // R->C {ops:[...], relay_seq, more}

    // Presence
    Presence     = 20,  // C->R {current_frame}
    PresenceList = 21,  // R->C {peers:[...]}

    // Project sharing
    PutManifest  = 30,  // C->R {manifest}
    GetManifest  = 31,  // C->R {}
    Manifest     = 32,  // R->C {present, manifest}
    HaveBlobs    = 33,  // C->R {hashes:[...]}  "I hold these"
    BlobsNeeded  = 34,  // R->C {hashes:[...]}  "send me these"
    BlobPut      = 35,  // C->R hdr{hash, offset, total} + raw
    BlobPutAck   = 36,  // R->C {hash, received, complete}
    BlobGet      = 37,  // C->R {hash, offset, len}
    BlobChunk    = 38,  // R->C hdr{hash, offset, total, eof} + raw

    // Either direction
    Error        = 90,  // {message}
    Bye          = 91,  // {}
};

inline const char *msg_name(Msg m) {
    switch (m) {
        case Msg::Hello:        return "Hello";
        case Msg::Challenge:    return "Challenge";
        case Msg::Auth:         return "Auth";
        case Msg::Welcome:      return "Welcome";
        case Msg::Deny:         return "Deny";
        case Msg::PushOps:      return "PushOps";
        case Msg::PushAck:      return "PushAck";
        case Msg::PullOps:      return "PullOps";
        case Msg::OpsBatch:     return "OpsBatch";
        case Msg::Presence:     return "Presence";
        case Msg::PresenceList: return "PresenceList";
        case Msg::PutManifest:  return "PutManifest";
        case Msg::GetManifest:  return "GetManifest";
        case Msg::Manifest:     return "Manifest";
        case Msg::HaveBlobs:    return "HaveBlobs";
        case Msg::BlobsNeeded:  return "BlobsNeeded";
        case Msg::BlobPut:      return "BlobPut";
        case Msg::BlobPutAck:   return "BlobPutAck";
        case Msg::BlobGet:      return "BlobGet";
        case Msg::BlobChunk:    return "BlobChunk";
        case Msg::Error:        return "Error";
        case Msg::Bye:          return "Bye";
    }
    return "Unknown";
}

// Rejects any type byte that is not one of the above, so a malformed or
// hostile peer is dropped at the frame layer rather than deeper in.
inline bool is_known_msg(uint8_t t) {
    switch (static_cast<Msg>(t)) {
        case Msg::Hello: case Msg::Challenge: case Msg::Auth:
        case Msg::Welcome: case Msg::Deny:
        case Msg::PushOps: case Msg::PushAck: case Msg::PullOps:
        case Msg::OpsBatch:
        case Msg::Presence: case Msg::PresenceList:
        case Msg::PutManifest: case Msg::GetManifest: case Msg::Manifest:
        case Msg::HaveBlobs: case Msg::BlobsNeeded:
        case Msg::BlobPut: case Msg::BlobPutAck:
        case Msg::BlobGet: case Msg::BlobChunk:
        case Msg::Error: case Msg::Bye:
            return true;
    }
    return false;
}

// ── Big-endian scalars ──

inline void put_u32(std::vector<uint8_t> &out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v >> 24));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v));
}

inline uint32_t get_u32(const uint8_t *p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           (static_cast<uint32_t>(p[3]));
}

// ── Encoding ──

struct Frame {
    Msg type = Msg::Error;
    std::vector<uint8_t> payload;

    // Parses the payload as JSON. Returns an empty object on malformed input
    // rather than throwing -- every caller is handling untrusted peer bytes.
    nlohmann::json json() const {
        return nlohmann::json::parse(payload.begin(), payload.end(), nullptr,
                                     /*allow_exceptions=*/false);
    }
};

inline std::vector<uint8_t> encode(Msg type, const std::string &payload) {
    std::vector<uint8_t> out;
    out.reserve(5 + payload.size());
    put_u32(out, static_cast<uint32_t>(1 + payload.size()));
    out.push_back(static_cast<uint8_t>(type));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

inline std::vector<uint8_t> encode(Msg type, const nlohmann::json &j) {
    return encode(type, j.dump());
}

inline std::vector<uint8_t> encode(Msg type) {
    return encode(type, std::string("{}"));
}

// Blob frame: [u32 hdr_len][hdr json][raw bytes]. Keeping the raw bytes
// unencoded is the whole point -- base64 would add ~33% to every byte of a
// multi-gigabyte media transfer.
inline std::vector<uint8_t> encode_blob(Msg type, const nlohmann::json &hdr,
                                        const void *data, size_t len) {
    const std::string h = hdr.dump();
    std::vector<uint8_t> payload;
    payload.reserve(4 + h.size() + len);
    put_u32(payload, static_cast<uint32_t>(h.size()));
    payload.insert(payload.end(), h.begin(), h.end());
    const uint8_t *p = static_cast<const uint8_t *>(data);
    payload.insert(payload.end(), p, p + len);

    std::vector<uint8_t> out;
    out.reserve(5 + payload.size());
    put_u32(out, static_cast<uint32_t>(1 + payload.size()));
    out.push_back(static_cast<uint8_t>(type));
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

// Splits a blob frame's payload back into its header and its raw bytes. The
// raw part is returned as a view into `f`, valid only while `f` lives.
inline bool decode_blob(const Frame &f, nlohmann::json &hdr,
                        const uint8_t *&data, size_t &len) {
    if (f.payload.size() < 4) return false;
    const uint32_t hdr_len = get_u32(f.payload.data());
    if (hdr_len > f.payload.size() - 4) return false;
    hdr = nlohmann::json::parse(f.payload.begin() + 4,
                                f.payload.begin() + 4 + hdr_len, nullptr,
                                /*allow_exceptions=*/false);
    if (hdr.is_discarded()) return false;
    data = f.payload.data() + 4 + hdr_len;
    len = f.payload.size() - 4 - hdr_len;
    return true;
}

// ── Decoding ──

enum class Decode {
    Ok,        // a whole frame was parsed; `consumed` bytes may be dropped
    NeedMore,  // a valid prefix so far; read more and retry
    Bad,       // malformed or hostile; close the connection
};

// Attempts to pull one frame off the front of `buf`.
//
// On Ok, `consumed` is how many bytes of `buf` the frame occupied. On NeedMore
// nothing is consumed. On Bad the caller must drop the connection -- there is
// no resynchronization point in a length-prefixed stream, so trying to skip
// ahead would just interpret payload bytes as a header.
inline Decode decode(const uint8_t *buf, size_t avail, Frame &out,
                     size_t &consumed, std::string *err = nullptr) {
    consumed = 0;
    if (avail < 4) return Decode::NeedMore;

    const uint32_t len = get_u32(buf);
    if (len < 1) {
        if (err) *err = "zero-length frame";
        return Decode::Bad;
    }
    if (len > kMaxFrameSize) {
        // Refuse before allocating: this is the memory-exhaustion guard.
        if (err)
            *err = "frame of " + std::to_string(len) + " bytes exceeds the " +
                   std::to_string(kMaxFrameSize) + " byte limit";
        return Decode::Bad;
    }
    if (avail < 4 + static_cast<size_t>(len)) return Decode::NeedMore;

    const uint8_t type = buf[4];
    if (!is_known_msg(type)) {
        if (err) *err = "unknown message type " + std::to_string(type);
        return Decode::Bad;
    }

    out.type = static_cast<Msg>(type);
    out.payload.assign(buf + 5, buf + 4 + len);
    consumed = 4 + static_cast<size_t>(len);
    return Decode::Ok;
}

inline Decode decode(const std::vector<uint8_t> &buf, Frame &out,
                     size_t &consumed, std::string *err = nullptr) {
    return decode(buf.data(), buf.size(), out, consumed, err);
}

// ── Handshake payload helpers ──
//
// The auth tag binds the nonce to the room AND the peer id, so a tag captured
// from one room cannot be replayed into another on the same relay.
inline std::string auth_message(const std::string &nonce_hex,
                                const std::string &room,
                                const std::string &peer) {
    return nonce_hex + "|" + room + "|" + peer;
}

}  // namespace proto
}  // namespace collab
