// test_collab_proto.cpp -- framing tests for src/collab_proto.h
//
// The decoder is the first thing an untrusted peer touches, so the interesting
// cases are all the malformed ones: truncation, oversize length prefixes,
// unknown types, and garbage.

#include "test_framework.h"

#include "collab_proto.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace collab::proto;

static void test_roundtrip_json() {
    nlohmann::json j;
    j["room"] = "rig-a";
    j["peer"] = "0123abcd";
    j["n"] = 42;

    const std::vector<uint8_t> wire = encode(Msg::Hello, j);

    Frame f;
    size_t consumed = 0;
    std::string err;
    EXPECT_TRUE(decode(wire, f, consumed, &err) == Decode::Ok);
    EXPECT_EQ(consumed, wire.size());
    EXPECT_TRUE(f.type == Msg::Hello);

    const nlohmann::json back = f.json();
    EXPECT_FALSE(back.is_discarded());
    EXPECT_STR_EQ(back.value("room", std::string{}), "rig-a");
    EXPECT_STR_EQ(back.value("peer", std::string{}), "0123abcd");
    EXPECT_EQ(back.value("n", 0), 42);
}

static void test_empty_payload() {
    const std::vector<uint8_t> wire = encode(Msg::Bye);
    Frame f;
    size_t consumed = 0;
    EXPECT_TRUE(decode(wire, f, consumed, nullptr) == Decode::Ok);
    EXPECT_TRUE(f.type == Msg::Bye);
    EXPECT_EQ(consumed, wire.size());
}

// Several frames in one buffer must come out one at a time, in order, with
// the right consumed counts -- this is exactly what a socket read produces.
static void test_multiple_frames_in_one_buffer() {
    std::vector<uint8_t> buf;
    for (int i = 0; i < 3; ++i) {
        nlohmann::json j;
        j["i"] = i;
        const std::vector<uint8_t> w = encode(Msg::Presence, j);
        buf.insert(buf.end(), w.begin(), w.end());
    }

    size_t off = 0;
    for (int i = 0; i < 3; ++i) {
        Frame f;
        size_t consumed = 0;
        EXPECT_TRUE(decode(buf.data() + off, buf.size() - off, f, consumed,
                           nullptr) == Decode::Ok);
        EXPECT_TRUE(f.type == Msg::Presence);
        EXPECT_EQ(f.json().value("i", -1), i);
        off += consumed;
    }
    EXPECT_EQ(off, buf.size());

    // Nothing left: a further decode wants more input, it does not error.
    Frame f;
    size_t consumed = 0;
    EXPECT_TRUE(decode(buf.data() + off, buf.size() - off, f, consumed,
                       nullptr) == Decode::NeedMore);
}

// Every prefix of a valid frame must report NeedMore and consume nothing.
// A decoder that consumed on a partial read would desynchronize the stream.
static void test_truncation_at_every_prefix() {
    nlohmann::json j;
    j["payload"] = std::string(200, 'z');
    const std::vector<uint8_t> wire = encode(Msg::PushOps, j);

    for (size_t n = 0; n < wire.size(); ++n) {
        Frame f;
        size_t consumed = 12345;
        const Decode r = decode(wire.data(), n, f, consumed, nullptr);
        EXPECT_TRUE(r == Decode::NeedMore);
        EXPECT_EQ(consumed, (size_t)0);
    }

    Frame f;
    size_t consumed = 0;
    EXPECT_TRUE(decode(wire.data(), wire.size(), f, consumed, nullptr) ==
                Decode::Ok);
}

// An attacker-controlled length prefix must be refused BEFORE any allocation.
static void test_oversize_length_rejected() {
    std::vector<uint8_t> buf;
    put_u32(buf, kMaxFrameSize + 1);
    buf.push_back(static_cast<uint8_t>(Msg::Hello));

    Frame f;
    size_t consumed = 0;
    std::string err;
    EXPECT_TRUE(decode(buf, f, consumed, &err) == Decode::Bad);
    EXPECT_TRUE(!err.empty());

    // 0xFFFFFFFF is the pathological case: it must not wrap or over-reserve.
    std::vector<uint8_t> huge;
    put_u32(huge, 0xFFFFFFFFu);
    huge.push_back(static_cast<uint8_t>(Msg::Hello));
    EXPECT_TRUE(decode(huge, f, consumed, &err) == Decode::Bad);

    // Exactly at the limit is still structurally acceptable (it will simply
    // wait for the rest of the bytes).
    std::vector<uint8_t> at_limit;
    put_u32(at_limit, kMaxFrameSize);
    at_limit.push_back(static_cast<uint8_t>(Msg::Hello));
    EXPECT_TRUE(decode(at_limit, f, consumed, &err) == Decode::NeedMore);
}

static void test_zero_length_rejected() {
    std::vector<uint8_t> buf;
    put_u32(buf, 0);
    Frame f;
    size_t consumed = 0;
    std::string err;
    EXPECT_TRUE(decode(buf, f, consumed, &err) == Decode::Bad);
}

static void test_unknown_type_rejected() {
    std::vector<uint8_t> buf;
    put_u32(buf, 3);
    buf.push_back(0xEE);  // not a defined Msg
    buf.push_back('h');
    buf.push_back('i');

    Frame f;
    size_t consumed = 0;
    std::string err;
    EXPECT_TRUE(decode(buf, f, consumed, &err) == Decode::Bad);
    EXPECT_TRUE(err.find("unknown") != std::string::npos);
}

// Random bytes must never be accepted as a frame with a plausible payload.
static void test_garbage() {
    std::vector<uint8_t> junk;
    for (int i = 0; i < 256; ++i) junk.push_back(static_cast<uint8_t>(i * 7));

    Frame f;
    size_t consumed = 0;
    const Decode r = decode(junk, f, consumed, nullptr);
    // The first four bytes are 0x00 0x07 0x0e 0x15 -> a huge length, so this
    // must be refused rather than parsed.
    EXPECT_TRUE(r == Decode::Bad || r == Decode::NeedMore);
    if (r != Decode::Ok) EXPECT_EQ(consumed, (size_t)0);
}

// Malformed JSON inside an otherwise well-framed message must surface as a
// discarded value, not an exception -- the parse runs on peer-controlled data.
static void test_bad_json_does_not_throw() {
    const std::vector<uint8_t> wire = encode(Msg::Hello, std::string("{not json"));
    Frame f;
    size_t consumed = 0;
    EXPECT_TRUE(decode(wire, f, consumed, nullptr) == Decode::Ok);
    EXPECT_TRUE(f.json().is_discarded());
}

// ── blob frames ──

static void test_blob_roundtrip() {
    std::vector<uint8_t> data(1000);
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<uint8_t>(i & 0xff);

    nlohmann::json hdr;
    hdr["hash"] = "deadbeef";
    hdr["offset"] = 4096;
    hdr["total"] = 1000000;

    const std::vector<uint8_t> wire =
        encode_blob(Msg::BlobPut, hdr, data.data(), data.size());

    Frame f;
    size_t consumed = 0;
    EXPECT_TRUE(decode(wire, f, consumed, nullptr) == Decode::Ok);
    EXPECT_TRUE(f.type == Msg::BlobPut);
    EXPECT_EQ(consumed, wire.size());

    nlohmann::json got_hdr;
    const uint8_t *raw = nullptr;
    size_t raw_len = 0;
    EXPECT_TRUE(decode_blob(f, got_hdr, raw, raw_len));
    EXPECT_STR_EQ(got_hdr.value("hash", std::string{}), "deadbeef");
    EXPECT_EQ(got_hdr.value("offset", 0), 4096);
    EXPECT_EQ(raw_len, data.size());
    bool same = true;
    for (size_t i = 0; i < raw_len; ++i)
        if (raw[i] != data[i]) { same = false; break; }
    EXPECT_TRUE(same);
}

// Raw bytes that happen to contain the framing pattern must survive intact --
// the length prefix, not scanning, is what delimits frames.
static void test_blob_with_framelike_bytes() {
    std::vector<uint8_t> data;
    put_u32(data, 5);
    data.push_back(static_cast<uint8_t>(Msg::Hello));
    for (int i = 0; i < 100; ++i) data.push_back(0x00);

    nlohmann::json hdr;
    hdr["hash"] = "abc";
    const std::vector<uint8_t> wire =
        encode_blob(Msg::BlobChunk, hdr, data.data(), data.size());

    Frame f;
    size_t consumed = 0;
    EXPECT_TRUE(decode(wire, f, consumed, nullptr) == Decode::Ok);
    EXPECT_EQ(consumed, wire.size());

    nlohmann::json got_hdr;
    const uint8_t *raw = nullptr;
    size_t raw_len = 0;
    EXPECT_TRUE(decode_blob(f, got_hdr, raw, raw_len));
    EXPECT_EQ(raw_len, data.size());
}

static void test_blob_empty_payload() {
    nlohmann::json hdr;
    hdr["hash"] = "abc";
    hdr["eof"] = true;
    const std::vector<uint8_t> wire =
        encode_blob(Msg::BlobChunk, hdr, nullptr, 0);

    Frame f;
    size_t consumed = 0;
    EXPECT_TRUE(decode(wire, f, consumed, nullptr) == Decode::Ok);

    nlohmann::json got_hdr;
    const uint8_t *raw = nullptr;
    size_t raw_len = 0;
    EXPECT_TRUE(decode_blob(f, got_hdr, raw, raw_len));
    EXPECT_EQ(raw_len, (size_t)0);
    EXPECT_TRUE(got_hdr.value("eof", false));
}

// A header length that runs past the end of the payload is the obvious
// hostile blob frame; it must be rejected, not read out of bounds.
static void test_blob_bad_header_length() {
    std::vector<uint8_t> payload;
    put_u32(payload, 9999);          // claims a 9999-byte header
    payload.push_back('{');
    payload.push_back('}');

    Frame f;
    f.type = Msg::BlobPut;
    f.payload = payload;

    nlohmann::json hdr;
    const uint8_t *raw = nullptr;
    size_t raw_len = 0;
    EXPECT_FALSE(decode_blob(f, hdr, raw, raw_len));

    // Too short to even hold the header length.
    Frame tiny;
    tiny.type = Msg::BlobPut;
    tiny.payload = {0x00, 0x01};
    EXPECT_FALSE(decode_blob(tiny, hdr, raw, raw_len));
}

// ── auth binding ──

// The tag must bind nonce, room, and peer together. If the message were just
// the nonce, a tag captured in one room would replay into another.
static void test_auth_message_binding() {
    EXPECT_STR_EQ(auth_message("aabb", "rig-a", "peer1"), "aabb|rig-a|peer1");
    EXPECT_TRUE(auth_message("aabb", "rig-a", "peer1") !=
                auth_message("aabb", "rig-b", "peer1"));
    EXPECT_TRUE(auth_message("aabb", "rig-a", "peer1") !=
                auth_message("aabb", "rig-a", "peer2"));
    EXPECT_TRUE(auth_message("aabb", "rig-a", "peer1") !=
                auth_message("ccdd", "rig-a", "peer1"));
}

static void test_msg_names() {
    EXPECT_STR_EQ(msg_name(Msg::Hello), "Hello");
    EXPECT_STR_EQ(msg_name(Msg::BlobChunk), "BlobChunk");
    EXPECT_TRUE(is_known_msg(static_cast<uint8_t>(Msg::Welcome)));
    EXPECT_FALSE(is_known_msg(0));
    EXPECT_FALSE(is_known_msg(200));
}

int main() {
    test_roundtrip_json();
    test_empty_payload();
    test_multiple_frames_in_one_buffer();
    test_truncation_at_every_prefix();
    test_oversize_length_rejected();
    test_zero_length_rejected();
    test_unknown_type_rejected();
    test_garbage();
    test_bad_json_does_not_throw();
    test_blob_roundtrip();
    test_blob_with_framelike_bytes();
    test_blob_empty_payload();
    test_blob_bad_header_length();
    test_auth_message_binding();
    test_msg_names();

    std::printf("test_collab_proto: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
