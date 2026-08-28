// test_collab_hash.cpp -- known-answer tests for src/collab_hash.h
//
// Vectors are the published ones: SHA-256 from FIPS 180-4 / NESSIE, and
// HMAC-SHA256 from RFC 4231. These are not self-consistency checks -- a
// from-scratch HMAC that is subtly wrong still round-trips against itself
// and would accept nothing, or worse, accept everything.

#include "test_framework.h"

#include "collab_hash.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace collab;

static std::string rep(char c, size_t n) { return std::string(n, c); }

static std::string bytes(uint8_t b, size_t n) {
    return std::string(n, static_cast<char>(b));
}

// ── SHA-256, FIPS 180-4 examples ──
static void test_sha256_vectors() {
    EXPECT_STR_EQ(
        sha256_hex(std::string("")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    EXPECT_STR_EQ(
        sha256_hex(std::string("abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    // 448-bit (two-block) message
    EXPECT_STR_EQ(
        sha256_hex(std::string(
            "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");

    // 896-bit message
    EXPECT_STR_EQ(
        sha256_hex(std::string(
            "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
            "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu")),
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");

    // One million 'a' -- exercises the multi-block streaming path and the
    // 64-bit length suffix.
    EXPECT_STR_EQ(
        sha256_hex(rep('a', 1000000)),
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// Feeding the same bytes in awkward chunk sizes must not change the digest.
// This is the property the file-streaming and socket paths actually rely on.
static void test_sha256_incremental() {
    const std::string msg = rep('x', 1000) + rep('y', 137) + rep('z', 64);
    const std::string want = sha256_hex(msg);

    const size_t chunks[] = {1, 7, 63, 64, 65, 127, 128, 999};
    for (size_t step : chunks) {
        Sha256 h;
        for (size_t i = 0; i < msg.size(); i += step) {
            const size_t n = (i + step > msg.size()) ? msg.size() - i : step;
            h.update(msg.data() + i, n);
        }
        EXPECT_STR_EQ(to_hex(h.finish()), want);
    }

    // Boundary lengths around the 55/56/64 padding cases.
    const size_t lens[] = {0, 1, 54, 55, 56, 57, 63, 64, 65, 119, 120, 128};
    for (size_t n : lens) {
        const std::string s = rep('a', n);
        Sha256 h;
        h.update(s.data(), s.size());
        EXPECT_STR_EQ(to_hex(h.finish()), sha256_hex(s));
    }
}

// ── HMAC-SHA256, RFC 4231 ──
static void test_hmac_vectors() {
    // Case 1
    EXPECT_STR_EQ(
        to_hex(hmac_sha256(bytes(0x0b, 20), std::string("Hi There"))),
        "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");

    // Case 2 -- short key
    EXPECT_STR_EQ(
        to_hex(hmac_sha256(std::string("Jefe"),
                           std::string("what do ya want for nothing?"))),
        "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");

    // Case 3
    EXPECT_STR_EQ(
        to_hex(hmac_sha256(bytes(0xaa, 20), bytes(0xdd, 50))),
        "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");

    // Case 4 -- key 0x01..0x19
    std::string k4;
    for (int i = 1; i <= 0x19; ++i) k4.push_back(static_cast<char>(i));
    EXPECT_STR_EQ(
        to_hex(hmac_sha256(k4, bytes(0xcd, 50))),
        "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");

    // Case 6 -- key longer than the 64-byte block, must be hashed down first.
    // This is the branch most from-scratch HMACs get wrong.
    EXPECT_STR_EQ(
        to_hex(hmac_sha256(
            bytes(0xaa, 131),
            std::string(
                "Test Using Larger Than Block-Size Key - Hash Key First"))),
        "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");

    // Case 7 -- oversize key AND oversize data
    EXPECT_STR_EQ(
        to_hex(hmac_sha256(
            bytes(0xaa, 131),
            std::string(
                "This is a test using a larger than block-size key and a "
                "larger than block-size data. The key needs to be hashed "
                "before being used by the HMAC algorithm."))),
        "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2");
}

// A key exactly at the block size must NOT be hashed down. Off-by-one here
// silently produces a different-but-stable tag, so both peers would agree
// and the bug would never surface until interop with a real HMAC.
static void test_hmac_key_at_block_boundary() {
    EXPECT_STR_EQ(to_hex(hmac_sha256(bytes(0xaa, 64), std::string("x"))),
                  to_hex(hmac_sha256(bytes(0xaa, 64), std::string("x"))));
    // 64-byte key and its SHA-256 must give DIFFERENT tags (i.e. the 64-byte
    // key was used verbatim rather than being hashed).
    const Digest kd = sha256(bytes(0xaa, 64));
    const std::string hashed(reinterpret_cast<const char *>(kd.data()),
                             kd.size());
    EXPECT_TRUE(to_hex(hmac_sha256(bytes(0xaa, 64), std::string("x"))) !=
                to_hex(hmac_sha256(hashed, std::string("x"))));
    // ...whereas a 65-byte key MUST equal its hashed form.
    const Digest kd65 = sha256(bytes(0xaa, 65));
    const std::string hashed65(reinterpret_cast<const char *>(kd65.data()),
                               kd65.size());
    EXPECT_STR_EQ(to_hex(hmac_sha256(bytes(0xaa, 65), std::string("x"))),
                  to_hex(hmac_sha256(hashed65, std::string("x"))));
}

// ── hex ──
static void test_hex() {
    const uint8_t raw[] = {0x00, 0x0f, 0x10, 0xa5, 0xff};
    EXPECT_STR_EQ(to_hex(raw, sizeof(raw)), "000f10a5ff");

    std::vector<uint8_t> back;
    EXPECT_TRUE(from_hex("000f10a5ff", back));
    EXPECT_EQ(back.size(), sizeof(raw));
    for (size_t i = 0; i < sizeof(raw); ++i) EXPECT_EQ(back[i], raw[i]);

    EXPECT_TRUE(from_hex("ABCDEF", back));  // uppercase accepted
    EXPECT_EQ(back.size(), (size_t)3);

    EXPECT_TRUE(from_hex("", back));
    EXPECT_EQ(back.size(), (size_t)0);

    // Malformed input from a peer must be rejected, not silently coerced.
    EXPECT_FALSE(from_hex("abc", back));      // odd length
    EXPECT_FALSE(from_hex("zz", back));       // non-hex
    EXPECT_FALSE(from_hex("00 11", back));    // embedded space
}

// ── constant-time compare ──
static void test_constant_time_equal() {
    const Digest a = sha256(std::string("same"));
    const Digest b = sha256(std::string("same"));
    const Digest c = sha256(std::string("different"));
    EXPECT_TRUE(constant_time_equal(a, b));
    EXPECT_FALSE(constant_time_equal(a, c));

    EXPECT_TRUE(constant_time_equal(std::string("abc"), std::string("abc")));
    EXPECT_FALSE(constant_time_equal(std::string("abc"), std::string("abd")));
    EXPECT_FALSE(constant_time_equal(std::string("abc"), std::string("ab")));

    // Differing only in the last byte must still compare false (i.e. the loop
    // does not stop early on the first match).
    uint8_t x[32], y[32];
    for (int i = 0; i < 32; ++i) x[i] = y[i] = static_cast<uint8_t>(i);
    y[31] ^= 0x01;
    EXPECT_FALSE(constant_time_equal(x, y, 32));
}

// ── file hashing ──
static void test_sha256_file() {
    const std::string path = "test_collab_hash_tmp.bin";

    // Larger than the 1 MiB read buffer, to cover the chunked loop.
    const std::string content = rep('q', (1u << 20) + 12345);
    {
        std::FILE *f = std::fopen(path.c_str(), "wb");
        EXPECT_TRUE(f != nullptr);
        if (f) {
            std::fwrite(content.data(), 1, content.size(), f);
            std::fclose(f);
        }
    }

    std::string hex, err;
    EXPECT_TRUE(sha256_file_hex(path, hex, &err));
    EXPECT_STR_EQ(hex, sha256_hex(content));

    // An empty file hashes to the empty-string digest.
    const std::string empty_path = "test_collab_hash_empty.bin";
    { std::FILE *f = std::fopen(empty_path.c_str(), "wb"); if (f) std::fclose(f); }
    EXPECT_TRUE(sha256_file_hex(empty_path, hex, &err));
    EXPECT_STR_EQ(
        hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    // A missing file reports failure rather than a bogus digest -- the clone
    // path decides "already present" from this.
    EXPECT_FALSE(sha256_file_hex("does_not_exist_hopefully.bin", hex, &err));
    EXPECT_TRUE(!err.empty());

    std::remove(path.c_str());
    std::remove(empty_path.c_str());
}

int main() {
    test_sha256_vectors();
    test_sha256_incremental();
    test_hmac_vectors();
    test_hmac_key_at_block_boundary();
    test_hex();
    test_constant_time_equal();
    test_sha256_file();

    std::printf("test_collab_hash: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
