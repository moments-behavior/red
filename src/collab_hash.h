#pragma once
// collab_hash.h -- SHA-256 and HMAC-SHA256 for the collaboration layer.
//
// Two consumers, one implementation:
//   - content addressing for project/media sharing (a blob's id IS its digest)
//   - relay authentication (HMAC challenge-response over a shared secret)
//
// Written from scratch: the tree has no crypto or hashing dependency and the
// no-external-dependencies rule keeps it that way. Correctness is not a matter
// of opinion here -- tests/test_collab_hash.cpp checks this against the
// FIPS 180-4 and RFC 4231 published vectors, and a wrong HMAC would otherwise
// fail silently and open.
//
// NOT a confidentiality mechanism. See the security note in collab_proto.h.

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace collab {

using Digest = std::array<uint8_t, 32>;

// =========================================================================
// SHA-256
// =========================================================================

class Sha256 {
  public:
    static constexpr size_t kDigestSize = 32;
    static constexpr size_t kBlockSize = 64;

    Sha256() { reset(); }

    void reset() {
        state_[0] = 0x6a09e667u; state_[1] = 0xbb67ae85u;
        state_[2] = 0x3c6ef372u; state_[3] = 0xa54ff53au;
        state_[4] = 0x510e527fu; state_[5] = 0x9b05688cu;
        state_[6] = 0x1f83d9abu; state_[7] = 0x5be0cd19u;
        total_ = 0;
        buflen_ = 0;
    }

    void update(const void *data, size_t len) {
        const uint8_t *p = static_cast<const uint8_t *>(data);
        total_ += len;
        while (len > 0) {
            size_t take = kBlockSize - buflen_;
            if (take > len) take = len;
            std::memcpy(buf_ + buflen_, p, take);
            buflen_ += take;
            p += take;
            len -= take;
            if (buflen_ == kBlockSize) {
                transform(buf_);
                buflen_ = 0;
            }
        }
    }

    void update(const std::string &s) { update(s.data(), s.size()); }

    // Finalizes into `out`. The object must be reset() before reuse.
    void finish(uint8_t out[kDigestSize]) {
        const uint64_t bitlen = total_ * 8ull;

        // Append the 0x80 terminator. buflen_ is at most 63 here, because a
        // full block is always transformed and cleared inside update().
        buf_[buflen_++] = 0x80;
        if (buflen_ > 56) {
            std::memset(buf_ + buflen_, 0, kBlockSize - buflen_);
            transform(buf_);
            buflen_ = 0;
        }
        std::memset(buf_ + buflen_, 0, 56 - buflen_);
        for (int i = 0; i < 8; ++i)
            buf_[56 + i] = static_cast<uint8_t>(bitlen >> (56 - 8 * i));
        transform(buf_);

        for (int i = 0; i < 8; ++i) {
            out[4 * i + 0] = static_cast<uint8_t>(state_[i] >> 24);
            out[4 * i + 1] = static_cast<uint8_t>(state_[i] >> 16);
            out[4 * i + 2] = static_cast<uint8_t>(state_[i] >> 8);
            out[4 * i + 3] = static_cast<uint8_t>(state_[i]);
        }
    }

    Digest finish() {
        Digest d;
        finish(d.data());
        return d;
    }

  private:
    static uint32_t rotr(uint32_t x, int n) {
        return (x >> n) | (x << (32 - n));
    }

    void transform(const uint8_t *block) {
        static const uint32_t K[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
            0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
            0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
            0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
            0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
            0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
            0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
            0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
            0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
            0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (static_cast<uint32_t>(block[4 * i]) << 24) |
                   (static_cast<uint32_t>(block[4 * i + 1]) << 16) |
                   (static_cast<uint32_t>(block[4 * i + 2]) << 8) |
                   (static_cast<uint32_t>(block[4 * i + 3]));
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 =
                rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 =
                rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    uint32_t state_[8];
    uint64_t total_;    // total bytes fed, for the length suffix
    size_t   buflen_;   // bytes currently held in buf_
    uint8_t  buf_[kBlockSize];
};

// =========================================================================
// Hex
// =========================================================================

inline std::string to_hex(const uint8_t *data, size_t len) {
    static const char *digits = "0123456789abcdef";
    std::string out;
    out.resize(len * 2);
    for (size_t i = 0; i < len; ++i) {
        out[2 * i]     = digits[data[i] >> 4];
        out[2 * i + 1] = digits[data[i] & 0x0f];
    }
    return out;
}

inline std::string to_hex(const Digest &d) { return to_hex(d.data(), d.size()); }

// Returns false on any non-hex character or an odd length, leaving `out`
// unspecified. Callers treat a false return as "malformed peer input".
inline bool from_hex(const std::string &hex, std::vector<uint8_t> &out) {
    if (hex.size() % 2 != 0) return false;
    out.clear();
    out.reserve(hex.size() / 2);
    auto nib = [](char c, int &v) -> bool {
        if (c >= '0' && c <= '9') { v = c - '0';        return true; }
        if (c >= 'a' && c <= 'f') { v = c - 'a' + 10;   return true; }
        if (c >= 'A' && c <= 'F') { v = c - 'A' + 10;   return true; }
        return false;
    };
    for (size_t i = 0; i < hex.size(); i += 2) {
        int hi = 0, lo = 0;
        if (!nib(hex[i], hi) || !nib(hex[i + 1], lo)) return false;
        out.push_back(static_cast<uint8_t>((hi << 4) | lo));
    }
    return true;
}

// =========================================================================
// Convenience wrappers
// =========================================================================

inline Digest sha256(const void *data, size_t len) {
    Sha256 h;
    h.update(data, len);
    return h.finish();
}

inline Digest sha256(const std::string &s) { return sha256(s.data(), s.size()); }

inline std::string sha256_hex(const void *data, size_t len) {
    return to_hex(sha256(data, len));
}

inline std::string sha256_hex(const std::string &s) {
    return to_hex(sha256(s));
}

// Streams a file through SHA-256 in 1 MiB chunks -- media files are far too
// large to slurp. Returns false (with *err set, if provided) if the file
// cannot be opened or a read fails partway.
inline bool sha256_file(const std::string &path, Digest &out,
                        std::string *err = nullptr) {
    std::FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    Sha256 h;
    std::vector<uint8_t> buf(1u << 20);
    for (;;) {
        const size_t n = std::fread(buf.data(), 1, buf.size(), f);
        if (n > 0) h.update(buf.data(), n);
        if (n < buf.size()) {
            if (std::ferror(f)) {
                if (err) *err = "read error on " + path;
                std::fclose(f);
                return false;
            }
            break;  // clean EOF
        }
    }
    std::fclose(f);
    out = h.finish();
    return true;
}

inline bool sha256_file_hex(const std::string &path, std::string &out_hex,
                            std::string *err = nullptr) {
    Digest d;
    if (!sha256_file(path, d, err)) return false;
    out_hex = to_hex(d);
    return true;
}

// =========================================================================
// HMAC-SHA256 (RFC 2104)
// =========================================================================

inline Digest hmac_sha256(const void *key, size_t keylen, const void *msg,
                          size_t msglen) {
    const size_t B = Sha256::kBlockSize;
    uint8_t k[B];
    std::memset(k, 0, B);

    // Keys longer than the block size are hashed down first.
    if (keylen > B) {
        const Digest kd = sha256(key, keylen);
        std::memcpy(k, kd.data(), kd.size());
    } else {
        std::memcpy(k, key, keylen);
    }

    uint8_t ipad[B], opad[B];
    for (size_t i = 0; i < B; ++i) {
        ipad[i] = static_cast<uint8_t>(k[i] ^ 0x36);
        opad[i] = static_cast<uint8_t>(k[i] ^ 0x5c);
    }

    Sha256 inner;
    inner.update(ipad, B);
    inner.update(msg, msglen);
    const Digest id = inner.finish();

    Sha256 outer;
    outer.update(opad, B);
    outer.update(id.data(), id.size());
    return outer.finish();
}

inline Digest hmac_sha256(const std::string &key, const std::string &msg) {
    return hmac_sha256(key.data(), key.size(), msg.data(), msg.size());
}

inline std::string hmac_sha256_hex(const std::string &key,
                                   const std::string &msg) {
    return to_hex(hmac_sha256(key, msg));
}

// Length-independent, data-independent comparison. Used for every auth-tag
// check: a plain == would leak the tag one byte at a time through timing.
inline bool constant_time_equal(const uint8_t *a, const uint8_t *b,
                                size_t len) {
    uint8_t diff = 0;
    for (size_t i = 0; i < len; ++i) diff |= static_cast<uint8_t>(a[i] ^ b[i]);
    return diff == 0;
}

inline bool constant_time_equal(const std::string &a, const std::string &b) {
    // The length itself is not secret (it is fixed by the protocol), but the
    // contents must not short-circuit.
    if (a.size() != b.size()) return false;
    return constant_time_equal(reinterpret_cast<const uint8_t *>(a.data()),
                               reinterpret_cast<const uint8_t *>(b.data()),
                               a.size());
}

inline bool constant_time_equal(const Digest &a, const Digest &b) {
    return constant_time_equal(a.data(), b.data(), a.size());
}

}  // namespace collab
