// test_collab_bundle.cpp -- manifests and transfer planning (src/collab_bundle.h)
//
// No relay here: this covers the decisions made before a byte moves, which is
// where "you already have this file" is decided and where a hostile manifest
// would have to be stopped.

#include "test_framework.h"

#include "collab_bundle.h"

#include <cstdio>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace collab;

struct TempDir {
    fs::path p;
    TempDir() {
        p = fs::temp_directory_path() / ("red_bundle_test_" + random_hex(8));
        fs::create_directories(p);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(p, ec);
    }
};

static void put(const fs::path &p, const std::string &content) {
    fs::create_directories(p.parent_path());
    write_file_atomic(p, content, nullptr);
}

// ── scanning ──

static void test_scan_dir() {
    TempDir t;
    put(t.p / "media" / "cam1.mp4", "aaaa");
    put(t.p / "media" / "cam2.mp4", "bbbb");
    put(t.p / "media" / "sub" / "cam3.mp4", "cccc");
    // Sync bookkeeping is per-machine and must never be shipped to a peer.
    put(t.p / ".collab" / "ops" / "local.log", "should not be included");
    put(t.p / "media" / ".collab" / "junk", "nor this");

    std::vector<FileRef> files;
    scan_dir(t.p / "media", t.p, category::kMedia, files);

    EXPECT_EQ(files.size(), (size_t)3);
    for (const auto &f : files) {
        EXPECT_TRUE(f.rel_path.find(".collab") == std::string::npos);
        EXPECT_STR_EQ(f.category, category::kMedia);
        // Paths must be POSIX-style so a manifest written on one platform
        // resolves on another.
        EXPECT_TRUE(f.rel_path.find('\\') == std::string::npos);
        EXPECT_TRUE(f.rel_path.rfind("media/", 0) == 0);
    }

    // A directory that does not exist is simply empty, not an error.
    std::vector<FileRef> none;
    scan_dir(t.p / "nope", t.p, category::kMedia, none);
    EXPECT_EQ(none.size(), (size_t)0);
}

// ── manifest ──

static void test_build_manifest() {
    TempDir t;
    put(t.p / "media" / "a.mp4", std::string(1000, 'a'));
    put(t.p / "calibration" / "c.yaml", "fx: 1\n");

    std::vector<FileRef> files;
    scan_dir(t.p / "media", t.p, category::kMedia, files);
    scan_dir(t.p / "calibration", t.p, category::kCalibration, files);

    Manifest m;
    std::string err;
    int done = 0;
    EXPECT_TRUE(build_manifest(files, m, &err, &done));
    EXPECT_EQ(done, 2);
    EXPECT_EQ(m.entries.size(), (size_t)2);

    for (const auto &e : m.entries) {
        EXPECT_EQ(e.sha256.size(), (size_t)64);
        EXPECT_TRUE(valid_blob_id(e.sha256));
        EXPECT_TRUE(e.size > 0);
    }
    EXPECT_EQ(m.media_bytes(), (uint64_t)1000);
    EXPECT_TRUE(m.total_bytes() > m.media_bytes());

    // The hash must be the file's real digest, not a name or size proxy.
    for (const auto &e : m.entries)
        if (e.category == category::kMedia)
            EXPECT_STR_EQ(e.sha256, sha256_hex(std::string(1000, 'a')));
}

static void test_build_manifest_cancels() {
    TempDir t;
    put(t.p / "m" / "a.bin", "x");
    std::vector<FileRef> files;
    scan_dir(t.p / "m", t.p, category::kMedia, files);

    Manifest m;
    std::string err;
    const bool cancel = true;
    EXPECT_FALSE(build_manifest(files, m, &err, nullptr, &cancel));
    EXPECT_STR_EQ(err, "cancelled");
}

static void test_manifest_json_roundtrip() {
    Manifest m;
    m.project_name = "rig";
    m.binding.skeleton_name = "Mouse22";
    m.binding.num_nodes = 22;
    m.binding.camera_names = {"cam1", "cam2"};
    m.created_by = "peer-1";
    m.created_by_name = "Alice";
    m.created_ms = 1234567;
    m.project_json = nlohmann::json{{"pump_offset_ms", 42}};

    BundleEntry e;
    e.rel_path = "media/a.mp4";
    e.size = 99;
    e.sha256 = std::string(64, 'a');
    e.category = category::kMedia;
    m.entries.push_back(e);

    nlohmann::json j;
    to_json(j, m);

    Manifest back;
    std::string err;
    EXPECT_TRUE(manifest_from_json(j, back, &err));
    EXPECT_STR_EQ(back.project_name, "rig");
    EXPECT_STR_EQ(back.binding.skeleton_name, "Mouse22");
    EXPECT_EQ(back.binding.num_nodes, 22);
    EXPECT_EQ(back.entries.size(), (size_t)1);
    EXPECT_STR_EQ(back.entries[0].rel_path, "media/a.mp4");
    EXPECT_EQ(back.project_json.value("pump_offset_ms", 0), 42);
}

// A manifest arrives from a peer. A blob id that is not a clean 64-char hex
// string would become a filesystem path, so it must be refused outright.
static void test_manifest_rejects_hostile_entries() {
    nlohmann::json j;
    j["version"] = kManifestVersion;
    j["entries"] = nlohmann::json::array();
    nlohmann::json e;
    e["rel_path"] = "media/a.mp4";
    e["size"] = 1;
    e["sha256"] = "../../../../etc/passwd";
    e["category"] = "media";
    j["entries"].push_back(e);

    Manifest m;
    std::string err;
    EXPECT_FALSE(manifest_from_json(j, m, &err));
    EXPECT_TRUE(err.find("malformed") != std::string::npos);

    // An empty relative path is equally unusable.
    j["entries"][0]["sha256"] = std::string(64, 'a');
    j["entries"][0]["rel_path"] = "";
    EXPECT_FALSE(manifest_from_json(j, m, &err));

    // A future manifest version must be refused with a readable reason rather
    // than silently half-parsed.
    nlohmann::json future;
    future["version"] = kManifestVersion + 1;
    EXPECT_FALSE(manifest_from_json(future, m, &err));
    EXPECT_TRUE(err.find("version") != std::string::npos);
}

static void test_valid_blob_id() {
    EXPECT_TRUE(valid_blob_id(std::string(64, 'a')));
    EXPECT_TRUE(valid_blob_id(sha256_hex(std::string("x"))));
    EXPECT_FALSE(valid_blob_id(""));
    EXPECT_FALSE(valid_blob_id(std::string(63, 'a')));
    EXPECT_FALSE(valid_blob_id(std::string(65, 'a')));
    EXPECT_FALSE(valid_blob_id(std::string(64, 'A')));   // uppercase
    EXPECT_FALSE(valid_blob_id(std::string(64, 'g')));   // not hex
    EXPECT_FALSE(valid_blob_id("../etc/passwd"));
}

// ── transfer planning ──

static void test_plan_transfer_detects_present_files() {
    TempDir src, dst;
    const std::string video = std::string(5000, 'V');
    put(src.p / "media" / "a.mp4", video);
    put(src.p / "calibration" / "c.yaml", "fx: 1\n");

    std::vector<FileRef> files;
    scan_dir(src.p / "media", src.p, category::kMedia, files);
    scan_dir(src.p / "calibration", src.p, category::kCalibration, files);

    Manifest m;
    std::string err;
    EXPECT_TRUE(build_manifest(files, m, &err));

    // Nothing at the destination yet.
    TransferPlan p0 = plan_transfer(m, dst.p, true);
    EXPECT_EQ(p0.needed.size(), (size_t)2);
    EXPECT_EQ(p0.already_present.size(), (size_t)0);
    EXPECT_EQ(p0.bytes_needed, m.total_bytes());
    EXPECT_EQ(p0.file_count(), (size_t)2);

    // Seed the big file out of band, exactly as a USB hand-off would.
    put(dst.p / "media" / "a.mp4", video);
    TransferPlan p1 = plan_transfer(m, dst.p, true);
    EXPECT_EQ(p1.needed.size(), (size_t)1);
    EXPECT_EQ(p1.already_present.size(), (size_t)1);
    EXPECT_TRUE(p1.bytes_needed < 5000);
}

// Same size, different content: size alone must NOT count as present, or a
// corrupted local copy would be silently kept forever.
static void test_plan_transfer_verifies_content_not_size() {
    TempDir src, dst;
    put(src.p / "media" / "a.mp4", std::string(4096, 'A'));

    std::vector<FileRef> files;
    scan_dir(src.p / "media", src.p, category::kMedia, files);
    Manifest m;
    std::string err;
    EXPECT_TRUE(build_manifest(files, m, &err));

    put(dst.p / "media" / "a.mp4", std::string(4096, 'B'));  // same length
    TransferPlan p = plan_transfer(m, dst.p, true);
    EXPECT_EQ(p.needed.size(), (size_t)1);
    EXPECT_EQ(p.already_present.size(), (size_t)0);

    // A truncated local copy is likewise not "present".
    put(dst.p / "media" / "a.mp4", std::string(100, 'A'));
    TransferPlan p2 = plan_transfer(m, dst.p, true);
    EXPECT_EQ(p2.needed.size(), (size_t)1);
}

static void test_plan_transfer_excludes_media() {
    TempDir src, dst;
    put(src.p / "media" / "a.mp4", std::string(9000, 'V'));
    put(src.p / "calibration" / "c.yaml", "fx: 1\n");

    std::vector<FileRef> files;
    scan_dir(src.p / "media", src.p, category::kMedia, files);
    scan_dir(src.p / "calibration", src.p, category::kCalibration, files);
    Manifest m;
    std::string err;
    EXPECT_TRUE(build_manifest(files, m, &err));

    TransferPlan p = plan_transfer(m, dst.p, /*include_media=*/false);
    EXPECT_EQ(p.needed.size(), (size_t)1);
    EXPECT_STR_EQ(p.needed[0].category, category::kCalibration);
    EXPECT_EQ(p.bytes_skipped, (uint64_t)9000);
}

// ── chunk resume ──

static void test_chunk_resume() {
    TempDir t;
    const std::string content = std::string(10000, 'Z');
    const std::string hash = sha256_hex(content);
    std::string err;

    EXPECT_EQ(resume_offset(t.p, hash), (uint64_t)0);

    // First half.
    EXPECT_TRUE(append_chunk(t.p, hash, 0, content.data(), 4000, &err));
    EXPECT_EQ(resume_offset(t.p, hash), (uint64_t)4000);

    // A chunk at the wrong offset means the two sides disagree about
    // progress; appending anyway would silently corrupt the file.
    EXPECT_FALSE(append_chunk(t.p, hash, 9999, content.data() + 4000, 10,
                              &err));
    EXPECT_TRUE(err.find("offset") != std::string::npos);
    EXPECT_EQ(resume_offset(t.p, hash), (uint64_t)4000);

    // Resume from the right place.
    EXPECT_TRUE(append_chunk(t.p, hash, 4000, content.data() + 4000, 6000,
                             &err));
    EXPECT_EQ(resume_offset(t.p, hash), (uint64_t)10000);

    const fs::path dest = t.p / "out" / "file.bin";
    EXPECT_TRUE(finalize_blob(t.p, hash, dest, &err));
    EXPECT_TRUE(file_exists(dest));

    std::string landed;
    EXPECT_TRUE(read_file(dest, landed, &err));
    EXPECT_TRUE(landed == content);

    // The staging file is consumed, so a repeat starts clean.
    EXPECT_EQ(resume_offset(t.p, hash), (uint64_t)0);
}

static void test_finalize_rejects_wrong_content() {
    TempDir t;
    const std::string claimed = sha256_hex(std::string("expected"));
    const std::string actual = "not what was promised";
    std::string err;

    EXPECT_TRUE(append_chunk(t.p, claimed, 0, actual.data(), actual.size(),
                             &err));
    EXPECT_FALSE(finalize_blob(t.p, claimed, t.p / "out.bin", &err));
    EXPECT_TRUE(err.find("hash mismatch") != std::string::npos);
    EXPECT_FALSE(file_exists(t.p / "out.bin"));
    EXPECT_EQ(resume_offset(t.p, claimed), (uint64_t)0);
}

static void test_blob_helpers_reject_bad_ids() {
    TempDir t;
    std::string err;
    EXPECT_FALSE(append_chunk(t.p, "../../evil", 0, "x", 1, &err));
    EXPECT_FALSE(finalize_blob(t.p, "../../evil", t.p / "o", &err));
    EXPECT_EQ(resume_offset(t.p, "../../evil"), (uint64_t)0);
}

// ── formatting ──

static void test_format_bytes() {
    EXPECT_STR_EQ(format_bytes(0), "0 B");
    EXPECT_STR_EQ(format_bytes(512), "512 B");
    EXPECT_STR_EQ(format_bytes(1024), "1.0 KB");
    EXPECT_STR_EQ(format_bytes(1024ull * 1024), "1.0 MB");
    EXPECT_STR_EQ(format_bytes(41231686042ull), "38.4 GB");
}

// ── atomic write, the primitive the rest of this relies on ──

static void test_atomic_write_leaves_no_temp_files() {
    TempDir t;
    const fs::path target = t.p / "sub" / "f.json";
    std::string err;
    EXPECT_TRUE(write_file_atomic(target, std::string("{\"a\":1}"), &err));
    EXPECT_TRUE(file_exists(target));

    std::string content;
    EXPECT_TRUE(read_file(target, content, &err));
    EXPECT_STR_EQ(content, "{\"a\":1}");

    // Overwriting must replace cleanly, not append or leave debris.
    EXPECT_TRUE(write_file_atomic(target, std::string("{\"b\":2}"), &err));
    EXPECT_TRUE(read_file(target, content, &err));
    EXPECT_STR_EQ(content, "{\"b\":2}");

    int temps = 0;
    for (const auto &e : fs::directory_iterator(t.p / "sub"))
        if (e.path().string().find(".tmp-") != std::string::npos) ++temps;
    EXPECT_EQ(temps, 0);
}

int main() {
    test_scan_dir();
    test_build_manifest();
    test_build_manifest_cancels();
    test_manifest_json_roundtrip();
    test_manifest_rejects_hostile_entries();
    test_valid_blob_id();
    test_plan_transfer_detects_present_files();
    test_plan_transfer_verifies_content_not_size();
    test_plan_transfer_excludes_media();
    test_chunk_resume();
    test_finalize_rejects_wrong_content();
    test_blob_helpers_reject_bad_ids();
    test_format_bytes();
    test_atomic_write_leaves_no_temp_files();

    std::printf("test_collab_bundle: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
