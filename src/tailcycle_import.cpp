// Built as C++20 with the rest of red; see tailcycle_export.cpp.

#include "tailcycle_import.h"

#if defined(RED_HAVE_PARQUET)
#include "tailcycle_schema.h"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#endif

namespace TailcycleImport {

#if !defined(RED_HAVE_PARQUET)
bool available() { return false; }
bool list_groups(const std::string &, std::vector<std::string> *, std::string *s) {
    if (s) *s = "This build has no Parquet support.";
    return false;
}
bool read_session(const std::string &, const std::string &, Session *, ImportStats *,
                  std::string *s) {
    if (s) *s = "This build has no Parquet support.";
    return false;
}
#else

bool available() { return true; }
namespace fs = std::filesystem;

namespace {

std::shared_ptr<arrow::Table> read_pq(const fs::path &p) {
    if (!fs::exists(p)) return nullptr;
    auto file = arrow::io::ReadableFile::Open(p.string());
    if (!file.ok()) return nullptr;
    auto reader = parquet::arrow::OpenFile(*file, arrow::default_memory_pool());
    if (!reader.ok()) return nullptr;
    auto t = (*reader)->ReadTable();
    return t.ok() ? *t : nullptr;
}

// A dictionary<int32,str> column, decoded row by row. Small tables arrive in
// one chunk; larger ones do not, so chunk offsets are tracked.
struct DictCol {
    std::vector<std::string> vals;
    bool ok = false;
    explicit DictCol(const std::shared_ptr<arrow::Table> &t, const char *name) {
        auto col = t->GetColumnByName(name);
        if (!col) return;
        for (int c = 0; c < col->num_chunks(); c++) {
            auto d = std::dynamic_pointer_cast<arrow::DictionaryArray>(col->chunk(c));
            if (!d) return;
            auto dv = std::dynamic_pointer_cast<arrow::StringArray>(d->dictionary());
            auto ix = std::dynamic_pointer_cast<arrow::Int32Array>(d->indices());
            if (!dv || !ix) return;
            for (int64_t i = 0; i < ix->length(); i++)
                vals.push_back(ix->IsNull(i) ? std::string() : dv->GetString(ix->Value(i)));
        }
        ok = true;
    }
};

struct NumCol {
    std::vector<double> vals;
    std::vector<bool> null;
    bool ok = false;
    explicit NumCol(const std::shared_ptr<arrow::Table> &t, const char *name) {
        auto col = t->GetColumnByName(name);
        if (!col) return;
        for (int c = 0; c < col->num_chunks(); c++) {
            auto a = col->chunk(c);
            for (int64_t i = 0; i < a->length(); i++) {
                null.push_back(a->IsNull(i));
                if (a->IsNull(i)) { vals.push_back(0.0); continue; }
                if (auto f = std::dynamic_pointer_cast<arrow::FloatArray>(a))
                    vals.push_back(f->Value(i));
                else if (auto d = std::dynamic_pointer_cast<arrow::DoubleArray>(a))
                    vals.push_back(d->Value(i));
                else if (auto n = std::dynamic_pointer_cast<arrow::Int32Array>(a))
                    vals.push_back(n->Value(i));
                else return;
            }
        }
        ok = true;
    }
};

// Minimal TOML reading: enough for the two files the format defines, which are
// flat key/value and arrays of numbers or strings. Pulling in a TOML library
// for this would be a dependency for one file each.
std::string toml_section(const std::string &text, const std::string &header) {
    const size_t b = text.find("[" + header + "]");
    if (b == std::string::npos) return {};
    const size_t e = text.find("\n[", b + 1);
    return text.substr(b, e == std::string::npos ? std::string::npos : e - b);
}

std::string toml_string(const std::string &text, const std::string &key) {
    std::string pat = key + " = \"";
    const size_t p = text.find(pat);
    if (p == std::string::npos) return {};
    const size_t s = p + pat.size();
    return text.substr(s, text.find('"', s) - s);
}

std::vector<double> toml_numbers(const std::string &text, const std::string &key) {
    std::vector<double> out;
    const size_t p = text.find(key + " = [");
    if (p == std::string::npos) return out;
    const size_t s = text.find('[', p);
    int depth = 0; size_t e = s;
    for (; e < text.size(); e++) {
        if (text[e] == '[') depth++;
        else if (text[e] == ']' && --depth == 0) break;
    }
    std::string body = text.substr(s, e - s + 1);
    for (char &c : body) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream in(body);
    double v;
    while (in >> v) out.push_back(v);
    return out;
}

std::vector<std::string> toml_strings(const std::string &text, const std::string &key) {
    std::vector<std::string> out;
    const size_t p = text.find(key + " = [");
    if (p == std::string::npos) return out;
    size_t e = text.find(']', p);
    // names/skeleton may nest; take to the last ] on the logical line
    const size_t line_end = text.find('\n', p);
    if (line_end != std::string::npos) e = text.rfind(']', line_end);
    std::string body = text.substr(p, e - p + 1);
    size_t q = 0;
    while ((q = body.find('"', q)) != std::string::npos) {
        const size_t r = body.find('"', q + 1);
        if (r == std::string::npos) break;
        out.push_back(body.substr(q + 1, r - q - 1));
        q = r + 1;
    }
    return out;
}

Eigen::Matrix3d rodrigues(const Eigen::Vector3d &r) {
    const double th = r.norm();
    if (th < 1e-12) return Eigen::Matrix3d::Identity();
    const Eigen::Vector3d k = r / th;
    Eigen::Matrix3d K;
    K <<     0, -k(2),  k(1),
          k(2),     0, -k(0),
         -k(1),  k(0),     0;
    return Eigen::Matrix3d::Identity() + std::sin(th) * K + (1 - std::cos(th)) * K * K;
}

} // namespace

bool list_groups(const std::string &session_dir, std::vector<std::string> *out,
                 std::string *status) {
    const fs::path g = fs::path(session_dir) / "groups";
    if (!fs::is_directory(g)) {
        if (status) *status = "No groups/ directory in " + session_dir;
        return false;
    }
    for (const auto &e : fs::directory_iterator(g))
        if (e.is_directory()) out->push_back(e.path().filename().string());
    std::sort(out->begin(), out->end());
    return !out->empty();
}

bool read_session(const std::string &session_dir, const std::string &group_id,
                  Session *out, ImportStats *stats, std::string *status) {
    ImportStats local;
    ImportStats &st = stats ? *stats : local;
    auto fail = [&](const std::string &m) { if (status) *status = m; return false; };
    const fs::path D(session_dir);

    if (!fs::exists(D / "session.toml")) return fail("No session.toml in " + session_dir);
    if (!fs::exists(D / "calibration.toml"))
        return fail("No calibration.toml -- rule 4 requires it, including for 2d sessions.");

    out->session_id = D.filename().string();
    out->split = D.parent_path().filename().string();

    // ── session.toml ──
    {
        std::ifstream f(D / "session.toml");
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        out->mode = toml_string(text, "mode");
        out->units = toml_string(text, "units");
        out->labels = toml_string(text, "labels");
        out->node_names = toml_strings(text, "names");
        if (out->node_names.empty()) return fail("session.toml has no `names`.");

        const std::vector<std::string> sk = toml_strings(text, "skeleton");
        for (size_t i = 0; i + 1 < sk.size(); i += 2) {
            const auto a = std::find(out->node_names.begin(), out->node_names.end(), sk[i]);
            const auto b = std::find(out->node_names.begin(), out->node_names.end(), sk[i + 1]);
            if (a == out->node_names.end() || b == out->node_names.end()) continue;
            out->edges.push_back({(int)(a - out->node_names.begin()),
                                  (int)(b - out->node_names.begin())});
        }
    }

    // ── calibration.toml ──
    {
        std::ifstream f(D / "calibration.toml");
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (int i = 0;; i++) {
            const std::string sec = toml_section(text, "cam_" + std::to_string(i));
            if (sec.empty()) break;
            const std::string name = toml_string(sec, "name");
            if (name.empty()) return fail("A camera in calibration.toml has no name (rule 4).");

            const auto off = toml_numbers(sec, "offset");
            if (off.size() == 2 && (off[0] != 0.0 || off[1] != 0.0))
                return fail("Camera " + name + " has offset [" + std::to_string(off[0]) + ", " +
                            std::to_string(off[1]) + "]. red has no crop model, so these "
                            "coordinates would be read against calibration that does not "
                            "describe them.");

            CameraParams c;
            const auto K = toml_numbers(sec, "matrix");
            const auto d = toml_numbers(sec, "distortions");
            const auto r = toml_numbers(sec, "rotation");
            const auto t = toml_numbers(sec, "translation");
            const auto size = toml_numbers(sec, "size");
            if (K.size() >= 9)
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++) c.k(a, b) = K[a * 3 + b];
            for (size_t j = 0; j < d.size() && j < 5; j++) c.dist_coeffs(j) = d[j];
            if (r.size() >= 3) {
                c.rvec = Eigen::Vector3d(r[0], r[1], r[2]);
                c.r = rodrigues(c.rvec);
            }
            if (t.size() >= 3) c.tvec = Eigen::Vector3d(t[0], t[1], t[2]);
            if (size.size() >= 2) {
                c.image_width = (int)size[0];
                c.image_height = (int)size[1];
            }
            c.projection_mat = red_math::projectionFromKRt(c.k, c.r, c.tvec);
            out->camera_names.push_back(name);
            out->calibration.push_back(c);
        }
    }
    if (out->camera_names.empty()) return fail("calibration.toml declares no cameras.");
    if (out->mode == "3d" && out->camera_names.size() < 2)
        return fail("mode is \"3d\" but only one camera is declared (rule 5).");

    // ── groups.pq ──
    auto gt = read_pq(D / "groups.pq");
    if (!gt) return fail("Cannot read groups.pq.");
    {
        auto gid = gt->GetColumnByName("group_id");
        std::vector<std::string> ids;
        for (int c = 0; c < gid->num_chunks(); c++) {
            auto a = std::dynamic_pointer_cast<arrow::StringArray>(gid->chunk(c));
            if (!a) return fail("groups.pq: group_id is not a string column.");
            for (int64_t i = 0; i < a->length(); i++) ids.push_back(a->GetString(i));
        }
        int row = -1;
        if (group_id.empty()) {
            if (ids.size() != 1)
                return fail("This session has " + std::to_string(ids.size()) +
                            " groups; a red project is one media folder, so pick one.");
            row = 0;
        } else {
            for (size_t i = 0; i < ids.size(); i++) if (ids[i] == group_id) row = (int)i;
            if (row < 0) return fail("No group \"" + group_id + "\" in groups.pq.");
        }
        out->group_id = ids[row];
        NumCol nf(gt, "n_frames"), fps(gt, "fps"), sfs(gt, "source_frame_start");
        if (nf.ok && row < (int)nf.vals.size()) out->n_frames = (int)nf.vals[row];
        if (fps.ok && row < (int)fps.vals.size() && !fps.null[row]) out->fps = (float)fps.vals[row];
        if (sfs.ok && row < (int)sfs.vals.size() && !sfs.null[row])
            out->source_frame_start = (int)sfs.vals[row];
        auto sv = gt->GetColumnByName("source_video");
        if (sv && sv->num_chunks() > 0)
            if (auto a = std::dynamic_pointer_cast<arrow::StringArray>(sv->chunk(0)))
                if (row < a->length() && !a->IsNull(row)) out->source_video = a->GetString(row);
    }
    if (out->n_frames <= 0) return fail("groups.pq: n_frames must be > 0.");

    const int NN = (int)out->node_names.size();
    const int NC = (int)out->camera_names.size();
    auto name_index = [&](const std::vector<std::string> &v, const std::string &s) {
        const auto it = std::find(v.begin(), v.end(), s);
        return it == v.end() ? -1 : (int)(it - v.begin());
    };
    auto frame_of = [&](u32 f) -> FrameAnnotation & {
        return get_or_create_frame(out->annotations, f, NN, NC);
    };

    std::set<std::string> animals;

    // ── keypoints.pq ──
    if (auto kt = read_pq(D / "keypoints.pq")) {
        out->has_2d = true;
        DictCol cam(kt, "camera"), bp(kt, "bodypart"), stt(kt, "status"), aid(kt, "animal_id"),
                gid(kt, "group_id");
        NumCol fr(kt, "frame"), x(kt, "x"), y(kt, "y"), sc(kt, "score");
        if (!cam.ok || !bp.ok || !stt.ok || !fr.ok || !x.ok || !y.ok)
            return fail("keypoints.pq: unexpected column types.");
        for (size_t i = 0; i < fr.vals.size(); i++) {
            if (gid.ok && gid.vals[i] != out->group_id) continue;
            if (aid.ok) animals.insert(aid.vals[i]);
            const std::string &s = stt.vals[i];
            // `unlabeled` is a progress marker a consumer treats as an absent
            // row; `missing` is an assessed occlusion red cannot represent.
            if (s == Tailcycle::status::kUnlabeled || s == Tailcycle::status::kMissing) continue;
            const int f = (int)fr.vals[i];
            if (f < 0 || f >= out->n_frames) continue;
            const int ci = name_index(out->camera_names, cam.vals[i]);
            const int ni = name_index(out->node_names, bp.vals[i]);
            if (ni < 0) return fail("keypoints.pq: bodypart \"" + bp.vals[i] +
                                    "\" is not in the session's names (rule 6).");
            if (ci < 0) return fail("keypoints.pq: camera \"" + cam.vals[i] +
                                    "\" is not in calibration.toml (rule 6).");
            if (x.null[i] || y.null[i]) continue;
            Keypoint2D &kp = frame_of((u32)f).cameras[ci].keypoints[ni];
            kp.x = x.vals[i];
            // Mirror of the export: the format stores y from the top of the
            // image, red works in ImPlot coordinates measured from the bottom.
            kp.y = (double)out->calibration[ci].image_height - y.vals[i];
            kp.labeled = true;
            if (sc.ok && i < sc.null.size() && !sc.null[i]) kp.confidence = (float)sc.vals[i];
            st.keypoint_rows++;
        }
    }

    // ── points3d.pq ──
    if (auto pt = read_pq(D / "points3d.pq")) {
        out->has_3d = true;
        DictCol bp(pt, "bodypart"), stt(pt, "status"), aid(pt, "animal_id"), gid(pt, "group_id");
        NumCol fr(pt, "frame"), x(pt, "x"), y(pt, "y"), z(pt, "z"), sc(pt, "score");
        if (!bp.ok || !stt.ok || !fr.ok || !x.ok || !y.ok || !z.ok)
            return fail("points3d.pq: unexpected column types.");
        for (size_t i = 0; i < fr.vals.size(); i++) {
            if (gid.ok && gid.vals[i] != out->group_id) continue;
            if (aid.ok) animals.insert(aid.vals[i]);
            if (stt.vals[i] != Tailcycle::status::kVisible) continue;
            const int f = (int)fr.vals[i];
            if (f < 0 || f >= out->n_frames) continue;
            const int ni = name_index(out->node_names, bp.vals[i]);
            if (ni < 0) return fail("points3d.pq: bodypart \"" + bp.vals[i] +
                                    "\" is not in the session's names (rule 6).");
            if (x.null[i] || y.null[i] || z.null[i]) continue;
            Keypoint3D &k3 = frame_of((u32)f).kp3d[ni];
            k3.x = x.vals[i]; k3.y = y.vals[i]; k3.z = z.vals[i];
            k3.set_imported(sc.ok && i < sc.null.size() && !sc.null[i] ? (float)sc.vals[i] : 1.0f);
            st.points3d_rows++;
        }
    }

    if (!out->has_2d && !out->has_3d)
        return fail("Session has neither keypoints.pq nor points3d.pq (§3).");
    if (animals.size() > 1)
        return fail("This session has " + std::to_string(animals.size()) +
                    " animals. red holds one instance per frame, so the others would be "
                    "silently discarded.");

    st.frames = (int)out->annotations.size();
    if (status) {
        *status = "Read " + out->session_id + "/" + out->group_id + ": " +
                  std::to_string(st.keypoint_rows) + " 2D rows, " +
                  std::to_string(st.points3d_rows) + " 3D rows, " +
                  std::to_string(st.frames) + " frames";
    }
    return true;
}

#endif // RED_HAVE_PARQUET

} // namespace TailcycleImport
