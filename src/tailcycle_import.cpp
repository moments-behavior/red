// Built as C++20 with the rest of red; see tailcycle_export.cpp.

#include "tailcycle_import.h"

#if defined(RED_HAVE_PARQUET)
#include "tailcycle_read.h"
#include "tailcycle_schema.h"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <algorithm>
#include <cctype>
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
bool scan_dataset(const std::string &, std::vector<SessionInfo> *, std::string *s) {
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

using Tailcycle::DictCol;
using Tailcycle::NumCol;
using Tailcycle::read_pq;

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
    // names may be on one line; stop at that array's closing bracket.
    int depth = 0;
    bool quoted = false;
    bool escaped = false;
    for (e = text.find('[', p); e != std::string::npos && e < text.size(); ++e) {
        const char c = text[e];
        if (quoted) {
            if (escaped) escaped = false;
            else if (c == '\\') escaped = true;
            else if (c == '"') quoted = false;
            continue;
        }
        if (c == '"') quoted = true;
        else if (c == '[') ++depth;
        else if (c == ']' && --depth == 0) break;
    }
    if (e == std::string::npos) return out;
    const std::string body = text.substr(p, e - p + 1);
    size_t q = 0;
    while ((q = body.find('"', q)) != std::string::npos) {
        const size_t r = body.find('"', q + 1);
        if (r == std::string::npos) break;
        out.push_back(body.substr(q + 1, r - q - 1));
        q = r + 1;
    }
    return out;
}

// Read an array of string arrays, preserving each inner array. Tailcycle uses
// those inner arrays as polylines: every consecutive pair is one skeleton edge.
std::vector<std::vector<std::string>> toml_string_arrays(
    const std::string &text, const std::string &key) {
    std::vector<std::vector<std::string>> out;
    const size_t p = text.find(key + " = [");
    if (p == std::string::npos) return out;
    const size_t s = text.find('[', p);
    if (s == std::string::npos) return out;

    int depth = 0;
    bool quoted = false;
    bool escaped = false;
    std::string value;
    std::vector<std::string> current;
    for (size_t i = s; i < text.size(); ++i) {
        const char c = text[i];
        if (quoted) {
            if (escaped) {
                value += c;
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                quoted = false;
                if (depth == 2) current.push_back(value);
                value.clear();
            } else {
                value += c;
            }
            continue;
        }
        if (c == '"') {
            quoted = true;
            value.clear();
        } else if (c == '[') {
            ++depth;
            if (depth == 2) current.clear();
        } else if (c == ']') {
            if (depth == 2) out.push_back(current);
            if (--depth == 0) break;
        }
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

namespace {
// Name order with runs of digits compared by value, so "rank2" < "rank10" and
// "..._7092" < "..._115308". Plain string order puts "10" before "2".
bool natural_less(const std::string &a, const std::string &b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const bool da = std::isdigit((unsigned char)a[i]);
        const bool db = std::isdigit((unsigned char)b[j]);
        if (da && db) {
            size_t ie = i, je = j;
            while (ie < a.size() && std::isdigit((unsigned char)a[ie])) ++ie;
            while (je < b.size() && std::isdigit((unsigned char)b[je])) ++je;
            // Compare by value without parsing (runs can exceed 64 bits):
            // drop leading zeros, then the longer run is larger, then digits.
            size_t is = i, js = j;
            while (is + 1 < ie && a[is] == '0') ++is;
            while (js + 1 < je && b[js] == '0') ++js;
            if (ie - is != je - js) return ie - is < je - js;
            const int c = a.compare(is, ie - is, b, js, je - js);
            if (c != 0) return c < 0;
            i = ie; j = je;
        } else {
            if (a[i] != b[j]) return a[i] < b[j];
            ++i; ++j;
        }
    }
    return a.size() - i < b.size() - j;
}

// train, val, test first, in the order a dataset is used; any other split
// after them, by name.
int split_rank(const std::string &split) {
    if (split == "train") return 0;
    if (split == "val") return 1;
    if (split == "test") return 2;
    return 3;
}
}  // namespace

bool scan_dataset(const std::string &root, std::vector<SessionInfo> *out,
                  std::string *status) {
    if (!fs::is_directory(root)) {
        if (status) *status = "Not a directory: " + root;
        return false;
    }
    // A session is any directory holding a session.toml. Pointing at a session
    // directly works too, which saves explaining the layout to someone who
    // already has the path.
    auto add = [&](const fs::path &d, const std::string &split) {
        if (!fs::exists(d / "session.toml")) return;
        SessionInfo si;
        si.dir = d.string();
        si.split = split;
        si.session_id = d.filename().string();
        std::ifstream f(d / "session.toml");
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        si.mode = toml_string(text, "mode");
        si.labels = toml_string(text, "labels");
        si.n_nodes = (int)toml_strings(text, "names").size();
        if (fs::exists(d / "calibration.toml")) {
            std::ifstream cf(d / "calibration.toml");
            std::string ctext((std::istreambuf_iterator<char>(cf)),
                              std::istreambuf_iterator<char>());
            for (size_t p = 0; (p = ctext.find("\nname = \"", p)) != std::string::npos; p++)
                si.n_cameras++;
            if (ctext.rfind("name = \"", 0) == 0) si.n_cameras++;   // first line
        }
        if (fs::is_directory(d / "groups"))
            for (const auto &g : fs::directory_iterator(d / "groups"))
                if (g.is_directory()) si.groups.push_back(g.path().filename().string());
        std::sort(si.groups.begin(), si.groups.end());
        si.has_2d = fs::exists(d / "keypoints.pq");
        si.has_3d = fs::exists(d / "points3d.pq");
        // One row, so this is cheap even for a large session.
        if (auto gt = read_pq(d / "groups.pq")) {
            NumCol nf(gt, "n_frames");
            if (nf.ok && !nf.vals.empty()) si.n_frames = (int)nf.vals[0];
        }
        out->push_back(std::move(si));
    };

    add(fs::path(root), fs::path(root).parent_path().filename().string());
    if (out->empty()) {
        for (const auto &split : fs::directory_iterator(root)) {
            if (!split.is_directory()) continue;
            for (const auto &sess : fs::directory_iterator(split.path()))
                if (sess.is_directory()) add(sess.path(), split.path().filename().string());
        }
    }
    std::sort(out->begin(), out->end(), [](const SessionInfo &a, const SessionInfo &b) {
        if (a.split != b.split) {
            const int ra = split_rank(a.split), rb = split_rank(b.split);
            return ra != rb ? ra < rb : a.split < b.split;
        }
        return natural_less(a.session_id, b.session_id);
    });
    if (out->empty() && status)
        *status = "No sessions under " + root + " (looked for <split>/<session>/session.toml)";
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

        const auto paths = toml_string_arrays(text, "skeleton");
        std::set<std::pair<int, int>> seen_edges;
        for (const auto &path : paths) {
            for (size_t i = 1; i < path.size(); ++i) {
                const auto a = std::find(out->node_names.begin(), out->node_names.end(),
                                         path[i - 1]);
                const auto b = std::find(out->node_names.begin(), out->node_names.end(),
                                         path[i]);
                if (a == out->node_names.end() || b == out->node_names.end()) continue;
                int ai = (int)(a - out->node_names.begin());
                int bi = (int)(b - out->node_names.begin());
                if (ai == bi) continue;
                if (ai > bi) std::swap(ai, bi);
                if (seen_edges.insert({ai, bi}).second)
                    out->edges.push_back({ai, bi});
            }
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
            const bool has_offset = sec.find("offset = [") != std::string::npos;
            if (has_offset && off.size() != 2)
                return fail("Camera " + name + " has an invalid offset; expected [x, y].");
            if (off.size() == 2 && (!std::isfinite(off[0]) || !std::isfinite(off[1])))
                return fail("Camera " + name + " has a non-finite crop offset.");

            CameraParams c;
            const auto K = toml_numbers(sec, "matrix");
            const auto d = toml_numbers(sec, "distortions");
            const auto r = toml_numbers(sec, "rotation");
            const auto t = toml_numbers(sec, "translation");
            const auto size = toml_numbers(sec, "size");
            if (K.size() >= 9)
                for (int a = 0; a < 3; a++)
                    for (int b = 0; b < 3; b++) c.k(a, b) = K[a * 3 + b];
            else if (size.size() >= 2)   // §5: a 2D camera may omit it; nominal pinhole
                c.k << std::max(size[0], size[1]), 0, size[0] / 2, 0,
                       std::max(size[0], size[1]), size[1] / 2, 0, 0, 1;
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
            // Tailcycle's matrix is expressed in full-sensor pixels while
            // labels and media are in stored-image (crop-local) pixels. Red
            // has no separate crop state, so normalize the calibration to the
            // stored image by translating the principal point. Focal lengths,
            // distortion, and extrinsics are unchanged by a pure pixel crop.
            if (off.size() == 2) {
                c.k(0, 2) -= off[0];
                c.k(1, 2) -= off[1];
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
    auto frame_of = [&](u32 f, int inst) -> FrameAnnotation & {
        return get_or_create_frame(out->annotations, f, NN, NC, inst);
    };

    // animal_id -> instance index, assigned in order of first appearance.
    std::map<std::string, int> animal_index;
    auto instance_of = [&](const std::string &aid) {
        auto it = animal_index.find(aid);
        if (it != animal_index.end()) return it->second;
        const int idx = (int)out->animal_ids.size();
        out->animal_ids.push_back(aid);
        animal_index.emplace(aid, idx);
        return idx;
    };

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
            const int inst = instance_of(aid.ok ? aid.vals[i] : std::string("a00"));
            const std::string &s = stt.vals[i];
            // `unlabeled` is semantically identical to an absent row (§7),
            // so red intentionally does not materialize it.
            if (s == Tailcycle::status::kUnlabeled) continue;
            const int f = (int)fr.vals[i];
            if (f < 0 || f >= out->n_frames) continue;
            const int ci = name_index(out->camera_names, cam.vals[i]);
            const int ni = name_index(out->node_names, bp.vals[i]);
            if (ni < 0) return fail("keypoints.pq: bodypart \"" + bp.vals[i] +
                                    "\" is not in the session's names (rule 6).");
            if (ci < 0) return fail("keypoints.pq: camera \"" + cam.vals[i] +
                                    "\" is not in calibration.toml (rule 6).");
            Keypoint2D &kp = frame_of((u32)f, inst).cameras[ci].keypoints[ni];
            if (s == Tailcycle::status::kMissing) {
                // Source is left None. It means "who produced these
                // coordinates", and a missing row has none -- assigning the
                // session's provenance here used to be harmless because a
                // A `missing` row carries null coordinates (§7), so has_pos
                // stays false and the setters are not used -- they would turn
                // it on. This point genuinely has no position, unlike one you
                // occlude in the UI with the coordinates on screen in front
                // of you; the two flags being independent is what lets both
                // be said.
                //
                // Only a tracked session's missing rows get an author, and it
                // is `predicted`. An annotated session's get none: bucket_2d
                // reads an occlusion with nothing claiming to predict it as a
                // person's call, so saying `manual` would add nothing and
                // would assert a click that never happened.
                if (out->labels == Tailcycle::labels::kTracked)
                    kp.author = Keypoint2D::Author::Predicted;
                kp.set_occluded();
                st.keypoint_rows++;
                continue;
            }
            if (s != Tailcycle::status::kVisible &&
                s != Tailcycle::status::kProjected)
                return fail("keypoints.pq: unknown status \"" + s + "\".");
            if (!Tailcycle::keypoint_row_loaded(s, !x.null[i] && !y.null[i])) continue;
            kp.x = x.vals[i];
            // Mirror of the export: the format stores y from the top of the
            // image, red works in ImPlot coordinates measured from the bottom.
            kp.y = (double)out->calibration[ci].image_height - y.vals[i];
            kp.vis = Keypoint2D::Vis::Unknown;
            // Two axes, from two fields, neither answering the other's
            // question.
            //
            // `status` is the per-point VISIBILITY, and it is the per-point
            // truth: a visible row stays visible even in a session declared
            // `tracked`, and the session label must not turn it into a
            // projected row on the next export. `projected` additionally says
            // the position was derived rather than observed.
            //
            // `labels` is the session's AUTHORSHIP, and it is the only thing
            // that knows who made these points. This used to read set_manual()
            // for any visible row -- taking "you can see the part here" to
            // mean "a person clicked here". So every point of a tracked
            // session imported as hand-made, and on export its visible points
            // bucketed annotated while its missing ones bucketed tracked: one
            // machine-produced session split in two, or refused outright.
            kp.has_pos = true;
            if (s == Tailcycle::status::kProjected) kp.reprojected = true;
            else                                    kp.vis = Keypoint2D::Vis::Observed;
            kp.author = (out->labels == Tailcycle::labels::kTracked)
                            ? Keypoint2D::Author::Predicted
                            : Keypoint2D::Author::Manual;
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
            const int inst = instance_of(aid.ok ? aid.vals[i] : std::string("a00"));
            if (stt.vals[i] != Tailcycle::status::kVisible) continue;
            const int f = (int)fr.vals[i];
            if (f < 0 || f >= out->n_frames) continue;
            const int ni = name_index(out->node_names, bp.vals[i]);
            if (ni < 0) return fail("points3d.pq: bodypart \"" + bp.vals[i] +
                                    "\" is not in the session's names (rule 6).");
            if (!Tailcycle::point3d_row_loaded(stt.vals[i], !x.null[i] && !y.null[i] && !z.null[i]))
                continue;
            Keypoint3D &k3 = frame_of((u32)f, inst).kp3d[ni];
            k3.x = x.vals[i]; k3.y = y.vals[i]; k3.z = z.vals[i];
            // The session's own claim decides. A session declaring
            // labels="annotated" holds 3D a person stands behind, solved from
            // their 2D -- Triangulated, even though red did not do the
            // solving. Treating every imported 3D as a prediction meant
            // exporting such a session through the Export Tool relabelled
            // human work as machine output, and "annotated" vs "tracked" is
            // the one field the format uses to tell them apart.
            const float conf =
                sc.ok && i < sc.null.size() && !sc.null[i] ? (float)sc.vals[i] : 1.0f;
            if (out->labels == Tailcycle::labels::kTracked)
                k3.set_predicted(conf);
            else
                k3.set_triangulated(conf);
            st.points3d_rows++;
        }
    }

    // ── instances.pq ── boxes only; red has no model for `present`/`absent`
    // without a box, and an in-place save keeps those rows (see
    // TailcycleExport's in_place). A `present` box loads like a `labeled` one;
    // a box on an `absent` row is not a positive and is skipped.
    if (auto it = read_pq(D / "instances.pq")) {
        out->has_boxes = true;
        DictCol cam(it, "camera"), aid(it, "animal_id"), gid(it, "group_id"), stt(it, "status");
        NumCol fr(it, "frame"), x0(it, "x0"), y0(it, "y0"), x1(it, "x1"), y1(it, "y1");
        bool warned = false;
        for (size_t i = 0; cam.ok && fr.ok && x0.ok && y0.ok && x1.ok && y1.ok &&
                           i < fr.vals.size(); i++) {
            if (gid.ok && gid.vals[i] != out->group_id) continue;
            const std::string s = stt.ok ? stt.vals[i] : Tailcycle::status::kLabeled;
            if (s != Tailcycle::status::kLabeled && s != Tailcycle::status::kPresent &&
                s != Tailcycle::status::kAbsent) {
                if (!warned)
                    st.warnings.push_back("instances.pq: unknown status \"" + s +
                                          "\"; its boxes skipped.");
                warned = true;
                continue;
            }
            const bool box = !x0.null[i] && !y0.null[i] && !x1.null[i] && !y1.null[i] &&
                             Tailcycle::box_nonempty(x0.vals[i], y0.vals[i], x1.vals[i], y1.vals[i]);
            if (!Tailcycle::instance_box_loaded(s, box)) continue;
            const int f = (int)fr.vals[i], ci = name_index(out->camera_names, cam.vals[i]);
            if (f < 0 || f >= out->n_frames || ci < 0) continue;
            CameraExtras &e = frame_of((u32)f, instance_of(aid.ok ? aid.vals[i] : "a00"))
                                  .cameras[ci].get_extras();
            e.bbox_x = x0.vals[i];
            e.bbox_y = y0.vals[i];
            e.bbox_w = x1.vals[i] - x0.vals[i];
            e.bbox_h = y1.vals[i] - y0.vals[i];
            e.has_bbox = true;
        }
    }

    // All annotation tables are optional. A session with none is a valid empty
    // annotation set (for example, after the user clears every correction).
    st.frames = (int)out->annotations.size();
    if (status) {
        *status = "Read " + out->session_id + "/" + out->group_id + ": " +
                  std::to_string(st.keypoint_rows) + " 2D rows, " +
                  std::to_string(st.points3d_rows) + " 3D rows, " +
                  std::to_string(st.frames) + " frames, " +
                  std::to_string(out->animal_ids.size()) + " animal(s)";
    }
    return true;
}

#endif // RED_HAVE_PARQUET

} // namespace TailcycleImport
