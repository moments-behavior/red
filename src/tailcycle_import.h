#ifndef RED_TAILCYCLE_IMPORT
#define RED_TAILCYCLE_IMPORT

// Read a `tailcycle-dataset` session (annotation_format.md) into red.
//
// Like tailcycle_export.h this is NOT header-only: Arrow requires C++20 and
// the Arrow includes stay in tailcycle_import.cpp. This header is plain C++17.
//
// The reader's job is the tables and the TOML. Turning a session into an
// openable red project -- the flat media folder red's image loader expects,
// the calibration YAMLs -- belongs to the caller; see
// import_tailcycle_project() in export_formats.h.

#include "annotation.h"
#include "camera.h"
#include <string>
#include <utility>
#include <vector>

namespace TailcycleImport {

struct Session {
    std::string session_id;      // the folder name, which IS the id
    std::string split;           // the parent folder name
    std::string mode;            // "2d" | "3d"
    std::string units;
    std::string labels;          // "annotated" | "tracked"

    std::vector<std::string> node_names;        // the keypoint axis
    std::vector<std::pair<int, int>> edges;     // resolved from names

    std::vector<std::string> camera_names;
    std::vector<CameraParams> calibration;      // parallel to camera_names

    // The chosen group. A session may hold several; the caller picks one,
    // because a red project is one media folder.
    std::string group_id;
    int n_frames = 0;
    float fps = 0.0f;
    int source_frame_start = 0;
    std::string source_video;

    // Frames are group-relative, matching the images in groups/<group_id>/.
    // Keeping them that way is what makes the imported project line up with
    // the extracted frames without any offset arithmetic.
    AnnotationMap annotations;

    bool has_2d = false;
    bool has_3d = false;

    // The session's animal_ids, in order of first appearance. red's
    // instance_id is the index into this: the format's ids are strings
    // ("a00".."a04") and red's are ints, so the mapping is positional and
    // stable within a session. Kept so an export can write the original names
    // back rather than inventing new ones.
    std::vector<std::string> animal_ids;
};

// One session's headline facts, read without touching the label tables --
// johnson-mouse-tracked's are 158k rows, and a browser must not pay for that
// to list what is available.
struct SessionInfo {
    std::string dir;          // <root>/<split>/<session>
    std::string split;        // the parent folder name
    std::string session_id;   // the folder name, which IS the id
    std::string mode;         // "2d" | "3d"
    std::string labels;       // "annotated" | "tracked"
    int n_cameras = 0;
    int n_nodes = 0;
    std::vector<std::string> groups;
    int n_frames = 0;         // of the first group
    bool has_2d = false;      // keypoints.pq present
    bool has_3d = false;      // points3d.pq present
};

// Walk <root>/<split>/<session>/ and summarise every session found. A session
// is any directory holding a session.toml, so the split level is whatever its
// parent happens to be called -- §2.1 makes split a directory name, not a
// closed vocabulary.
bool scan_dataset(const std::string &root, std::vector<SessionInfo> *out,
                  std::string *status);

struct ImportStats {
    int keypoint_rows = 0;
    int points3d_rows = 0;
    int frames = 0;
    std::vector<std::string> warnings;
};

// List the group ids in a session, so the caller can offer a choice.
bool list_groups(const std::string &session_dir, std::vector<std::string> *out,
                 std::string *status);

// Read `session_dir` (a <split>/<session>/ directory). `group_id` empty takes
// the only group, and fails if there is more than one.
//
// Refuses rather than silently misreading:
//   - a non-zero camera `offset`: red has no crop model, and loading these
//     coordinates against uncorrected calibration is the 2.38mm -> 16.9mm
//     error the format's §5 documents
//   - a bodypart outside the session's `names`
bool read_session(const std::string &session_dir, const std::string &group_id,
                  Session *out, ImportStats *stats, std::string *status);

bool available();

} // namespace TailcycleImport

#endif // RED_TAILCYCLE_IMPORT
