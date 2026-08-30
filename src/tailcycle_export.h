#ifndef RED_TAILCYCLE_EXPORT
#define RED_TAILCYCLE_EXPORT

// Export a red project as a `tailcycle-dataset` session (annotation_format.md).
//
// Deliberately NOT header-only, unlike jarvis_export.h. Arrow 25 requires
// C++20 and red is C++17 everywhere, so every Arrow include lives in
// tailcycle_export.cpp, which CMake builds as its own C++20 target. This
// header stays plain C++17 so the GUI can include it.

#include "annotation.h"
#include "camera.h"
#include <string>
#include <utility>
#include <vector>

namespace TailcycleExport {

// What to put in groups/<group_id>/ for each camera.
//
// §3 encourages symlinking whole camera directories, and for a group that is a
// whole recording that is cheap and correct -- but only while the dataset stays
// on the machine that wrote it. Ship it anywhere and the links dangle, leaving
// labels with no pixels, so Copy is the default and Symlink is opt-in.
//
// None leaves the folder empty for the caller to populate: red's export window
// uses it, because a group that is a frame range must contain exactly its own
// frames (a consumer reads group frame f as the f-th frame of the media there,
// and source_frame_start is provenance, not an offset to apply).
enum class MediaMode { Copy, Symlink, None };

struct ExportConfig {
    std::string output_folder;      // dataset root; <root>/<split>/<session>/
    std::string split = "train";    // train | val | test -- a directory level (§2.1)
    std::string session_id;         // becomes the folder name, which IS the id

    std::vector<std::string> camera_names;
    std::vector<CameraParams> calibration;   // parallel to camera_names
    std::vector<std::string> video_paths;    // parallel to camera_names

    std::vector<std::string> node_names;         // the keypoint axis (§4)
    std::vector<std::pair<int, int>> edges;      // resolved to name pairs on write

    std::string group_id;           // folder under groups/; defaults to session_id
    int n_frames = 0;               // from the media, NOT the annotation range
    float fps = 0.0f;
    int source_frame_start = 0;     // red's frame_number is absolute; §6 rebases
    std::string source_video;

    std::string units = "mm";

    // §8 says a consumer derives 3D from 2D by triangulation and that "neither
    // derivation is stored", so red's triangulated solve is excluded by
    // default and the session ships keypoints.pq plus the calibration.
    //
    // Note what "off" means in practice: triangulation is the ONLY thing in
    // red that produces 3D, so a labelling project exports no 3D layer at all
    // rather than a reduced one. Turn this on when the consumer wants red's
    // specific solve -- triangulation is not unique, and two implementations
    // can differ on outlier rejection and which views they use -- or when it
    // does not triangulate for itself.
    bool include_triangulated_3d = false;

    // `labels` is closed at annotated|tracked and a session that is both must
    // be two sessions (§2.6). Rows partition by LabelSource/Kp3DSource; a
    // project with both produces <session>_annotated and <session>_tracked.
    bool export_annotated = true;
    bool export_tracked = true;

    MediaMode media = MediaMode::Copy;

    std::string provenance_source;
    std::string annotator;          // empty when one annotator authored the root (§2.11)
};

struct ExportStats {
    int sessions_written = 0;
    int keypoint_rows = 0;
    int points3d_rows = 0;
    int instance_rows = 0;
    int frames_with_labels = 0;
    std::vector<std::string> warnings;
    std::vector<std::string> sessions;   // paths actually written
    double elapsed_seconds = 0.0;
};

// Returns false and fills `status` on refusal. Refusals are for cases where
// writing anything would produce a file that loads cleanly and is wrong:
// telecentric calibration (no aniposelib representation), an improper
// rotation (not expressible as Rodrigues), or a camera whose declared size
// disagrees with its media (validation rule 8).
bool export_session(const ExportConfig &config, const AnnotationMap &amap,
                    ExportStats *stats, std::string *status);

// True when red was built with Arrow/Parquet. The GUI hides the window
// otherwise rather than offering a button that cannot work.
bool available();

} // namespace TailcycleExport

#endif // RED_TAILCYCLE_EXPORT
