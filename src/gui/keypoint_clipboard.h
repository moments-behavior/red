#pragma once
// keypoint_clipboard.h — multi-select + copy/paste/delete of keypoint SETS.
//
// Transient (never serialized) UI state backing three related gestures in the
// Keypoints window and the Labeling Tool:
//   * a File-Explorer-style multi-selection of keypoint columns (nodes),
//   * copying that selection from one frame and pasting it onto another,
//   * deleting keypoints (a hovered cell, a whole column, or the selection).
//
// The app is single-project / single active selection, so — like the existing
// g_keypoint_colormap / g_sync_fix globals — this lives in one process-wide
// instance reached through keypoint_clipboard() rather than being threaded
// through the AppContext reference bundle.
//
// A "keypoint" here is one node's full vertical slice: its 2D Keypoint2D in
// every camera plus the triangulated Keypoint3D. Copy snapshots that slice;
// paste overwrites it on the target frame.
#include "annotation.h"
#include "types.h"
#include <algorithm>
#include <string>
#include <vector>

struct KeypointClipboard {
    // ── Selection: which nodes are selected in the Keypoints window ──
    // Sized to skeleton.num_nodes; 1 = selected. `anchor` is the last
    // plain/ctrl-clicked node, used as the pivot for Shift-range selection.
    std::vector<char> selected;
    int anchor = -1;

    // ── Clipboard: a snapshot taken at copy time ──
    // Decoupled from `selected` so navigating frames or changing the selection
    // after Ctrl+C does not affect what Ctrl+V pastes.
    struct NodeClip {
        int node = -1;                 // node index this slice came from
        std::vector<Keypoint2D> cams;  // [num_cameras] 2D label per camera
        Keypoint3D kp3d;               // triangulated 3D for the node
    };
    std::vector<NodeClip> clip;
    // Skeleton identity of the snapshot, so a paste after a skeleton switch is
    // refused rather than silently re-indexing keypoints.
    int         clip_num_nodes = 0;
    int         clip_num_cams  = 0;
    std::string clip_skeleton;

    // Keep `selected` sized to the current skeleton; resizing (skeleton change)
    // drops any stale selection.
    void ensure_size(int num_nodes) {
        if (num_nodes < 0) num_nodes = 0;
        if ((int)selected.size() != num_nodes) {
            selected.assign((size_t)num_nodes, 0);
            anchor = -1;
        }
    }
    void clear_selection() {
        std::fill(selected.begin(), selected.end(), (char)0);
        anchor = -1;
    }
    int count() const {
        int n = 0;
        for (char c : selected) n += (c ? 1 : 0);
        return n;
    }
    bool any() const {
        for (char c : selected)
            if (c) return true;
        return false;
    }
    bool is_selected(int node) const {
        return node >= 0 && node < (int)selected.size() && selected[(size_t)node];
    }
    bool has_clip() const { return !clip.empty(); }
};

// The one process-wide instance (see file header for the rationale).
inline KeypointClipboard &keypoint_clipboard() {
    static KeypointClipboard kc;
    return kc;
}

// ── Copy ────────────────────────────────────────────────────────────────────
// Snapshot the currently-selected nodes of `fa` into the clipboard. Nodes that
// are labeled in NO camera are skipped (paste must never fabricate a label).
// Returns the number of node-slices copied.
inline int copy_selected_keypoints(KeypointClipboard &kc, const FrameAnnotation &fa,
                                   int num_nodes, int num_cams,
                                   const std::string &skeleton_name) {
    kc.clip.clear();
    kc.clip_num_nodes = num_nodes;
    kc.clip_num_cams  = num_cams;
    kc.clip_skeleton  = skeleton_name;

    for (int node = 0; node < num_nodes; ++node) {
        if (!kc.is_selected(node)) continue;

        bool any_labeled = false;
        for (int c = 0; c < num_cams && c < (int)fa.cameras.size(); ++c)
            if (node < (int)fa.cameras[c].keypoints.size() &&
                fa.cameras[c].keypoints[node].labeled) {
                any_labeled = true;
                break;
            }
        if (!any_labeled) continue;

        KeypointClipboard::NodeClip nc;
        nc.node = node;
        nc.cams.resize((size_t)num_cams);
        for (int c = 0; c < num_cams && c < (int)fa.cameras.size(); ++c)
            if (node < (int)fa.cameras[c].keypoints.size())
                nc.cams[(size_t)c] = fa.cameras[c].keypoints[node];
        if (node < (int)fa.kp3d.size())
            nc.kp3d = fa.kp3d[node];
        kc.clip.push_back(std::move(nc));
    }
    return (int)kc.clip.size();
}

// ── Paste ───────────────────────────────────────────────────────────────────
// Overwrite the clipboard's node-slices onto `fa`. Assumes the caller has
// already verified skeleton identity (paste_identity_ok). Returns the count
// pasted.
inline int paste_keypoints(const KeypointClipboard &kc, FrameAnnotation &fa,
                           int num_nodes, int num_cams) {
    int n = 0;
    for (const auto &nc : kc.clip) {
        const int node = nc.node;
        if (node < 0 || node >= num_nodes) continue;
        for (int c = 0; c < num_cams && c < (int)fa.cameras.size() &&
                        c < (int)nc.cams.size(); ++c)
            if (node < (int)fa.cameras[c].keypoints.size())
                fa.cameras[c].keypoints[node] = nc.cams[(size_t)c];
        if (node < (int)fa.kp3d.size())
            fa.kp3d[node] = nc.kp3d;
        ++n;
    }
    return n;
}

// True if the clipboard was copied under a skeleton compatible with the target.
inline bool paste_identity_ok(const KeypointClipboard &kc, int num_nodes,
                              int num_cams, const std::string &skeleton_name) {
    return kc.clip_num_nodes == num_nodes && kc.clip_num_cams == num_cams &&
           kc.clip_skeleton == skeleton_name;
}

// ── Delete ──────────────────────────────────────────────────────────────────
// Delete one node from one camera (mirrors the image-view 'R' key: resets that
// camera's 2D label only, leaves kp3d — other cameras may still support it).
inline void delete_node_from_camera(FrameAnnotation &fa, int node, int cam) {
    if (cam >= 0 && cam < (int)fa.cameras.size() &&
        node >= 0 && node < (int)fa.cameras[cam].keypoints.size())
        fa.cameras[cam].keypoints[node] = Keypoint2D{};
}

// Delete one node from every camera AND clear its (now unsupported) 3D, so the
// triangulated "T" marker does not linger with no 2D behind it.
inline void delete_node_all_cameras(FrameAnnotation &fa, int node, int num_cams) {
    for (int c = 0; c < num_cams && c < (int)fa.cameras.size(); ++c)
        if (node >= 0 && node < (int)fa.cameras[c].keypoints.size())
            fa.cameras[c].keypoints[node] = Keypoint2D{};
    if (node >= 0 && node < (int)fa.kp3d.size())
        fa.kp3d[node].clear();
}

// Delete every selected node from every camera. Returns the count deleted.
inline int delete_selected_all_cameras(const KeypointClipboard &kc,
                                       FrameAnnotation &fa, int num_nodes,
                                       int num_cams) {
    int n = 0;
    for (int node = 0; node < num_nodes; ++node) {
        if (!kc.is_selected(node)) continue;
        delete_node_all_cameras(fa, node, num_cams);
        ++n;
    }
    return n;
}
