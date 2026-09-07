#pragma once
#include "app_context.h"
#include "gui/gui_helpers.h"
#include "annotation.h"
#include "annotation_csv.h"
#include "gui/gui_keypoints.h"
#include "gui/keypoint_clipboard.h"
#include "keypoints_table.h"
#include "gui/shortcuts.h"
#include "IconsForkAwesome.h"
#include "implot.h"
#include "implot_internal.h"

#include <imgui.h>
#include <ctime>

struct LabelingToolState {
    std::time_t last_saved = static_cast<std::time_t>(-1);
    bool save_requested = false;
    bool timeline_reset_pending = false;
    // Height of the keypoints table, in pixels. 0 = follow the window: take
    // whatever the frame overview below does not need. Set by dragging the
    // splitter under the table; double-clicking it goes back to 0.
    float table_height = 0.0f;
};

inline void DrawLabelingToolWindow(
    LabelingToolState &state, AppContext &ctx) {
    int current_frame_num = ctx.current_frame_num;
    bool keypoints_find = ctx.annotations.find(ctx.current_frame_num) != ctx.annotations.end();
    auto &pm = ctx.pm;
    auto *scene = ctx.scene;
    auto *dc_context = ctx.dc_context;
    auto &skeleton = ctx.skeleton;
    auto &annotations = ctx.annotations;
    auto &ps = ctx.ps;
    auto &popups = ctx.popups;
    auto &toasts = ctx.toasts;

    state.save_requested = false;

    if (ImGui::Begin("Labeling Tool")) {
        // Helper: find prev/next frame matching a predicate (with wraparound)
        struct PrevNext { int prev = -1; int next = -1; };
        auto find_prev_next = [&](auto predicate) -> PrevNext {
            PrevNext pn;
            // Next: search forward from current, wrap to beginning
            for (auto it = annotations.upper_bound(current_frame_num);
                 it != annotations.end(); ++it)
                if (predicate(it->second)) { pn.next = (int)it->first; break; }
            if (pn.next < 0)
                for (auto it = annotations.begin();
                     it != annotations.upper_bound(current_frame_num); ++it)
                    if (predicate(it->second)) { pn.next = (int)it->first; break; }
            // Prev: search backward from current, wrap to end
            auto lb = annotations.lower_bound(current_frame_num);
            if (lb != annotations.begin())
                for (auto it = std::prev(lb);;) {
                    if (predicate(it->second)) { pn.prev = (int)it->first; break; }
                    if (it == annotations.begin()) break;
                    --it;
                }
            if (pn.prev < 0 && !annotations.empty())
                for (auto it = std::prev(annotations.end());;) {
                    if (it->first <= (u32)current_frame_num) break;
                    if (predicate(it->second)) { pn.prev = (int)it->first; break; }
                    if (it == annotations.begin()) break;
                    --it;
                }
            return pn;
        };

        // Helper: render Prev [Jump] Next buttons. id_suffix for unique widget IDs.
        auto jump_buttons = [&](PrevNext pn, const char *id_suffix) {
            ImGui::BeginDisabled(pn.prev < 0);
            char prev_id[32]; snprintf(prev_id, sizeof(prev_id), ICON_FK_CHEVRON_LEFT " Prev##%s", id_suffix);
            if (ImGui::SmallButton(prev_id)) {
                ps.play_video = false;
                seek_all_cameras(scene, pn.prev, dc_context->video_fps, ps, true);
            }
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("Jump");
            ImGui::SameLine();
            ImGui::BeginDisabled(pn.next < 0);
            char next_id[32]; snprintf(next_id, sizeof(next_id), "Next " ICON_FK_CHEVRON_RIGHT "##%s", id_suffix);
            if (ImGui::SmallButton(next_id)) {
                ps.play_video = false;
                seek_all_cameras(scene, pn.next, dc_context->video_fps, ps, true);
            }
            ImGui::EndDisabled();
        };

        // ─── Animals in this frame ───
        // Only shown once a frame holds more than one, so a single-animal
        // project sees nothing new. The selector picks which animal placing,
        // dragging and triangulating apply to; the others draw dimmed.
        {
            auto fit = annotations.find((u32)current_frame_num);
            const int n = fit == annotations.end() ? 0 : (int)fit->second.size();
            if (ctx.active_instance >= n) ctx.active_instance = 0;

            if (n > 1) {
                ImGui::SeparatorText("Animals");
                for (int i = 0; i < n; i++) {
                    if (i) ImGui::SameLine();
                    ImGui::PushID(i);
                    const ImVec4 t = instance_tint(i);
                    ImGui::PushStyleColor(ImGuiCol_Text, t);
                    char lbl[16];
                    snprintf(lbl, sizeof(lbl), "%d", fit->second[(size_t)i].instance_id);
                    if (ImGui::RadioButton(lbl, ctx.active_instance == i))
                        ctx.active_instance = i;
                    ImGui::PopStyleColor();
                    ImGui::PopID();
                }
                ImGui::SameLine();
                ImGui::TextDisabled("(editing #%d)", ctx.active_instance);
            }

            if (fit != annotations.end() && skeleton.has_skeleton) {
                if (n > 1) ImGui::SameLine();
                if (ImGui::SmallButton(ICON_FK_PLUS " Animal")) {
                    // A new animal gets the next unused id, so ids stay stable
                    // even after one is removed.
                    int next_id = 0;
                    for (const auto &fa : fit->second)
                        next_id = std::max(next_id, fa.instance_id + 1);
                    get_or_create_frame(annotations, (u32)current_frame_num,
                                        skeleton.num_nodes,
                                        (int)scene->num_cams, next_id);
                    ctx.active_instance = (int)fit->second.size() - 1;
                }
                if (n > 1) {
                    ImGui::SameLine();
                    if (ImGui::SmallButton(ICON_FK_TRASH " Animal")) {
                        fit->second.erase(fit->second.begin() + ctx.active_instance);
                        ctx.active_instance = 0;
                    }
                }
            }
        }

        // Find prev/next for keypoints
        auto kp_pn = find_prev_next([](const FrameInstances &fis) {
            return any_instance_has_keypoints(fis);
        });
        bool has_next = kp_pn.next >= 0;
        bool has_prev = kp_pn.prev >= 0;
        int next_frame = kp_pn.next;
        int prev_frame = kp_pn.prev;

        // === Top row: Save, Triangulate, Prev/Next label ===
        if (ImGui::Button(ICON_FK_FLOPPY_O " Save")) {
            state.save_requested = true;
        }

        if (scene->num_cams > 1) {
            ImGui::SameLine();

            bool keypoint_triangulated_all = true;
            if (keypoints_find && scene->num_cams > 1) {
                const auto &fa = instance_or_first(
                    annotations.at(current_frame_num), ctx.active_instance);
                for (int j = 0; j < skeleton.num_nodes; j++) {
                    if (!fa.kp3d[j].triangulated) {
                        keypoint_triangulated_all = false;
                        break;
                    }
                }
            } else {
                keypoint_triangulated_all = false;
            }
            bool apply_color =
                !keypoint_triangulated_all && keypoints_find;
            if (apply_color) {
                ImGui::PushStyleColor(
                    ImGuiCol_Button,
                    (ImVec4)ImColor::HSV(0.8, 1.0f, 1.0f));
                ImGui::PushStyleColor(
                    ImGuiCol_ButtonHovered,
                    (ImVec4)ImColor::HSV(0.8, 0.9f, 0.8f));
                ImGui::PushStyleColor(
                    ImGuiCol_ButtonActive,
                    (ImVec4)ImColor::HSV(0.8, 0.9f, 0.5f));
            }

            bool can_triangulate = keypoints_find &&
                                   !pm.camera_params.empty();
            ImGui::BeginDisabled(!can_triangulate);
            if (ImGui::Button("Triangulate")) {
                // The animal being edited, same as the T key. Using front()
                // here triangulated animal 0 however many were selected.
                reprojection(instance_or_first(annotations.at(current_frame_num),
                                               ctx.active_instance),
                             &skeleton, pm.camera_params, scene);
            }
            ImGui::EndDisabled();
            if (keypoints_find && pm.camera_params.empty()) {
                ImGui::SameLine();
                ImGui::TextDisabled("(no calibration)");
            }

            if (apply_color) {
                ImGui::PopStyleColor(3);
            }
        }

        // Prev / Jump to Label / Next
        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        ImGui::BeginDisabled(!has_prev);
        if (ImGui::Button(ICON_FK_CHEVRON_LEFT " Prev")) {
            ps.play_video = false;
            seek_all_cameras(scene, prev_frame,
                             dc_context->video_fps, ps, true);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.7f, 1.0f, 1.0f), "Jump");
        ImGui::SameLine();

        ImGui::BeginDisabled(!has_next);
        if (ImGui::Button("Next " ICON_FK_CHEVRON_RIGHT)) {
            ps.play_video = false;
            seek_all_cameras(scene, next_frame,
                             dc_context->video_fps, ps, true);
        }
        ImGui::EndDisabled();

        ImGui::SameLine();
        ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
        ImGui::SameLine();

        ImGui::BeginDisabled(!has_prev);
        if (ImGui::Button("Copy Prev")) {
            // Copy annotations from prev frame into current frame
            const auto &prev_fa = annotations.at(prev_frame).front();
            FrameAnnotation new_fa = make_frame(skeleton.num_nodes, scene->num_cams, current_frame_num);
            // Copy keypoints from prev frame
            for (int c = 0; c < scene->num_cams && c < (int)prev_fa.cameras.size(); ++c) {
                for (int k = 0; k < skeleton.num_nodes && k < (int)prev_fa.cameras[c].keypoints.size(); ++k) {
                    new_fa.cameras[c].keypoints[k] = prev_fa.cameras[c].keypoints[k];
                }
                new_fa.cameras[c].active_id = prev_fa.cameras[c].active_id;
            }
            for (int k = 0; k < skeleton.num_nodes && k < (int)prev_fa.kp3d.size(); ++k) {
                new_fa.kp3d[k] = prev_fa.kp3d[k];
            }
            annotations[current_frame_num] = FrameInstances{std::move(new_fa)};
        }
        ImGui::EndDisabled();

        // Copy / Paste a SELECTED set of keypoints (selection is built in the
        // keypoints table by clicking column names). Overwrite on paste.
        // Mirrors the Ctrl+C / Ctrl+V hotkeys handled in the keypoints table.
        {
            KeypointClipboard &kc = keypoint_clipboard();
            ImGui::SameLine();
            ImGui::SeparatorEx(ImGuiSeparatorFlags_Vertical);
            ImGui::SameLine();

            int sel = kc.count();
            ImGui::BeginDisabled(!(keypoints_find && sel > 0));
            char copy_id[32];
            snprintf(copy_id, sizeof(copy_id), "Copy Sel (%d)", sel);
            if (ImGui::Button(copy_id)) {
                int n = copy_selected_keypoints(
                    kc, instance_or_first(annotations.at(current_frame_num),
                                          ctx.active_instance),
                    skeleton.num_nodes, scene->num_cams, skeleton.name);
                if (n == 0)
                    toasts.push("None of the selected keypoints are labeled here",
                                Toast::Warning, 4.0f);
                else
                    toasts.pushSuccess("Copied " + std::to_string(n) +
                                       " keypoint(s)");
            }
            ImGui::EndDisabled();

            ImGui::SameLine();
            ImGui::BeginDisabled(!kc.has_clip());
            char paste_id[32];
            snprintf(paste_id, sizeof(paste_id), "Paste (%d)",
                     (int)kc.clip.size());
            if (ImGui::Button(paste_id)) {
                if (!paste_identity_ok(kc, skeleton.num_nodes, scene->num_cams,
                                       skeleton.name)) {
                    toasts.push("Clipboard is from a different skeleton \xE2\x80\x94 "
                                "cannot paste",
                                Toast::Warning, 5.0f);
                } else {
                    FrameAnnotation &fa = get_or_create_frame(
                        annotations, (u32)current_frame_num,
                        skeleton.num_nodes, scene->num_cams);
                    int n = paste_keypoints(kc, fa, skeleton.num_nodes,
                                            scene->num_cams);
                    toasts.pushSuccess("Pasted " + std::to_string(n) +
                                       " keypoint(s)");
                }
            }
            ImGui::EndDisabled();
        }

        if (state.last_saved != static_cast<std::time_t>(-1)) {
            char time_buf[32];
            struct tm tm_buf;
#ifdef _WIN32
            localtime_s(&tm_buf, &state.last_saved);
#else
            localtime_r(&state.last_saved, &tm_buf);
#endif
            strftime(time_buf, sizeof(time_buf), "%H:%M:%S", &tm_buf);
            ImGui::TextDisabled("Last saved: %s", time_buf);
        }

        ImGui::Separator();

        // === Collect labeled frames (counts + timeline) ===
        // needs_improvement frames (promoted predictions awaiting a manual fix)
        // are collected separately so they get their own section below.
        // Keypoint-label state, used to color the timeline ticks:
        //   GREEN  = every keypoint placed on every camera AND fully triangulated
        //   PURPLE = not complete, but every placed keypoint IS triangulated
        //   YELLOW = some placed keypoint is not (yet) triangulated
        // (2D projects have no triangulation: complete -> GREEN, else YELLOW.)
        enum KpLabelState { KP_YELLOW = 0, KP_PURPLE = 1, KP_GREEN = 2 };
        auto classify_kp_state = [&](const FrameAnnotation &fa) -> int {
            if (!skeleton.has_skeleton)
                return KP_YELLOW;
            if (project_is_2d(pm))
                return frame_is_complete(fa) ? KP_GREEN : KP_YELLOW;
            bool green = scene->num_cams > 1 && frame_is_complete(fa) &&
                         frame_is_fully_triangulated(fa, skeleton.num_nodes);
            if (green)
                return KP_GREEN;
            // Purple iff every placed node (labeled in >=1 camera) is
            // triangulated. Triangulated implies placed, so this means the
            // placed and triangulated sets coincide.
            int placed = 0, placed_untriangulated = 0;
            for (int n = 0; n < skeleton.num_nodes; ++n) {
                bool node_placed = false;
                for (const auto &cam : fa.cameras)
                    if (n < (int)cam.keypoints.size() &&
                        cam.keypoints[n].labeled) {
                        node_placed = true;
                        break;
                    }
                bool node_tri =
                    n < (int)fa.kp3d.size() && fa.kp3d[n].triangulated;
                if (node_placed) {
                    ++placed;
                    if (!node_tri) ++placed_untriangulated;
                }
            }
            if (placed > 0 && placed_untriangulated == 0)
                return KP_PURPLE;
            return KP_YELLOW;
        };

        struct LabeledFrameInfo { int frame; int state; };
        std::vector<LabeledFrameInfo> labeled_frames;
        std::vector<LabeledFrameInfo> needs_fix_frames;
        // The frame list is per frame, not per animal: a frame appears once,
        // classified by the animal being labelled.
        for (const auto &[fnum, fis] : annotations) {
            if (fis.empty() || !frame_has_any_keypoints(fis.front()))
                continue;
            const FrameAnnotation &fa = fis.front();
            int state = classify_kp_state(fa);
            if (fa.needs_improvement)
                needs_fix_frames.push_back({(int)fnum, state});
            else
                labeled_frames.push_back({(int)fnum, state});
        }

        // === Collect bounding box frames ===
        struct BBoxFrameInfo { int frame; bool has_bbox; bool has_obb; };
        std::vector<BBoxFrameInfo> bbox_frames;
        for (const auto &[fnum, fis] : annotations) {
            bool any_bbox = false, any_obb = false;
            for (const auto &fa : fis)
              for (const auto &cam : fa.cameras) {
                if (cam.has_bbox()) any_bbox = true;
                if (cam.has_obb())  any_obb  = true;
            }
            if (any_bbox || any_obb)
                bbox_frames.push_back({(int)fnum, any_bbox, any_obb});
        }

        // === Merge into one cell per frame ===
        // One record per frame rather than one per annotation type, so the
        // counts and the unfinished list below do not double-count a frame
        // that is annotated, boxed and flagged all at once.
        struct FrameCell {
            int frame = 0;
            int kp_state = 0;
            bool has_kp = false;
            bool needs_fix = false;
            bool has_bbox = false;
            bool has_obb = false;
        };
        std::map<int, FrameCell> cell_by_frame;
        for (auto &lf : labeled_frames) {
            auto &c = cell_by_frame[lf.frame];
            c.frame = lf.frame; c.kp_state = lf.state; c.has_kp = true;
        }
        for (auto &nf : needs_fix_frames) {
            auto &c = cell_by_frame[nf.frame];
            c.frame = nf.frame; c.kp_state = nf.state; c.has_kp = true;
            c.needs_fix = true;
        }
        for (auto &bf : bbox_frames) {
            auto &c = cell_by_frame[bf.frame];
            c.frame = bf.frame;
            c.has_bbox = bf.has_bbox; c.has_obb = bf.has_obb;
        }
        std::vector<FrameCell> cells;
        cells.reserve(cell_by_frame.size());
        for (auto &[f, c] : cell_by_frame) cells.push_back(c);

        // === What still needs work ===
        // 8000 frames across ~250px of timeline is 0.03px each, so the one
        // frame that is still yellow is invisible among the green -- and one
        // that was never labelled has no annotation entry, so it has no tick
        // at all and you are hunting an absence. A button that walks the list
        // is the only thing that reliably reaches it.
        const int total_frames = dc_context->estimated_num_frames;
        // "Every frame", with slack: an image set can be off by one or two
        // against the estimated count.
        const bool near_total =
            total_frames > 0 &&
            (double)cells.size() >= (double)total_frames * 0.98;

        std::vector<int> unfinished;
        // Frames with no annotation at all. Kept separately because the
        // timeline has to plot them as their own series: they are the one
        // thing it cannot show by colouring a tick, having no tick to colour.
        std::vector<int> unlabeled;
        {
            constexpr size_t kMaxUnfinished = 4096;  // bound the pathological case
            for (const auto &c : cells)
                if (c.needs_fix ||
                    (skeleton.has_skeleton && (!c.has_kp || c.kp_state != KP_GREEN)))
                    unfinished.push_back(c.frame);
            // Never-labelled frames only count when they are the exception.
            // On a sparsely labelled recording "unfinished" would be almost
            // every frame, which is not a thing anyone navigates.
            if (near_total) {
                int expected = 0;
                for (const auto &c : cells) {
                    for (; expected < c.frame &&
                           unlabeled.size() < kMaxUnfinished; ++expected)
                        unlabeled.push_back(expected);
                    expected = c.frame + 1;
                }
                for (; expected < total_frames &&
                       unlabeled.size() < kMaxUnfinished; ++expected)
                    unlabeled.push_back(expected);
                unfinished.insert(unfinished.end(), unlabeled.begin(),
                                  unlabeled.end());
            }
            std::sort(unfinished.begin(), unfinished.end());
            unfinished.erase(std::unique(unfinished.begin(), unfinished.end()),
                             unfinished.end());
        }

        // Annotation type colors, shared by the section labels and the
        // timeline ticks.
        const ImVec4 color_green(0.2f, 0.8f, 0.3f, 1.0f);
        const ImVec4 color_yellow(0.95f, 0.85f, 0.15f, 1.0f);
        const ImVec4 color_purple(0.63f, 0.35f, 0.86f, 1.0f);
        const ImVec4 color_lilac(0.78f, 0.59f, 1.0f, 1.0f);
        const ImVec4 color_red(0.90f, 0.28f, 0.28f, 1.0f);
        const ImVec4 color_orange(1.00f, 0.55f, 0.10f, 1.0f);

        // === Keypoints table ===
        // Height follows the window by default -- whatever the frame overview
        // below does not need -- until you drag the splitter under it, after
        // which your height wins. Double-click the splitter to go back.
        {
            const float line_h = ImGui::GetTextLineHeightWithSpacing();
            int overview_lines = 1;  // "Keypoint Labels" is always shown
            if (!needs_fix_frames.empty()) overview_lines++;
            if (!bbox_frames.empty())      overview_lines++;
            if (!unfinished.empty())       overview_lines++;
            const float timeline_block =
                (dc_context->estimated_num_frames > 0) ? 78.0f : 0.0f;
            const float splitter_h = 6.0f;
            const float reserved =
                overview_lines * line_h + timeline_block + splitter_h +
                ImGui::GetStyle().ItemSpacing.y * 4.0f;

            const float avail = ImGui::GetContentRegionAvail().y;
            const float auto_h = avail - reserved;
            float table_h = (state.table_height > 0.0f) ? state.table_height
                                                        : auto_h;
            // Clamp for display only -- the stored height is left alone, so
            // shrinking the window and growing it again gets it back.
            table_h = ImClamp(table_h, 60.0f,
                              ImMax(60.0f, avail - splitter_h - line_h));

            DrawKeypointsTable(ctx, table_h);

            // Splitter. Dragging seeds the stored height from the height
            // actually in use, so the first drag from auto does not jump.
            ImGui::InvisibleButton("##kp_table_splitter",
                                   ImVec2(-1.0f, splitter_h));
            const bool split_active = ImGui::IsItemActive();
            const bool split_hover = ImGui::IsItemHovered();
            if (split_active || split_hover)
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            if (split_active)
                state.table_height = table_h + ImGui::GetIO().MouseDelta.y;
            if (split_hover && ImGui::IsMouseDoubleClicked(0))
                state.table_height = 0.0f;
            if (split_hover && !split_active)
                ImGui::SetTooltip("Drag to resize the table, double-click to "
                                  "fit the window");
            {
                ImVec2 mn = ImGui::GetItemRectMin();
                ImVec2 mx = ImGui::GetItemRectMax();
                float y = (mn.y + mx.y) * 0.5f;
                ImU32 col = ImGui::GetColorU32(
                    split_active  ? ImGuiCol_SeparatorActive
                    : split_hover ? ImGuiCol_SeparatorHovered
                                  : ImGuiCol_Separator);
                ImGui::GetWindowDrawList()->AddLine(ImVec2(mn.x, y),
                                                    ImVec2(mx.x, y), col,
                                                    split_active ? 3.0f : 2.0f);
            }
        }

        // === Frame overview ===
        // Counts and jump buttons per annotation type. Below them: a strip of
        // clickable cells for the frames around the current one, then the
        // timeline for the whole recording. The strip used to be every
        // annotated frame at once, which on a fully-labelled recording meant
        // thousands of squares filling the panel.
        if (!needs_fix_frames.empty()) {
            auto fix_pn = find_prev_next([](const FrameInstances &fis) {
                for (const auto &fa : fis)
                    if (fa.needs_improvement) return true;
                return false;
            });
            ImGui::TextColored(color_red, "Needs Improvement (%zu)",
                               needs_fix_frames.size());
            ImGui::SameLine();
            jump_buttons(fix_pn, "needsfix");

            // "Mark fixed" for the current frame, if it is one of them.
            auto cur_it = annotations.find((u32)current_frame_num);
            bool cur_needs_fix = cur_it != annotations.end() &&
                                 !cur_it->second.empty() &&
                                 cur_it->second.front().needs_improvement;
            if (cur_needs_fix) {
                ImGui::SameLine();
                if (ImGui::SmallButton("Mark fixed"))
                    cur_it->second.front().needs_improvement = false;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Clear the Needs-Improvement flag on the "
                                      "current frame (moves it to Keypoint Labels).");
            }
        }

        // No jump buttons here: the Prev / Jump / Next row above already walks
        // the annotated frames. The per-state counts stand in for the colour
        // key the grid squares used to carry in their tooltips.
        {
            size_t n_green = 0, n_purple = 0, n_yellow = 0;
            for (auto &lf : labeled_frames) {
                if (lf.state == KP_GREEN) n_green++;
                else if (lf.state == KP_PURPLE) n_purple++;
                else n_yellow++;
            }
            ImGui::Text("Keypoint Labels (%zu)", labeled_frames.size());
            auto count_chip = [&](const ImVec4 &col, size_t n, const char *tip) {
                if (n == 0) return;
                ImGui::SameLine();
                ImGui::TextColored(col, "%zu", n);
                if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tip);
            };
            count_chip(color_green, n_green,
                       "complete (all placed & triangulated)");
            count_chip(color_purple, n_purple,
                       "all placed keypoints triangulated");
            count_chip(color_yellow, n_yellow,
                       "some keypoints not triangulated");
        }

        // The reliable way to reach the one frame in 8000 that is not done.
        // No amount of pixel-hunting beats a button that walks the list.
        if (!unfinished.empty()) {
            PrevNext un_pn;
            auto it = std::upper_bound(unfinished.begin(), unfinished.end(),
                                       current_frame_num);
            un_pn.next = (it != unfinished.end()) ? *it : unfinished.front();
            auto lb = std::lower_bound(unfinished.begin(), unfinished.end(),
                                       current_frame_num);
            un_pn.prev = (lb != unfinished.begin()) ? *std::prev(lb)
                                                    : unfinished.back();
            ImGui::TextColored(color_yellow, "Unfinished (%zu)",
                               unfinished.size());
            if (ImGui::IsItemHovered())
                ImGui::SetTooltip(
                    near_total
                        ? "Frames that are unlabelled, or labelled but not "
                          "complete."
                        : "Labelled frames that are not complete.");
            if (!unlabeled.empty()) {
                // Same orange as the timeline ticks, so the colour there means
                // something without needing a legend of its own.
                ImGui::SameLine();
                ImGui::TextColored(color_orange, "%zu", unlabeled.size());
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("not labelled at all");
            }
            ImGui::SameLine();
            jump_buttons(un_pn, "unfinished");
        }

        if (!bbox_frames.empty()) {
            auto bbox_pn = find_prev_next([](const FrameInstances &fis) {
                for (const auto &fa : fis)
                    for (const auto &cam : fa.cameras)
                        if (cam.has_bbox() || cam.has_obb()) return true;
                return false;
            });
            ImGui::Text("Bounding Box Labels (%zu)", bbox_frames.size());
            ImGui::SameLine();
            jump_buttons(bbox_pn, "bbox");
        }

        // === Timeline minimap (ImPlot — all annotation types) ===
        const bool has_any_annotations = !labeled_frames.empty() ||
                                         !bbox_frames.empty() ||
                                         !needs_fix_frames.empty();
        if (total_frames > 0 && has_any_annotations) {
            ImGui::Spacing();

            // Reserve space for rotated "Timeline" label on the left
            float label_font = ImGui::GetFontSize();
            float label_margin = label_font + 6.0f;
            float timeline_w = ImGui::GetContentRegionAvail().x - label_margin;
            float timeline_h = 60.0f;

            // Draw rotated "Timeline" label on the left
            {
                ImVec2 label_pos = ImGui::GetCursorScreenPos();
                float text_w = ImGui::CalcTextSize("Timeline").x;
                ImVec2 tp(label_pos.x + (label_margin - label_font) * 0.5f,
                          label_pos.y + (timeline_h + text_w) * 0.5f);
                ImPlot::AddTextVertical(ImGui::GetWindowDrawList(), tp,
                    ImGui::GetColorU32(ImGuiCol_Text), "Timeline");
                ImGui::Dummy(ImVec2(label_margin, timeline_h));
                ImGui::SameLine();
            }

            // Build tick arrays for each annotation type
            std::vector<double> kp_yellow_x, kp_purple_x, green_x,
                purple_x, lilac_x, needs_fix_x;
            for (auto &nf : needs_fix_frames)
                needs_fix_x.push_back((double)nf.frame);
            std::vector<double> unlabeled_x;
            for (int f : unlabeled) unlabeled_x.push_back((double)f);
            for (auto &lf : labeled_frames) {
                if (lf.state == KP_GREEN) green_x.push_back((double)lf.frame);
                else if (lf.state == KP_PURPLE)
                    kp_purple_x.push_back((double)lf.frame);
                else kp_yellow_x.push_back((double)lf.frame);
            }
            for (auto &bf : bbox_frames) {
                if (bf.has_bbox) purple_x.push_back((double)bf.frame);
                if (bf.has_obb) lilac_x.push_back((double)bf.frame);
            }

            // Collect all annotated frames for click-to-seek
            std::vector<int> all_annotated_frames;
            for (auto &lf : labeled_frames) all_annotated_frames.push_back(lf.frame);
            for (auto &bf : bbox_frames) all_annotated_frames.push_back(bf.frame);
            for (auto &nf : needs_fix_frames) all_annotated_frames.push_back(nf.frame);
            for (int f : unlabeled) all_annotated_frames.push_back(f);
            std::sort(all_annotated_frames.begin(), all_annotated_frames.end());
            all_annotated_frames.erase(
                std::unique(all_annotated_frames.begin(), all_annotated_frames.end()),
                all_annotated_frames.end());

            if (state.timeline_reset_pending) {
                ImPlot::SetNextAxesLimits(0, total_frames, 0, 1);
                state.timeline_reset_pending = false;
            }

            ImPlot::PushStyleVar(ImPlotStyleVar_PlotPadding, ImVec2(4, 2));
            ImPlotFlags plot_flags = ImPlotFlags_NoLegend | ImPlotFlags_NoTitle |
                                     ImPlotFlags_NoMouseText;
            if (ImPlot::BeginPlot("##timeline", ImVec2(timeline_w, timeline_h), plot_flags)) {
                ImPlotAxisFlags x_flags = ImPlotAxisFlags_NoLabel;
                ImPlotAxisFlags y_flags = ImPlotAxisFlags_NoLabel |
                                          ImPlotAxisFlags_NoTickLabels |
                                          ImPlotAxisFlags_NoTickMarks |
                                          ImPlotAxisFlags_NoGridLines |
                                          ImPlotAxisFlags_Lock;
                ImPlot::SetupAxes("frame number", nullptr, x_flags, y_flags);
                ImPlot::SetupAxisLimits(ImAxis_X1, 0, total_frames, ImPlotCond_Once);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1, ImPlotCond_Always);
                ImPlot::SetupAxisZoomConstraints(ImAxis_X1, 50, total_frames);

                // Ticks, widened by rarity. At 8000 frames across ~250px a
                // frame is 0.03px, so the one that is still yellow among 7999
                // green ones is invisible at a fixed weight -- and the green
                // series, drawn after it, paints over it. Both are fixed here:
                // a series gets at most a quarter of the bar's width shared
                // between its ticks, so a lone frame draws fat and a common
                // state stays hairline; and the series are drawn commonest
                // first, so the rare ones land on top.
                struct Series {
                    const char *id;
                    const std::vector<double> *xs;
                    const ImVec4 *color;
                };
                Series series[] = {
                    {"##kp_yellow", &kp_yellow_x, &color_yellow},
                    {"##kp_purple", &kp_purple_x, &color_purple},
                    {"##green",     &green_x,     &color_green},
                    {"##bbox",      &purple_x,    &color_purple},
                    {"##obb",       &lilac_x,     &color_lilac},
                    {"##needsfix",  &needs_fix_x, &color_red},
                    {"##unlabeled", &unlabeled_x, &color_orange},
                };
                std::stable_sort(
                    std::begin(series), std::end(series),
                    [](const Series &a, const Series &b) {
                        return a.xs->size() > b.xs->size();
                    });
                for (const Series &sr : series) {
                    if (sr.xs->empty()) continue;
                    const float w = ImClamp(
                        timeline_w * 0.25f / (float)sr.xs->size(), 2.0f, 10.0f);
                    ImPlot::PlotInfLines(sr.id, sr.xs->data(), (int)sr.xs->size(),
                                         red_line_spec(*sr.color, w));
                }

                // Current frame indicator, drawn last so it stays visible
                // over a widened tick.
                double cf = (double)current_frame_num;
                ImPlot::PlotInfLines("##current", &cf, 1,
                                     red_line_spec(ImVec4(1, 1, 1, 0.6f), 1.5f));

                // Double-click to reset to full video range
                if (ImPlot::IsPlotHovered() && ImGui::IsMouseDoubleClicked(0))
                    state.timeline_reset_pending = true;

                // Click to seek + tooltip (combined to avoid duplicate computation)
                if (ImPlot::IsPlotHovered()) {
                    ImPlotPoint mp = ImPlot::GetPlotMousePos();
                    ImPlotRect lims = ImPlot::GetPlotLimits();
                    double px_per_frame = timeline_w / (lims.X.Max - lims.X.Min);
                    double tolerance = 5.0 / px_per_frame;

                    auto nearest_of = [&](const std::vector<int> &v,
                                          double tol) {
                        int best = -1;
                        double best_d = tol + 1.0;
                        for (int f : v) {
                            double d = fabs((double)f - mp.x);
                            if (d < best_d) { best_d = d; best = f; }
                        }
                        return std::pair<int, double>(best, best_d);
                    };

                    // Hit-testing follows the draw order: rare ticks are drawn
                    // on top, so a click lands on them first. Without this,
                    // all_annotated_frames holds every frame once the
                    // recording is fully labelled, the nearest is whichever
                    // green neighbour the cursor happens to be over, and the
                    // one orange frame is unclickable however wide it is drawn.
                    // The wider tolerance matches the width those ticks are
                    // given, so clicking the tick itself works, not just its
                    // centre line.
                    // Only while they are rare enough to be targets rather
                    // than a field -- the same condition that draws them fat.
                    // Prioritising 500 of them would just make the ordinary
                    // frames hard to hit instead.
                    const double pri_tol = 8.0 / px_per_frame;
                    const bool prioritize =
                        !unfinished.empty() &&
                        timeline_w * 0.25f / (float)unfinished.size() > 2.5f;
                    int nearest = -1;
                    double nearest_dist = 0.0;
                    if (prioritize) {
                        auto [pri, pri_d] = nearest_of(unfinished, pri_tol);
                        if (pri >= 0 && pri_d <= pri_tol) {
                            nearest = pri;
                            nearest_dist = pri_d;
                            tolerance = pri_tol;
                        }
                    }
                    if (nearest < 0) {
                        auto [any, any_d] =
                            nearest_of(all_annotated_frames, tolerance);
                        nearest = any;
                        nearest_dist = any_d;
                    }
                    if (nearest >= 0 && nearest_dist <= tolerance) {
                        if (std::binary_search(unlabeled.begin(),
                                               unlabeled.end(), nearest))
                            ImGui::SetTooltip(
                                "Frame %d \xE2\x80\x94 not labelled", nearest);
                        else
                            ImGui::SetTooltip("Frame %d", nearest);
                        if (ImGui::IsMouseClicked(0)) {
                            ps.play_video = false;
                            seek_all_cameras(scene, nearest,
                                             dc_context->video_fps, ps, true);
                        }
                    }
                }

                ImPlot::EndPlot();
            }
            ImPlot::PopStyleVar();
        }

    }
    ImGui::End();

    // Ctrl+S save handling
    if (pm.plot_keypoints_flag && keys::pressed(keys::Sc::SaveLabels)) {
        state.save_requested = true;
    }

    // Toolbar Save button (from main menu bar)
    if (ctx.save_requested) {
        state.save_requested = true;
        ctx.save_requested = false;
    }

    if (state.save_requested) {
        std::string save_err;
        std::string saved_folder = AnnotationCSV::save_all(
            pm.keypoints_root_folder, skeleton.name,
            annotations, scene->num_cams, skeleton.num_nodes,
            pm.camera_names, &save_err);
        if (saved_folder.empty()) {
            toasts.pushError("Save failed: " + save_err);
        } else {
            state.last_saved = time(NULL);
            toasts.pushSuccess("Labels saved");
        }
    }
}
