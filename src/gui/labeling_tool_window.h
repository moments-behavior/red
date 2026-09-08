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
    // Which annotation class the timeline is filtered to, as an index into the
    // fixed class list below; -1 shows everything. Selecting one is also what
    // Prev/Next walk, so there is one jump control rather than a row of them.
    int selected_class = -1;
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
        // Only the previous frame is still needed up here, for "Copy Prev";
        // walking the labelled frames is jump_buttons' job now.
        const bool has_prev = kp_pn.prev >= 0;
        const int prev_frame = kp_pn.prev;

        // === Top row: Save, Triangulate, Prev/Next label ===
        // Saves red's CSVs into the project's label folder. A tailcycle
        // session has none -- it is saved back into its own tables from the
        // tailcycle Dataset panel -- so the button says why rather than
        // failing when pressed.
        const bool can_save_csv = !pm.keypoints_root_folder.empty();
        ImGui::BeginDisabled(!can_save_csv);
        if (ImGui::Button(ICON_FK_FLOPPY_O " Save")) {
            state.save_requested = true;
        }
        ImGui::EndDisabled();
        if (!can_save_csv &&
            ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("This session has no red label folder. Use "
                              "\"Save corrections to this session\" in the "
                              "tailcycle Dataset panel.");

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

        // (Prev / Jump / Next lives with the other jump controls in the frame
        // overview below. Up here it was an unlabelled pair of buttons that
        // looked exactly like the per-type ones but walked a different set,
        // with nothing on screen to say so. There is one jump control now, and
        // the selected chip under the timeline says what it walks.)
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
        // The states themselves are documented on KpProgress in annotation.h,
        // which the Frame Buffer classifies with too.
        auto classify_kp_state = [&](const FrameAnnotation &fa) {
            return frame_kp_progress(fa, skeleton.num_nodes, (int)scene->num_cams,
                                     project_is_2d(pm), skeleton.has_skeleton);
        };

        struct LabeledFrameInfo { int frame; KpProgress state; };
        std::vector<LabeledFrameInfo> labeled_frames;
        std::vector<LabeledFrameInfo> needs_fix_frames;
        // The frame list is per frame, not per animal: a frame appears once,
        // classified by the animal being labelled.
        for (const auto &[fnum, fis] : annotations) {
            if (fis.empty() || !frame_has_any_keypoints(fis.front()))
                continue;
            const FrameAnnotation &fa = fis.front();
            KpProgress state = classify_kp_state(fa);
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
        // counts below do not double-count a frame that is annotated, boxed
        // and flagged all at once.
        struct FrameCell {
            int frame = 0;
            KpProgress kp_state = KpProgress::None;
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

        // === Are unlabelled frames the exception? ===
        // 8000 frames across ~250px of timeline is 0.03px each, so the one
        // frame that is still yellow is invisible among the green -- and one
        // that was never labelled has no annotation entry, so it has no tick
        // at all and you are hunting an absence. A button that walks the list
        // is the only thing that reliably reaches it.
        const int total_frames = dc_context->estimated_num_frames;
        // Are the never-labelled frames the exception, or the norm? On a
        // video where 100 frames of 8000 are labelled they are the norm: an
        // orange tick on each would paint the bar and bury the 100 that
        // matter. On a proofread that is nearly done they are exactly what
        // you are looking for. The test is which side outnumbers the other,
        // not a coverage percentage -- 7000 of 8000 is a pass with 1000
        // frames left to do, and hiding them because it missed a 98% cutoff
        // helps nobody.
        // The cap is part of the test, not a truncation applied after it:
        // enumerating 40000 gaps and plotting the first 4096 would read as
        // "these are the gaps" while quietly hiding nine tenths of them.
        constexpr int kMaxGaps = 4096;
        const int gap_count = ImMax(0, total_frames - (int)cells.size());
        const bool gaps_are_exceptional =
            total_frames > 0 && gap_count > 0 &&
            gap_count < (int)cells.size() && gap_count <= kMaxGaps;

        // Frames with no annotation at all. The timeline plots them as
        // their own class, and only while they are the exception -- they are
        // the one thing it cannot show by colouring a tick, having no tick.
        std::vector<int> unlabeled;
        if (gaps_are_exceptional) {
            int expected = 0;
            for (const auto &c : cells) {
                for (; expected < c.frame; ++expected)
                    unlabeled.push_back(expected);
                expected = c.frame + 1;
            }
            for (; expected < total_frames; ++expected)
                unlabeled.push_back(expected);
        }

        // The shared palette (keypoint_colors.h). Aliased locally so the rest
        // of this function reads the same as before.
        const ImVec4 &color_green  = kLabelComplete;
        const ImVec4 &color_teal   = kLabelTriangulated;
        const ImVec4 &color_yellow = kLabelUntriangulated;
        const ImVec4 &color_red    = kLabelNeedsFix;
        const ImVec4 &color_orange = kLabelGap;
        const ImVec4 &color_purple = kLabelBBox;
        const ImVec4 &color_lilac  = kLabelOBB;

        // === Keypoints table ===
        // Height follows the window by default -- whatever the frame overview
        // below does not need -- until you drag the splitter under it, after
        // which your height wins. Double-click the splitter to go back.
        {
            const float line_h = ImGui::GetTextLineHeightWithSpacing();
            // What sits below the table now: one Jump row, a Mark-fixed row
            // when the current frame carries that flag, then the timeline.
            int overview_lines = 1;
            if (!needs_fix_frames.empty()) overview_lines++;
            // Plot plus up to two wrapped rows of class chips.
            const float timeline_block =
                (dc_context->estimated_num_frames > 0) ? 78.0f + 2.0f * line_h
                                                       : 0.0f;
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

        // === Classes ===
        // Every annotation type as one list, in a fixed order so an index into
        // it is stable. These drive three things at once: the counts, which
        // frames the timeline plots, and what Prev/Next walks. They used to be
        // three separate count rows each with its own Prev/Jump/Next, which on
        // a densely labelled recording meant several jump controls that all
        // stepped frame by frame.
        struct FrameClass {
            const char *label;
            const ImVec4 *color;
            const char *tip;
            std::vector<int> frames;
        };
        FrameClass classes[] = {
            {"complete", &color_green,
             "every keypoint placed in every camera, and triangulated. A "
             "keypoint hidden from one camera keeps a frame out of this state "
             "however finished it is.", {}},
            {"triangulated", &color_teal,
             "every keypoint you placed is triangulated, but not all keypoints "
             "are placed", {}},
            {"untriangulated", &color_yellow,
             "some placed keypoint has no 3D yet", {}},
            {"needs fixing", &color_red,
             "a promoted prediction still waiting on a manual fix", {}},
            {"gap", &color_orange,
             "no annotation on this frame -- a hole in otherwise complete "
             "coverage. Only listed while unlabelled frames are the minority.", {}},
            {"bbox", &color_purple, "an axis-aligned bounding box", {}},
            {"obb", &color_lilac, "an oriented bounding box", {}},
        };
        constexpr int kFull = 0, kTri = 1, kUntri = 2, kFix = 3, kGap = 4,
                      kBBox = 5, kOBB = 6;
        constexpr int kNumClasses = 7;

        for (const auto &lf : labeled_frames) {
            if (lf.state == KpProgress::Complete) classes[kFull].frames.push_back(lf.frame);
            else if (lf.state == KpProgress::Triangulated)
                classes[kTri].frames.push_back(lf.frame);
            else classes[kUntri].frames.push_back(lf.frame);
        }
        for (const auto &nf : needs_fix_frames) classes[kFix].frames.push_back(nf.frame);
        classes[kGap].frames = unlabeled;
        for (const auto &bf : bbox_frames) {
            if (bf.has_bbox) classes[kBBox].frames.push_back(bf.frame);
            if (bf.has_obb)  classes[kOBB].frames.push_back(bf.frame);
        }

        if (state.selected_class >= 0 &&
            (state.selected_class >= kNumClasses ||
             classes[state.selected_class].frames.empty()))
            state.selected_class = -1;   // the class went away under us

        // No separate totals line: the chips under the timeline carry every
        // count, including the "all labelled" one, so a header repeating them
        // was one more row saying what was already on screen.
        if (!needs_fix_frames.empty()) {
            auto cur_it = annotations.find((u32)current_frame_num);
            const bool cur_needs_fix = cur_it != annotations.end() &&
                                       !cur_it->second.empty() &&
                                       cur_it->second.front().needs_improvement;
            if (cur_needs_fix) {
                if (ImGui::SmallButton("Mark fixed"))
                    cur_it->second.front().needs_improvement = false;
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Clear the Needs-Improvement flag on this "
                                      "frame.");
            }
        }

        // === Jump ===
        // Walks the selected class, or every labelled frame when nothing is
        // selected. One control, and what it steps through is whatever the
        // legend below says is selected.
        {
            const std::vector<int> *walk =
                state.selected_class >= 0 ? &classes[state.selected_class].frames
                                          : nullptr;
            std::vector<int> all_labeled;
            if (!walk) {
                for (const auto &lf : labeled_frames)
                    all_labeled.push_back(lf.frame);
                walk = &all_labeled;
            }
            PrevNext pn;
            if (!walk->empty()) {
                auto nx = std::upper_bound(walk->begin(), walk->end(),
                                           current_frame_num);
                pn.next = (nx != walk->end()) ? *nx : walk->front();
                auto lb = std::lower_bound(walk->begin(), walk->end(),
                                           current_frame_num);
                pn.prev = (lb != walk->begin()) ? *std::prev(lb) : walk->back();
            }
            ImGui::TextDisabled("Jump");
            ImGui::SameLine();
            if (state.selected_class >= 0)
                ImGui::TextColored(*classes[state.selected_class].color, "%s",
                                   classes[state.selected_class].label);
            else
                ImGui::TextDisabled("all labelled");
            ImGui::SameLine();
            jump_buttons(pn, "class");
        }

        // === Timeline minimap ===
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
            // One array of doubles per class, and only the selected one when
            // a class is selected -- which is the point of selecting it: on a
            // densely labelled recording the interesting series is buried
            // under the common one.
            std::vector<double> class_x[kNumClasses];
            for (int c = 0; c < kNumClasses; c++) {
                if (state.selected_class >= 0 && state.selected_class != c)
                    continue;
                class_x[c].reserve(classes[c].frames.size());
                for (int f : classes[c].frames)
                    class_x[c].push_back((double)f);
            }

            // Collect all annotated frames for click-to-seek
            std::vector<int> all_annotated_frames;
            for (int c = 0; c < kNumClasses; c++) {
                if (state.selected_class >= 0 && state.selected_class != c)
                    continue;
                all_annotated_frames.insert(all_annotated_frames.end(),
                                            classes[c].frames.begin(),
                                            classes[c].frames.end());
            }
            std::sort(all_annotated_frames.begin(), all_annotated_frames.end());
            all_annotated_frames.erase(
                std::unique(all_annotated_frames.begin(), all_annotated_frames.end()),
                all_annotated_frames.end());

            // Ticks are drawn centred on their frame, so one at frame 0 or at
            // the last frame loses half its width to the plot edge and the
            // surviving half lands flush under the border -- reading as absent.
            // The wider the tick the worse it is, so the widening that made a
            // lone frame visible is what buried it at the extremes, and frame 0
            // is exactly the frame someone hunting a deleted label lands on.
            // Pad the axis by half the widest tick so the ends sit inside the
            // plot.
            const double px_per_frame_full =
                timeline_w / (double)ImMax(1, total_frames);
            const double x_pad = 6.0 / ImMax(px_per_frame_full, 1e-9);
            const double x_lo = -x_pad, x_hi = (double)total_frames + x_pad;

            if (state.timeline_reset_pending) {
                ImPlot::SetNextAxesLimits(x_lo, x_hi, 0, 1);
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
                ImPlot::SetupAxisLimits(ImAxis_X1, x_lo, x_hi, ImPlotCond_Once);
                ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 1, ImPlotCond_Always);
                ImPlot::SetupAxisZoomConstraints(ImAxis_X1, 50, x_hi - x_lo);

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
                Series series[kNumClasses];
                static const char *kIds[kNumClasses] = {
                    "##full", "##tri", "##untri", "##fix",
                    "##gap", "##bbox", "##obb"};
                for (int c = 0; c < kNumClasses; c++)
                    series[c] = {kIds[c], &class_x[c], classes[c].color};
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
                    // Nothing to prioritise once a class is selected: the
                    // plot holds only that class, so nearest is already it.
                    int nearest = -1;
                    double nearest_dist = 0.0;
                    {
                        auto [any, any_d] =
                            nearest_of(all_annotated_frames, tolerance);
                        nearest = any;
                        nearest_dist = any_d;
                    }
                    if (nearest >= 0 && nearest_dist <= tolerance) {
                        if (std::binary_search(classes[kGap].frames.begin(),
                                               classes[kGap].frames.end(),
                                               nearest))
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

            // === Legend, and the class selector ===
            // Each entry carries its own count and is clickable: selecting one
            // filters the timeline to it and points Prev/Next at it. Clicking
            // it again clears the filter.
            {
                const float h = ImGui::GetTextLineHeight();
                const float sw = h * 0.62f;
                const float pad = 4.0f;
                const float step = ImGui::GetStyle().ItemSpacing.x;
                const float right = ImGui::GetWindowPos().x +
                                    ImGui::GetWindowContentRegionMax().x;

                // -1 is the "all labelled" chip: the total, and the way back
                // to an unfiltered timeline.
                bool first = true;
                for (int c = -1; c < kNumClasses; c++) {
                    const FrameClass *fcp = (c >= 0) ? &classes[c] : nullptr;
                    if (fcp && fcp->frames.empty()) continue;

                    char text[64];
                    if (fcp)
                        snprintf(text, sizeof(text), "%s (%zu)", fcp->label,
                                 fcp->frames.size());
                    else
                        snprintf(text, sizeof(text), "all (%zu)",
                                 labeled_frames.size());
                    const float text_w = ImGui::CalcTextSize(text).x;
                    const float chip_w = sw + pad + text_w + pad * 2.0f;

                    if (!first && ImGui::GetItemRectMax().x + step + chip_w < right)
                        ImGui::SameLine(0, step);
                    first = false;

                    const bool selected = (state.selected_class == c);
                    ImGui::PushID(c);
                    ImVec2 p0 = ImGui::GetCursorScreenPos();
                    if (ImGui::Selectable("##chip", selected,
                                          ImGuiSelectableFlags_None,
                                          ImVec2(chip_w, h)))
                        state.selected_class = (selected || !fcp) ? -1 : c;
                    if (ImGui::IsItemHovered())
                        ImGui::SetTooltip(
                            fcp ? "%s\n\nClick to show only these on the timeline "
                                  "and step through them with Jump."
                                : "%s",
                            fcp ? fcp->tip
                                : "Every frame carrying keypoints. Jump steps "
                                  "through all of them.");
                    ImDrawList *dl = ImGui::GetWindowDrawList();
                    if (fcp)
                        dl->AddRectFilled(
                            ImVec2(p0.x + pad, p0.y + (h - sw) * 0.5f),
                            ImVec2(p0.x + pad + sw, p0.y + (h + sw) * 0.5f),
                            ImGui::ColorConvertFloat4ToU32(*fcp->color), 2.0f);
                    else
                        dl->AddRect(
                            ImVec2(p0.x + pad, p0.y + (h - sw) * 0.5f),
                            ImVec2(p0.x + pad + sw, p0.y + (h + sw) * 0.5f),
                            ImGui::GetColorU32(ImGuiCol_TextDisabled), 2.0f);
                    dl->AddText(ImVec2(p0.x + pad + sw + pad, p0.y),
                                ImGui::GetColorU32(selected ? ImGuiCol_Text
                                                            : ImGuiCol_TextDisabled),
                                text);
                    ImGui::PopID();
                }
            }
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
