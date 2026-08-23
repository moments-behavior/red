# Cropped-calibration review notes — branch `calib-review-fixes`

**For:** Wilson Chen (wchen27) — please review, then merge into `xp` (or push back on anything).
**From:** Rob + Claude, 2026-08-23.
**Scope:** an automated + manual review of the cropped-sensor calibration / bee-rig work
(`d05c668`, `29bc205`, and the `1dd1204` merge) turned up the findings below. The
low-risk fixes are implemented on this branch as three commits; the judgment calls are
listed at the end as open questions for you — nothing behavior-changing was decided
unilaterally there.

All commits build clean on macOS (`./build.sh`) and pass `test_crop_refinement`
(416 checks), `test_pump_events` (79), and the full annotation suite.

---

## Fixed on this branch

### Commit 1 — data plumbing

| Finding | Fix |
|---|---|
| **"Extrapolated" post flag was dead code.** `load_coverage_radii` (crop_calibration.h) requires `residuals.obs_x/obs_y` in `calibration_data.json`, but the writer only emitted `camera_idx/landmark_id/error` — so the Posts table column was always "–". | Writer (`calibration_pipeline.h`, `write_calibration_data`) now emits `obs_x/obs_y`; the `Residual` struct already carried them, so this is 2 parallel arrays in the JSON. **Note:** existing stage-1 runs predate the field — re-running stage 1 is needed before the flag lights up. |
| **Malformed `calibration_data.json` threw into the Triangulate button handler** — the `.get<>()` calls sat outside the reader's `try`. | Whole parse now inside the `try`, plus an `obs_x/obs_y/camera_idx` length check. Degrades to "no coverage data". |
| **`resolve_calibration_folder` could throw in the render loop** — the outer `fs::directory_iterator` had no `error_code` overload (unreadable/network dir ⇒ exception every frame). | Non-throwing overload. |
| **media_loader dims-mismatch warning could misattribute the source** — when the calibration file has no image size, the "calibration says WxH" values were actually back-filled from the first media loaded. | Warning reworded to say so. |

### Commit 2 — wizard state safety + crop-spec integrity

| Finding | Fix |
|---|---|
| **Apply Crop / Export ROI silently rewrote the spec.** `apply_shared_dims` forces every row to the shared `crop_w/h` and floors offsets to multiples of 16 at Apply time. An imported `crop_info.json` with offset 100 or heterogeneous per-camera dims (which `apply_crop_to_calibration` fully supports) got baked in with up to 15 px of principal-point error, no warning — and the table showed the raw values until the moment they were destroyed. | Apply / Apply+Verify / Export ROI now diff the spec against what snapping would produce (`spec_snap_changes`). No difference ⇒ unchanged one-click flow. Any difference ⇒ a **"Snap crop spec?"** modal listing the exact per-camera rewrites, with Snap & Continue / Cancel. Designer-time snapping (checkbox, Crop W/H inputs, Fill-from-media) is untouched — that's the editing tool's documented convention. |
| **Crop spec wasn't persisted across project reopen.** The table reseeded with zero offsets while "Cropped calibration ready" showed; pressing Apply again would overwrite `cropped_calibration/` with an **unshifted** calibration. | One-shot restore from `proj.crop_info_file` when step 2 first draws (same merge-by-serial as the Import button). Belt-and-braces: an orange warning appears next to Apply whenever all offsets are 0 but a cropped calibration already exists. |
| **Step-1 buttons acted on whatever media was loaded.** `load_posts_media(cropped_stage=true)` didn't retire `fullframe_videos_loaded`/`fullframe_skeleton_ready`, so after loading cropped media, Triangulate / Auto-center / Export ROI / Verify stayed enabled and y-flipped or shifted with the cropped dims ⇒ silently wrong `posts_3d.csv` / verification. | Media stages are now mutually exclusive: loading cropped media clears the full-frame flags and vice versa. Auto-center and Export ROI additionally require full-frame media (Export clamps against full sensor dims read from the loaded scene). |
| **`CroppedState` never reset between projects** — stale spec/report/results carried over, and the refine worker holds `&cs.status`, so a reset while it ran would have dangled. | On tool close/project switch: valid futures are `wait()`ed (they're seconds-long Ceres solves; see open question 4), then `state.cropped = {}`. |
| **Stage-1 folder resolution ran a directory walk every UI frame** (resolve + N× `fs::exists`). | Cached, refreshed every ~60 frames or on folder change. |

### Commit 3 — this document.

---

## Open questions for you (nothing changed on the branch)

1. **`reproj_threshold_scale` applies to every rig above 1080p, not just the bee rig.**
   It multiplies the PnP (10 px), BA outlier, triangulation (50 px) and quality (1 px)
   gates by `diag/2202.9` — so an existing 2448×2048 ArUco rig's thresholds change
   ×1.45 on its next recalibration, silently. If that's intended, fine; if not, our
   suggestion is to gate it behind a per-calibration setting (on by default for the
   cropped/bee workflow, off for existing subtypes) so legacy rigs reproduce old
   results. **Your call — you know whether any legacy >2 MP rigs matter.**

2. **`apply_crop_to_calibration` copies the full-frame `calibration_data.json` into
   `cropped_calibration/`.** Anything reading it from there (calib viewer, the new
   coverage radii) interprets full-frame pixels as crop pixels. Options: don't copy;
   copy with a `"frame": "full"` tag; or shift the residual coords. Depends what your
   tooling expects to find in that folder.

3. **ArUco slot-map sync**: design looks right (fail-closed to constant offset), one
   nit — the Detect button now runs `get_video_frame_count` per camera on the UI
   thread. Worth moving off-thread if it's noticeable on your rigs.

4. **The future-wait on project close** (commit 2) blocks the UI for the tail of an
   in-flight verify/refine solve. Acceptable? The alternative is a detach/generation
   counter, but that needs the worker to stop taking `&cs.status` — happy to do that
   as a follow-up if you'd rather not block.

5. **`orange_status` async race** (pre-existing pattern, unchanged): the refine worker
   writes `cs.status` through a pointer while the UI thread reads it. Same pattern as
   `kp_status` elsewhere, so we left it; flagging in case you want a shared fix.

6. **Snap-16**: the confirm-modal fix assumes snap-16 + shared dims are orange
   hardware constraints, and that *specs already satisfying them* should apply
   silently. If offsets have a different alignment constraint per camera model, the
   `spec_snap_changes` helper is the one place to encode it.

## How to review

```
git fetch origin calib-review-fixes
git log --oneline origin/xp..origin/calib-review-fixes   # 3 commits
git diff origin/xp...origin/calib-review-fixes
```

Merge with a plain `git merge calib-review-fixes` on `xp` (it branches from current
`xp` head `c481e28`, so it should be conflict-free unless `xp` moves first).
