# Cropped-Sensor Calibration Tutorial (Rainbow Rig)

Two-stage calibration for rigs that record a sensor ROI crop: calibrate at
**full frame** with a ChArUco board, anchor fixed **posts** in 3D, design the
**crop** in RED, export it to **orange**, and get a calibration valid for the
cropped recordings — plus a later drift-correction path once real cropped
recordings exist.

Why it works: the ROI crop is a pure pixel translation, so the full-frame
calibration transforms *exactly* into the crop frame (`cx -= OffsetX`,
`cy -= OffsetY`; focal, distortion, and extrinsics unchanged). The posts give
you a physical reference to verify that transform and to absorb small pointing
drift later — refined with extrinsics locked, since a handful of posts in a
narrow crop can't constrain camera geometry (only the full-frame board can).

---

## What you need

- The cameras (4 for the test rig) running **full frame** in orange.
- A large ChArUco board (the bigger it appears in the views, the better).
- A printed post target section mounted rigidly in the arena
  (`scripts/post_target/` — e.g. section 2 for center-looking cameras,
  12 posts).
- The folder of orange configs (`{serial}.json`) you want the crop written to.

## Step 0 — Record two full-frame datasets

**A. ChArUco sweep** (before the plexiglass goes in, if possible):
wave the board slowly through the arena. Two things matter:
1. **Overlap**: the board must be visible in **≥2 cameras at once** at many
   poses — that's what chains the extrinsics. Sweep through the overlap zones
   between every adjacent camera pair.
2. **Sensor coverage**: also bring the board close to each camera so corners
   cover the whole sensor — that pins down distortion.

**B. Posts** (with the plexiglass/final enclosure in place): a short recording
or still images of the arena with the post target mounted. Full frame, same
camera settings. Images must be named `{serial}_{number}.png/jpg`
(e.g. `2002496_0.png`); videos `Cam{serial}.mp4`. One sharp image per camera
is enough — the posts don't move.

Record B immediately after A, before anything gets bumped.

## Stage 1 — Full-frame ChArUco calibration

1. Launch RED → **ArUco Calibration** on the welcome screen.
2. Point it at the charuco media folder, set the board parameters
   (squares, square/marker size in mm, dictionary), run the pipeline.
3. Sanity-check the result: mean reprojection well under ~0.5 px, all cameras
   registered. Output lands in a timestamped folder of `Cam{serial}.yaml`
   files — **this is your stage-1 calibration folder**.

## Stage 2 — The Cropped-Sensor Refinement wizard

Welcome screen → **Cropped-Sensor Refinement**. In the create dialog:
- **Stage-1 Calibration YAMLs**: the folder from Stage 1 (the output root with
  the timestamped subfolder works too).
- Optionally set the posts media folder and orange config folder now (both can
  be set later).

### Wizard step 1 — Posts at full frame

1. Set the **Media Folder** to the posts recording → **Load Media**
   (images or videos are auto-detected).
2. Set **Num Posts** (12 for one printed section) → **Setup Posts**.
   Each post becomes a numbered, colored skeleton node.
3. Click every post's **tip apex** in every camera view, **in id order,
   left-to-right along the base** (post 0 = leftmost). Skip posts a camera
   can't see. Press `W` to place the active node and auto-advance.
4. **Save Labels**, then **Triangulate Posts**.
5. Read the per-post table: green rows are good; **red rows** mean bad clicks,
   a swapped pair, or <2 views — fix the clicks and re-triangulate. Mean
   reprojection of a few px or less validates both your clicks *and* the
   stage-1 calibration. This writes `posts_3d.csv` into the project.

### Wizard step 2 — Design the crop, export, verify

1. Tick **Design crop on camera views**. An orange rectangle appears on every
   camera view — exactly the region orange will record.
2. Set the shared **Crop W / Crop H** (all cameras get the same size;
   everything snaps to multiples of 16, orange's convention).
3. Position each camera's rectangle: drag it, use **Auto-center on posts**,
   or type offsets in the table. The overlay label shows `posts N/M` — make
   sure every camera keeps **all its posts inside** (it turns red otherwise).
4. **Orange configs**: browse to your orange config folder → **Export ROI**.
   This rewrites `width`, `height`, `offsetx`, `offsety` in each
   `{serial}.json` in place, preserving every other key.
   ⚠ Orange currently applies only width/height — it hardcodes offsets to 0
   and must be patched to read `offsetx`/`offsety` before off-center crops
   take effect on the camera.
5. **Apply Crop + Verify**. This writes the cropped calibration to
   `<project>/cropped_calibration` and re-projects your full-frame post clicks
   through it. Expected result: **~0 px mean reprojection** and no principal
   point movement. If posts are reported outside a crop or residuals are
   large, the crop design or the click order is wrong — fix and repeat.

You now have a ready-to-use calibration for the cropped recordings:
`<project>/cropped_calibration`.

### Wizard step 3 — Later: refine on real cropped recordings (optional)

Weeks later, cameras may have drifted slightly. Record the posts again — now
**cropped**, through orange with the exported ROI — and:

1. Step 3: **Load Media** on the cropped recording. (If you accidentally load
   it against the full-frame calibration, RED warns about the dims mismatch.)
2. **Setup Posts** (this clears the step-1 clicks — full-frame coordinates
   would be wrong here) and click the **same posts in the same order**.
3. Choose refine options — defaults refine only the principal point (cx, cy),
   which absorbs pointing drift; **Free focal length** additionally absorbs
   the plexiglass depth shift. Distortion and extrinsics stay locked by
   design.
4. **Run Refinement** → step 4 shows before/after reprojection per camera,
   the cx/cy/f deltas, and a holdout check (red `OVERFITTING` flag if the
   held-out posts disagree). Output: `<project>/cropped_refined`.

If a camera's principal point moved more than a few tens of px, something
physically moved — redo Stage 1 rather than trusting the refinement.

## Using the calibration

In your annotation/tracking project, set `calibration_folder` to
`<project>/cropped_calibration` (or `cropped_refined` once step 3 has run).
Triangulation, reprojection, and export all work unchanged — the YAMLs are
standard `Cam{serial}.yaml` files with per-camera image sizes.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Red rows in the post table | Bad click or swapped ids — check node colors, re-click, re-triangulate |
| `posts N/M OUTSIDE` on a view | Crop too small / misplaced — drag or Auto-center |
| Verification ≫ 0 px | Click order differs between checks, or wrong stage-1 folder |
| `[load_videos] WARNING: ... wrong calibration for this crop?` | You loaded media against a calibration with different dims — pick the matching one |
| `OVERFITTING` in step 4 | Too few posts for the freed parameters — lock focal, or add posts |
| Orange records full frame despite export | Orange not yet patched to read `offsetx`/`offsety` (and check the config folder is the one orange loads) |
| Camera has no posts in its crop | It gets no verification/refinement — calibration still valid from stage 1, just unchecked |
