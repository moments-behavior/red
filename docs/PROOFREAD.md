# Proofreading Workflow (dashboard-driven)

Branch: `pose_proofread_client`. Adds a server-driven proofreading mode so
red can pull *bad frames* — frames whose pose the pipeline got wrong — from
the [mouse_dashboard](https://github.com/JohnsonLabJanelia/mouse_pose_dashboard)
server, jump straight to them on the original recording, and re-label them
with red's normal tools. It reuses existing predictions and calibration, so
the point of a proofread project is correction, not labeling from scratch.

## What it does, end to end

1. **Connect** to the dashboard (default `http://10.102.10.138:8000`).
2. **Pick** an `(animal, session)` — the list is pulled from the server.
3. **Auto-fetch calibration** for that session into a local cache. The yamls
   are already red's OpenCV format (`camera_matrix` / `rc_ext` / `tc_ext`),
   so triangulation works with **no conversion**.
4. **Create a normal red project** pointed at the recording. Every red tool
   works: 2D/3D labeling, JARVIS, PoseTail, save, and export.
5. **Walk the bad frames** in the *Proofread Queue* panel — one click seeks
   all cameras to a bad frame; you fix it and move to the next.

## UI

Two entry points, both open the **Create Proofread Project** form:

- Top-level **`Proofread`** menu → `Create Proofread Project` / `Load
  Proofread Project`.
- Blue **Proofread** section on the Welcome screen.

### Create Proofread Project (`gui/proofread_dialog.h`)

Server-driven form (mirrors the annotation-project dialog):

- **Server URL** + `Refresh`. On open it auto-pulls the animal/session list.
- **Animal** / **Session** combos, populated from the server. The session
  combo shows each session's bad-frame count.
- On session-select, **calibration is fetched automatically** from
  `/api/session_calib_zip` into `~/.cache/red/proofread/<animal>/<session>/`,
  and `calibration_folder` is pointed at it. The cache is keyed per session
  (not per date) because the server resolves calibration per session — two
  same-date sessions can use different calibrations.
- **Only calibrated cameras load**: a camera is included only if it has both
  a `Cam<ID>.yaml` in the fetched calib *and* a `Cam<ID>.mp4` in the
  recording. Uncalibrated cams on disk are skipped.
- **Project Name** — defaults to **`<session>_proofread`** (editable). The
  `_proofread` suffix keeps an exported/uploaded dataset from colliding with
  the original session's data.
- **Project Root Path** (editable, with a Browse folder picker) and a
  read-only **Full Path** = `root / name`.
- **Skeleton** — preset or JSON file, same as a normal project.

`media_folder` is derived as `/mnt/free/<animal>/<session>`.

### Proofread Queue (`gui/proofread_window.h`)

Opens after a proofread project is created/loaded. Scoped to the loaded
`(animal, session)`.

- **`bad by:` source toggle — `IK residual` | `Scorer`.** Both are offered
  because scorer coverage is still partial — a session with no
  `scorer.parquet` simply won't appear under `Scorer`.
  - **IK residual**: a frame is bad if its triangulation residual ≥ a
    threshold in **mm**. Filter: `residual ≥ mm` + `min gap`.
  - **Scorer**: a frame is bad if a core keypoint's learned score is below a
    threshold. Filter: `score <` + `min bad kps` + `min gap`. Rows show the
    worst keypoint's score and name.
- **Frame table** with a **`Seek`** per row → issues an accurate
  `seek_all_cameras` to that video frame. The seek path is source-agnostic.
- `min gap` debounces clusters of adjacent bad frames down to one pick.

### Tailcycle prediction overlay

With **Overlay tailcycle prediction on Seek** on (default), seeking to a
frame that has no labels places the pipeline's tailcycle 3D prediction on
it and reprojects it onto every camera that isn't excluded. Keypoints are
marked Predicted (`P` in the saved CSV). The user drags the wrong ones and
presses T, rather than labelling from scratch. Frames that already have
labels are never overwritten; **Load on this frame** replaces the current
frame's keypoints with the prediction on purpose.

- Predictions for every frame in the queue are fetched in one go when the
  queue loads (and again on Refresh / Apply / source change); a frame
  outside the queue is fetched on demand.
- Keypoints map to the skeleton by name (case-insensitive). A skeleton with
  47+ nodes asks for `tailcycle47`, otherwise `tailcycle`; the server falls
  back to whichever exists.
- The server converts the 3D into the world frame of the calibration it
  serves (canonical vs. a date's native frame, via
  `jarvis_calibrations/world_alignment/<date>.json`), so the overlay lines
  up with the views.

### Export: corrected tailcycle CSV

The end product is the 3D pose. **Export corrected CSV** (Proofread Queue)
writes the session's tailcycle prediction with the user's corrections
swapped in, in exactly the tailcycle `data3D.csv` format (two header rows,
one row per video frame, x/y/z/confidence per keypoint, same world frame as
the original):

- **On the server**, next to the original, as its own prediction source:
  `<session>/tailcycle_proofread/` (or `tailcycle47_proofread/`) with
  `data3D.csv`, `corrections.json` (every corrected frame/keypoint, who,
  when, excluded cameras) and a copy of the source's `info.yaml` (same
  calibration). The original `data3D.csv` is never modified. In the
  dashboard the session gets a **PR** badge and the viewer a **proofread**
  toggle next to *tailcycle*, to compare the two side by side.
- **Locally**, a copy of both files in `<project>/proofread_export/`.

What counts as corrected: a keypoint the user **dragged** on a used camera
and then **Triangulated (T)**; it gets red's 3D and confidence 1.0. All
other keypoints/frames keep the prediction's values. Exports accumulate:
each one merges into `corrections.json` and rebuilds the CSV. red only sends
the corrected keypoints (a small JSON POST), so it works from any machine
that can reach the dashboard.

Fixing a point: drag it in **at least 2 cameras**, then press T. In a
proofread project a keypoint with 2+ dragged views is triangulated from
those views only — the other views hold reprojections of the prediction's
own (wrong) 3D and would pull the fix back.

### Bad-calibration cameras

A proofread session's calibration is sometimes wrong for one or two cameras,
which drags every triangulated keypoint off. The **user decides** which
cameras are bad:

1. Every camera loads, and the tailcycle prediction is drawn on all of
   them, so a camera whose points sit off the animal stands out.
2. In the Proofread Queue's **Cameras** section, untick **Use** on that
   camera. Its keypoints are cleared on every frame and it gets no more
   predictions; it is saved to `.redproj` right away, and from then on
   Triangulate / T / Refine 3D, save and export ignore it. Tick it to bring
   it back: frames you haven't touched get the prediction on it again;
   frames you already fixed get its points on the next T.
3. Fix the points on the good cameras and press T.
4. Export: excluded cameras get no images, labels or calibration.

**Apply (reload project)** (optional) saves labels and reloads so excluded
cameras stop being loaded at all; it returns to the same frame.

**Error px** is a hint from red's own check, not a decision: every
Triangulate records the frame's raw 2D (before it is overwritten by
reprojections), and a camera is marked **suggest exclude** when it is the
outlier — its median error against a triangulation from the other cameras
is ≥ 8 px and ≥ 3× the typical camera's, or leaving it out makes the rest
agree ≥ 3× better. *Scan annotated frames* adds every frame with
un-triangulated 2D.

What exclusion changes:

| Where | Effect |
| --- | --- |
| Load (`setup_project`) | Excluded cameras are removed from `camera_names` — not decoded, no view, no calibration. |
| Triangulate / T / Refine 3D | Excluded cameras are left out, and their 2D is never overwritten with a reprojection through the bad calibration. |
| Save (`save_all`) | An excluded-but-loaded camera's CSV is written blank; `excluded_cameras.txt` lists them. `annotations.json` stores `cam_name`, so bbox/mask data maps by name when the camera set changes. |
| Export (JARVIS window, all formats in Export) | Excluded cameras get no images, labels or calibration yaml. |

Persisted as `"excluded_cameras": ["Cam2006515"]` in `.redproj`.

### Load Proofread Project

Reopens a saved `.redproj`; the Proofread Queue re-fetches the bad-frame list
from the stamped `proofread_server_url` so the queue reflects the current
server state.

## Server endpoints used

All on the dashboard (`mouse_dashboard/app.py`):

| Endpoint | Used for |
| --- | --- |
| `GET /api/bad_frames_all` | Cross-session IK-residual bad frames (fills the pickers + the IK queue). |
| `GET /api/scorer_bad_frames_all` | Cross-session **scorer**-labelled bad frames (the `Scorer` source). Mirrors `bad_frames_all` but sourced from `scorer.parquet`. |
| `GET /api/session_calib_zip` | ZIP of the session's `Cam*.yaml` calibration. |
| `POST /api/session_corrections` | Merge corrected keypoints (`{frame: {keypoint: [x,y,z]}}`, served-calibration frame) and rewrite `<source>_proofread/data3D.csv` in the prediction's frame. |
| `GET /api/session_corrected_file` | Download that `data3D.csv` (`which=csv`) or `corrections.json` (`which=info`). |
| `GET /api/session_prediction` | Tailcycle 3D for `frames=` (comma-separated), in the served calibration's world frame (`mouse_dashboard/session_prediction.py`). 404 when the session has no tailcycle prediction; 409 when its frame can't be matched to the calibration. |

### Auth / trusted-IP bypass

The dashboard requires an HTTP Basic login, **but requests from trusted LAN
IPs skip it** — red sends no credentials. The trusted set defaults to
`127.0.0.0/8, ::1, 10.102.10.0/24` (the whole lab subnet), so any machine on
the lab network reaches the dashboard without a password. Override with the
`MOUSE_DASHBOARD_TRUSTED_IPS` env var on the server, e.g.:

```bash
MOUSE_DASHBOARD_TRUSTED_IPS="10.102.10.138,10.102.10.88" \
  uvicorn app:app --host 0.0.0.0 --port 8000
```

Anything off the list falls back to the normal login.

## Persisted fields (`.redproj`)

`Create Proofread Project` stamps these so `Load` can refetch:

```json
"proofread_server_url": "http://10.102.10.138:8000",
"proofread_animal":     "rat",
"proofread_session":    "2026_05_21_12_57_09",
"excluded_cameras":     ["Cam2006515"]
```

## Code layout

| Path | Purpose |
| --- | --- |
| `src/proofread_client.h` | Header-only client. `proofread_fetch` (bad-frame list, IK **or** scorer via a `Source` selector) and `proofread_fetch_calib` (download + unzip `Cam*.yaml`). |
| `src/gui/proofread_dialog.h` | `Create Proofread Project` form; auto-fetch calib on select; camera set from calibrated ∩ recorded. |
| `src/gui/proofread_window.h` | `Proofread Queue` panel: source toggle, filters, bad-frame table with per-row Seek. |
| `src/gui/main_menu_bar.h`, `welcome_window.h` | `Proofread` menu + Welcome-screen section. |
| `src/project.h` | `proofread_*` fields, persisted in `.redproj`. |
| `src/red.cpp` | Seek handler: turns a queue `Seek` into an accurate `seek_all_cameras`; *Apply (reload project)* handler. |
| `src/camera_check.h` | Raw-2D samples for the camera check, thresholds, exclusion mask. |
| `src/gui/gui_keypoints.h` | `reprojection` / `refine_3d_ba` take the exclusion mask; `analyze_camera_check`; `triangulate_frame`. |

## Known limits

- The `Scorer` source only lists sessions that have been scored
  (`scorer.parquet` present); everything else needs the `IK residual` source.
- The server URL is persisted per project; credentials are not entered in red
  at all — connectivity relies on the trusted-IP bypass above.
