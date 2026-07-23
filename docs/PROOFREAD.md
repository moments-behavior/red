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
"proofread_session":    "2026_05_21_12_57_09"
```

## Code layout

| Path | Purpose |
| --- | --- |
| `src/proofread_client.h` | Header-only client. `proofread_fetch` (bad-frame list, IK **or** scorer via a `Source` selector) and `proofread_fetch_calib` (download + unzip `Cam*.yaml`). |
| `src/gui/proofread_dialog.h` | `Create Proofread Project` form; auto-fetch calib on select; camera set from calibrated ∩ recorded. |
| `src/gui/proofread_window.h` | `Proofread Queue` panel: source toggle, filters, bad-frame table with per-row Seek. |
| `src/gui/main_menu_bar.h`, `welcome_window.h` | `Proofread` menu + Welcome-screen section. |
| `src/project.h` | `proofread_*` fields, persisted in `.redproj`. |
| `src/red.cpp` | Seek handler: turns a queue `Seek` into an accurate `seek_all_cameras`. |

## Known limits

- The `Scorer` source only lists sessions that have been scored
  (`scorer.parquet` present); everything else needs the `IK residual` source.
- The server URL is persisted per project; credentials are not entered in red
  at all — connectivity relies on the trusted-IP bypass above.
