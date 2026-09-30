# tracktail

Branch: `multianimal_posetail` (= `multianimal` + the tracktail forward
temporal tracker ported from `pose_proofread_client`).

tracktail takes the 3D pose on one frame and tracks it forward through the
next frames across all cameras at once. In red it lives in
**Tools → tracktail** and works on whatever is labeled on the current
frame — no JARVIS, no prediction store, no separate detector.

## Workflow

1. Open a 3D project (calibration loaded) and its videos.
2. Label the current frame in at least two cameras for the animal you want
   to track. With several animals in the frame, select the one to track in
   the Labeling Tool (the panel shows `animal i/n (id k)`).
3. Press **T** to triangulate, or leave *Triangulate 2D labels first if there
   is no 3D* ticked and the panel does it for you.
4. **Tools → tracktail**, set the server URL, click **tracktail Forward**.
   Tick **Predict backwards** to track into the frames before the current one
   instead.
5. Step forward (or back): frames `current+1 … current+N` (or
   `current−N … current−1`) now carry the predicted 3D and its 2D reprojection
   in every camera, marked *Predicted*, for that animal only. Fix them in the
   Labeling Tool like any other label; Save writes them with the rest.

The seed is the active animal's 3D keypoints on the current frame, whether
triangulated or predicted by an earlier click (so clicking again from the
last predicted frame continues the track). Only those keypoints are tracked;
nodes with no 3D are left alone. Other
animals in the target frames are untouched (the prediction is written into
the FrameAnnotation whose `instance_id` matches the seed, created if the
frame has none yet). A tracked keypoint placed by hand on a target frame is
kept, and so are occluded marks, unless *Overwrite hand-placed labels* is
ticked.

Every image sent is looked up in the display buffer by frame number. The
buffer holds frames from the last seek onwards, so when the clip is not all
there (always for *Predict backwards*) red seeks to the clip's first frame,
waits for every camera to decode it (the panel shows *Loading frames …*;
20 s timeout, and starting playback cancels), then returns to the seed frame
and sends the request. N is shortened at the start/end of the video.

## Server

Inference runs on the tracktail HTTP server, `server/server.py` in
[AI-HHMI/tracktail](https://github.com/AI-HHMI/tracktail); red needs no GPU
or model of its own. Each click sends the seed plus *N frames to keep* (1–24,
default 4) on one side, rounded up to an even count and at most `n_frames`,
and writes those N predictions; a larger N is clamped to `n_frames − 1`.
Forward, the seed is the clip's first frame; backward, its last, and
`query_times` says so. The
round trip takes ~1–3 s, during which the UI stalls.

The chunk length (`n_frames`) and crop size (`image_size`) belong to the
model, not to red: `GET /info` reports them and red uses whatever it says.
Set the URL (`http://host:8000`) and click **Probe** to read them; **Forward**
probes by itself when the URL has not been probed yet. The status line goes
green/orange; per-call `encode / request / decode` timings show under it. The
URL is not persisted across launches.

## Wire format

tracktail's `server/SERVER.md` is the authoritative description.

One `POST /predict` (`multipart/form-data`) per click:

- `metadata`: JSON with `cameras` (`mat` = K scaled to the crop, `dist`
  = 5 coefficients, `ext` = `[R|t; 0 0 0 1]` world→camera, `offset` = crop
  origin), `coords` (seed 3D), `query_times`.
- `images`: `n_frames` × N_cams PNGs named `<cam_name>__<t>.png`, each the
  `image_size` × `image_size` crop for that camera (square crop around the
  projected seed bbox + 20 px padding, expanded to ≥ `image_size`, resized to
  `image_size`). `cam_name` is the project's
  camera name.
- Response: `.npz` (uncompressed ZIP of `.npy`); red reads `coords_pred`,
  `vis_pred`, `conf_pred` (`conf_pred` accepted with or without the trailing
  `1` dim).

## Code layout

| Path | Purpose |
|---|---|
| `src/gui/tracktail_window.h` | `TracktailWindowState` + `DrawTracktailWindow()`. UI only, no CUDA / httplib includes (it is pulled into `test_gui` via `window_states.h`). |
| `src/tracktail_actions.h` | `TracktailRuntime` (HTTP state) and `tracktail_handle_requests()`, called once per tick from `red.cpp`. Seed collection, frame staging from the display buffer, the request, and the write-back into `AnnotationMap`. |
| `src/tracktail_server_client.h` | HTTP client: `tracktail_server_probe()`, `tracktail_server_predict_chunk()`. Crop-box geometry, own crop/resize + `stb_image_write` PNG encode (this branch has no OpenCV), minimal `.npy` parser, `miniz` `.npz` reader. |
| `lib/httplib/` | cpp-httplib v0.18.5 (git submodule; `git submodule update --init lib/httplib`). |
| `lib/miniz/` | miniz 3.0.2, amalgamated. `miniz.c` is compiled into `red` (`project()` now lists `C`). |
| `CMakeLists.txt` | `DIR_HTTPLIB` / `DIR_MINIZ` include paths and `MINIZ_SRC` on all three platforms; `ws2_32` on Windows. |

Platform notes baked into the headers:

- `TRACKTAIL_HAS_CUDA` is defined only when not on macOS and not
  `RED_NO_CUDA` (`-DRED_ENABLE_CUDA=OFF`). Without it the GPU-resident
  display buffer cannot be read back, so use the CPU frame buffer
  (Settings) — the default — for tracktail on such builds.
- The display buffer is BGRA on macOS (`RED_FRAME_BGRA`, see `decoder.h`);
  the crop swaps channels accordingly so the model always sees RGB.
- Telecentric cameras: the 2D write-back uses `reproject_3d_to_cam()` and is
  correct; the crop box and the server metadata assume pinhole cameras.

## Build

No new system dependencies: cpp-httplib is a submodule (fetched by
`git clone --recursive`) and miniz is vendored.

On Linux with CMake < 3.24 and a CUDA build, `CMAKE_CUDA_STANDARD` is pinned
to 17 (the `.cu` files are C++17; CMake 3.22 cannot map `CUDA20` to an nvcc
flag and configure fails).

## Known limits

- One chunk per click, so at most `n_frames − 1` frames either way; click
  again from the last predicted frame to go further.
- The request runs on the main thread; the window is unresponsive while a
  request is in flight.
- Timeouts are hardcoded: 3 s for `/info`, 10 s connect + 120 s read for
  `/predict`.
- One animal per click. Switch the active animal and click again for the
  next one.
