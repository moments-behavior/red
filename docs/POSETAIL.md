# PoseTail Tracker

Branch: `multianimal_posetail` (= `multianimal` + the PoseTail forward
temporal tracker ported from `pose_proofread_client`).

PoseTail takes the 3D pose on one frame and tracks it forward through the
next frames across all cameras at once. In red it lives in
**Tools → PoseTail Tracker** and works on whatever is labeled on the current
frame — no JARVIS, no prediction store, no separate detector.

## Workflow

1. Open a 3D project (calibration loaded) and its videos.
2. Label the current frame in at least two cameras for the animal you want
   to track. With several animals in the frame, select the one to track in
   the Labeling Tool (the panel shows `animal i/n (id k)`).
3. Press **T** to triangulate, or leave *Triangulate 2D labels first if there
   is no 3D* ticked and the panel does it for you.
4. **Tools → PoseTail Tracker**, set the server URL, click **PoseTail Forward**.
5. Step forward: frames `current+1 … current+N` now carry the predicted 3D and
   its 2D reprojection in every camera, marked *Predicted*, for that animal
   only. Fix them in the Labeling Tool like any other label; Save writes them
   with the rest.

The seed is the active animal's triangulated keypoints on the current frame.
Only those keypoints are tracked; untriangulated ones are left alone. Other
animals in the future frames are untouched (the prediction is written into
the FrameAnnotation whose `instance_id` matches the seed, created if the
frame has none yet).

All cameras must have the next 16 frames in the display buffer, which is the
normal case when paused on a frame. A camera / frame that is not staged is
sent as a grey image and a warning is printed.

## Server

Inference runs on the PoseTail HTTP server, `server/server.py` in
[AI-HHMI/tracktail](https://github.com/AI-HHMI/tracktail); red needs no GPU
or model of its own. Each click sends one 16-frame chunk and gets back up to
15 future frames (*N future frames to keep*), in a ~1–3 s round trip during
which the UI stalls.

Set the URL (`http://host:8000`), click **Probe** to `GET /info` and confirm
the model is the 16-frame × 256×256 one red expects. The status line goes
green/orange; per-call `encode / request / decode` timings show under it. The
URL is not persisted across launches.

## Wire format

tracktail's `server/SERVER.md` is the authoritative description.

One `POST /predict` (`multipart/form-data`) per click:

- `metadata`: JSON with `cameras` (`mat` = K scaled to the 256-crop, `dist`
  = 5 coefficients, `ext` = `[R|t; 0 0 0 1]` world→camera, `offset` = crop
  origin), `coords` (seed 3D), `query_times`.
- `images`: 16 × N_cams PNGs named `<cam_name>__<t>.png`, each the 256×256
  crop for that camera (square crop around the projected seed bbox + 20 px
  padding, expanded to ≥ 256, resized to 256). `cam_name` is the project's
  camera name.
- Response: `.npz` (uncompressed ZIP of `.npy`); red reads `coords_pred`,
  `vis_pred`, `conf_pred` (`conf_pred` accepted with or without the trailing
  `1` dim).

## Code layout

| Path | Purpose |
|---|---|
| `src/gui/posetail_window.h` | `PosetailWindowState` + `DrawPosetailWindow()`. UI only, no CUDA / httplib includes (it is pulled into `test_gui` via `window_states.h`). |
| `src/posetail_actions.h` | `PosetailRuntime` (HTTP state) and `posetail_handle_requests()`, called once per tick from `red.cpp`. Seed collection, frame staging from the display buffer, the request, and the write-back into `AnnotationMap`. |
| `src/posetail_server_client.h` | HTTP client: `posetail_server_probe()`, `posetail_server_predict_chunk()`. Crop-box geometry, own crop/resize + `stb_image_write` PNG encode (this branch has no OpenCV), minimal `.npy` parser, `miniz` `.npz` reader. |
| `lib/httplib/` | cpp-httplib v0.18.5 (git submodule; `git submodule update --init lib/httplib`). |
| `lib/miniz/` | miniz 3.0.2, amalgamated. `miniz.c` is compiled into `red` (`project()` now lists `C`). |
| `CMakeLists.txt` | `DIR_HTTPLIB` / `DIR_MINIZ` include paths and `MINIZ_SRC` on all three platforms; `ws2_32` on Windows. |

Platform notes baked into the headers:

- `POSETAIL_HAS_CUDA` is defined only when not on macOS and not
  `RED_NO_CUDA` (`-DRED_ENABLE_CUDA=OFF`). Without it the GPU-resident
  display buffer cannot be read back, so use the CPU frame buffer
  (Settings) — the default — for PoseTail on such builds.
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

- One chunk per click, so at most +15 frames; click again from the last
  predicted frame to go further.
- The request runs on the main thread; the window is unresponsive while a
  request is in flight.
- Timeouts are hardcoded: 3 s for `/info`, 10 s connect + 120 s read for
  `/predict`.
- One animal per click. Switch the active animal and click again for the
  next one.
