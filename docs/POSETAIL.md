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
4. **Tools → PoseTail Tracker**, choose a backend, click **PoseTail Forward**.
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

## Backends

| | Server (HTTP) — default | Local ONNX |
|---|---|---|
| Needs | a running `posetail/server/server.py` | `lib/onnxruntime` bundle at build time, the exported `*tracker*.onnx`, a GPU with ~6–8 GB free (or *Use CPU*) |
| Per click | one 16-frame chunk → up to 15 future frames (*N future frames to keep*) | chunks are chained to reach *N forward frames* (1–200) |
| Cost | ~1–3 s round trip; the UI stalls for that long | ~30 ms/frame on GPU, ~3 s/frame on CPU, plus ~1 s model reload after every run |

**Server**: set the URL (`http://host:8000`), click **Probe** to `GET /info`
and confirm the model is the 16-frame × 256×256 one red expects. The status
line goes green/orange; per-call `encode / request / decode` timings show
under it. The URL is not persisted across launches.

**Local ONNX**: the path box is pre-filled from `~/src/posetail-onnx` or
`~/posetail-onnx` (first `*tracker*.onnx` found, 3 levels deep). **Load**
picks the CUDA device with the most free VRAM (printed to stderr with the
device name; CUDA ids ≠ `nvidia-smi` ids unless `CUDA_DEVICE_ORDER=PCI_BUS_ID`).
*Queries per pass* splits the keypoints into batches so no single tensor
exceeds the driver's per-allocation cap (24 kp → 2 × 12 by default). The
session is dropped and reloaded after every Forward: reusing it produced
slowly drifting predictions.

## Wire format (server backend)

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
| `src/gui/posetail_window.h` | `PosetailWindowState` + `DrawPosetailWindow()`. UI only, no ONNX / CUDA / httplib includes (it is pulled into `test_gui` via `window_states.h`). |
| `src/posetail_actions.h` | `PosetailRuntime` (ONNX session + HTTP state) and `posetail_handle_requests()`, called once per tick from `red.cpp`. Seed collection, frame staging from the display buffer, the two backends, and the write-back into `AnnotationMap`. |
| `src/posetail_infer.h` | Local ONNX Runtime path: crop-box geometry, chunk inference, chunk chaining (`posetail_forward`). Compiles without ONNX Runtime (everything returns "not available"). |
| `src/posetail_server_client.h` | HTTP client: `posetail_server_probe()`, `posetail_server_predict_chunk()`. Own crop/resize + `stb_image_write` PNG encode (this branch has no OpenCV), minimal `.npy` parser, `miniz` `.npz` reader. |
| `lib/httplib/httplib.h` | cpp-httplib v0.18.5, single header. |
| `lib/miniz/` | miniz 3.0.2, amalgamated. `miniz.c` is compiled into `red` (`project()` now lists `C`). |
| `CMakeLists.txt` | `DIR_HTTPLIB` / `DIR_MINIZ` include paths and `MINIZ_SRC` on all three platforms; optional `lib/onnxruntime` detection → `RED_HAS_ONNXRUNTIME` via `red_link_onnxruntime()`; Linux rpath onto the bundle. |

Platform notes baked into the headers:

- `POSETAIL_HAS_CUDA` is defined only when not on macOS and not
  `RED_NO_CUDA` (`-DRED_ENABLE_CUDA=OFF`). Without it the GPU-resident
  display buffer cannot be read back, so use the CPU frame buffer
  (Settings) — the default — for PoseTail on such builds.
- The display buffer is BGRA on macOS (`RED_FRAME_BGRA`, see `decoder.h`);
  both the local crop and the PNG encode swap channels accordingly so the
  model always sees RGB.
- Telecentric cameras: the 2D write-back uses `reproject_3d_to_cam()` and is
  correct; the crop box and the server metadata assume pinhole cameras.

## Build

No new system dependencies. For the local backend, unpack an ONNX Runtime
(GPU) release so that `lib/onnxruntime/include/onnxruntime_cxx_api.h` and
`lib/onnxruntime/lib/libonnxruntime.so` exist (the directory is gitignored),
then configure; CMake prints which backend set it found:

```
-- ONNX Runtime found at .../lib/onnxruntime -- local PoseTail backend enabled
-- ONNX Runtime not found at .../lib/onnxruntime -- PoseTail server backend only
```

On Linux with CMake < 3.24 and a CUDA build, `CMAKE_CUDA_STANDARD` is pinned
to 17 (the `.cu` files are C++17; CMake 3.22 cannot map `CUDA20` to an nvcc
flag and configure fails).

## Known limits

- Server backend: one chunk per click, so at most +15 frames; use the local
  backend to go further.
- Both backends run on the main thread; the window is unresponsive while a
  request is in flight.
- Timeouts are hardcoded: 3 s for `/info`, 10 s connect + 120 s read for
  `/predict`.
- One animal per click. Switch the active animal and click again for the
  next one.
