# PoseTail Server Client

Branch: `linux_posetail_client`. Adds an HTTP backend for the PoseTail
forward temporal tracker so red can offload inference to a shared GPU
host (`posetail/server/server.py`) instead of running ONNX locally.

## What's new in the UI

`JARVIS Predict` panel, `PoseTail (forward temporal tracker)` section:

- **`Backend:` radio**: `Local ONNX` | `Server (HTTP)`.
  Defaults to `Server (HTTP)` on a fresh launch.
- **Server-only widgets** (shown when `Server (HTTP)` is selected):
  - `URL` text field (default `http://10.102.10.88:8000`).
  - `Probe` button — `GET /info` to verify the server is reachable and
    its model matches red's hardcoded 16-frame × 256×256 input.
  - Status line: green on success, orange on errors / warnings.
  - Cached model info: `n_frames`, `image_size`, `device`, `mode_3d`.
  - Per-call timings: `total / encode / request / decode` in ms.
  - `N future frames to keep` slider (1–15). Chunk is always 16
    timepoints — `t=0` is the seed (= current frame), so `t=1..15` are
    the available future predictions.
- **Forward button** label updates to reflect the mode and N:
  - Local: `PoseTail Forward +N`
  - Server: `PoseTail Forward (server, +N)`

Behaviour matches the local path: the button seeds from the current
frame's triangulated 3D keypoints, predicts forward, and writes the
3D + reprojected 2D into annotations for frames
`[current+1 .. current+N]`. All cameras must have frames staged in the
display buffer.

## Wire format

Single round-trip per click:

- `POST /predict` (`multipart/form-data`):
  - `metadata` field: JSON with `cameras` (mat, dist, ext, offset),
    `coords`, `query_times` — same schema as
    `posetail/server/SERVER.md`.
  - `images`: 16 × N\_cams files named `<cam_name>__<frame_idx>.png`,
    each a 256×256 PNG of the per-camera crop (square crop centered on
    the projected seed bbox + 20 px padding, expanded to ≥ 256, resized
    to 256×256). Encoded at PNG compression level 1.
- Response: `.npz` (uncompressed ZIP of `.npy`). Client only reads
  `coords_pred`, `vis_pred`, `conf_pred`.

`conf_pred` is accepted in both `(1, 16, N, 1)` and `(1, 16, N)`
layouts — the live `posetail-odyssey` checkpoint omits the trailing-1
dim that `SERVER.md` documents.

## Code layout

| Path | Purpose |
| --- | --- |
| `src/posetail_server_client.h` | Header-only client (`posetail_server_probe`, `posetail_server_predict_chunk`). Mirrors `PosetailChunkResult` from `posetail_infer.h` so callers swap backends with no shape changes. Includes a minimal `.npy` parser and `mz_zip`-based `.npz` extractor. |
| `src/gui/jarvis_predict_window.h` | `JarvisPredictState` fields for server mode (URL, status, timings, n\_keep slider). Backend radio + server widgets. |
| `src/red.cpp` | `PosetailServerState` instance, probe handler, server forward handler. Server path branches off `posetail_use_server` before the local handler. |
| `lib/httplib/httplib.h` | cpp-httplib v0.18.5 (BSD), single header. Used for the multipart POST. |
| `lib/miniz/{miniz.h,miniz.c}` | miniz 3.0.2 (MIT/unlicense), amalgamated. Used to read the `.npz` ZIP container. |
| `CMakeLists.txt` | Added `C` to project languages (so `miniz.c` compiles), wired include paths and the `miniz.c` source on both Linux and macOS. |

## Build

No new system dependencies. cpp-httplib and miniz are vendored as
header-only / single-source. Standard `./build.sh` (or
`cmake -S . -B release && cmake --build release -j`) picks them up.

## Performance notes

Per-frame encode now crops the RGBA buffer first (zero-copy `cv::Mat`
view) and only runs `cvtColor` on the ~256×256 region. The naive
version converted the full 3216×2208 RGBA frame for every (cam, frame)
upload — ~28 MB × cams × 16 of throwaway work that dominated the
client-side latency before the request even left.

Remaining cost on a 16-cam, full-resolution session is
`cudaMemcpy`-staging the GPU display buffer to host (~7 GB total at
3216×2208 RGBA) and the raw HTTP upload (~3–8 MB of PNG depending on
content). Either can be the bottleneck on slow networks; the timing
line printed after each call breaks down `encode / request / decode`.

## Known limits

- One chunk per click. No chaining for `N > 15` — the local ONNX path
  is still the way to predict deep into the future.
- Connect / read timeouts are hardcoded (3 s for `/info`, 10 s connect
  + 120 s read for `/predict`). Most real failures surface as the
  `Status` line going orange.
- The server URL is not persisted across runs; on next launch it falls
  back to the compiled-in default. Add it to user\_settings if this
  becomes annoying.
