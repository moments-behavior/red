# JARVIS Predict Panel — UX Assessment

*Johnson Lab, HHMI Janelia Research Campus — March 2026*
*Post-audit review of the model loading and prediction workflow.*

This document walks through every user scenario for the JARVIS Predict panel,
evaluating the experience from the perspective of a researcher who needs to load
trained pose estimation models and run inference on multi-camera video. The panel
is the primary interface between RED's annotation environment and JARVIS-trained
neural networks.

---

## Panel Architecture Overview

The JARVIS Predict panel has four visual sections, shown top-to-bottom:

1. **Project Models** — a combo dropdown of previously imported models (hidden if none exist)
2. **Import Model** — a folder browser + "Import to Project" button for bringing in new models
3. **Model Info** — metadata display for the active model (hidden until a model is loaded)
4. **Prediction** — confidence threshold slider, timing display, and "Predict Current Frame" button

The panel is registered in the Tools menu and docks into the main workspace like
all other tool windows. Default size is 480x600 pixels, which comfortably fits
all content without scrolling even when all four sections are visible.

---

## Scenario A: Returning User with an Imported Model (Most Common Case)

This is the daily-driver scenario. A researcher has already imported a trained
model in a previous session and is returning to continue labeling and predicting.

### What happens:

1. The user opens their project (File > Open or recent projects).
2. The `.redproj` file contains `jarvis_models` array and `active_jarvis_model` index.
3. The user opens Tools > JARVIS Predict.
4. **Auto-load fires immediately** (lines 94-120 of `jarvis_predict_window.h`):
   - Checks `!any_loaded && pm.active_jarvis_model >= 0`
   - Resolves the model directory from the project's relative path
   - Parses `model_info.json` once via `parse_jarvis_model_info()`
   - On macOS: prefers CoreML (`.mlpackage`) for GPU/ANE acceleration
   - Falls back to ONNX Runtime if no CoreML files exist
   - Properly cleans up the opposite backend before loading (defensive)

### What the user sees:

```
┌─────────────────────────────────────────┐
│ Project Models                          │
│ [mouseJan30 ▾]  (24 joints, 704x704)   │
│                                         │
│ Import Model                            │
│ Models Folder [                    ] [...│
│ ─────────────────────────────────────── │
│                                         │
│ Model Info                              │
│ Project:        mouseJan30              │
│ Backend:        CoreML (GPU/ANE) [green]│
│ Joints:         24                      │
│ Architecture:   EfficientTrack-medium   │
│ Center input:   320 x 320              │
│ Keypoint input: 704 x 704              │
│ Precision:      float16                 │
│ Normalization:  ImageNet (baked)        │
│ Model files:    jarvis_models/mouseJan30│
│                                         │
│ Prediction                              │
│ Confidence Threshold [====|====] 0.10   │
│ [Predict Current Frame]  Press 6 (hotkey│
└─────────────────────────────────────────┘
```

### Assessment:

- **Seamless.** Zero clicks required — the model auto-loads from saved project state.
- **Informative.** The Model Info section tells the user everything they need to know
  about what's running: backend, architecture, precision, input sizes.
- **Backend is highlighted.** CoreML (GPU/ANE) appears in green text, making it
  immediately obvious that hardware acceleration is active.
- **Predict button is ready.** The user can immediately click "Predict Current Frame"
  or press the hotkey (6) to run inference.
- **Import section is empty but not distracting.** The folder field is blank, and the
  "Import to Project" button is grayed out. This section is visually present but
  clearly inactive, signaling that no action is needed.

---

## Scenario B: First-Time Import (New Project, No Models Yet)

A researcher has trained a JARVIS model externally and wants to bring it into
RED for the first time. They have a folder containing `.onnx` or `.mlpackage`
files plus a `model_info.json`.

### What happens:

1. The user opens a new or existing project with no imported models.
2. Opens Tools > JARVIS Predict.
3. Auto-load check: `pm.jarvis_models.empty()` so `active_jarvis_model < 0` — skipped.
4. **Project Models section is hidden** (the `if (!pm.jarvis_models.empty())` guard).
5. The user sees only the Import section and the (disabled) Prediction section.

### Step-by-step user flow:

1. **Browse to model folder.** User clicks the `...` button next to "Models Folder",
   navigates to their JARVIS project's model output directory (e.g.,
   `/data/jarvis/mouseJan30/models/onnx/`), and selects it.

2. **File detection.** The panel scans the selected folder (cached, only on change):
   - Checks `models/onnx/` subdirectory first (JARVIS export convention)
   - Then checks the folder directly
   - Also checks for CoreML `.mlpackage` directories
   - Falls back to checking for `.pth` checkpoints

3. **Status feedback appears immediately:**
   - Green: "Found ONNX + CoreML models" (best case — both formats available)
   - Green: "Found CoreML models (.mlpackage)" (CoreML only)
   - Green: "Found ONNX models" (ONNX only)
   - Yellow: "Found .pth checkpoints (no ONNX/CoreML files)" (needs conversion)
   - Red: "No ONNX, CoreML, or .pth files found" (wrong folder)

4. **"Import to Project" button enables** (green status means loadable files exist).

5. **User clicks "Import to Project".** This does four things atomically:
   - **Loads** the model into memory (CoreML preferred on macOS, ONNX fallback)
   - **Copies** model files (`.onnx`, `.onnx.data`, `model_info.json`, `.mlpackage/`)
     into `<project>/jarvis_models/<model_name>/`
   - **Registers** the model in the project's `.redproj` metadata
   - **Saves** the updated `.redproj`

6. **Form clears.** After successful import, `models_folder` is cleared, signaling
   completion and preventing accidental re-import. Any stale conversion status
   messages are also cleared.

7. **Project Models combo appears** with the newly imported model selected.

8. **Model Info populates** with full details from the parsed `model_info.json`.

9. **"Predict Current Frame" enables.**

### Assessment:

- **Clear guided flow.** The user can't miss the steps: browse → see green status →
  click import → model appears in combo → predict.
- **Form clearing signals completion.** After import, the empty form makes it obvious
  the import succeeded. The model is now in the combo above.
- **Button label is honest.** "Import to Project" clearly communicates that files
  will be copied into the project directory. Previous label "Load & Import" was
  ambiguous about what "load" meant in this context.
- **Error path is clear.** If the wrong folder is selected, red text immediately
  tells the user no models were found, and the button stays disabled.
- **No unnecessary fields.** The old `config.yaml` input field (which was never used
  by any code path) has been removed. One fewer thing for the user to wonder about.

---

## Scenario C: Converting .pth Checkpoints to ONNX

A researcher has JARVIS `.pth` checkpoint files but hasn't exported to ONNX yet.
This is common when the researcher trains on a GPU server and wants to run
inference on their Mac laptop.

### What happens:

1. User browses to the JARVIS `models/` directory.
2. Scan finds `CenterDetect/Run_001/*_final.pth` and `KeypointDetect/Run_001/*_final.pth`.
3. No ONNX or CoreML files found.

### What the user sees:

```
Models Folder [/data/jarvis/mouse/models ] [...]
──────────────────────────────────────────────
Found .pth checkpoints (no ONNX/CoreML files)  [yellow]
[Import to Project]  (grayed out)
──────────────────────────────────────────────
ONNX files not found. You can convert .pth
checkpoints to ONNX using the JARVIS export script.
[Convert to ONNX]
```

### Step-by-step flow:

1. **"Import to Project" is disabled** — no loadable model format exists yet.

2. **Conversion section appears** with explanatory text and a "Convert to ONNX" button.

3. **User clicks "Convert to ONNX":**
   - A background thread runs `conda run -n jarvis python -m jarvis.utils.onnx_export`
   - The button shows "Converting..." (grayed out) during the operation
   - The UI remains fully responsive (thread is detached, communicates via atomic flags)

4. **Thread safety.** The conversion thread uses a `ConvertJob` struct shared via
   `std::shared_ptr`. The thread writes results to the job object using atomic flags;
   the main thread polls on each frame. No reference capture of UI state — the thread
   is safe even if the panel or application exits during conversion.

5. **On completion:**
   - Success: Green "Conversion complete. Click Import to Project." message
   - Failure: Red error message with exit code and first 200 chars of output

6. **Filesystem rescan triggers automatically** (`force_rescan` flag), detecting the
   new ONNX files. The status changes from yellow to green, and "Import to Project"
   becomes clickable.

7. **User clicks "Import to Project"** — proceeds as in Scenario B.

8. **After import, both the conversion message and import form clear.**

### Assessment:

- **Progressive disclosure.** The conversion section only appears when relevant
  (`.pth` files exist, no loadable format available, no model already loaded).
- **Non-blocking.** The conversion can take minutes (conda environment activation +
  PyTorch export). The background thread keeps the UI responsive.
- **Clear next step.** The success message explicitly says "Click Import to Project"
  rather than just "Done", guiding the user to the next action.
- **Thread safety is correct.** Previous implementation captured `&state` by reference
  in a detached thread — a use-after-free risk. The new `ConvertJob` + `shared_ptr`
  pattern ensures the job data outlives both the thread and the UI state.

---

## Scenario D: Switching Between Multiple Models

A researcher has imported multiple models (e.g., different training runs, or
models for different species) and wants to compare predictions.

### What happens:

1. Project has 2+ entries in `jarvis_models` (e.g., "mouseJan30", "ratFeb15").
2. The combo shows the currently active model.
3. User clicks the combo and selects a different model.

### Behind the scenes (lines 132-156):

1. `active_jarvis_model` index updates.
2. `model_info.json` is parsed once via `parse_jarvis_model_info()`.
3. `model_dir_display` updates to the new model's relative path.
4. **Backend cleanup:** Before loading the new model, the old backend is properly
   released:
   - Switching to CoreML: `jarvis_cleanup(jarvis)` releases ONNX sessions
   - Switching to ONNX: `jarvis_coreml_cleanup(jarvis_coreml)` releases CoreML models
5. New model loads with the pre-parsed config.
6. Model Info section updates immediately.

### Assessment:

- **Proper resource management.** The previous implementation set `.loaded = false`
  directly, orphaning ONNX sessions (unique_ptr leak) and CoreML MLModel objects
  (CFRetain leak). Now both `jarvis_cleanup()` and `jarvis_coreml_cleanup()` are
  called at every switch point, properly releasing GPU/ANE resources.
- **Config parsed once.** `parse_jarvis_model_info()` is called exactly once per
  switch. The result is passed to the init function and stored in the backend's
  `config` member. No redundant JSON I/O.
- **Model Info updates atomically.** The display reads from `jarvis_active_config()`,
  which returns a const reference to whichever backend's config is active. All fields
  (architecture, precision, normalization) update together.
- **Timing from previous model persists** until the next prediction. This is standard
  behavior for profiling displays — the last measurement is shown until replaced.

---

## Scenario E: Load Failure (Corrupt or Missing Model Files)

The model folder exists and contains files with the right names, but the files
are corrupt, incompatible, or incomplete.

### What happens:

1. File scan finds `.onnx` files → green status, button enables.
2. User clicks "Import to Project".
3. `jarvis_init()` or `jarvis_coreml_init()` fails internally.
4. `loaded_any` remains false.
5. The file copy and project registration block is skipped entirely.
6. The status line next to the button shows the error message in red.

### What the user sees:

```
Found ONNX models  [green]
[Import to Project]  ONNX error: invalid model format  [red]
```

### Assessment:

- **Error is visible immediately.** Red text next to the button, at the point of action.
- **No partial state.** If loading fails, no files are copied, no project metadata
  is modified, and the import form is not cleared. The user can fix the issue and
  retry without any cleanup.
- **Error color is correct.** Previous implementation used yellow (warning) for errors.
  Now uses red `ImVec4(1, 0.3f, 0.3f, 1)` — universally recognized as error.
- **Both backends report errors.** On macOS, if CoreML fails, `jarvis_coreml.status`
  is shown. If ONNX fails, `jarvis.status` is shown. Both use the same red color.

---

## Scenario F: No Project Open

Edge case: the user opens the JARVIS Predict panel before opening a project.

### What happens:

1. `pm.project_path` is empty.
2. Auto-load: `pm.jarvis_models` is empty, so auto-load is skipped.
3. Project Models section is hidden.
4. User can still browse to a model folder and load it into memory.
5. "Import to Project" will load the model but skip the file copy
   (`if (loaded_any && !pm.project_path.empty())` guard).
6. The model works for prediction but isn't persisted to any project.

### Assessment:

- **Graceful degradation.** The user can still use inference without a project.
  This is useful for quick one-off predictions or model validation.
- **No crash.** The empty project path is guarded at every point where file
  operations occur.

---

## Technical Design Decisions

### Single-parse model configuration

The `JarvisModelConfig` struct and `parse_jarvis_model_info()` function live in
`jarvis_model_config.h` — a lightweight header with no heavy dependencies. Both
the ONNX backend (`jarvis_inference.h`) and CoreML backend (`jarvis_coreml.h`)
include it. The predict window parses the JSON once and passes the result to
whichever init function it calls.

Previous design parsed `model_info.json` up to 4 times per model load (once in
each init function, once in the metadata display helper, and once inline for the
project name). The unified config eliminates all redundant I/O and ensures
consistency — every consumer sees the same parsed values.

### Backend-agnostic Model Info display

The `jarvis_active_config()` helper returns a const reference to whichever
backend's config is active. The ONNX backend stores config as `JarvisState::config`
(a `JarvisModelConfig` member). The CoreML backend stores it as
`JarvisCoreMLState::config` (also a `JarvisModelConfig` member, populated during
init). The display code doesn't need to know which backend is active — it just
reads from the returned config.

CoreML also maintains flat copies of `center_input_size`, `keypoint_input_size`,
and `num_joints` for hot-path inference access. These are set alongside the config
during init and are only used by the inference code in `jarvis_coreml.mm`.

### Thread-safe conversion

The `ConvertJob` struct uses `std::atomic<bool>` for `running` and `finished`
flags. Non-atomic fields (`message`, `success`, `force_rescan`) are only read
after `finished` is true, which provides a happens-before guarantee via the atomic
load. The job is shared via `std::shared_ptr`, so the thread and UI thread each
hold a reference — neither can outlive the data.

The main thread polls `finished` once per frame in the draw function. When
finished, it copies the result into `convert_status` (a plain string on the UI
side) and resets the shared_ptr.

### Resource cleanup at every switch point

There are three code paths that load a model: auto-load, combo selection, and
Import to Project. Each path calls `jarvis_cleanup()` before loading CoreML (to
release any ONNX sessions) and `jarvis_coreml_cleanup()` before loading ONNX (to
release any CoreML MLModel objects). Additionally, `jarvis_coreml_init()` itself
calls `jarvis_coreml_cleanup()` at the top as a safety guard against pointer
overwrite if called twice without an explicit cleanup.

---

## Remaining Polish Opportunities

These are not bugs or UX problems, but potential improvements for future iterations:

1. **Batch prediction.** Currently only predicts one frame at a time. A "Predict All
   Frames" button with a progress bar would complete the active learning loop.

2. **First-use help text.** When the panel opens with no models and an empty Import
   section, a brief explanation of what folder to browse to (and what file formats
   are expected) would help new users. A collapsible "Getting Started" section
   or tooltip on the browse button would suffice.

3. **Model validation before import.** Check that model input sizes match the
   project's video dimensions, or warn if the joint count doesn't match the
   skeleton definition.

4. **Import progress for large models.** Copying `.mlpackage` directories (which
   can be 10+ MB) is synchronous. For very large models on slow disks, a progress
   indicator would prevent the UI from appearing frozen.

5. **Drag-and-drop model folder.** Allow dragging a folder onto the Models Folder
   input field to auto-populate it.
