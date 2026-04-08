# Hand Tracking Studio - Agent Context

This document gives full engineering context for AI agents that need to build new features on top of this project.

## 1) Project intent

Hand Tracking Studio is a desktop, real-time, webcam-based hand tracking app with:

- dual-hand tracking (left and right)
- finger flexion metrics
- bone vectors/lengths
- triangular hand mesh generation
- live 2D overlay on camera feed
- live 3D rendering in a separate panel
- estimated camera-space hand position and inter-hand distance

The current architecture is optimized for interactive experimentation and feature extension (gestures, XR control, robotics control, teleoperation, etc.).

---

## 2) Runtime stack

- Python 3.10+
- MediaPipe Tasks (`hand_landmarker` model)
- OpenCV (camera capture + 2D drawing)
- PySide6 (desktop UI)
- pyqtgraph + OpenGL (3D view)
- NumPy (math)

Dependencies are pinned in `requirements.txt`:

- `numpy>=1.26.0`
- `opencv-python>=4.9.0`
- `mediapipe>=0.10.14`
- `PySide6>=6.6.0`
- `pyqtgraph>=0.13.4`
- `PyOpenGL>=3.1.7`

---

## 3) Project layout and ownership

### Root

- `README.md` - user-facing setup and run instructions
- `requirements.txt` - Python dependencies
- `context.md` - this deep context for AI agents
- `hand_tracker/` - all application code

### `hand_tracker/`

- `main.py`
  - entry point (`python -m hand_tracker.main`)
  - CLI args (`--camera`, `--fps`, `--mode`)
  - global app font + log noise suppression env vars

- `gui.py`
  - all UI and rendering orchestration
  - camera device handling and recovery
  - mode switch and camera dropdown
  - 3D scene objects (2 hand slots)
  - tables (finger and bone metrics)

- `tracker.py`
  - MediaPipe hand detection/tracking integration
  - per-hand state assembly
  - handedness extraction
  - smoothing filters
  - camera-space reconstruction and depth estimate
  - 2D overlay drawing

- `geometry.py`
  - geometry utilities:
    - finger joint angles
    - finger flexion
    - bone measurements
    - mesh triangulation

- `constants.py`
  - hand skeleton connectivity (`BONE_CONNECTIONS`)
  - finger joint triplets
  - fallback mesh faces
  - hand side order

- `models/hand_landmarker.task`
  - auto-downloaded model artifact on first run

---

## 4) Data model contracts (important)

### `TrackerConfig` (`tracker.py`)

Core tunables:

- `max_num_hands` (default `2`)
- confidence thresholds
- depth reconstruction assumptions:
  - `camera_fov_degrees` (default `62.0`)
  - `default_hand_width_m` (default `0.085`)
  - min/max depth clamp
- smoothing parameters (`slow_alpha`, `fast_alpha`, velocity scale)

### `HandState` (`tracker.py`)

Per detected hand, per frame:

- `side`: `Left` / `Right` / fallback inferred
- `handedness_score`: confidence or `None`
- `landmarks`: normalized image landmarks (`x,y,z`)
- `world_landmarks`: raw world landmarks when returned (kept for diagnostics)
- `finger_flexion`: per-finger angle-like bend metric
- `bones`: vector + direction + length for each bone connection
- `mesh_faces`: triangle index list
- `camera_vertices`: reconstructed 3D points in camera space (meters)
- `camera_center`: palm center in camera space
- `estimated_depth_m`: scalar depth estimate

### `HandFrameState` (`tracker.py`)

- `hands: list[HandState]`

The GUI consumes this object every frame.

---

## 5) Coordinate systems and semantics

This project uses multiple spaces. Mixing them incorrectly will break 3D alignment.

1. **Normalized image space** (MediaPipe landmarks)
   - `x` in [0,1] left->right
   - `y` in [0,1] top->bottom
   - `z` relative depth-like value (model-specific, non-metric)

2. **World landmarks from MediaPipe**
   - available as `results.hand_world_landmarks`
   - treated as hand-local for this app
   - **not used as the global inter-hand metric frame**

3. **Camera space (app canonical 3D space)**
   - built from normalized landmarks + camera model assumptions
   - units in approximate meters
   - used for:
     - 3D rendering
     - bone and finger geometry metrics
     - inter-hand distance

Design rule: if you add features that compare two hands spatially, use `camera_vertices` / `camera_center`, not raw `world_landmarks`.

---

## 6) Tracking pipeline (frame-by-frame)

Implemented mainly in `HandTracker.process()`:

1. Convert BGR frame -> RGB.
2. Run MediaPipe Tasks `HandLandmarker.detect_for_video(...)` with monotonic timestamp.
3. For each detected hand:
   - extract normalized landmarks
   - infer handedness (`Left`/`Right`) from task output
   - apply adaptive smoothing (per hand identity key)
   - estimate absolute depth from palm width in pixels
   - convert normalized coords to camera-space metric 3D
   - compute finger flexion and bones from camera-space vertices
   - generate triangle mesh faces from normalized 2D points
4. Sort hands left-to-right by camera-space `x` center.
5. Draw 2D overlay and metrics text.
6. Return `(overlay_frame, HandFrameState)`.

---

## 7) 3D reconstruction details

Depth estimation uses a pinhole approximation:

- focal length from horizontal FOV
- palm width pair `(landmark 5, landmark 17)` as a scale cue
- global hand depth inferred from apparent pixel width
- per-landmark relative depth from MediaPipe `z` offsets

Then each point is projected to camera-space:

- `X = (u-cx) * Z / f`
- `Y = -(v-cy) * Z / f`
- `Z = depth + relative_z_offset`

This gives perspective-consistent hand placement and enables inter-hand distance estimation.

Important practical note:

- Relative movement is usually good.
- Absolute meter accuracy depends on actual camera FOV and true hand width calibration.

---

## 8) GUI architecture

Main class: `HandTrackerWindow` in `gui.py`.

### Update loop

- `QTimer` drives `_tick()`.
- each tick:
  - read camera frame
  - run tracker
  - update video label
  - update stats, tables, and 3D items

### 3D scene strategy

- pre-allocates **2 hand slots** for landmarks, bones, meshes
- visibility toggled depending on current detected hand count
- dynamic scene center and camera distance based on hand centers/spread

### Camera handling

Windows backend priority:

1. `CAP_MSMF`
2. `CAP_DSHOW`
3. `CAP_ANY`

Includes:

- startup index selection
- dropdown switching
- rescan button
- repeated-read failure recovery
- fallback capture profiles (high->medium->low)

---

## 9) Performance modes

CLI and UI support 3 modes:

- `balanced`
- `precision`
- `max`

Each mode sets:

- tracker confidence thresholds
- smoothing aggressiveness
- capture resolution/FPS
- default visibility toggles

Mode switch at runtime reinitializes camera + tracker with the selected profile.

---

## 10) Hand geometry and mesh generation

From `geometry.py` and `constants.py`:

- `BONE_CONNECTIONS`: standard 21-point hand skeleton edges
- `FINGER_JOINT_TRIPLETS`: triplets used to compute bend angles
- flexion metric: `180 - joint_angle`, averaged per finger
- mesh: Delaunay triangulation in normalized XY with quality filtering
- fallback mesh topology used if triangulation is unstable

---

## 11) Current known limitations

1. Monocular depth is estimated, not ground-truth.
2. Absolute inter-hand distance can drift with camera/FOV mismatch.
3. Only up to two hands are rendered in UI slots (tracker can be extended, UI currently fixed at two).
4. No automated unit/integration tests yet; current verification is manual runtime + `compileall`.
5. Gesture recognition layer is not implemented yet (only raw kinematics/geometry).

---

## 12) Known warnings/noise

Possible runtime logs:

- MediaPipe/TFLite warnings (`W0000 ...`) may appear and are often non-fatal.
- Some are reduced via env vars in `main.py`, but native libraries may still print occasional warnings.

---

## 13) Extension points for future agents

### Add gesture recognition

Best place:

- add classifier logic in `tracker.py` after `finger_flexion`/`bones` computation
- carry result in `HandState`
- display in `gui.py` status/table overlays

### Add recording/export

- add recorder class in new module, e.g. `hand_tracker/recording.py`
- hook from `_tick()` in `gui.py`
- write per-frame `HandFrameState` JSON/CSV + timestamps

### Add network streaming (OSC/WebSocket/ROS)

- create protocol adapter module (e.g. `transport.py`)
- publish `camera_vertices`, `bones`, `side`, and confidence
- avoid direct UI-thread blocking calls

### Add calibration workflow

Recommended:

- add UI controls for `camera_fov_degrees` and `default_hand_width_m`
- persist calibration in JSON config
- reinitialize `HandTracker` when calibration changes

### Add downstream control systems

- use `camera_center` and bone vectors as canonical control inputs
- apply temporal filtering and deadzones outside UI layer

---

## 14) Agent implementation guidelines

1. Keep camera-space as the single source of truth for cross-hand geometry.
2. Do not block Qt UI thread with heavy computation or network I/O.
3. Preserve runtime mode switching behavior.
4. Keep changes modular: tracking math in `tracker.py`, presentation in `gui.py`.
5. Prefer additive data fields in `HandState` over ad hoc globals.
6. When adding dependencies, update both `requirements.txt` and `README.md`.

---

## 15) How to run and validate quickly

### Windows (PowerShell)

```powershell
cd "C:\Users\shaik\OneDrive\Desktop\opencode-folder-2026-04-08"
.\.venv\Scripts\python.exe -m hand_tracker.main --camera 1 --mode precision
```

### Basic manual validation

1. Show one hand: verify 2D overlay, 3D mesh, finger/bone tables update.
2. Show both hands: verify distinct left/right 3D structures and non-zero distance badge.
3. Move one hand closer/farther from camera: verify `Z` and distance change.
4. Switch camera dropdown + mode dropdown: verify live reconfiguration without crash.
5. Toggle mesh/bones/landmarks: verify visibility control.

---

## 16) Suggested next roadmap

High-value improvements for future agents:

1. Calibration UI for FOV/hand width to improve metric accuracy.
2. Gesture state machine (pinch, grab, point, open palm, etc.).
3. Frame recording + replay for debugging and model tuning.
4. Optional CPU/GPU profiling overlay.
5. Automated tests around geometry functions and state contracts.

---

## 17) Prompt template for other AI agents

Use this prompt when asking another agent to build on this project:

```text
You are working on Hand Tracking Studio in this repo.
Read context.md first and treat camera-space coordinates in HandState as canonical.
Do not break runtime mode switching, camera dropdown switching, or dual-hand rendering.
Implement <FEATURE> with minimal UI-thread blocking and keep tracking math in tracker.py.
Update README.md and requirements.txt if behavior/deps change.
Return exact files changed and manual verification steps.
```
