# Hand Tracking Studio

A real-time hand tracking desktop app with:

- Live webcam tracking for both left and right hands using MediaPipe Tasks.
- Finger movement metrics (per-finger flexion angles for each hand).
- Bone movement generation (per-bone vectors and lengths for each hand).
- Dynamic hand mesh generation over tracked landmarks for each detected hand.
- 3D visualization panel for dual-hand landmarks, bones, and mesh.
- Camera-space 3D estimation for hand position and depth, including inter-hand distance.

## Stack

- Python 3.10+
- PySide6 (GUI)
- OpenCV (camera and frame drawing)
- MediaPipe (hand landmarks)
- pyqtgraph (3D rendering)
- NumPy (math)
- pynput (system mouse/keyboard control)

## Setup

1. Open a terminal in this folder:

```bash
cd "OneDrive/Desktop/opencode-folder-2026-04-08"
```

2. (Recommended) create and activate a virtual environment:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m hand_tracker.main
```

Optional camera selection:

```bash
python -m hand_tracker.main --camera 1
```

Tracking modes:

```bash
# balanced default
python -m hand_tracker.main --mode balanced

# higher stability and precision
python -m hand_tracker.main --mode precision

# highest responsiveness / lowest latency
python -m hand_tracker.main --mode max
```

## Air Mouse mode

You can also run the new hand-tracked air-mouse controller (uses the same `hand_tracker` pipeline):

```bash
python -m Air_Mouse.main
```

Optional controls:

```bash
# choose camera, mode, and controlling hand
python -m Air_Mouse.main --camera 1 --mode precision --control-hand right

# disable horizontal mirroring if you prefer direct mapping
python -m Air_Mouse.main --no-mirror

# ask for GPU delegate (falls back automatically if unavailable when using auto)
python -m Air_Mouse.main --delegate gpu
```

### Air Mouse gestures

- `Index + Thumb pinch` -> left click
- `Middle + Thumb pinch` -> right click
- `Index+Thumb + Middle+Thumb pinch` -> double click
- `Fist` -> click-and-drag while moving your hand
- `Open palm (front to camera)` -> free pointer movement (no shortcut)
- Pointer mapping now focuses on a smaller center camera region (boosted mapping), so you do not need to use the entire camera FOV to reach the full screen.
- Click intent now applies micro-movement damping and a short pinch confirmation window for more reliable click registration.

Shortcut gestures (edge-triggered + cooldown protected):

- `Peace sign` -> browser back (`Alt+Left`)
- `Three fingers` -> browser forward (`Alt+Right`)
- `Thumbs up` -> play/pause media
- `German three sign (thumb + index + middle open; ring + pinky closed)` (brief hold) -> show desktop (`Win/Cmd+D`, Linux fallback uses `Ctrl+Alt+D`)

Pointer selection behavior:

- `--control-hand right` only accepts MediaPipe-labeled right hand for pointer control.
- `--control-hand left` only accepts MediaPipe-labeled left hand for pointer control.
- `--control-hand auto` prefers right, then falls back to left.

Inference delegate:

- `--delegate auto` tries GPU first, then CPU fallback.
- `--delegate gpu` requires GPU delegate support and errors if unavailable.
- `--delegate cpu` forces CPU inference.

Note: Air Mouse sends real OS mouse/keyboard events. On macOS/Linux you may need to grant accessibility/input-control permissions to your terminal or Python interpreter.

If you already installed dependencies earlier, install any new ones after pulling updates:

```bash
pip install -r requirements.txt
```

## Controls

- `Show Landmarks`: toggles landmark points in 3D panel.
- `Show Bones`: toggles bone lines in 3D panel.
- `Show Mesh`: toggles generated triangular mesh in 3D panel.
- `Camera` dropdown: switch active camera live.
- `Rescan`: refresh available camera list.
- `Mode` dropdown: switch `balanced`, `precision`, `max` live.
- Stats bar shows live hand count and detected hand sides.

## Camera reliability

- On Windows, the app now tries multiple camera backends automatically (`MSMF`, `DSHOW`, then `AUTO`).
- If frame reads fail repeatedly, it auto-recovers and reconnects to the selected camera.
- It also falls back to lower capture profiles when a webcam rejects high-resolution settings.

## 3D placement behavior

- The 3D view now uses camera-space coordinates so hand placement reflects left/right offset and depth relative to camera POV.
- Depth (`Z`) is estimated from hand scale in image space and/or world landmarks when available.
- Live hand-to-hand distance is shown in the top status badges and frame overlay when both hands are visible.

## Notes

- The mesh is generated each frame via Delaunay triangulation of hand landmarks, with a fallback topology if triangulation becomes unstable.
- For best tracking quality, use good lighting and keep one hand fully visible in frame.
- On first run, the app downloads the MediaPipe hand-landmarker model into `hand_tracker/models/hand_landmarker.task`.
