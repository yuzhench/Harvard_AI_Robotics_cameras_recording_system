# RealSense RGBD Recording Pipeline

A web-based multi-camera recording system for Intel RealSense D435I cameras. Streams live RGB and aligned depth video in the browser, supports up to 4 simultaneous cameras, and saves synchronized RGB + depth + IMU data to disk.

---

## Hardware Requirements

- Intel RealSense D435I (1–4 cameras)
- USB 3.x recommended for full 30 fps; USB 2.x works at reduced bandwidth
- Linux (Ubuntu 20.04 / 22.04 / 24.04)

---

## Installation

### 1. Install librealsense (build from source — required for IMU support)

```bash
sudo apt install -y git cmake build-essential libusb-1.0-0-dev libudev-dev \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libgtk-3-dev

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense

# USB udev rules (allows non-root camera access)
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON \
         -DPYTHON_EXECUTABLE=$(which python3) \
         -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=ON

make -j$(($(nproc)-1)) && sudo make install
sudo ldconfig
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify camera is detected

```bash
realsense-viewer
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

---

## Running the Server

```bash
./run.sh                        # real cameras, default settings
./run.sh --mock                 # mock cameras (no hardware needed)
./run.sh --mock --num-mock 4    # simulate 4 cameras
./run.sh --fps 15               # change frame rate
./run.sh --width 1280 --height 720  # change resolution
./run.sh --port 8001            # use a different port
```

Then open **http://localhost:8000** in your browser.

---

## Web Interface

### Layout

The interface is split into two panes separated by a draggable divider:

- **Left pane** — live camera feeds (2×2 grid)
- **Right pane** — recording controls, settings, intrinsics, orientation

---

### Camera Grid (Left Pane)

Up to 4 camera panels are shown at all times.

| Element | Description |
|---------|-------------|
| **LIVE** badge | Camera is connected and streaming |
| **OFFLINE** badge | Camera not detected |
| **RGB** button | Switch panel to color stream |
| **Aligned Depth** button | Switch panel to depth stream (colorized, aligned to RGB) |
| **⊙ Cloud** button | Open interactive 3D point cloud viewer for this camera |

Cameras are activated automatically when detected — no manual configuration needed.

---

### Stats Bar (Top)

Shows one chip per task. Each chip displays:
- Task name
- Number of recorded demos
- Average demo duration

Click a chip to select that task in the session setup dropdown.

---

### Right Pane — Controls

#### Clock Card
- Current time and date
- Recording status indicator (red pulsing dot when recording)
- Elapsed recording time

#### Session Setup
- **Task** — select from task1–task10
- **Text Prompt** — describe what is happening in this demo (required to start recording)

#### Camera Settings
- **FPS** — frames per second (default 30)
- **Resolution** — 640×480 / 1280×720 / 848×480 / 424×240

#### Recording Buttons
- **▶ Start** — begin recording (requires a text prompt)
- **■ Stop** — stop and save; shows "⏳ Saving…" while compressing depth data

#### Camera Intrinsics
Displays focal lengths (fx, fy), principal point (cx, cy), distortion model, depth scale, and serial number for the first active camera.

#### Orientation — Pitch / Roll
2×2 grid showing real-time pitch and roll for each camera, read from the D435I's built-in IMU. Updates at 2 Hz via WebSocket.

---

### Point Cloud Viewer

Click **⊙ Cloud** on any camera panel to open the interactive 3D viewer.

| Control | Action |
|---------|--------|
| **Drag** | Orbit (rotate) |
| **Scroll wheel** | Zoom in/out |
| **Pinch** (touch) | Zoom |
| **⟳ Capture** button | Snapshot current frame as point cloud |
| **Density slider** | Left = low density (fast), Right = high density (full res) |

The viewer shows:
- **Point cloud** — colored by the RGB frame
- **Coordinate axes** — X (red), Y (green), Z (blue) at camera origin
- **Camera sensor body** — small rectangle at the sensor face
- **Frustum wireframe** — showing the camera's field of view

The status bar at the bottom confirms the renderer (e.g. `WebGL 2.0 (OpenGL ES 3.0)`).

---

## Recording a Demo

1. Connect RealSense camera(s) and start the server with `./run.sh`
2. Open **http://localhost:8000**
3. Wait for camera panels to show **LIVE**
4. Select a **Task** from the dropdown
5. Type a **Text Prompt** describing the demo
6. (Optional) Adjust FPS and resolution
7. Click **▶ Start**
8. Perform the demo
9. Click **■ Stop** — data is saved automatically

---

## Output Data Structure

```
data/
└── task1/
    └── 04_07_2026/
        └── 14_32_05/
            ├── prompt.txt          # text description of the demo
            ├── cam1/
            │   ├── rgb.mp4         # color video (H.264)
            │   ├── depth.npz       # aligned depth frames + timestamps
            │   └── intrinsics.json # camera calibration
            └── cam2/
                ├── rgb.mp4
                ├── depth.npz
                └── intrinsics.json
```

### depth.npz format

```python
import numpy as np
d = np.load("depth.npz")
depth      = d["depth"]       # shape (N, H, W), uint16, units = depth_scale metres
timestamps = d["timestamps"]  # shape (N,), float64, Unix timestamps
```

Depth is **aligned to the color frame** — each pixel in `depth` corresponds to the same pixel in the RGB frame.

### intrinsics.json format

```json
{
  "color": { "fx": 615.0, "fy": 615.0, "ppx": 320.0, "ppy": 240.0, ... },
  "depth": { "fx": 615.0, "fy": 615.0, "ppx": 320.0, "ppy": 240.0, "depth_scale": 0.001 },
  "serial": "344422072270",
  "name": "Intel RealSense D435I"
}
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--fps` | 30 | Frame rate |
| `--width` | 640 | Frame width |
| `--height` | 480 | Frame height |
| `--port` | 8000 | Server port |
| `--mock` | off | Use synthetic cameras (no hardware) |
| `--num-mock` | 2 | Number of mock cameras |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard |
| `GET` | `/status` | Camera state, FPS, resolution, active cameras |
| `GET` | `/tasks` | List of available task names |
| `GET` | `/stats` | Per-task demo counts and durations |
| `POST` | `/start` | Start recording `{"task": "task1", "prompt": "..."}` |
| `POST` | `/stop` | Stop recording and save |
| `GET` | `/pointcloud?cam=1&stride=2` | Current point cloud as binary blob |
| `GET` | `/orientation` | Pitch/roll for all cameras |
| `WS` | `/ws/stream?cam=1&mode=color` | Live JPEG stream |
| `WS` | `/ws/orientation` | Real-time IMU orientation at 2 Hz |
