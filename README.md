# Camera Record Pipeline

Multi-camera recording server for **Intel RealSense D435I** — streams
live RGB and aligned depth in the browser, records up to 4 cameras
simultaneously, and optionally coordinates with an on-robot Jetson
running [go2_record_pipeline](https://github.com/yuzhench/Harvard_AI_Robotics_go2_recording_system)
for fully synchronized multi-modal capture.

```
┌────────────────────── Laptop ──────────────────────┐      ┌────── Jetson (optional) ──────┐
│  Browser UI (index.html)                           │      │  go2_record_pipeline          │
│      │                                             │      │  :8010                        │
│      │ HTTP + WS                                   │      │                               │
│      ▼                                             │      │  ├── IMU / joints / contacts  │
│  FastAPI backend (this repo)  :8000                │◀───▶ │  ├── Ego RGB-D D435I          │
│      │                                             │ HTTP │  └── LiDAR                    │
│      └── USB3 ──▶ 4× RealSense D435I (third-person)│      │                               │
└────────────────────────────────────────────────────┘      └───────────────────────────────┘
                         ↕ chrony NTP
                   (< 5 ms clock alignment)
```

If no Jetson is configured (`JETSON_URL` unset), this runs stand-alone
as a 4-camera recorder. When pointed at a Jetson, the Start/Stop buttons
fan-out to both machines and data lands in matching session directories
for post-hoc merging.

---

## Quick start

```bash
# 1. Build librealsense (first time only — see Installation below)

# 2. Install Python deps
pip install -r requirements.txt

# 3. Run
./run.sh                          # real cameras
./run.sh --mock                   # synthetic cameras, no hardware
./run.sh --mock --num-mock 4      # simulate 4 cameras

# 4. Open http://localhost:8000 in your browser
```

---

## What this does

- **Live preview** — 4-panel grid of RGB or aligned-depth streams over WebSocket
- **Recording** — synchronized session capture of RGB video (H.264), aligned depth (uint16 NPZ), and per-frame timestamps
- **3D point cloud viewer** — WebGL, interactive, per-camera snapshot
- **Task/prompt workflow** — 10 pre-configured tasks, required text prompt per demo, per-task stats bar
- **Jetson integration** — forwards `/start` / `/stop` / `/resync_clock` to a separate on-robot daemon, with warning vs. error semantics when the robot side is slow or unreachable
- **Event log** — collapsible footer panel that persists every toast message (successes, warnings, errors) so nothing disappears after 3 s

---

## Using with Go2 (optional Jetson integration)

Set `JETSON_URL` before launching:

```bash
JETSON_URL=http://10.100.206.170:8010 ./run.sh
```

The UI gains a **Robot** row (state/FPS/sample counts) and three extra
controls: a `⟳ Resync` button for clock sync, a `↓ Sync` button for
rsync-pulling the Jetson's data to this laptop, and a Force-Stop button
for reconciling diverged states.

Protocol version must match on all three sides (frontend, this backend,
Jetson daemon). Currently `PROTOCOL_VERSION = 4`.

See **[go2_record_pipeline](https://github.com/yuzhench/Harvard_AI_Robotics_go2_recording_system)**
for the robot-side deployment.

---

## Session output

```
<DATA_ROOT>/<task>/<MM_DD_YYYY>/<HH_MM_SS>/
├── third_person/          ← this repo (4× RealSense D435I)
│   ├── cam0/
│   │   ├── rgb.mp4               H.264 video
│   │   ├── rgb_timestamps.npy    (N,) float64 Unix epoch
│   │   ├── depth.npz             {depth: (N,H,W) uint16, timestamps: (N,) float64}
│   │   └── intrinsics.json
│   ├── cam1/  ...
│   ├── cam2/  ...
│   ├── cam3/  ...
│   └── session_meta.json
└── first_person/          ← written by go2_record_pipeline (if used)
    └── ...
```

`DATA_ROOT` defaults to `~/GO2_DATA`; override with `DATA_ROOT=/path/to/dir ./run.sh`.

---

<details>
<summary><b>Installation (librealsense from source)</b></summary>

On x86_64 Linux `pip install pyrealsense2` is usually sufficient (Intel
ships wheels on PyPI). For IMU support or aarch64 (Jetson) you must
build from source:

```bash
sudo apt install -y git cmake build-essential libusb-1.0-0-dev libudev-dev \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libgtk-3-dev python3-dev

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

mkdir build && cd build
cmake .. -DBUILD_PYTHON_BINDINGS=ON \
         -DPYTHON_EXECUTABLE=$(which python3) \
         -DCMAKE_BUILD_TYPE=Release
make -j$(($(nproc)-1)) && sudo make install
sudo ldconfig

# Copy the Python binding into your active conda/venv
cp wrappers/python/pyrealsense2*.so $CONDA_PREFIX/lib/python*/site-packages/
```

Verify:
```bash
realsense-viewer
python3 -c "import pyrealsense2 as rs; print(rs.__version__)"
```

</details>

<details>
<summary><b>Web UI guide</b></summary>

### Layout

Two-pane split with a draggable divider.

**Left pane** — 2×2 camera grid. Each panel shows a `LIVE` / `OFFLINE`
badge, a RGB/Aligned-Depth toggle, and a `⊙ Cloud` button that opens
the 3D point cloud viewer.

**Right pane** — collapsible cards for state, session setup, camera
settings, intrinsics, orientation (pitch/roll from D435I IMU),
optional robot row (when `JETSON_URL` is set), and Sync/Resync controls.

### Event log

Persistent collapsible footer. Every toast (success / warning / error /
info) is also logged here with a timestamp, colored by type. Not
persisted across reloads — by design, to avoid stale context.

### Point cloud viewer

Pure WebGL 2, no external libraries. Drag to orbit, scroll to zoom.
Density slider controls back-projection stride (lower = denser = slower).
`⟳ Capture` snapshots the current live frame.

</details>

<details>
<summary><b>CLI options</b></summary>

| Flag | Default | Description |
|---|---|---|
| `--fps` | 30 | Frame rate |
| `--width` | 640 | Frame width |
| `--height` | 480 | Frame height |
| `--port` | 8000 | Server port |
| `--mock` | off | Synthetic cameras, no hardware |
| `--num-mock` | 2 | Number of mock cameras |

Environment variables:

| Var | Default | Description |
|---|---|---|
| `DATA_ROOT` | `~/GO2_DATA` | Session output root |
| `JETSON_URL` | `http://10.100.206.170:8010` | Jetson daemon (empty disables forwarding) |
| `JETSON_TIMEOUT_S` | `5.0` | Default HTTP timeout to Jetson |
| `JETSON_STOP_TIMEOUT_S` | `60.0` | Longer timeout for `/stop` (save can be slow) |
| `JETSON_RSYNC_BWLIMIT_KBPS` | `5000` | Rate-limit rsync to avoid saturating WiFi |
| `JETSON_SSH_USER` | `unitree` | SSH user for rsync from Jetson |
| `JETSON_DATA_ROOT` | `/home/unitree/GO2_DATA` | Remote path for rsync |

</details>

<details>
<summary><b>HTTP API</b></summary>

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web dashboard |
| `GET` | `/status` | Camera state, FPS, resolution, active cameras |
| `GET` | `/tasks` | Available task names |
| `GET` | `/stats` | Per-task demo counts and durations |
| `POST` | `/start` | Start recording. Body: `{"task":..., "prompt":...}` |
| `POST` | `/stop` | Stop and save. Returns per-camera + robot results. |
| `POST` | `/jetson/stop` | Force-stop only the Jetson (for state reconciliation) |
| `POST` | `/jetson/resync_clock` | Trigger chrony restart on Jetson, return offset |
| `GET` | `/jetson/sync_status` | Check how much data Jetson has pending for rsync |
| `POST` | `/jetson/sync` | Run rsync pull from Jetson to laptop |
| `GET` | `/pointcloud?cam=N&stride=S` | Current point cloud as binary blob |
| `GET` | `/orientation` | Pitch/roll for all cameras |
| `WS` | `/ws/stream?cam=N&mode=color` | Live JPEG stream |
| `WS` | `/ws/orientation` | Real-time IMU orientation at 2 Hz |

</details>

<details>
<summary><b>Time synchronization with the Jetson</b></summary>

Both machines rely on a shared `time.time()` (Unix UTC epoch) for
timestamping. `chrony` on each host keeps clocks aligned to < 5 ms
(typically < 100 μs on LAN).

The UI **⟳ Resync** button triggers a `systemctl restart chrony` on the
Jetson via the daemon, then waits for convergence and reports the
resulting offset. Intended to be pressed once per recording session as
a belt-and-braces check.

Full setup and rationale lives in the robot repo:
[go2_record_pipeline/GO_NOTES/control_architecture.md §19](https://github.com/yuzhench/Harvard_AI_Robotics_go2_recording_system/blob/main/GO_NOTES/control_architecture.md).

</details>

<details>
<summary><b>Troubleshooting</b></summary>

**Cameras show OFFLINE**
- `realsense-viewer` sees them? If not, udev/permissions. Re-run the `setup_udev_rules.sh` step.
- USB 2.0 only? Force to USB 3 port (blue) — D435I at 30 fps / 640×480 needs USB 3.

**Stop returns a yellow warning** ("robot save status unknown")
- Jetson is still flushing to disk; not an error. Wait 30 s and the session directory on Jetson will have all files.

**Sync button fails with "connection closed" or "broken pipe"**
- WiFi saturation killing SSH. Lower `JETSON_RSYNC_BWLIMIT_KBPS` to `2000` and retry.

**Timestamps misaligned between `first_person/` and `third_person/`**
- Run the ⟳ Resync button before the next recording. Check `chronyc tracking` on the Jetson — `System time` should be < 0.005 s.

**Version mismatch red banner**
- Protocol version between frontend, this backend, and the Jetson daemon must match (`PROTOCOL_VERSION` in each `config.py`). After upgrading, restart the Jetson daemon and hard-refresh the browser.

</details>

---

## Requirements

- Linux (Ubuntu 20.04 / 22.04 / 24.04)
- Python 3.10+
- Intel RealSense D435I × 1–4, on USB 3.x ports
- Optional: Unitree Go2 EDU on the LAN with [go2_record_pipeline](https://github.com/yuzhench/Harvard_AI_Robotics_go2_recording_system) running on its Jetson

---

## Related

- [go2_record_pipeline](https://github.com/yuzhench/Harvard_AI_Robotics_go2_recording_system) — first-person (on-robot) data collection, to be paired with this repo for full synchronized capture
