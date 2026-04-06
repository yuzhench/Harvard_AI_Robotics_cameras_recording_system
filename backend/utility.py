"""
utility.py — shared helper functions for the recording pipeline.
"""

import json
import math
from pathlib import Path

import numpy as np

from .config import DATA_ROOT, TASKS

# ------------------------------------------------------------------ #
# Camera enumeration
# ------------------------------------------------------------------ #

def enumerate_cameras() -> list[dict]:
    """
    Return info for all connected RealSense devices.

    Each entry:
        {
            "name":   "Intel RealSense D435I",
            "serial": "344422072270",
            "usb":    "3.2",
        }
    """
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        result = []
        for d in devices:
            result.append({
                "name":   d.get_info(rs.camera_info.name),
                "serial": d.get_info(rs.camera_info.serial_number),
                "usb":    d.get_info(rs.camera_info.usb_type_descriptor),
            })
        return result
    except Exception as e:
        return [{"error": str(e)}]


def print_cameras():
    """Print connected cameras to stdout. Useful as a CLI check."""
    devices = enumerate_cameras()
    print(f"Found {len(devices)} device(s)")
    for d in devices:
        if "error" in d:
            print(f"  ERROR: {d['error']}")
        else:
            print(f"  - {d['name']} | Serial: {d['serial']} | USB: {d['usb']}")


# ------------------------------------------------------------------ #
# IMU / Orientation helpers
# ------------------------------------------------------------------ #

def compute_pitch_roll(ax: float, ay: float, az: float) -> tuple[float, float]:
    """
    Compute pitch and roll from accelerometer readings (static / slow-moving only).

    Assumes the accelerometer measures gravity when stationary.
    RealSense D435I coordinate frame:
        X → right, Y → down, Z → forward (into the scene)

    Returns:
        pitch_deg : rotation around X axis (camera tilting up/down)
        roll_deg  : rotation around Z axis (camera tilting left/right)

    NOTE: Yaw (rotation around Y / vertical axis) cannot be computed from
    accelerometer alone — gravity has no horizontal component to detect it.
    """
    pitch_rad = math.atan2(ay, math.sqrt(ax**2 + az**2))
    roll_rad  = math.atan2(-ax, az)
    return math.degrees(pitch_rad), math.degrees(roll_rad)


def get_orientation(cam) -> dict:
    """
    Return orientation for a camera.
    Always returns a dict (camera is assumed online).
    'available' is False when IMU data was never successfully read
    (e.g. USB 2.x hardware limitation).
    """
    imu = cam.get_imu()
    if imu is None or imu.get("accel") is None:
        return {"available": False, "reason": "IMU unavailable (USB 2.x)"}

    ax, ay, az = imu["accel"]
    pitch, roll = compute_pitch_roll(ax, ay, az)

    return {
        "available": True,
        "accel":  imu["accel"],
        "gyro":   imu["gyro"],
        "pitch":  round(pitch, 2),
        "roll":   round(roll, 2),
        "yaw":    None,
    }


def get_all_orientations(camera_manager) -> dict[int, dict | None]:
    """
    Read orientation for all cameras simultaneously.
    Each camera's IMU is read independently (thread-safe via camera._lock).

    Args:
        camera_manager: CameraManager instance

    Returns:
        {cam_idx: orientation_dict_or_None, ...}

    Example:
        {
            1: {"pitch": 3.2, "roll": -1.1, "accel": (...), "gyro": (...), "yaw": None},
            2: {"pitch": 0.5, "roll":  0.3, ...},
        }
    """
    return {
        idx: get_orientation(cam)
        for idx, cam in camera_manager.cameras.items()
    }


def print_orientations(camera_manager):
    """Print pitch/roll for all connected cameras. Useful as a CLI check."""
    results = get_all_orientations(camera_manager)
    for idx, data in results.items():
        if data is None:
            print(f"  cam{idx}: no IMU data yet")
        else:
            print(f"  cam{idx}: pitch={data['pitch']:+.1f}°  roll={data['roll']:+.1f}°  "
                  f"accel=({data['accel'][0]:.2f}, {data['accel'][1]:.2f}, {data['accel'][2]:.2f}) m/s²")


# ------------------------------------------------------------------ #
# Stats helpers
# ------------------------------------------------------------------ #

STATS_FILE = Path(DATA_ROOT) / "stats.json"


def load_stats() -> dict:
    """Load stats.json; return empty structure if missing or corrupt."""
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text())
        except Exception:
            pass
    return {t: {"count": 0, "total_duration_s": 0.0, "demos": []} for t in TASKS}


def save_stats(stats: dict):
    """Write stats dict to stats.json."""
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(stats, indent=2))


def rebuild_stats_from_disk() -> dict:
    """
    Scan data/ directory and rebuild stats.json from what's on disk.
    Useful after manual file moves, crashes, or first-run sync.
    """
    stats = {t: {"count": 0, "total_duration_s": 0.0, "demos": []} for t in TASKS}
    data_root = Path(DATA_ROOT)

    for task in TASKS:
        task_dir = data_root / task
        if not task_dir.exists():
            continue
        for day_dir in sorted(task_dir.iterdir()):
            if not day_dir.is_dir():
                continue
            for session_dir in sorted(day_dir.iterdir()):
                if not session_dir.is_dir():
                    continue

                prompt = ""
                prompt_f = session_dir / "prompt.txt"
                if prompt_f.exists():
                    prompt = prompt_f.read_text().strip()

                duration_s = 0.0
                depth_f = session_dir / "depth.npz"
                if depth_f.exists():
                    try:
                        d = np.load(str(depth_f), allow_pickle=False)
                        ts = d["timestamps"]
                        if len(ts) > 1:
                            duration_s = round(float(ts[-1] - ts[0]), 2)
                    except Exception:
                        pass

                stats[task]["count"] += 1
                stats[task]["total_duration_s"] = round(
                    stats[task]["total_duration_s"] + duration_s, 2
                )
                stats[task]["demos"].append({
                    "dir": str(session_dir),
                    "date": day_dir.name,
                    "time": session_dir.name,
                    "duration_s": duration_s,
                    "prompt": prompt,
                })

    save_stats(stats)
    return stats


# ------------------------------------------------------------------ #
# Point cloud helper
# ------------------------------------------------------------------ #

def make_pointcloud(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    intrinsics: dict,
    stride: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project aligned color + depth frames into a 3-D point cloud.

    The depth frame must already be aligned to the color frame (as produced
    by RealSenseCamera._capture_loop via rs.align).

    Args:
        color_bgr : (H, W, 3) uint8 BGR image
        depth     : (H, W)    uint16 depth image (sensor units)
        intrinsics: camera intrinsics dict with 'color' and 'depth' sub-dicts
        stride    : sub-sampling stride (3 → ~1/9 of pixels)

    Returns:
        xyz : (N, 3) float32  XYZ in metres  (X right, Y down, Z forward)
        rgb : (N, 3) uint8    RGB colours matching each point
    """
    ci = intrinsics["color"]
    depth_scale: float = intrinsics["depth"]["depth_scale"]
    fx: float = ci["fx"]
    fy: float = ci["fy"]
    cx: float = ci["ppx"]
    cy: float = ci["ppy"]

    H, W = depth.shape
    rows, cols = np.mgrid[0:H:stride, 0:W:stride]

    z = depth[rows, cols].astype(np.float32) * depth_scale  # metres

    # Keep only points with valid depth (0.1 m – 10 m)
    valid = (z > 0.1) & (z < 10.0)
    c = cols[valid].astype(np.float32)
    r = rows[valid].astype(np.float32)
    z = z[valid]

    x = (c - cx) * z / fx
    y = (r - cy) * z / fy

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)          # (N, 3)
    rgb = color_bgr[rows[valid], cols[valid]][:, ::-1].copy()      # BGR → RGB

    return xyz, rgb.astype(np.uint8)


def print_stats():
    """Print per-task demo counts and durations to stdout."""
    stats = load_stats()
    print(f"{'Task':<25} {'Demos':>6} {'Total (s)':>10} {'Avg (s)':>8}")
    print("-" * 55)
    for task, data in stats.items():
        count = data["count"]
        total = data["total_duration_s"]
        avg = round(total / count, 1) if count > 0 else 0.0
        print(f"{task:<25} {count:>6} {total:>10.1f} {avg:>8.1f}")
