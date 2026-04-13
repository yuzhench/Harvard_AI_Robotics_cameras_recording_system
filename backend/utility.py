"""
utility.py — shared helper functions for the recording pipeline.
"""

import json
import math
from pathlib import Path

import cv2
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


# ------------------------------------------------------------------ #
# Video trimming helpers
# ------------------------------------------------------------------ #

def _compute_motion_scores(video_path: Path) -> tuple[np.ndarray, float, int, int]:
    """
    Read all frames and compute per-frame motion score.

    Score = mean absolute pixel difference between consecutive frames
    in grayscale, in range [0, 255]. Frame 0 always gets score 0.0.

    Returns:
        scores  : (N,) float32 array, one value per frame
        fps     : float
        width   : int
        height  : int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scores: list[float] = []
    prev_gray = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            scores.append(0.0)
        else:
            diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            scores.append(diff)
        prev_gray = gray

    cap.release()
    return np.array(scores, dtype=np.float32), fps, width, height


def _find_trim_points(
    scores: np.ndarray,
    motion_threshold: float = 2.0,
    min_still_frames: int = 15,
) -> tuple[int, int]:
    """
    Find the first and last frames of sustained motion.

    A rolling average (window = min_still_frames) is applied before
    thresholding so that brief noise spikes don't trigger false motion
    and short robot pauses in the middle don't look like the video end.

    Strategy:
        - trim_start: scan *forward*  — find the first smoothed-moving frame
        - trim_end:   scan *backward* — find the last  smoothed-moving frame

    This guarantees that any pause in the middle of the demo is kept,
    because we never scan inward from both ends simultaneously.

    Returns:
        (start_frame, end_frame)  both inclusive.
        Falls back to (0, N-1) if no motion is detected at all.
    """
    n = len(scores)
    if n == 0:
        return 0, 0

    # Smooth scores to suppress single-frame noise
    window = max(1, min(min_still_frames, n))
    smooth = np.convolve(scores, np.ones(window) / window, mode='same')
    is_moving = smooth > motion_threshold

    # Scan forward for the start of motion
    start_frame = 0
    for i in range(n):
        if is_moving[i]:
            start_frame = i
            break

    # Scan backward for the end of motion
    end_frame = n - 1
    for i in range(n - 1, -1, -1):
        if is_moving[i]:
            end_frame = i
            break

    # If the detected window is degenerate, keep everything
    if start_frame >= end_frame:
        return 0, n - 1

    return start_frame, end_frame


def _write_trimmed_video(
    src_path: Path,
    dst_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    width: int,
    height: int,
):
    """Write frames [start_frame .. end_frame] inclusive from src to dst."""
    cap = cv2.VideoCapture(str(src_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(dst_path), fourcc, fps, (width, height))

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if start_frame <= idx <= end_frame:
            out.write(frame)
        elif idx > end_frame:
            break
        idx += 1

    cap.release()
    out.release()


def trim_recording(
    cam_dir: Path,
    motion_threshold: float = 2.0,
    min_still_frames: int = 15,
) -> dict:
    """
    Trim the still lead-in and lead-out from one camera's recording.

    Input layout (cam_dir/):
        rgb.mp4       — color video recorded by Recorder
        depth.npz     — aligned depth frames + timestamps  (optional)

    Output (written into the same cam_dir/):
        rgb_raw.mp4   — original video renamed (never modified)
        rgb.mp4       — trimmed color video
        depth_raw.npz — original depth renamed (if depth.npz existed)
        depth.npz     — trimmed depth array matching the same frame window
        trim_info.json — JSON summary of what was cut

    Motion detection is done on the color video only; the same
    frame indices are applied to depth.

    Args:
        cam_dir          : path to a camN/ directory
        motion_threshold : mean grayscale pixel-diff (0–255) above which
                           a frame is considered "moving" (default 2.0)
        min_still_frames : smoothing window size — also the minimum number
                           of consecutive still frames that constitute a
                           "still" period (default 15)

    Returns:
        trim_info dict (same content as trim_info.json)
    """
    cam_dir    = Path(cam_dir)
    video_path = cam_dir / "rgb.mp4"
    depth_path = cam_dir / "depth.npz"

    if not video_path.exists():
        raise FileNotFoundError(f"rgb.mp4 not found in {cam_dir}")

    # 1. Compute per-frame motion scores from the color video
    scores, fps, width, height = _compute_motion_scores(video_path)
    total_frames = len(scores)

    # 2. Detect trim points
    start_frame, end_frame = _find_trim_points(scores, motion_threshold, min_still_frames)
    kept_frames = end_frame - start_frame + 1

    # 3. Rename originals before writing anything new
    raw_video_path = cam_dir / "rgb_raw.mp4"
    video_path.rename(raw_video_path)

    raw_depth_path = None
    if depth_path.exists():
        raw_depth_path = cam_dir / "depth_raw.npz"
        depth_path.rename(raw_depth_path)

    # 4. Write trimmed color video
    _write_trimmed_video(
        raw_video_path, cam_dir / "rgb.mp4",
        start_frame, end_frame, fps, width, height,
    )

    # 5. Trim depth array to the same frame window
    if raw_depth_path is not None:
        d          = np.load(str(raw_depth_path), allow_pickle=False)
        depth_arr  = d["depth"]       # (N, H, W) uint16
        timestamps = d["timestamps"]  # (N,) float64

        # Clamp indices to actual depth frame count (video and depth may
        # have a one-frame difference due to pipeline startup timing)
        s = min(start_frame, len(depth_arr) - 1)
        e = min(end_frame,   len(depth_arr) - 1)

        np.savez_compressed(
            str(cam_dir / "depth.npz"),
            depth=depth_arr[s : e + 1],
            timestamps=timestamps[s : e + 1],
        )

    # 6. Save trim summary
    trim_info = {
        "raw_video":              "rgb_raw.mp4",
        "trimmed_video":          "rgb.mp4",
        "raw_depth":              "depth_raw.npz" if raw_depth_path else None,
        "trimmed_depth":          "depth.npz"     if raw_depth_path else None,
        "total_frames":           total_frames,
        "fps":                    round(fps, 3),
        "trim_start_frame":       start_frame,
        "trim_end_frame":         end_frame,
        "kept_frames":            kept_frames,
        "trim_start_seconds":     round(start_frame / fps, 3),
        "trim_end_seconds":       round(end_frame   / fps, 3),
        "duration_raw_seconds":   round(total_frames / fps, 3),
        "duration_trimmed_seconds": round(kept_frames / fps, 3),
        "motion_threshold":       motion_threshold,
        "min_still_frames":       min_still_frames,
        "motion_scores_mean":     round(float(np.mean(scores)), 3),
        "motion_scores_max":      round(float(np.max(scores)),  3),
    }
    (cam_dir / "trim_info.json").write_text(json.dumps(trim_info, indent=2))
    return trim_info


def trim_session(
    session_dir: Path,
    motion_threshold: float = 2.0,
    min_still_frames: int = 15,
) -> dict[str, dict]:
    """
    Trim all cameras in a session directory.

    Expects the standard session layout:
        session_dir/
            prompt.txt
            cam1/  rgb.mp4  depth.npz
            cam2/  rgb.mp4  depth.npz
            ...

    Args:
        session_dir      : path containing camN/ sub-directories
        motion_threshold : passed through to trim_recording()
        min_still_frames : passed through to trim_recording()

    Returns:
        {cam_name: trim_info_dict, ...}
        If a camera fails, its entry is {"error": "<message>"}.
    """
    session_dir = Path(session_dir)
    cam_dirs = sorted(
        d for d in session_dir.iterdir()
        if d.is_dir() and d.name.startswith("cam")
    )
    if not cam_dirs:
        raise FileNotFoundError(f"No camN/ directories found in {session_dir}")

    results: dict[str, dict] = {}
    for cam_dir in cam_dirs:
        print(f"  [{cam_dir.name}] analysing motion...")
        try:
            info = trim_recording(cam_dir, motion_threshold, min_still_frames)
            results[cam_dir.name] = info
            print(
                f"  [{cam_dir.name}] kept {info['kept_frames']}/{info['total_frames']} frames"
                f"  ({info['trim_start_seconds']:.2f}s – {info['trim_end_seconds']:.2f}s)"
            )
        except Exception as exc:
            results[cam_dir.name] = {"error": str(exc)}
            print(f"  [{cam_dir.name}] ERROR: {exc}")

    return results
