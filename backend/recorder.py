"""
recorder.py — per-camera recording (MP4 + NPZ).

Each Recorder handles one camera writing to an explicit save_dir.
The server is responsible for creating session dirs and coordinating
multiple Recorder instances (one per camera).
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np


class Recorder:
    """Saves RGB (MP4) + Depth (NPZ) + intrinsics for a single camera."""

    def __init__(self):
        self._lock = threading.Lock()
        self._recording = False
        self._video_writer: cv2.VideoWriter | None = None
        self._depth_frames: list[np.ndarray] = []
        self._depth_timestamps: list[float] = []
        self._start_time: float = 0.0
        self._frame_count: int = 0

    def start(self, save_dir: Path, intrinsics: dict, width: int, height: int, fps: int):
        """
        Start recording into save_dir.
        Saves intrinsics.json immediately; rgb.mp4 and depth.npz written on stop().
        """
        import json
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording")

            save_dir.mkdir(parents=True, exist_ok=True)

            with open(save_dir / "intrinsics.json", "w") as f:
                json.dump(intrinsics, f, indent=2)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(save_dir / "rgb.mp4"), fourcc, fps, (width, height)
            )
            self._depth_frames = []
            self._depth_timestamps = []
            self._start_time = time.time()
            self._frame_count = 0
            self._recording = True

    def write_frame(self, color_bgr: np.ndarray, depth_mm: np.ndarray, timestamp: float):
        with self._lock:
            if not self._recording:
                return
            self._video_writer.write(color_bgr)
            self._depth_frames.append(depth_mm.copy())
            self._depth_timestamps.append(timestamp)
            self._frame_count += 1

    def stop(self) -> dict:
        """Stop recording, flush files, return info dict."""
        with self._lock:
            if not self._recording:
                raise RuntimeError("Not recording")

            self._recording = False
            self._video_writer.release()
            self._video_writer = None

            elapsed = round(time.time() - self._start_time, 2)
            info = {
                "frame_count": self._frame_count,
                "elapsed_seconds": elapsed,
                "actual_fps": round(self._frame_count / elapsed, 2) if elapsed > 0 else 0,
            }

            # Return depth data for caller to save (avoids storing save_dir in self)
            depth_out = (
                np.stack(self._depth_frames, axis=0),
                np.array(self._depth_timestamps),
            ) if self._depth_frames else (None, None)

            self._depth_frames = []
            self._depth_timestamps = []

            return info, depth_out

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def elapsed(self) -> float:
        if not self._recording:
            return 0.0
        return time.time() - self._start_time

    @property
    def frame_count(self) -> int:
        return self._frame_count
