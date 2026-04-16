import threading
import time
import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

import cv2


class RealSenseCamera:
    """Wraps a single Intel RealSense camera for color + depth streaming."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, serial: str = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial

        self._pipeline = None
        self._config = None
        self._profile = None
        self._running = False
        self._lock = threading.Lock()

        self.color_frame: np.ndarray | None = None
        self.depth_frame: np.ndarray | None = None
        self.timestamp: float = 0.0
        self.intrinsics: dict = {}

        # Latest IMU readings (updated independently of video frames)
        self.accel: tuple[float, float, float] | None = None  # (ax, ay, az) m/s²
        self.gyro:  tuple[float, float, float] | None = None  # (gx, gy, gz) rad/s

    def start(self):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not installed")

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._imu_in_pipeline = False

        if self.serial:
            self._config.enable_device(self.serial)

        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
        self._config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)

        try:
            self._profile = self._pipeline.start(self._config)
            self._imu_in_pipeline = True
        except Exception:
            # Fall back to video-only if IMU streams can't be combined
            self._pipeline = rs.pipeline()
            self._config = rs.config()
            if self.serial:
                self._config.enable_device(self.serial)
            self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self._profile = self._pipeline.start(self._config)

        self._running = True

        color_stream = self._profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = self._profile.get_stream(rs.stream.depth).as_video_stream_profile()
        ci = color_stream.get_intrinsics()
        di = depth_stream.get_intrinsics()
        depth_scale = self._profile.get_device().first_depth_sensor().get_depth_scale()

        self.intrinsics = {
            "color": {
                "width": ci.width, "height": ci.height,
                "fx": ci.fx, "fy": ci.fy,
                "ppx": ci.ppx, "ppy": ci.ppy,
                "distortion_model": str(ci.model),
                "coeffs": list(ci.coeffs),
            },
            "depth": {
                "width": di.width, "height": di.height,
                "fx": di.fx, "fy": di.fy,
                "ppx": di.ppx, "ppy": di.ppy,
                "distortion_model": str(di.model),
                "coeffs": list(di.coeffs),
                "depth_scale": depth_scale,
            },
            "serial": self.get_serial(),
            "name": self.get_name(),
        }

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        align = rs.align(rs.stream.color)
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)

                # Extract IMU data whenever it arrives in the frameset
                if self._imu_in_pipeline:
                    accel_f = frames.first_or_default(rs.stream.accel)
                    gyro_f  = frames.first_or_default(rs.stream.gyro)
                    if accel_f and accel_f.is_motion_frame():
                        a = accel_f.as_motion_frame().get_motion_data()
                        with self._lock:
                            self.accel = (a.x, a.y, a.z)
                    if gyro_f and gyro_f.is_motion_frame():
                        g = gyro_f.as_motion_frame().get_motion_data()
                        with self._lock:
                            self.gyro = (g.x, g.y, g.z)

                # Skip framesets that have no video (motion-only)
                if not frames.get_color_frame() or not frames.get_depth_frame():
                    continue

                aligned = align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue
                with self._lock:
                    self.color_frame = np.asanyarray(color.get_data())
                    self.depth_frame = np.asanyarray(depth.get_data())
                    self.timestamp = time.time()
            except Exception:
                pass

    def get_frames(self):
        with self._lock:
            if self.color_frame is None:
                return None, None, 0
            return self.color_frame.copy(), self.depth_frame.copy(), self.timestamp

    def get_imu(self) -> dict | None:
        """
        Returns last cached IMU reading (set by read_imu_once).
            {"accel": (ax, ay, az), "gyro": (gx, gy, gz)}
        or None if read_imu_once has never been called.
        """
        with self._lock:
            if self.accel is None:
                return None
            return {"accel": self.accel, "gyro": self.gyro}

    def read_imu_once(self) -> dict | None:
        """
        Return the latest cached IMU reading.
        If IMU is running in the main pipeline, data is already being updated
        continuously — no pipeline restart needed.
        Falls back to a stop/restart approach if IMU is not in the main pipeline.
        """
        if getattr(self, '_imu_in_pipeline', False):
            # IMU is live in the capture loop — just return the latest cached value.
            # Wait briefly on first call so at least one frame has arrived.
            for _ in range(10):
                with self._lock:
                    if self.accel is not None:
                        return {"accel": self.accel, "gyro": self.gyro}
                time.sleep(0.1)
            return None

        # --- fallback: stop video, read IMU-only pipeline, restart video ---
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join(timeout=3.0)
        try:
            self._pipeline.stop()
        except Exception:
            pass

        imu_data = None
        try:
            imu_pipeline = rs.pipeline()
            imu_config = rs.config()
            if self.serial:
                imu_config.enable_device(self.serial)
            imu_config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)
            imu_config.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f, 200)
            imu_pipeline.start(imu_config)
            for _ in range(30):
                frames = imu_pipeline.wait_for_frames(timeout_ms=500)
                accel_f = frames.first_or_default(rs.stream.accel)
                gyro_f  = frames.first_or_default(rs.stream.gyro)
                if accel_f and gyro_f:
                    a = accel_f.as_motion_frame().get_motion_data()
                    g = gyro_f.as_motion_frame().get_motion_data()
                    imu_data = {"accel": (a.x, a.y, a.z), "gyro": (g.x, g.y, g.z)}
            imu_pipeline.stop()
        except Exception as e:
            print(f"[Camera {self.serial}] read_imu_once failed: {e}")

        if imu_data:
            with self._lock:
                self.accel = imu_data["accel"]
                self.gyro  = imu_data["gyro"]

        self._pipeline = rs.pipeline()
        self._config   = rs.config()
        if self.serial:
            self._config.enable_device(self.serial)
        self._config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._profile  = self._pipeline.start(self._config)
        self._running  = True
        self._thread   = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return imu_data

    def get_jpeg(self) -> bytes:
        color, _, _ = self.get_frames()
        if color is None:
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No signal", (20, self.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buf = cv2.imencode(".jpg", placeholder)
            return bytes(buf)
        _, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return bytes(buf)

    def get_depth_jpeg(self) -> bytes:
        _, depth, _ = self.get_frames()
        if depth is None:
            placeholder = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            _, buf = cv2.imencode(".jpg", placeholder)
            return bytes(buf)
        depth_clipped = np.clip(depth, 0, 4000).astype(np.float32) / 4000.0
        depth_u8 = (depth_clipped * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        _, buf = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return bytes(buf)

    def get_serial(self) -> str:
        if self._profile:
            return self._profile.get_device().get_info(rs.camera_info.serial_number)
        return "unknown"

    def get_name(self) -> str:
        if self._profile:
            return self._profile.get_device().get_info(rs.camera_info.name)
        return "unknown"

    def stop(self):
        self._running = False
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Mock camera
# ---------------------------------------------------------------------------

class MockCamera:
    """Synthetic color + depth frames for UI testing."""

    # Different hues per camera index so they look distinct in the UI
    _HUE_MAP = {1: 0, 2: 60, 3: 120, 4: 200}

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, cam_id: int = 1):
        self.width = width
        self.height = height
        self.fps = fps
        self.cam_id = cam_id
        self.serial = f"MOCK-{cam_id:04d}"
        self._frame_idx = 0
        self._running = False
        self._lock = threading.Lock()
        self.color_frame = None
        self.depth_frame = None
        self.accel = (0.0, -9.81, 0.0)   # synthetic: gravity along -Y
        self.gyro  = (0.0, 0.0, 0.0)
        self.timestamp = 0.0
        self.intrinsics = {
            "color": {
                "width": width, "height": height,
                "fx": 600.0, "fy": 600.0,
                "ppx": width / 2, "ppy": height / 2,
                "distortion_model": "Brown Conrady",
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "depth": {
                "width": width, "height": height,
                "fx": 600.0, "fy": 600.0,
                "ppx": width / 2, "ppy": height / 2,
                "distortion_model": "Brown Conrady",
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                "depth_scale": 0.001,
            },
            "serial": f"MOCK-{cam_id:04d}",
            "name": f"Mock RealSense D435 (cam{cam_id})",
        }

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._gen_loop, daemon=True)
        self._thread.start()

    def _gen_loop(self):
        interval = 1.0 / self.fps
        hue = self._HUE_MAP.get(self.cam_id, 0)
        while self._running:
            t = time.time()
            # Animated HSV color shifting over time
            h = int((hue + t * 20) % 180)
            hsv = np.full((self.height, self.width, 3), [h, 200, 180], dtype=np.uint8)
            color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.putText(color, f"MOCK cam{self.cam_id} #{self._frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            depth = np.tile(
                np.linspace(500, 3000, self.width, dtype=np.uint16),
                (self.height, 1)
            )

            with self._lock:
                self.color_frame = color
                self.depth_frame = depth
                self.timestamp = t
                self._frame_idx += 1

            time.sleep(max(0, interval - (time.time() - t)))

    def get_frames(self):
        with self._lock:
            if self.color_frame is None:
                return None, None, 0
            return self.color_frame.copy(), self.depth_frame.copy(), self.timestamp

    def get_imu(self) -> dict | None:
        with self._lock:
            return {"accel": self.accel, "gyro": self.gyro}

    def read_imu_once(self) -> dict | None:
        """Mock: returns synthetic IMU instantly, no pipeline restart needed."""
        return self.get_imu()

    def get_jpeg(self) -> bytes:
        color, _, _ = self.get_frames()
        if color is None:
            color = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return bytes(buf)

    def get_depth_jpeg(self) -> bytes:
        _, depth, _ = self.get_frames()
        if depth is None:
            depth = np.zeros((self.height, self.width), dtype=np.uint16)
        depth_clipped = np.clip(depth, 0, 4000).astype(np.float32) / 4000.0
        depth_u8 = (depth_clipped * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
        _, buf = cv2.imencode(".jpg", depth_color, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return bytes(buf)

    def get_serial(self) -> str:
        return self.serial

    def get_name(self) -> str:
        return self.intrinsics["name"]

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Camera Manager — handles N cameras
# ---------------------------------------------------------------------------

class CameraManager:
    """Manages all connected RealSense cameras, indexed 1–N."""

    def __init__(self):
        self.cameras: dict[int, RealSenseCamera | MockCamera] = {}

    def start_all(self, width: int, height: int, fps: int,
                  mock: bool = False, num_mock: int = 2):
        """
        Enumerate and start all connected cameras.
        In mock mode, starts `num_mock` synthetic cameras.
        """
        if mock or not REALSENSE_AVAILABLE:
            for i in range(1, num_mock + 1):
                cam = MockCamera(width=width, height=height, fps=fps, cam_id=i)
                cam.start()
                self.cameras[i] = cam
            print(f"[CameraManager] Started {num_mock} mock camera(s)")
            return

        if not REALSENSE_AVAILABLE:
            return

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            print("[CameraManager] No RealSense devices found, falling back to 1 mock camera")
            cam = MockCamera(width=width, height=height, fps=fps, cam_id=1)
            cam.start()
            self.cameras[1] = cam
            return

        # Stagger pipeline.start() calls to avoid simultaneous USB bandwidth
        # negotiation. When N RealSense cameras come up at the same instant,
        # the kernel's USB scheduler can under-allocate bandwidth to some of
        # them, dropping them to reduced fps / USB 2.0 mode. Sleeping between
        # starts lets each device finish its SuperSpeed isoc negotiation and
        # lock in its slot before the next one starts.
        STARTUP_STAGGER_S = 1.0
        for i, device in enumerate(devices, start=1):
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            try:
                cam = RealSenseCamera(width=width, height=height, fps=fps, serial=serial)
                cam.start()
                self.cameras[i] = cam
                print(f"[CameraManager] cam{i}: {name} (serial {serial}) started")
            except Exception as e:
                print(f"[CameraManager] cam{i}: failed to start {name} ({serial}): {e}")
            # Give the USB stack time to settle before starting the next one.
            if i < len(devices):
                time.sleep(STARTUP_STAGGER_S)

    def get(self, idx: int):
        return self.cameras.get(idx)

    def stop_all(self):
        for cam in self.cameras.values():
            cam.stop()

    @property
    def count(self) -> int:
        return len(self.cameras)

    @property
    def indices(self) -> list[int]:
        return sorted(self.cameras.keys())

    def info(self) -> dict:
        """Return per-camera info dict for the /status endpoint."""
        return {
            idx: {
                "name": cam.get_name(),
                "serial": cam.get_serial(),
                "intrinsics": cam.intrinsics,
            }
            for idx, cam in self.cameras.items()
        }

    def read_all_imu_once(self) -> dict[int, dict | None]:
        """
        Read IMU for every camera sequentially (must be sequential —
        each camera stops its video pipeline during the read).
        Returns {cam_idx: {"accel": ..., "gyro": ...} or None}.
        """
        results = {}
        for idx, cam in self.cameras.items():
            print(f"[CameraManager] Reading IMU for cam{idx}...")
            results[idx] = cam.read_imu_once()
            print(f"[CameraManager] cam{idx} IMU: {results[idx]}")
        return results
