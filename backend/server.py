"""
FastAPI server for the RealSense RGBD recording pipeline.
Supports N cameras simultaneously.
"""

import asyncio
import json
import struct
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from .config import (
    TASKS, DEFAULT_FPS, DEFAULT_WIDTH, DEFAULT_HEIGHT, DATA_ROOT,
    JETSON_URL, JETSON_TIMEOUT_S,
)
from .camera import CameraManager
from .recorder import Recorder
from .utility import load_stats, save_stats, rebuild_stats_from_disk, enumerate_cameras, get_all_orientations, make_pointcloud


# ---------------------------------------------------------------------------
# Jetson record-daemon forwarding helpers
# ---------------------------------------------------------------------------
# If JETSON_URL is non-empty, POST /start and POST /stop are mirrored to the
# Jetson daemon so the robot and cameras record as a single session.
# Blocking `requests` calls are wrapped in run_in_executor to keep the
# FastAPI event loop responsive.

def _jetson_enabled() -> bool:
    return bool(JETSON_URL)


def _jetson_post(path: str, json_body: Optional[dict] = None) -> tuple[int, dict]:
    """Blocking POST to the Jetson daemon. Call via run_in_executor."""
    try:
        r = requests.post(
            f"{JETSON_URL}{path}",
            json=json_body,
            timeout=JETSON_TIMEOUT_S,
        )
    except requests.RequestException as e:
        return -1, {"error": f"Jetson unreachable: {e}"}
    try:
        body = r.json()
    except Exception:
        body = {"text": r.text}
    return r.status_code, body


async def _jetson_post_async(path: str, json_body: Optional[dict] = None) -> tuple[int, dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _jetson_post, path, json_body)

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

app = FastAPI(title="RealSense Recording Pipeline")

camera_manager = CameraManager()

# Per-camera recorders: {cam_idx: Recorder}
_recorders: dict[int, Recorder] = {}

# Current session metadata
_session_dir: Path | None = None
_session_task: str = ""
_session_prompt: str = ""
_session_start: float = 0.0
_session_active: bool = False
_session_used_jetson: bool = False   # True iff the current session started Jetson
_cameras_saving: bool = False        # True while _do_stop is flushing files

current_fps = DEFAULT_FPS
current_width = DEFAULT_WIDTH
current_height = DEFAULT_HEIGHT
use_mock = False
num_mock = 2  # how many mock cameras when --mock

# WebSocket clients
_ws_clients: set[WebSocket] = set()
_orient_ws_clients: set[WebSocket] = set()

# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    camera_manager.start_all(
        width=current_width,
        height=current_height,
        fps=current_fps,
        mock=use_mock,
        num_mock=num_mock,
    )
    for idx in camera_manager.indices:
        _recorders[idx] = Recorder()

    # Read IMU once at startup (stops/restarts each camera's pipeline briefly)
    print("[Server] Reading IMU at startup...")
    camera_manager.read_all_imu_once()
    print("[Server] IMU read complete. Starting streams.")

    asyncio.create_task(frame_broadcast_loop())
    asyncio.create_task(orientation_broadcast_loop())


@app.on_event("shutdown")
async def shutdown():
    camera_manager.stop_all()


# ---------------------------------------------------------------------------
# Frame broadcast loop
# ---------------------------------------------------------------------------

async def frame_broadcast_loop():
    interval = 1.0 / current_fps
    while True:
        t0 = time.monotonic()

        # --- stream frames to WebSocket clients ---
        if _ws_clients:
            frame_cache: dict[tuple, bytes] = {}  # (cam_idx, mode) -> jpeg
            dead = set()
            for ws in list(_ws_clients):
                try:
                    cam_idx = getattr(ws, '_cam_idx', 1)
                    mode = getattr(ws, '_stream_mode', 'color')
                    key = (cam_idx, mode)
                    if key not in frame_cache:
                        cam = camera_manager.get(cam_idx)
                        if cam:
                            frame_cache[key] = (
                                cam.get_depth_jpeg() if mode == 'depth' else cam.get_jpeg()
                            )
                    if key in frame_cache:
                        await ws.send_bytes(frame_cache[key])
                except Exception:
                    dead.add(ws)
            _ws_clients.difference_update(dead)

        # --- write frames to active recorders ---
        if _session_active:
            for cam_idx, recorder in _recorders.items():
                if not recorder.is_recording:
                    continue
                cam = camera_manager.get(cam_idx)
                if cam is None:
                    continue
                color, depth, ts = cam.get_frames()
                if color is not None:
                    recorder.write_frame(color, depth, ts)

        await asyncio.sleep(max(0, interval - (time.monotonic() - t0)))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def index():
    return (FRONTEND_DIR / "index.html").read_text()


async def orientation_broadcast_loop():
    while True:
        if _orient_ws_clients:
            data = get_all_orientations(camera_manager)
            dead = set()
            for ws in list(_orient_ws_clients):
                try:
                    await ws.send_json(data)
                except Exception:
                    dead.add(ws)
            _orient_ws_clients.difference_update(dead)
        await asyncio.sleep(0.5)  # 2 Hz


@app.websocket("/ws/orientation")
async def ws_orientation(websocket: WebSocket):
    await websocket.accept()
    _orient_ws_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        _orient_ws_clients.discard(websocket)


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket, cam: int = 1, mode: str = "color"):
    await websocket.accept()
    websocket._cam_idx = cam
    websocket._stream_mode = mode
    _ws_clients.add(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        _ws_clients.discard(websocket)


@app.get("/tasks")
async def get_tasks():
    return {"tasks": TASKS}


@app.post("/jetson/stop")
async def force_stop_jetson():
    """Force-stop the Jetson record daemon regardless of local camera state.
    Used to clear orphaned recordings when local/Jetson states diverge."""
    if not _jetson_enabled():
        raise HTTPException(status_code=503, detail="JETSON_URL unset")
    code, body = await _jetson_post_async("/stop")
    if code == 200:
        return {"status": "stopped", "jetson": body}
    if code == 409:
        # Jetson was already idle — not an error, just re-sync UI.
        return {"status": "already_idle", "jetson": body}
    if code == -1:
        raise HTTPException(status_code=502, detail=body.get("error", "Jetson unreachable"))
    raise HTTPException(status_code=502, detail=f"Jetson /stop failed (HTTP {code}): {body}")


@app.get("/jetson/status")
async def get_jetson_status():
    """Proxy the Jetson record daemon's /status so the frontend can poll it
    same-origin without CORS concerns. Returns {"enabled": bool, ...}."""
    if not _jetson_enabled():
        return {"enabled": False, "reachable": False, "detail": "JETSON_URL unset"}
    loop = asyncio.get_event_loop()
    try:
        r = await loop.run_in_executor(
            None,
            lambda: requests.get(f"{JETSON_URL}/status", timeout=JETSON_TIMEOUT_S),
        )
        data = r.json()
    except requests.RequestException as e:
        return {"enabled": True, "reachable": False, "detail": str(e)}
    except Exception as e:
        return {"enabled": True, "reachable": False, "detail": f"parse error: {e}"}
    return {"enabled": True, "reachable": True, **data}


@app.get("/status")
async def get_status():
    recording = _session_active
    elapsed = round(time.time() - _session_start, 1) if _session_active else 0.0
    total_frames = sum(r.frame_count for r in _recorders.values())

    # Live per-camera fps (sliding 1s window). Zero when not recording.
    live_fps = {
        str(idx): round(_recorders[idx].recent_fps, 1)
        for idx in camera_manager.indices
        if idx in _recorders
    }

    # Derived high-level state for UI state-light row.
    if _cameras_saving:
        cam_state = "saving"
    elif recording:
        cam_state = "recording"
    elif camera_manager.count == 0:
        cam_state = "offline"
    else:
        cam_state = "idle"

    return {
        "recording": recording,
        "state": cam_state,
        "elapsed": elapsed,
        "frame_count": total_frames,
        "fps": current_fps,
        "width": current_width,
        "height": current_height,
        "active_cameras": camera_manager.indices,
        "cameras": camera_manager.info(),
        "live_fps": live_fps,
        "used_jetson": _session_used_jetson,
    }


@app.get("/stats")
async def get_stats():
    return load_stats()


@app.post("/stats/rebuild")
async def rebuild_stats():
    return rebuild_stats_from_disk()


@app.get("/cameras")
async def get_cameras_endpoint():
    return {"cameras": enumerate_cameras()}


@app.get("/orientation")
async def get_orientation_endpoint():
    """Return cached pitch/roll for all connected cameras (from last IMU read)."""
    return get_all_orientations(camera_manager)


@app.post("/orientation/refresh")
async def refresh_orientation():
    """
    Re-read IMU for all cameras (stops/restarts each pipeline briefly).
    Do NOT call while recording.
    """
    if _session_active:
        raise HTTPException(status_code=409, detail="Cannot refresh IMU while recording")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, camera_manager.read_all_imu_once)
    return get_all_orientations(camera_manager)


@app.get("/pointcloud")
async def get_pointcloud(cam: int = 1, stride: int = 3):
    """
    Return current point cloud for one camera as a compact binary blob.

    Binary layout (little-endian):
        [4 bytes]  uint32  N  — number of valid points
        [N*12 bytes] float32 × N*3  — XYZ in metres (X right, Y down, Z fwd)
        [N*3  bytes] uint8  × N*3  — RGB colours

    Use stride >= 2 to sub-sample; stride=3 gives ~1/9 of pixels (~30 K pts
    for 640×480), which renders smoothly at interactive rates.
    """
    camera = camera_manager.get(cam)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera {cam} not found")

    color, depth, _ = camera.get_frames()
    if color is None:
        raise HTTPException(status_code=503, detail="No frames available yet")

    loop = asyncio.get_event_loop()
    xyz, rgb = await loop.run_in_executor(
        None, make_pointcloud, color, depth, camera.intrinsics, stride
    )

    N = len(xyz)
    buf = struct.pack("<I", N) + xyz.tobytes() + rgb.tobytes()
    return Response(content=buf, media_type="application/octet-stream")


class StartRequest(BaseModel):
    task: str
    prompt: str
    # Optional per-session override for the local camera data root.
    # If omitted or empty, falls back to DATA_ROOT from config.py.
    # Jetson's DATA_ROOT is fixed at daemon-startup time and unaffected.
    data_root: Optional[str] = None
    # Per-session opt-out of Jetson forwarding, even if JETSON_URL is set.
    # True (default) = record cameras + robot; False = record cameras only.
    use_jetson: bool = True


@app.post("/start")
async def start_recording(req: StartRequest):
    global _session_dir, _session_task, _session_prompt, _session_start
    global _session_active, _session_used_jetson

    if req.task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {req.task}")
    if _session_active:
        raise HTTPException(status_code=409, detail="Already recording")
    if camera_manager.count == 0:
        raise HTTPException(status_code=503, detail="No cameras available")

    # --- Step 1: ask the Jetson daemon to start first (fail fast) -----------
    # Forward to Jetson only if both the server-level URL is configured AND
    # the caller opted in via use_jetson (default True).
    jetson_session = None
    forward_jetson = _jetson_enabled() and req.use_jetson
    if forward_jetson:
        code, body = await _jetson_post_async(
            "/start", {"task": req.task, "prompt": req.prompt}
        )
        if code != 200:
            # Don't touch local state — nothing started yet.
            raise HTTPException(
                status_code=502,
                detail=f"Jetson /start failed (HTTP {code}): {body}",
            )
        jetson_session = body.get("session_dir")

    # --- Step 2: prepare session dir + start local camera recorders --------
    # Effective data root: request override if provided, otherwise config default.
    effective_data_root = (req.data_root or DATA_ROOT).strip() if isinstance(req.data_root, str) else DATA_ROOT
    if not effective_data_root:
        effective_data_root = DATA_ROOT

    now = datetime.now()
    session_dir = Path(effective_data_root) / req.task / now.strftime("%m_%d_%Y") / now.strftime("%H_%M_%S")

    # All local setup is wrapped in a single try so any failure rolls back the
    # already-started Jetson recording.
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "prompt.txt").write_text(req.prompt)

        for cam_idx in camera_manager.indices:
            cam = camera_manager.get(cam_idx)
            cam_dir = session_dir / f"cam{cam_idx}"
            _recorders[cam_idx].start(
                save_dir=cam_dir,
                intrinsics=cam.intrinsics,
                width=current_width,
                height=current_height,
                fps=current_fps,
            )
    except Exception as e:
        # Local setup or recorder start failed — roll back Jetson so it doesn't
        # stay recording with no corresponding laptop data.
        if forward_jetson:
            await _jetson_post_async("/stop")
        raise HTTPException(
            status_code=500,
            detail=f"Local recorder start failed: {e}",
        )

    _session_dir = session_dir
    _session_task = req.task
    _session_prompt = req.prompt
    _session_start = time.time()
    _session_active = True
    _session_used_jetson = forward_jetson

    return {
        "status": "started",
        "session_dir": str(session_dir),
        "cameras": camera_manager.indices,
        "jetson_session_dir": jetson_session,
        "used_jetson": forward_jetson,
    }


@app.post("/stop")
async def stop_recording():
    global _session_dir, _session_task, _session_prompt, _session_start
    global _session_active, _session_used_jetson

    if not _session_active:
        raise HTTPException(status_code=409, detail="Not recording")

    _session_active = False
    used_jetson_this_session = _session_used_jetson
    _session_used_jetson = False
    elapsed = round(time.time() - _session_start, 2)
    session_dir_snap = _session_dir  # snapshot before globals are cleared below

    # Run the CPU-heavy stop + NPZ compression in a thread so the event loop
    # stays responsive (status polling, WebSocket streams, etc.)
    def _do_stop():
        results = {}
        for cam_idx, recorder in _recorders.items():
            if not recorder.is_recording:
                continue
            info, (depth_array, timestamps) = recorder.stop()
            cam_dir = session_dir_snap / f"cam{cam_idx}"
            if depth_array is not None:
                np.savez_compressed(
                    str(cam_dir / "depth.npz"),
                    depth=depth_array,
                    timestamps=timestamps,
                )
            results[f"cam{cam_idx}"] = info

        return results

    # Stop local cameras and Jetson daemon concurrently. Local stop is the
    # source of truth for our NPZ files; Jetson stop is best-effort — if it
    # fails, we surface the error but don't roll the local save back (the
    # cameras already stopped).
    global _cameras_saving
    _cameras_saving = True

    async def _local_with_flag():
        try:
            return await loop.run_in_executor(None, _do_stop)
        finally:
            # Clear saving flag as soon as local save completes, before
            # waiting on the (usually slower) Jetson path in gather().
            global _cameras_saving
            _cameras_saving = False

    loop = asyncio.get_event_loop()
    local_task = _local_with_flag()
    jetson_task = (
        _jetson_post_async("/stop")
        if used_jetson_this_session
        else asyncio.sleep(0, result=(None, None))
    )
    results, (jet_code, jet_body) = await asyncio.gather(local_task, jetson_task)

    if jet_code is None:
        # forwarding disabled (JETSON_URL unset)
        jetson_result, jetson_error = None, None
    elif jet_code == 200:
        jetson_result, jetson_error = jet_body, None
    elif jet_code == -1:
        # connection failure, jet_body already has {"error": "..."}
        jetson_result, jetson_error = None, jet_body.get("error", "Jetson unreachable")
    else:
        jetson_result = None
        jetson_error = f"Jetson /stop failed (HTTP {jet_code}): {jet_body}"

    # Update stats
    stats = load_stats()
    task = _session_task
    if task not in stats:
        stats[task] = {"count": 0, "total_duration_s": 0.0, "demos": []}
    stats[task]["count"] += 1
    stats[task]["total_duration_s"] = round(stats[task]["total_duration_s"] + elapsed, 2)
    stats[task]["demos"].append({
        "dir": str(_session_dir),
        "date": _session_dir.parent.name,
        "time": _session_dir.name,
        "duration_s": elapsed,
        "prompt": _session_prompt,
        "cameras": camera_manager.indices,
    })
    save_stats(stats)

    session_dir_str = str(_session_dir)
    _session_dir = None
    _session_task = ""
    _session_prompt = ""

    return {
        "status": "stopped",
        "session_dir": session_dir_str,
        "elapsed_seconds": elapsed,
        "cameras": results,
        "jetson": jetson_result,
        "jetson_error": jetson_error,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global use_mock, current_fps, current_width, current_height, num_mock

    parser = argparse.ArgumentParser(description="RealSense Recording Server")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--num-mock", type=int, default=2, help="Number of mock cameras")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    use_mock = args.mock
    num_mock = args.num_mock
    current_fps = args.fps
    current_width = args.width
    current_height = args.height

    uvicorn.run("backend.server:app", host="0.0.0.0", port=args.port, reload=False)


if __name__ == "__main__":
    main()
