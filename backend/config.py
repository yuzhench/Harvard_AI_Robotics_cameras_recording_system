import os

# Bump whenever the HTTP response shape changes in a way that the frontend
# or the Jetson daemon would need to co-update. Frontend enforces strict
# equality; mismatch → UI error banner + disabled Start button.
PROTOCOL_VERSION = 3

TASKS = [
    "task1",
    "task2",
    "task3",
    "task4",
    "task5",
    "task6",
    "task7",
    "task8",
    "task9",
    "task10",
]

DEFAULT_FPS = 30
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# Default data root for session files and stats.json. Lives under the current
# user's home so no sudo/chown is needed, and works on any machine (yuzhench,
# harvardair, etc.). Overridable per-session via the frontend "Save Directory"
# input or at process startup via the DATA_ROOT env var.
DATA_ROOT = os.environ.get("DATA_ROOT") or os.path.expanduser("~/GO2_DATA")

# When set (non-empty), POST /start and POST /stop on this server are
# forwarded to the Jetson record daemon so the robot and cameras record
# together. Override at runtime, e.g.:
#     JETSON_URL=http://100.112.18.112:8010 ./run.sh          # Tailscale
#     JETSON_URL=""                          ./run.sh          # disable forwarding
JETSON_URL = os.environ.get("JETSON_URL", "http://10.100.206.170:8010")
JETSON_TIMEOUT_S = float(os.environ.get("JETSON_TIMEOUT_S", "5.0"))

# /stop on the Jetson runs save() on every collector (LiDAR .npy + depth
# .npz compression especially can take 10-30s). A timeout here does NOT
# mean the save failed — it just means we stopped waiting. The UI surfaces
# this as a warning rather than an error.
JETSON_STOP_TIMEOUT_S = float(os.environ.get("JETSON_STOP_TIMEOUT_S", "60.0"))

# Rsync-pull configuration for the "Sync" button on the UI. The Jetson host
# is reused from JETSON_URL; SSH user and remote data root come from env
# vars so the same source tree works for different Jetson images.
# Requires passwordless SSH (key-based) from laptop to the Jetson.
JETSON_SSH_USER  = os.environ.get("JETSON_SSH_USER",  "unitree")
JETSON_DATA_ROOT = os.environ.get("JETSON_DATA_ROOT", "/home/unitree/GO2_DATA")

# Cap rsync bandwidth (KB/sec) so the transfer doesn't saturate the WiFi
# and knock the Jetson HTTP daemon off the air mid-sync. Set to "0" to
# disable the cap. 5000 KB/s ~= 40 Mbps — plenty for data pull, leaves
# headroom for /status polling.
JETSON_RSYNC_BWLIMIT_KBPS = int(os.environ.get("JETSON_RSYNC_BWLIMIT_KBPS", "5000"))
