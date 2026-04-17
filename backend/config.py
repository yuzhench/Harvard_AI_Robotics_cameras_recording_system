import os

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
