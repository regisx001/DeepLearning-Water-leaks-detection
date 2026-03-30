# ── Serial ───────────────────────────────────────────────────
SERIAL_PORT     = "/dev/ttyUSB0"   # or /dev/ttyACM0
SERIAL_BAUDRATE = 9600

# ── Model ────────────────────────────────────────────────────
MODEL_PATH      = "models/bestmodel.keras"
THRESHOLD_PATH  = "models/bestthreshold.npy"
DATA_PATH       = "data/dataengineered.csv"

# ── Window ───────────────────────────────────────────────────
WINDOW_SIZE     = 20
STEP_SIZE       = 5

# ── Features (must match training order) ────────────────────
FEATURES = [
    "Pressure bar",
    "Flow Rate Ls",
    "Temperature C",
    "pressure_delta",
    "flow_zscore",
    "pf_ratio",
]

# ── Sensor stats for z-score (computed from training data) ───
# These will be loaded dynamically from dataengineered.csv
# but you can hardcode them per sensor if running offline

# ── Safety ───────────────────────────────────────────────────
# Consecutive leak windows before closing valve
LEAK_CONFIRMATION_COUNT = 2

# Seconds to wait before re-opening valve after manual reset
REOPEN_COOLDOWN_SEC     = 30

# ── Logging ──────────────────────────────────────────────────
LOG_FILE = "leak_detection.log"