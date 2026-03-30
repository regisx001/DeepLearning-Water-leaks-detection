#!/usr/bin/env python3
"""
Water Leak Detection — Raspberry Pi Inference Script
Reads sensor data from Arduino over Serial,
runs CNN-BiLSTM-Attention model, and closes valve on leak.
"""

import os
import time
import logging
import signal
import sys
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import serial
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from config import (
    SERIAL_PORT, SERIAL_BAUDRATE,
    MODEL_PATH, THRESHOLD_PATH, DATA_PATH,
    WINDOW_SIZE, STEP_SIZE, FEATURES,
    LEAK_CONFIRMATION_COUNT, REOPEN_COOLDOWN_SEC,
    LOG_FILE,
)

# ── Logging Setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ============================================================
#  Feature Engineering (mirrors Notebook 02)
# ============================================================
class FeatureBuffer:
    """
    Maintains a rolling buffer of raw readings and computes
    engineered features on the fly, matching the training pipeline.
    """

    def __init__(self, sensor_id: str, train_df: pd.DataFrame):
        self.sensor_id = sensor_id
        self.raw_buffer: deque = deque(maxlen=WINDOW_SIZE + 10)

        # Compute per-sensor stats from training data for flow z-score
        sensor_data = train_df[train_df["SensorID"] == sensor_id]["Flow Rate Ls"]
        self.flow_mean = float(sensor_data.mean()) if len(sensor_data) > 0 else 0.0
        self.flow_std  = float(sensor_data.std())  if len(sensor_data) > 0 else 1.0
        if self.flow_std == 0:
            self.flow_std = 1.0

        log.info(
            f"[{sensor_id}] flow_mean={self.flow_mean:.3f}  "
            f"flow_std={self.flow_std:.3f}"
        )

    def add_reading(self, pressure: float, flow: float, temperature: float):
        self.raw_buffer.append({
            "Pressure bar":  pressure,
            "Flow Rate Ls":  flow,
            "Temperature C": temperature,
        })

    def _engineer(self) -> pd.DataFrame:
        df = pd.DataFrame(list(self.raw_buffer))

        # pressure_delta: temporal diff per row
        df["pressure_delta"] = df["Pressure bar"].diff().fillna(0.0)

        # pf_ratio: pressure / flow
        df["pf_ratio"] = df["Pressure bar"] / (df["Flow Rate Ls"] + 1e-8)

        # flow_zscore: sensor-normalised flow anomaly
        df["flow_zscore"] = (
            (df["Flow Rate Ls"] - self.flow_mean) / (self.flow_std + 1e-8)
        )

        return df[FEATURES]

    def get_window(self) -> np.ndarray | None:
        """Returns the latest (WINDOW_SIZE, n_features) window or None."""
        if len(self.raw_buffer) < WINDOW_SIZE:
            return None
        engineered = self._engineer()
        return engineered.iloc[-WINDOW_SIZE:].values.astype(np.float32)


# ============================================================
#  Scaler (fit on training data — same as Notebook 04)
# ============================================================
def build_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """
    Fit StandardScaler on training data only,
    exactly as done during model training.
    """
    log.info("Fitting StandardScaler on training data...")

    # Replicate feature engineering on full training set
    dfs = []
    for sid, group in train_df.groupby("SensorID"):
        g = group.copy().sort_values("Timestamp").reset_index(drop=True)
        g["pressure_delta"] = g["Pressure bar"].diff().fillna(0.0)
        g["pf_ratio"]       = g["Pressure bar"] / (g["Flow Rate Ls"] + 1e-8)
        flow_mean = g["Flow Rate Ls"].mean()
        flow_std  = g["Flow Rate Ls"].std() or 1.0
        g["flow_zscore"] = (g["Flow Rate Ls"] - flow_mean) / (flow_std + 1e-8)
        dfs.append(g)

    full = pd.concat(dfs)[FEATURES].dropna()
    scaler = StandardScaler()
    scaler.fit(full.values)
    log.info("Scaler ready.")
    return scaler


# ============================================================
#  Valve Controller
# ============================================================
class ValveController:
    def __init__(self, ser: serial.Serial):
        self.ser = ser
        self.is_closed = False
        self._last_close_time = 0.0

    def close(self):
        if not self.is_closed:
            self._send("CLOSE")
            self.is_closed = True
            self._last_close_time = time.time()
            log.warning("🚨 VALVE CLOSED — leak confirmed!")

    def open(self):
        elapsed = time.time() - self._last_close_time
        if elapsed < REOPEN_COOLDOWN_SEC:
            log.info(
                f"Cooldown active — {REOPEN_COOLDOWN_SEC - elapsed:.0f}s remaining."
            )
            return
        if self.is_closed:
            self._send("OPEN")
            self.is_closed = False
            log.info("✅ VALVE OPENED — normal flow restored.")

    def status(self) -> str:
        self._send("STATUS")
        time.sleep(0.1)
        if self.ser.in_waiting:
            return self.ser.readline().decode("utf-8", errors="ignore").strip()
        return "UNKNOWN"

    def ping(self) -> bool:
        self._send("PING")
        time.sleep(0.1)
        if self.ser.in_waiting:
            resp = self.ser.readline().decode("utf-8", errors="ignore").strip()
            return resp == "PONG"
        return False

    def _send(self, cmd: str):
        self.ser.write((cmd + "\n").encode("utf-8"))
        log.debug(f"→ Arduino: {cmd}")


# ============================================================
#  Main Inference Loop
# ============================================================
class LeakDetector:
    def __init__(self):
        # ── Load model ───────────────────────────────────────
        log.info(f"Loading model from {MODEL_PATH} ...")
        self.model = tf.keras.models.load_model(MODEL_PATH)

        # ── Load threshold ───────────────────────────────────
        self.threshold = float(np.load(THRESHOLD_PATH))
        log.info(f"Decision threshold: {self.threshold:.4f}")

        # ── Load training data for scaler + sensor stats ─────
        log.info(f"Loading training data from {DATA_PATH} ...")
        train_df = pd.read_csv(DATA_PATH)
        train_df["Timestamp"] = pd.to_datetime(train_df["Timestamp"])

        # ── Scaler ───────────────────────────────────────────
        self.scaler = build_scaler(train_df)

        # ── Serial connection ────────────────────────────────
        log.info(f"Opening serial port {SERIAL_PORT} @ {SERIAL_BAUDRATE} baud...")
        self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=2)
        time.sleep(2)   # wait for Arduino reset
        log.info("Serial ready.")

        # ── Valve controller ─────────────────────────────────
        self.valve = ValveController(self.ser)

        # ── Per-sensor buffers ───────────────────────────────
        self.buffers: dict[str, FeatureBuffer] = {}
        for sid in train_df["SensorID"].unique():
            self.buffers[sid] = FeatureBuffer(sid, train_df)

        # ── Leak confirmation counter ────────────────────────
        # Key: sensor_id → consecutive leak window count
        self.leak_counter: dict[str, int] = {}

        # ── Step counter (for STEP_SIZE logic) ───────────────
        self.step_counter: dict[str, int] = {}

        log.info("LeakDetector initialised. Listening for sensor data...")
        self._ping_arduino()

    def _ping_arduino(self):
        if self.valve.ping():
            log.info("Arduino responded to PING ✓")
        else:
            log.warning("No PONG from Arduino — check serial connection.")

    def _parse_line(self, line: str) -> tuple | None:
        """
        Parse Arduino CSV line: SENSOR_ID,pressure,flow,temperature
        Returns (sensor_id, pressure, flow, temp) or None on error.
        """
        parts = line.strip().split(",")
        if len(parts) != 4:
            return None
        try:
            sid      = parts[0].strip()
            pressure = float(parts[1])
            flow     = float(parts[2])
            temp     = float(parts[3])
            return sid, pressure, flow, temp
        except ValueError:
            return None

    def _run_inference(self, sensor_id: str) -> float | None:
        """
        Get the latest window from the buffer, scale it, run model.
        Returns probability of leak, or None if window not ready.
        """
        buf = self.buffers.get(sensor_id)
        if buf is None:
            # New sensor seen at runtime — create buffer with default stats
            log.warning(f"Unknown sensor {sensor_id} — creating default buffer.")
            dummy_df = pd.DataFrame({"SensorID": [sensor_id], "Flow Rate Ls": [100.0]})
            self.buffers[sensor_id] = FeatureBuffer(sensor_id, dummy_df)
            buf = self.buffers[sensor_id]

        window = buf.get_window()
        if window is None:
            return None   # not enough data yet

        # Scale: reshape to 2D → scale → reshape back
        n, f = window.shape
        scaled = self.scaler.transform(window.reshape(-1, f)).reshape(n, f)

        # Model expects (batch, timesteps, features)
        x = scaled[np.newaxis, :, :].astype(np.float32)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        return prob

    def _handle_prediction(self, sensor_id: str, prob: float):
        is_leak = prob >= self.threshold
        label   = "LEAK" if is_leak else "normal"

        log.info(
            f"[{sensor_id}] prob={prob:.4f}  threshold={self.threshold:.4f}  → {label}"
        )

        if is_leak:
            self.leak_counter[sensor_id] = (
                self.leak_counter.get(sensor_id, 0) + 1
            )
            if self.leak_counter[sensor_id] >= LEAK_CONFIRMATION_COUNT:
                self.valve.close()
        else:
            # Reset counter on normal window
            self.leak_counter[sensor_id] = 0
            # Attempt to re-open if cooldown has passed
            if self.valve.is_closed:
                self.valve.open()

    def run(self):
        """Main blocking loop."""
        while True:
            try:
                if self.ser.in_waiting == 0:
                    time.sleep(0.05)
                    continue

                raw = self.ser.readline().decode("utf-8", errors="ignore")

                # ── Filter Arduino log/ack lines ──────────────
                if raw.startswith(("LOG:", "ACK:", "STATUS:", "PONG", "READY")):
                    log.debug(f"← Arduino: {raw.strip()}")
                    continue

                parsed = self._parse_line(raw)
                if parsed is None:
                    continue

                sid, pressure, flow, temp = parsed

                # ── Add to rolling buffer ─────────────────────
                buf = self.buffers.setdefault(
                    sid, FeatureBuffer(sid, pd.DataFrame())
                )
                buf.add_reading(pressure, flow, temp)

                # ── Run inference every STEP_SIZE readings ────
                self.step_counter[sid] = self.step_counter.get(sid, 0) + 1
                if self.step_counter[sid] % STEP_SIZE == 0:
                    prob = self._run_inference(sid)
                    if prob is not None:
                        self._handle_prediction(sid, prob)

            except serial.SerialException as e:
                log.error(f"Serial error: {e} — retrying in 5s...")
                time.sleep(5)

            except KeyboardInterrupt:
                log.info("Interrupted. Shutting down...")
                self.valve.open()   # safety: open valve on exit
                self.ser.close()
                sys.exit(0)


# ============================================================
#  Graceful shutdown on SIGTERM (systemd / kill)
# ============================================================
detector_instance = None

def handle_sigterm(sig, frame):
    log.info("SIGTERM received. Opening valve and exiting...")
    if detector_instance:
        detector_instance.valve.open()
        detector_instance.ser.close()
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)


# ============================================================
if __name__ == "__main__":
    detector_instance = LeakDetector()
    detector_instance.run()