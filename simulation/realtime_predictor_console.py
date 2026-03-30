from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "Timestamp",
        "timestamp",
        "time",
        "datetime",
    ),
    "sensor_id": (
        "SensorID",
        "Sensor_ID",
        "sensor_id",
        "sensor",
        "sensorid",
    ),
    "pressure": (
        "Pressure bar",
        "Pressure (bar)",
        "pressure",
        "pressure_bar",
    ),
    "flow": (
        "Flow Rate Ls",
        "Flow Rate (L/s)",
        "flow",
        "flow_rate",
    ),
    "temperature": (
        "Temperature C",
        "Temperature (C)",
        "temperature",
        "temp",
        "temperature_c",
    ),
    "leak_status": (
        "Leak Status",
        "leak_status",
        "label",
        "target",
        "y",
    ),
}

FEATURE_ORDER: tuple[str, ...] = (
    "Pressure bar",
    "Flow Rate Ls",
    "Temperature C",
    "pressure_delta",
    "flow_zscore",
    "pf_ratio",
)

MODEL_CANDIDATES: tuple[str, ...] = (
    "models/best_model.keras",
    "models/bestmodel.keras",
    "simulation/best_model.keras",
    "simulation/bestmodel.keras",
)

THRESHOLD_CANDIDATES: tuple[str, ...] = (
    "models/best_threshold.npy",
    "models/bestthreshold.npy",
)

TRAIN_DATA_CANDIDATES: tuple[str, ...] = (
    "data/data_engineered.csv",
    "data/dataengineered.csv",
)


@dataclass
class SensorState:
    flow_mean: float
    flow_std: float
    raw_buffer: deque[dict[str, float]]
    step_counter: int = 0
    leak_counter: int = 0


@dataclass(frozen=True)
class StreamRow:
    timestamp: pd.Timestamp
    sensor_id: str
    pressure: float
    flow: float
    temperature: float
    leak_status: int | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime prediction and row console logging",
    )

    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--threshold-path", type=str, default=None)
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/data_clean.csv",
        help="Realtime stream source CSV",
    )

    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--leak-confirmation-count", type=int, default=2)

    parser.add_argument("--speed", type=float, default=600.0)
    parser.add_argument("--max-sleep-sec", type=float, default=2.0)
    parser.add_argument("--fixed-interval-sec", type=float, default=1.0)
    parser.add_argument("--no-timestamp-sleep", action="store_true")

    parser.add_argument("--sensor", action="append", default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.window_size <= 1:
        raise ValueError("window-size must be > 1")
    if args.step_size <= 0:
        raise ValueError("step-size must be > 0")
    if args.leak_confirmation_count <= 0:
        raise ValueError("leak-confirmation-count must be > 0")

    model_path = _resolve_path(args.model_path, MODEL_CANDIDATES, "model")
    threshold_path = _resolve_path(args.threshold_path, THRESHOLD_CANDIDATES, "threshold")
    train_data_path = _resolve_path(args.train_data_path, TRAIN_DATA_CANDIDATES, "train-data")
    stream_data_path = _to_absolute_path(args.data_path)

    model = _load_model_with_fallback(
        model_path=model_path,
        window_size=args.window_size,
        n_features=len(FEATURE_ORDER),
    )
    threshold = _load_threshold(threshold_path)

    train_frame = load_existing_data(train_data_path)
    scaler, sensor_stats, global_stats = _build_scaler_and_stats(train_frame)

    stream_iter = iter_stream_from_csv(
        csv_path=stream_data_path,
        speed=args.speed,
        use_timestamp_sleep=not args.no_timestamp_sleep,
        fixed_interval_sec=args.fixed_interval_sec,
        max_sleep_sec=args.max_sleep_sec,
        sensor_filter=args.sensor,
        loop=args.loop,
    )

    sensor_state: dict[str, SensorState] = {}

    for index, row in enumerate(stream_iter, start=1):
        state = sensor_state.get(row.sensor_id)
        if state is None:
            flow_mean, flow_std = sensor_stats.get(row.sensor_id, global_stats)
            state = SensorState(
                flow_mean=flow_mean,
                flow_std=flow_std,
                raw_buffer=deque(maxlen=args.window_size + 10),
            )
            sensor_state[row.sensor_id] = state

        state.raw_buffer.append(
            {
                "Pressure bar": row.pressure,
                "Flow Rate Ls": row.flow,
                "Temperature C": row.temperature,
            }
        )
        state.step_counter += 1

        output = {
            "timestamp": row.timestamp.isoformat(sep=" "),
            "sensor_id": row.sensor_id,
            "pressure": row.pressure,
            "flow": row.flow,
            "temperature": row.temperature,
            "ground_truth": row.leak_status,
            "probability": None,
            "prediction": "WARMUP",
            "confirmed_leak": False,
            "consecutive_leaks": state.leak_counter,
        }

        if len(state.raw_buffer) >= args.window_size and state.step_counter % args.step_size == 0:
            window = _build_window(state, args.window_size)
            probability = _predict_probability(model, scaler, window)
            is_leak = probability >= threshold

            if is_leak:
                state.leak_counter += 1
            else:
                state.leak_counter = 0

            confirmed = state.leak_counter >= args.leak_confirmation_count

            output["probability"] = round(probability, 6)
            output["prediction"] = "LEAK" if is_leak else "NO_LEAK"
            output["confirmed_leak"] = confirmed
            output["consecutive_leaks"] = state.leak_counter

        elif len(state.raw_buffer) >= args.window_size:
            output["prediction"] = "WAIT_STEP"

        print(json.dumps(output, ensure_ascii=True))

        if args.max_rows is not None and index >= args.max_rows:
            break

    return 0


def _build_window(state: SensorState, window_size: int) -> np.ndarray:
    frame = pd.DataFrame(list(state.raw_buffer))
    frame["pressure_delta"] = frame["Pressure bar"].diff().fillna(0.0)
    frame["pf_ratio"] = frame["Pressure bar"] / (frame["Flow Rate Ls"] + 1e-8)
    frame["flow_zscore"] = (frame["Flow Rate Ls"] - state.flow_mean) / (
        state.flow_std + 1e-8
    )
    window = frame[list(FEATURE_ORDER)].iloc[-window_size:]
    return window.values.astype(np.float32)


def _predict_probability(model, scaler: StandardScaler, window: np.ndarray) -> float:
    n_rows, n_features = window.shape
    scaled = scaler.transform(window.reshape(-1, n_features)).reshape(n_rows, n_features)
    batch = scaled[np.newaxis, :, :].astype(np.float32)
    return float(model.predict(batch, verbose=0)[0][0])


def _build_scaler_and_stats(
    train_frame: pd.DataFrame,
) -> tuple[StandardScaler, dict[str, tuple[float, float]], tuple[float, float]]:
    per_sensor_features: list[pd.DataFrame] = []
    sensor_stats: dict[str, tuple[float, float]] = {}

    for sensor_id, group in train_frame.groupby("SensorID", sort=False):
        local = group.sort_values("Timestamp").reset_index(drop=True).copy()

        flow_mean = float(local["Flow Rate Ls"].mean())
        flow_std = float(local["Flow Rate Ls"].std())
        if not np.isfinite(flow_std) or flow_std == 0.0:
            flow_std = 1.0

        sensor_stats[str(sensor_id)] = (flow_mean, flow_std)

        local["pressure_delta"] = local["Pressure bar"].diff().fillna(0.0)
        local["pf_ratio"] = local["Pressure bar"] / (local["Flow Rate Ls"] + 1e-8)
        local["flow_zscore"] = (local["Flow Rate Ls"] - flow_mean) / (flow_std + 1e-8)

        per_sensor_features.append(local[list(FEATURE_ORDER)])

    if not per_sensor_features:
        raise ValueError("No training rows available for scaler fitting")

    scaler_frame = pd.concat(per_sensor_features, ignore_index=True).dropna()
    scaler = StandardScaler()
    scaler.fit(scaler_frame.values)

    global_mean = float(train_frame["Flow Rate Ls"].mean())
    global_std = float(train_frame["Flow Rate Ls"].std())
    if not np.isfinite(global_std) or global_std == 0.0:
        global_std = 1.0

    return scaler, sensor_stats, (global_mean, global_std)


def _load_threshold(threshold_path: Path) -> float:
    value = np.load(threshold_path)
    flat = np.array(value).reshape(-1)
    if flat.size == 0:
        raise ValueError(f"Empty threshold file: {threshold_path}")
    return float(flat[0])


def _load_model_with_fallback(model_path: Path, window_size: int, n_features: int):
    errors: list[Exception] = []

    for loader in (
        lambda: tf.keras.models.load_model(model_path),
        lambda: tf.keras.models.load_model(model_path, safe_mode=False),
        lambda: _load_with_global_unsafe(model_path),
    ):
        try:
            return loader()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    if _has_lambda_shape_issue(errors):
        model = _build_cnn_bilstm_attention(window_size, n_features)
        model.load_weights(model_path)
        return model

    raise errors[-1]


def _load_with_global_unsafe(model_path: Path):
    tf.keras.config.enable_unsafe_deserialization()
    return tf.keras.models.load_model(model_path)


def _has_lambda_shape_issue(errors: list[Exception]) -> bool:
    for exc in errors:
        msg = str(exc)
        if "Could not automatically infer the shape of the Lambda" in msg:
            return True
        if "Lambda.call" in msg and "output_shape" in msg:
            return True
    return False


def _build_cnn_bilstm_attention(window_size: int, n_features: int):
    inp = tf.keras.layers.Input(shape=(window_size, n_features), name="input")

    x = tf.keras.layers.Conv1D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(inp)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Conv1D(
        32,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Dropout(0.30, name="drop_cnn")(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(32, return_sequences=True, dropout=0.25),
        name="bilstm",
    )(x)

    attn_scores = tf.keras.layers.Dense(1, activation="tanh", name="attn_score")(x)
    attn_weights = tf.keras.layers.Softmax(axis=1, name="attn_weight")(attn_scores)
    x = tf.keras.layers.Multiply(name="attn_mul")([x, attn_weights])
    x = tf.keras.layers.Lambda(
        lambda z: tf.reduce_sum(z, axis=1),
        output_shape=lambda s: (s[0], s[2]),
        name="attn_context",
    )(x)

    x = tf.keras.layers.Dense(16, activation="relu", name="dense1")(x)
    x = tf.keras.layers.Dropout(0.40, name="drop_dense")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return tf.keras.Model(inputs=inp, outputs=out)


def _resolve_path(
    override: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> Path:
    if override:
        path = _to_absolute_path(override)
        if not path.exists():
            raise FileNotFoundError(f"Provided {label} path does not exist: {path}")
        return path

    for rel in candidates:
        candidate = (PROJECT_ROOT / rel).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not resolve {label} path from candidates: {list(candidates)}")


def _to_absolute_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_existing_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    frame = pd.read_csv(csv_path)

    rename_map = {
        _resolve_column(frame.columns, "timestamp"): "Timestamp",
        _resolve_column(frame.columns, "sensor_id"): "SensorID",
        _resolve_column(frame.columns, "pressure"): "Pressure bar",
        _resolve_column(frame.columns, "flow"): "Flow Rate Ls",
        _resolve_column(frame.columns, "temperature"): "Temperature C",
    }

    leak_col = _resolve_column(frame.columns, "leak_status", required=False)
    if leak_col is not None:
        rename_map[leak_col] = "Leak Status"

    normalized = frame.rename(columns=rename_map).copy()

    normalized["Timestamp"] = pd.to_datetime(normalized["Timestamp"], errors="coerce")
    normalized["SensorID"] = normalized["SensorID"].astype(str).str.strip()

    normalized["Pressure bar"] = pd.to_numeric(normalized["Pressure bar"], errors="coerce")
    normalized["Flow Rate Ls"] = pd.to_numeric(normalized["Flow Rate Ls"], errors="coerce")
    normalized["Temperature C"] = pd.to_numeric(normalized["Temperature C"], errors="coerce")

    normalized = normalized.dropna(
        subset=[
            "Timestamp",
            "SensorID",
            "Pressure bar",
            "Flow Rate Ls",
            "Temperature C",
        ]
    )

    if "Leak Status" in normalized.columns:
        normalized["Leak Status"] = pd.to_numeric(
            normalized["Leak Status"], errors="coerce"
        ).round().astype("Int64")

    normalized = normalized.sort_values(["Timestamp", "SensorID"]).reset_index(drop=True)

    keep_cols = [
        "Timestamp",
        "SensorID",
        "Pressure bar",
        "Flow Rate Ls",
        "Temperature C",
    ]
    if "Leak Status" in normalized.columns:
        keep_cols.append("Leak Status")

    return normalized[keep_cols]


def iter_stream_from_csv(
    csv_path: Path,
    speed: float,
    use_timestamp_sleep: bool,
    fixed_interval_sec: float,
    max_sleep_sec: float,
    sensor_filter: Iterable[str] | None,
    loop: bool,
) -> Iterator[StreamRow]:
    frame = load_existing_data(csv_path)

    if sensor_filter:
        allowed = {str(sensor).strip() for sensor in sensor_filter}
        frame = frame[frame["SensorID"].isin(allowed)]

    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows available to stream after filtering")

    while True:
        previous_timestamp: pd.Timestamp | None = None

        for _, row in frame.iterrows():
            timestamp = pd.Timestamp(row["Timestamp"])

            if use_timestamp_sleep:
                if previous_timestamp is not None:
                    delta = (timestamp - previous_timestamp).total_seconds()
                    if delta > 0:
                        time.sleep(min(delta / speed, max_sleep_sec))
            elif fixed_interval_sec > 0:
                time.sleep(fixed_interval_sec)

            previous_timestamp = timestamp

            leak_status: int | None = None
            if "Leak Status" in frame.columns:
                leak_value = row["Leak Status"]
                if pd.notna(leak_value):
                    leak_status = int(leak_value)

            yield StreamRow(
                timestamp=timestamp,
                sensor_id=str(row["SensorID"]),
                pressure=float(row["Pressure bar"]),
                flow=float(row["Flow Rate Ls"]),
                temperature=float(row["Temperature C"]),
                leak_status=leak_status,
            )

        if not loop:
            return


def _normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def _resolve_column(
    columns: Iterable[str],
    key: str,
    required: bool = True,
) -> str | None:
    normalized_columns = {_normalize_name(col): col for col in columns}
    for alias in COLUMN_ALIASES.get(key, ()):  # pragma: no branch
        normalized_alias = _normalize_name(alias)
        if normalized_alias in normalized_columns:
            return normalized_columns[normalized_alias]

    if required:
        raise ValueError(
            f"Missing required column alias for '{key}'. Available columns: {list(columns)}"
        )
    return None


if __name__ == "__main__":
    raise SystemExit(main())
