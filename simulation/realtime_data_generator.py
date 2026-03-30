from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Iterable, Iterator

import pandas as pd


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


@dataclass(frozen=True)
class StreamRow:
    timestamp: datetime
    sensor_id: str
    pressure: float
    flow: float
    temperature: float
    leak_status: int | None


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
        previous_timestamp: datetime | None = None

        for _, row in frame.iterrows():
            timestamp = pd.Timestamp(row["Timestamp"]).to_pydatetime()

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime row generator from existing dataset",
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="data/data_clean.csv",
        help="CSV source path",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=600.0,
        help="Timestamp speed factor (higher = faster playback)",
    )
    parser.add_argument(
        "--max-sleep-sec",
        type=float,
        default=2.0,
        help="Upper bound for each sleep interval",
    )
    parser.add_argument(
        "--fixed-interval-sec",
        type=float,
        default=1.0,
        help="Used only with --no-timestamp-sleep",
    )
    parser.add_argument(
        "--no-timestamp-sleep",
        action="store_true",
        help="Disable timestamp-driven sleep and use fixed interval",
    )
    parser.add_argument(
        "--sensor",
        action="append",
        default=None,
        help="Filter sensor IDs (repeatable)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Replay forever",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Stop after N streamed rows",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_path = _to_absolute_path(args.data_path)

    row_iter = iter_stream_from_csv(
        csv_path=data_path,
        speed=args.speed,
        use_timestamp_sleep=not args.no_timestamp_sleep,
        fixed_interval_sec=args.fixed_interval_sec,
        max_sleep_sec=args.max_sleep_sec,
        sensor_filter=args.sensor,
        loop=args.loop,
    )

    for index, row in enumerate(row_iter, start=1):
        payload = {
            "timestamp": row.timestamp.isoformat(sep=" "),
            "sensor_id": row.sensor_id,
            "pressure": row.pressure,
            "flow": row.flow,
            "temperature": row.temperature,
            "leak_status": row.leak_status,
        }
        print(json.dumps(payload, ensure_ascii=True))

        if args.max_rows is not None and index >= args.max_rows:
            break

    return 0


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


def _to_absolute_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
