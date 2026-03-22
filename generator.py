import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ── Seeded for reproducibility ────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ── Class-conditional parameters (derived from EDA) ──────────────────────
CLASS_PARAMS = {
    "normal": {
        "pressure": {"mean": 3.26, "std": 0.43, "min": 2.0,  "max": 4.0},
        "flow":     {"mean": 124.2,"std": 43.1, "min": 50.0, "max": 200.0},
        "temp":     {"mean": 17.4, "std": 4.3,  "min": 10.0, "max": 25.0},
    },
    "leak": {
        "pressure": {"mean": 2.19, "std": 0.45, "min": 1.0,  "max": 3.1},
        "flow":     {"mean": 148.7,"std": 57.6, "min": 70.0, "max": 240.0},
        "temp":     {"mean": 17.9, "std": 4.4,  "min": 10.0, "max": 25.0},
    },
    "burst": {
        "pressure": {"mean": 1.35, "std": 0.43, "min": 0.5,  "max": 2.3},
        "flow":     {"mean": 166.0,"std": 76.8, "min": 85.0, "max": 340.0},
        "temp":     {"mean": 18.6, "std": 3.8,  "min": 10.0, "max": 25.0},
    },
}

SENSORS = [f"S{str(i).zfill(3)}" for i in range(1, 11)]

# small per-sensor bias to simulate hardware variation
SENSOR_BIAS = {s: {"pressure": np.random.uniform(-0.1, 0.1),
                   "flow":     np.random.uniform(-3.0, 3.0),
                   "temp":     np.random.uniform(-0.5, 0.5)} for s in SENSORS}


def clipped_normal(mean, std, lo, hi, size=1):
    """Sample from a Gaussian, hard-clipped to [lo, hi]."""
    samples = np.random.normal(mean, std, size)
    return np.clip(samples, lo, hi)


def generate_normal_segment(n_steps, prev_pressure=None, prev_flow=None, prev_temp=None, sensor=None):
    """
    Generate n_steps of normal readings with AR(1) temporal correlation.
    AR(1): x_t = phi * x_{t-1} + (1-phi) * mean + noise
    """
    p = CLASS_PARAMS["normal"]
    phi = 0.6  # autocorrelation coefficient

    pressures, flows, temps = [], [], []
    pr = prev_pressure if prev_pressure else p["pressure"]["mean"]
    fl = prev_flow     if prev_flow     else p["flow"]["mean"]
    tp = prev_temp     if prev_temp     else p["temp"]["mean"]

    bias_p = SENSOR_BIAS[sensor]["pressure"] if sensor else 0
    bias_f = SENSOR_BIAS[sensor]["flow"]     if sensor else 0
    bias_t = SENSOR_BIAS[sensor]["temp"]     if sensor else 0

    for _ in range(n_steps):
        pr = phi * pr + (1 - phi) * p["pressure"]["mean"] + np.random.normal(0, p["pressure"]["std"] * 0.4)
        fl = phi * fl + (1 - phi) * p["flow"]["mean"]     + np.random.normal(0, p["flow"]["std"]     * 0.4)
        tp = phi * tp + (1 - phi) * p["temp"]["mean"]     + np.random.normal(0, p["temp"]["std"]     * 0.4)

        pressures.append(np.clip(pr + bias_p, p["pressure"]["min"], p["pressure"]["max"]))
        flows.append(    np.clip(fl + bias_f, p["flow"]["min"],     p["flow"]["max"]))
        temps.append(    np.clip(tp + bias_t, p["temp"]["min"],     p["temp"]["max"]))

    return pressures, flows, temps


def generate_event_segment(event_type, n_steps, ramp_steps=4, sensor=None):
    """
    Generate an anomaly event with a realistic ramp-in and ramp-out transition.
    Pressure ramps DOWN before the event, Flow ramps UP.
    """
    p_norm  = CLASS_PARAMS["normal"]
    p_event = CLASS_PARAMS[event_type]

    bias_p = SENSOR_BIAS[sensor]["pressure"] if sensor else 0
    bias_f = SENSOR_BIAS[sensor]["flow"]     if sensor else 0
    bias_t = SENSOR_BIAS[sensor]["temp"]     if sensor else 0

    pressures, flows, temps = [], [], []

    for i in range(n_steps):
        # ramp-in: gradually shift from normal → event values
        if i < ramp_steps:
            alpha = i / ramp_steps  # 0 → 1
        # ramp-out: gradually return
        elif i >= n_steps - ramp_steps:
            alpha = (n_steps - i) / ramp_steps  # 1 → 0
        else:
            alpha = 1.0

        # interpolate mean
        p_mean = (1 - alpha) * p_norm["pressure"]["mean"] + alpha * p_event["pressure"]["mean"]
        f_mean = (1 - alpha) * p_norm["flow"]["mean"]     + alpha * p_event["flow"]["mean"]
        t_mean = (1 - alpha) * p_norm["temp"]["mean"]     + alpha * p_event["temp"]["mean"]

        pr = clipped_normal(p_mean, p_event["pressure"]["std"] * 0.6,
                            p_event["pressure"]["min"], p_event["pressure"]["max"])[0]
        fl = clipped_normal(f_mean, p_event["flow"]["std"] * 0.6,
                            p_event["flow"]["min"], p_event["flow"]["max"])[0]
        tp = clipped_normal(t_mean, p_event["temp"]["std"] * 0.6,
                            p_event["temp"]["min"], p_event["temp"]["max"])[0]

        pressures.append(pr + bias_p)
        flows.append(    fl + bias_f)
        temps.append(    tp + bias_t)

    return pressures, flows, temps


def generate_dataset(
    n_rows=5000,
    leak_ratio=0.15,
    burst_ratio=0.08,
    start_time=datetime(2024, 1, 1),
    interval_minutes=5,
    event_min_len=3,
    event_max_len=12,
):
    """
    Main generator function.

    Parameters
    ----------
    n_rows         : total number of rows to generate
    leak_ratio     : fraction of rows that are Leak events
    burst_ratio    : fraction of rows that are Burst events
    start_time     : starting timestamp
    interval_minutes: sampling interval
    event_min_len  : min consecutive steps per anomaly event
    event_max_len  : max consecutive steps per anomaly event

    Returns
    -------
    pd.DataFrame with same schema as original dataset
    """
    timestamps = [start_time + timedelta(minutes=interval_minutes * i) for i in range(n_rows)]

    # ── Build an event plan ───────────────────────────────────────────────
    # We decide upfront which index ranges are events vs normal
    labels = np.zeros(n_rows, dtype=int)  # 0=normal, 1=leak, 2=burst
    i = 0
    n_leak_target  = int(n_rows * leak_ratio)
    n_burst_target = int(n_rows * burst_ratio)
    n_leak_placed, n_burst_placed = 0, 0

    while i < n_rows:
        # decide randomly whether to place an event at position i
        roll = random.random()
        if roll < leak_ratio and n_leak_placed < n_leak_target:
            length = random.randint(event_min_len, event_max_len)
            end = min(i + length, n_rows)
            labels[i:end] = 1
            n_leak_placed += (end - i)
            i = end + random.randint(5, 30)  # gap of normal readings
        elif roll < leak_ratio + burst_ratio and n_burst_placed < n_burst_target:
            length = random.randint(event_min_len, event_max_len)
            end = min(i + length, n_rows)
            labels[i:end] = 2
            n_burst_placed += (end - i)
            i = end + random.randint(5, 30)
        else:
            i += 1

    # ── Generate readings segment by segment ─────────────────────────────
    pressures = np.zeros(n_rows)
    flows     = np.zeros(n_rows)
    temps     = np.zeros(n_rows)
    sensors   = []

    # assign sensors in rotation with some randomness
    i = 0
    prev_p = prev_f = prev_t = None
    while i < n_rows:
        sensor = random.choice(SENSORS)
        current_label = labels[i]

        # find the end of this run
        j = i
        while j < n_rows and labels[j] == current_label:
            j += 1
        seg_len = j - i

        if current_label == 0:
            p_seg, f_seg, t_seg = generate_normal_segment(seg_len, prev_p, prev_f, prev_t, sensor)
        elif current_label == 1:
            p_seg, f_seg, t_seg = generate_event_segment("leak", seg_len, ramp_steps=min(3, seg_len//2+1), sensor=sensor)
        else:
            p_seg, f_seg, t_seg = generate_event_segment("burst", seg_len, ramp_steps=min(3, seg_len//2+1), sensor=sensor)

        pressures[i:j] = p_seg
        flows[i:j]     = f_seg
        temps[i:j]     = t_seg
        sensors.extend([sensor] * seg_len)

        prev_p, prev_f, prev_t = pressures[j-1], flows[j-1], temps[j-1]
        i = j

    df_gen = pd.DataFrame({
        "Timestamp":        timestamps,
        "Sensor_ID":        sensors,
        "Pressure (bar)":   np.round(pressures, 6),
        "Flow Rate (L/s)":  np.round(flows, 6),
        "Temperature (°C)": np.round(temps, 6),
        "Leak Status":      (labels == 1).astype(int),
        "Burst Status":     (labels == 2).astype(int),
    })

    return df_gen


# ── Run the generator ─────────────────────────────────────────────────────
df_generated = generate_dataset(
    n_rows=5000,
    leak_ratio=0.15,
    burst_ratio=0.08,
)
df_generated.to_csv("output/water_leak_generated_5000.csv", index=False)

# ── Verify output ─────────────────────────────────────────────────────────
print("Shape:", df_generated.shape)
print("\nClass distribution:")
print("  Normal:", (df_generated['Leak Status']==0) & (df_generated['Burst Status']==0), )
n = ((df_generated['Leak Status']==0) & (df_generated['Burst Status']==0)).sum()
l = df_generated['Leak Status'].sum()
b = df_generated['Burst Status'].sum()
print(f"  Normal: {n} ({n/5000*100:.1f}%)")
print(f"  Leak:   {l} ({l/5000*100:.1f}%)")
print(f"  Burst:  {b} ({b/5000*100:.1f}%)")
print("\nFeature stats by class:")
print(df_generated.groupby('Leak Status')[['Pressure (bar)', 'Flow Rate (L/s)', 'Temperature (°C)']].mean())
print(df_generated.head(10).to_string())