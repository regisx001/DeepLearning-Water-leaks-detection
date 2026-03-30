Here is the **complete system documentation** for your Water Leak Detection project:

***
# 📘 Water Leak Detection System — Full Technical Documentation
**Project:** IoT-based Water Leak Detection with Automated Valve Shutdown
**Stack:** Arduino Uno · Raspberry Pi · CNN-BiLSTM-Attention · Python
**Version:** 1.0 · March 2026

***
## 1. System Overview
This system detects water pipe leaks in real time using sensor data, a deep learning model running on a Raspberry Pi, and an Arduino that physically closes a solenoid valve when a leak is confirmed. The pipeline runs entirely on-device — no cloud dependency — making it suitable for industrial and residential IoT deployments. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/84862668/e08fc2f2-dbc6-4896-bd1c-774928aadf3a/04-Model-Training-Evaluation.ipynb)

***
## 2. System Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                     SENSORS LAYER                            │
│   MPX5700AP (Pressure)  ·  YF-S201 (Flow)  ·  LM35 (Temp)  │
└───────────────────────────┬──────────────────────────────────┘
                            │ Analog / Digital signals
┌───────────────────────────▼──────────────────────────────────┐
│                    ARDUINO UNO / MEGA                        │
│  · Reads sensors every 500ms                                 │
│  · Formats CSV: "S001,3.215,124.5,17.3\n"                   │
│  · Listens for CLOSE/OPEN commands                           │
│  · Controls relay → solenoid valve                           │
│  · Triggers buzzer + LED alerts                              │
└───────────────────────────┬──────────────────────────────────┘
                            │ Serial USB (9600 baud)
┌───────────────────────────▼──────────────────────────────────┐
│                    RASPBERRY PI                              │
│  · Reads serial stream (pyserial)                            │
│  · Rolling window buffer (20 timesteps)                      │
│  · Feature engineering (pressure_delta, flow_zscore, ratio)  │
│  · StandardScaler (fit on training data)                     │
│  · CNN-BiLSTM-Attention inference (23K params)               │
│  · Decision threshold: prob > 0.0333                         │
│  · 2× consecutive windows → confirmed leak → CLOSE command   │
└───────────────────────────┬──────────────────────────────────┘
                            │ Serial command "CLOSE\n"
┌───────────────────────────▼──────────────────────────────────┐
│                   ACTUATOR LAYER                             │
│   Solenoid Valve (12V NC)  ·  Red LED  ·  Buzzer alert       │
└──────────────────────────────────────────────────────────────┘
```
---
## 3. Hardware Components
### 3.1 Sensors
| Sensor | Model | Type | Arduino Pin | Purpose |
|---|---|---|---|---|
| **Pressure** | MPX5700AP | Analog (0.5–4.5V) | `A0` | Detects pressure drops (primary leak signal) |
| **Flow Rate** | YF-S201 | Digital pulse | `D2` | Detects flow anomalies (secondary signal) |
| **Temperature** | LM35 | Analog (10mV/°C) | `A1` | Ambient context (weak leak signal) |

#### Sensor Calibration

**Pressure (MPX5700AP):**
```
Voltage range: 0.5V (0 bar) → 4.5V (5 bar)
Formula: pressure = (voltage - 0.5) × (5.0 / 4.0)
```

**Flow (YF-S201):**
```
Calibration factor: 7.5 pulses per litre
Formula: flow (L/s) = (pulse_count / 7.5) / 60
```

**Temperature (LM35):**
```
Formula: temperature (°C) = voltage × 100
```

***
### 3.2 Arduino Uno / Mega
| Role | Detail |
|---|---|
| Microcontroller | ATmega328P (Uno) or ATmega2560 (Mega) |
| Operating voltage | 5V |
| Analog resolution | 10-bit ADC (0–1023 → 0–5V) |
| Serial baudrate | 9600 baud |
| Sample interval | 500ms per reading |
| Flow interrupt | `digitalPinToInterrupt(D2)` — RISING edge |

#### Pin Wiring Table

| Component | Arduino Pin | Mode | Notes |
|---|---|---|---|
| Pressure sensor | `A0` | Analog IN | 0.5–4.5V output |
| Flow sensor | `D2` | Digital IN | Interrupt-capable |
| Temperature | `A1` | Analog IN | LM35 |
| Relay module | `D7` | Digital OUT | LOW = valve closes |
| Buzzer | `D8` | Digital OUT | Active buzzer |
| Red LED | `D9` | Digital OUT | Leak alert |
| Green LED | `D10` | Digital OUT | Normal state |
| RPi TX | `RX (D0)` | Serial IN | Commands from RPi |

***
### 3.3 Relay + Solenoid Valve
| Component | Spec |
|---|---|
| Relay module | 5V single-channel, LOW-active |
| Solenoid valve | 12V DC, Normally Closed (NC) |
| Valve behavior | NC = closed when unpowered — **safe default** |
| Power supply | Separate 12V supply for solenoid (NOT from Arduino) |

> ⚠️ **Always use a separate 12V power supply for the solenoid.** Never power it directly from the Arduino 5V rail.

***
### 3.4 Raspberry Pi
| Spec | Value |
|---|---|
| Recommended model | Raspberry Pi 4B (2GB+) or Pi 3B+ |
| OS | Raspberry Pi OS (64-bit) |
| Python | 3.10+ |
| Serial port | `/dev/ttyUSB0` or `/dev/ttyACM0` |
| Model file | `models/bestmodel.keras` (92.88 KB) |
| Threshold file | `models/bestthreshold.npy` (0.0333) |

***
## 4. Software Architecture
### 4.1 Arduino Sketch — Responsibilities
```
setup()
  ├── Serial.begin(9600)
  ├── Pin modes (relay, LEDs, buzzer, flow interrupt)
  └── openValve() → green LED ON

loop() every tick
  ├── [every 1s]    compute flowRate from pulse ISR
  ├── [every 500ms] readPressure() + readTemperature()
  │                 → Serial.println("S001,p,f,t")
  └── [on Serial]   parse command:
                    CLOSE → closeValve() → relay LOW → 3 beeps
                    OPEN  → openValve()  → relay HIGH
                    STATUS → "STATUS:CLOSED/OPEN"
                    PING   → "PONG"
```
### 4.2 Python Inference Script — Responsibilities
```
LeakDetector.__init__()
  ├── Load CNN-BiLSTM-Attention model (.keras)
  ├── Load decision threshold (0.0333)
  ├── Fit StandardScaler on training data
  ├── Open serial port
  └── Build FeatureBuffer per sensor

LeakDetector.run() — main loop
  ├── Read serial line → parse "SID,p,f,t"
  ├── Add reading to FeatureBuffer
  ├── Every 5 readings → run_inference()
  │     ├── get_window() → (20, 6) array
  │     ├── StandardScaler.transform()
  │     ├── model.predict() → probability
  │     └── return prob
  └── _handle_prediction()
        ├── prob > 0.0333 → leak_counter++
        ├── leak_counter >= 2 → valve.close() → "CLOSE\n"
        └── normal → leak_counter = 0
```
### 4.3 Serial Communication Protocol
```
Direction         Message             Meaning
─────────────────────────────────────────────────────────────
Arduino → RPi     S001,3.215,124.5,17.3   Sensor reading (CSV)
Arduino → RPi     LOG:VALVE_CLOSED         State change log
Arduino → RPi     ACK:VALVE_CLOSED         Command acknowledged
Arduino → RPi     PONG                     Heartbeat response

RPi → Arduino     CLOSE\n                  Close valve command
RPi → Arduino     OPEN\n                   Open valve command
RPi → Arduino     STATUS\n                 Query valve state
RPi → Arduino     PING\n                   Heartbeat check
```

***
## 5. Machine Learning Model
### 5.1 Architecture — CNN-BiLSTM-Attention
```
Input (20 timesteps × 6 features)
    ↓
Conv1D(32, kernel=3) + BatchNorm + ReLU
    ↓
Conv1D(32, kernel=5) + BatchNorm + ReLU
    ↓
Dropout(0.30)
    ↓
BiLSTM(32 units per direction = 64 total) + Dropout(0.25)
    ↓
Temporal Attention (Softmax over 20 timesteps)
    ↓
Context vector (weighted sum)
    ↓
Dense(16) + ReLU + Dropout(0.40)
    ↓
Dense(1) + Sigmoid → leak probability
```

Total parameters: **23,778** (92.88 KB) [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/84862668/e08fc2f2-dbc6-4896-bd1c-774928aadf3a/04-Model-Training-Evaluation.ipynb)
### 5.2 Features (in order)
| # | Feature | Description | Key Signal |
|---|---|---|---|
| 1 | `Pressure bar` | Raw pressure reading | Drops during leaks |
| 2 | `Flow Rate Ls` | Raw flow rate (L/s) | Rises during leaks |
| 3 | `Temperature C` | Ambient temperature | Weak signal |
| 4 | `pressure_delta` | Temporal pressure diff | Sudden drop = leak |
| 5 | `flow_zscore` | Sensor-normalized flow | Cross-sensor anomaly |
| 6 | `pf_ratio` | Pressure ÷ Flow | Drops during leaks |
### 5.3 Windowing Strategy
| Parameter | Value |
|---|---|
| Window size | 20 timesteps |
| Step size | 5 timesteps |
| Window label | 1 if ANY timestep is a leak |
| Train windows | 863 (38.1% leak) |
| Test windows | 221 (41.2% leak) |
### 5.4 Model Performance
| Metric | CNN-BiLSTM | Random Forest | Logistic Reg. |
|---|---|---|---|
| **AUC-ROC** | 0.9412 | **0.9710** | 0.9524 |
| **AUC-PR** | 0.9583 | **0.9652** | 0.9595 |
| **Accuracy** | **94.57%** | 90.50% | 92.31% |
| **Precision** | **96.47%** | 94.87% | 94.05% |
| **Recall** | **90.11%** | 81.32% | 86.81% |
| **F1-Score** | **93.18%** | 87.57% | 90.29% |
| False Alarm Rate | **2.31%** | 3.08% | 3.85% |

> The CNN-BiLSTM-Attention model was chosen for deployment due to its superior **Recall** (fewest missed leaks) and lowest **False Alarm Rate** — the two most critical metrics for a safety-first valve control system. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/84862668/ddf959ed-e59f-464c-b7c2-c9502d476e54/05-Train-benchmark-models.ipynb)

***
## 6. Inference Latency
| Stage | Time | Notes |
|---|---|---|
| Sensor sampling | 500ms | Arduino sample interval (configurable) |
| Serial TX | ~10ms | USB at 9600 baud |
| Feature engineering | ~5ms | Pandas ops on 20-row buffer |
| Scaling | ~2ms | StandardScaler transform |
| Model inference | ~80ms | TensorFlow on RPi CPU |
| Decision + command | ~3ms | Threshold compare + serial write |
| **Total** | **~600ms** | Per reading cycle |

> Leak confirmation requires **2 consecutive positive windows × 5-step interval × 500ms** = approximately **5 seconds** from leak onset to valve closure.

***
## 7. Installation & Setup
### 7.1 Raspberry Pi Setup
```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Python dependencies
pip install tensorflow numpy pandas scikit-learn pyserial

# 3. Give serial port permissions
sudo usermod -aG dialout $USER

# 4. Copy project files
mkdir ~/leak_detection
cp inference.py config.py ~/leak_detection/
cp -r models/ data/ ~/leak_detection/

# 5. Run manually (test)
cd ~/leak_detection
python3 inference.py

# 6. Install as systemd service (auto-start on boot)
sudo cp leak-detection.service /etc/systemd/system/
sudo systemctl enable leak-detection
sudo systemctl start leak-detection
```
### 7.2 Arduino Setup
```
1. Open Arduino IDE
2. Paste the sketch from Section 3.1
3. Set SENSOR_ID to match your device (S001–S010)
4. Verify FLOW_CALIBRATION matches your sensor model
5. Verify pressure voltage range matches your sensor datasheet
6. Upload via USB
7. Open Serial Monitor @ 9600 baud to verify "READY" message
```
### 7.3 Wiring Diagram (Text)
```
                    ARDUINO UNO
                 ┌─────────────┐
  MPX5700AP ─── │ A0          │
  YF-S201   ─── │ D2 (INT0)   │
  LM35      ─── │ A1          │
                │             │
  Relay IN  ─── │ D7          │ ──→ Relay → 12V Solenoid Valve
  Buzzer    ─── │ D8          │
  Red LED   ─── │ D9          │
  Green LED ─── │ D10         │
                │             │
  RPi (USB) ─── │ USB         │ (Serial RX/TX)
                └─────────────┘
```

***
## 8. Configuration Reference (`config.py`)
| Parameter | Default | Description |
|---|---|---|
| `SERIAL_PORT` | `/dev/ttyUSB0` | Arduino USB serial port |
| `SERIAL_BAUDRATE` | `9600` | Must match Arduino sketch |
| `WINDOW_SIZE` | `20` | Timesteps per inference window |
| `STEP_SIZE` | `5` | Readings between each inference call |
| `LEAK_CONFIRMATION_COUNT` | `2` | Consecutive positive windows to confirm |
| `REOPEN_COOLDOWN_SEC` | `30` | Seconds before auto-reopen attempt |
| `FEATURES` | `[6 features]` | Must match training order exactly |

***
## 9. Safety & Operational Notes
- **Solenoid is Normally Closed (NC):** In case of power failure, the valve closes automatically — safe default [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/84862668/e08fc2f2-dbc6-4896-bd1c-774928aadf3a/04-Model-Training-Evaluation.ipynb)
- **Graceful shutdown:** On `SIGTERM` or `KeyboardInterrupt`, the Python script sends `OPEN` before exiting
- **Cooldown period:** After closing, the valve won't auto-reopen for 30 seconds (prevents valve flapping)
- **Confirmation window:** 2 consecutive leak detections required (reduces false trips)
- **Log file:** All events written to `leak_detection.log` for audit/debugging
- **Unknown sensors:** New sensor IDs seen at runtime get a default FeatureBuffer (no crash)

***
## 10. Project File Structure
```
leak_detection/
├── inference.py              # Main RPi inference loop
├── config.py                 # All configurable parameters
├── requirements.txt          # Python dependencies
├── leak-detection.service    # systemd service file
├── models/
│   ├── bestmodel.keras       # Trained CNN-BiLSTM-Attention (93KB)
│   └── bestthreshold.npy     # Decision threshold (0.0333)
├── data/
│   └── dataengineered.csv    # Training data (scaler fitting + sensor stats)
├── arduino/
│   └── leak_detection.ino    # Arduino sketch
└── leak_detection.log        # Runtime log (auto-created)
```