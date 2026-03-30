// ============================================================
//  Water Leak Detection - Arduino Sketch
//  Sensors : Pressure, Flow Rate, Temperature
//  Actuator: Solenoid Valve (via relay) + Buzzer + LED
//  Comm    : Serial (USB/UART) with Raspberry Pi
// ============================================================

#include <Arduino.h>

// ── Pin Configuration ────────────────────────────────────────
#define PRESSURE_PIN      A0   // Analog pressure sensor
#define FLOW_PIN          2    // Digital flow sensor (interrupt)
#define TEMP_PIN          A1   // Analog temperature sensor (e.g. LM35)

#define VALVE_RELAY_PIN   7    // Relay IN → solenoid valve (LOW = active)
#define BUZZER_PIN        8    // Buzzer
#define LED_RED_PIN       9    // Red LED  (leak alert)
#define LED_GREEN_PIN     10   // Green LED (normal)

// ── Flow Sensor ──────────────────────────────────────────────
volatile unsigned long pulseCount = 0;
float flowRate = 0.0;
// Calibration factor (pulses per litre) — adjust to your sensor model
// Common value for YF-S201: 7.5
#define FLOW_CALIBRATION  7.5

// ── Sampling Configuration ───────────────────────────────────
#define SAMPLE_INTERVAL_MS  500   // Send reading every 500ms
#define SENSOR_ID           "S001" // Change per device

// ── State ────────────────────────────────────────────────────
bool valveClosed = false;
unsigned long lastSampleTime = 0;
unsigned long lastFlowCalcTime = 0;

// ── ISR: count flow pulses ────────────────────────────────────
void IRAM_ATTR flowPulseISR() {
  pulseCount++;
}

// ============================================================
void setup() {
  Serial.begin(9600);

  // Pin modes
  pinMode(VALVE_RELAY_PIN, OUTPUT);
  pinMode(BUZZER_PIN,      OUTPUT);
  pinMode(LED_RED_PIN,     OUTPUT);
  pinMode(LED_GREEN_PIN,   OUTPUT);
  pinMode(FLOW_PIN,        INPUT_PULLUP);

  // Default state: valve OPEN, green LED ON
  openValve();

  // Attach interrupt for flow sensor
  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), flowPulseISR, RISING);

  Serial.println("READY");
}

// ============================================================
void loop() {
  unsigned long now = millis();

  // ── 1. Compute flow rate every second ──────────────────────
  if (now - lastFlowCalcTime >= 1000) {
    noInterrupts();
    unsigned long pulses = pulseCount;
    pulseCount = 0;
    interrupts();

    // L/s = (pulses/s) / calibration
    flowRate = (pulses / FLOW_CALIBRATION) / 60.0;
    lastFlowCalcTime = now;
  }

  // ── 2. Send sensor reading to Raspberry Pi ──────────────────
  if (now - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    float pressure = readPressure();
    float temperature = readTemperature();

    // Format: SENSOR_ID,pressure,flowRate,temperature
    // Example: S001,3.21,124.50,17.30
    Serial.print(SENSOR_ID);
    Serial.print(",");
    Serial.print(pressure, 3);
    Serial.print(",");
    Serial.print(flowRate, 3);
    Serial.print(",");
    Serial.println(temperature, 3);

    lastSampleTime = now;
  }

  // ── 3. Listen for commands from Raspberry Pi ─────────────────
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "CLOSE") {
      closeValve();
      Serial.println("ACK:VALVE_CLOSED");
    }
    else if (cmd == "OPEN") {
      openValve();
      Serial.println("ACK:VALVE_OPENED");
    }
    else if (cmd == "STATUS") {
      Serial.print("STATUS:");
      Serial.println(valveClosed ? "CLOSED" : "OPEN");
    }
    else if (cmd == "PING") {
      Serial.println("PONG");
    }
  }
}

// ============================================================
//  Valve Control
// ============================================================
void closeValve() {
  digitalWrite(VALVE_RELAY_PIN, LOW);   // Relay ON → valve closes
  valveClosed = true;
  digitalWrite(LED_RED_PIN,   HIGH);
  digitalWrite(LED_GREEN_PIN, LOW);
  alertBuzzer(3);                        // 3 short beeps
  Serial.println("LOG:VALVE_CLOSED");
}

void openValve() {
  digitalWrite(VALVE_RELAY_PIN, HIGH);  // Relay OFF → valve opens
  valveClosed = false;
  digitalWrite(LED_RED_PIN,   LOW);
  digitalWrite(LED_GREEN_PIN, HIGH);
  digitalWrite(BUZZER_PIN,    LOW);
  Serial.println("LOG:VALVE_OPENED");
}

// ============================================================
//  Sensor Readers
// ============================================================

// Pressure sensor (0–5 bar range, 0.5V–4.5V output — adjust to yours)
float readPressure() {
  int raw = analogRead(PRESSURE_PIN);
  float voltage = raw * (5.0 / 1023.0);
  // Linear mapping: 0.5V = 0 bar, 4.5V = 5 bar
  float pressure = (voltage - 0.5) * (5.0 / 4.0);
  return max(0.0f, pressure);
}

// Temperature sensor LM35: 10mV per °C
float readTemperature() {
  int raw = analogRead(TEMP_PIN);
  float voltage = raw * (5.0 / 1023.0);
  return voltage * 100.0;  // LM35: 10mV/°C → *100
}

// ============================================================
//  Buzzer: n short beeps
// ============================================================
void alertBuzzer(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(150);
    digitalWrite(BUZZER_PIN, LOW);
    delay(100);
  }
}