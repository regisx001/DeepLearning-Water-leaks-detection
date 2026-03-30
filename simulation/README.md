# Realtime Stream Scripts

This folder now contains two scripts only.

## 1) Realtime data generator

File: `simulation/realtime_data_generator.py`

What it does:

- Reads existing CSV data (default: `data/data_clean.csv`)
- Streams rows in timestamp order
- Sleeps between rows based on real timestamp deltas (with speed factor)
- Can loop forever to simulate continuous realtime feed

Quick run:

```bash
python -m simulation.realtime_data_generator --max-rows 20
```

## 2) Realtime predictor + row logger

File: `simulation/realtime_predictor_console.py`

What it does:

- Consumes stream rows from the generator
- Builds rolling windows per sensor
- Predicts leak or no leak using the trained model
- Prints each row and prediction as JSON to console

Quick run:

```bash
python -m simulation.realtime_predictor_console --max-rows 120 --sensor S001
```

## Notes

- Model path fallback supports both naming styles:
  `best_model.keras` and `bestmodel.keras`
- Threshold path fallback supports both naming styles:
  `best_threshold.npy` and `bestthreshold.npy`
- Training data fallback supports both naming styles:
  `data_engineered.csv` and `dataengineered.csv`
- The first predictions start after window warmup (`window_size`, default 20)
