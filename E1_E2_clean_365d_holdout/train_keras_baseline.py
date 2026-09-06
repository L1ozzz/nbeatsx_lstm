"""Train a leakage-controlled LSTM or TCN baseline for the clean 365-day test.

The final 365 rows are never used for fitting scalers, fitting weights, early
stopping, or model selection. The preceding 730 rows are the validation period.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
DATA = PROJECT / "data" / "imputed_data_KNN_1.csv"
TRAIN_END = 7874
VAL_END = 8604
TEST_END = 8969
SEQUENCE_LENGTH = 7
E4_MISSING_RUNS = [(model_name, seed) for seed in (2, 3) for model_name in ("lstm", "tcn")]


def metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    error = actual - forecast
    mse = float(np.mean(error**2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(np.mean(np.abs(error) / ((np.abs(actual) + np.abs(forecast)) / 2)) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def build_sequences(features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values, y_values, target_indices = [], [], []
    for target_index in range(SEQUENCE_LENGTH, len(features)):
        x_values.append(features[target_index - SEQUENCE_LENGTH : target_index])
        y_values.append(target[target_index])
        target_indices.append(target_index)
    return np.asarray(x_values), np.asarray(y_values), np.asarray(target_indices)


def build_model(name: str, num_features: int) -> Any:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential

    if name == "lstm":
        model = Sequential(
            [
                tf.keras.Input(shape=(SEQUENCE_LENGTH, num_features)),
                LSTM(256, activation="tanh", return_sequences=True),
                Dropout(0.2),
                LSTM(256, activation="tanh"),
                Dropout(0.2),
                Dense(1),
            ]
        )
        batch_size = 128
    elif name == "tcn":
        from tcn_keras3_compat import TCN

        model = Sequential(
            [
                tf.keras.Input(shape=(SEQUENCE_LENGTH, num_features)),
                TCN(
                    dilations=[1, 2, 4, 8, 16],
                    nb_filters=32,
                    kernel_size=3,
                    dropout_rate=0.2,
                    return_sequences=True,
                ),
                Dropout(0.2),
                TCN(
                    dilations=[1, 2, 4, 8, 16],
                    nb_filters=32,
                    kernel_size=3,
                    dropout_rate=0.2,
                    return_sequences=False,
                ),
                Dropout(0.2),
                Dense(1),
            ]
        )
        batch_size = 32
    else:
        raise ValueError(name)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae", metrics=["mae"])
    model._registered_batch_size = batch_size
    return model


def train_one(args: argparse.Namespace) -> None:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    for directory in (OUT / "runs", OUT / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(DATA)
    raw["Day(Local_Date)"] = pd.to_datetime(raw["Day(Local_Date)"])
    if len(raw) != TEST_END:
        raise RuntimeError(f"Expected {TEST_END} rows, found {len(raw)}")
    dates = raw["Day(Local_Date)"].copy()
    numeric = raw.drop(columns=["Day(Local_Date)", "Season"])
    target_raw = numeric["SoilM(%)"].to_numpy(dtype=float)
    feature_raw = numeric.drop(columns=["SoilM(%)"]).to_numpy(dtype=float)

    scaler_x = MinMaxScaler().fit(feature_raw[:TRAIN_END])
    scaler_y = MinMaxScaler().fit(target_raw[:TRAIN_END].reshape(-1, 1))
    features_scaled = scaler_x.transform(feature_raw)
    target_scaled = scaler_y.transform(target_raw.reshape(-1, 1)).reshape(-1)
    x_all, y_all, indices = build_sequences(features_scaled, target_scaled)
    train = indices < TRAIN_END
    validation = (indices >= TRAIN_END) & (indices < VAL_END)
    test = indices >= VAL_END

    model = build_model(args.model, x_all.shape[2])
    callback = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, mode="min")
    started = time.time()
    history = model.fit(
        x_all[train],
        y_all[train],
        validation_data=(x_all[validation], y_all[validation]),
        epochs=100,
        batch_size=model._registered_batch_size,
        callbacks=[callback],
        shuffle=False,
        verbose=2,
    )
    elapsed = time.time() - started

    run_id = f"{args.model}_seed{args.seed}"
    checkpoint = OUT / "checkpoints" / f"{run_id}.keras"
    model.save(checkpoint)
    pd.DataFrame(history.history).assign(epoch=np.arange(1, len(history.history["loss"]) + 1)).to_csv(
        OUT / "runs" / f"{run_id}_history.csv", index=False
    )

    split_predictions = {}
    metric_rows = []
    for split_name, selector in [("validation", validation), ("test", test)]:
        scaled_forecast = model.predict(x_all[selector], batch_size=1024, verbose=0).reshape(-1, 1)
        forecast = scaler_y.inverse_transform(scaled_forecast).reshape(-1)
        actual = target_raw[indices[selector]]
        frame = pd.DataFrame(
            {
                "date": dates.iloc[indices[selector]].dt.strftime("%Y-%m-%d").to_numpy(),
                "model": args.model,
                "variant": "released_fixed_config",
                "seed": args.seed,
                "actual": actual,
                "forecast": forecast,
                "split": split_name,
            }
        )
        path = OUT / "runs" / f"{run_id}_{split_name}_forecasts.csv"
        frame.to_csv(path, index=False, float_format="%.10f")
        split_predictions[split_name] = str(path)
        metric_rows.append({"split": split_name, **metrics(actual, forecast)})

    metadata = {
        "run_id": run_id,
        "model": args.model,
        "variant": "released_fixed_config",
        "seed": args.seed,
        "train_rows_raw": TRAIN_END,
        "validation_rows": VAL_END - TRAIN_END,
        "test_rows": TEST_END - VAL_END,
        "train_end_date": dates.iloc[TRAIN_END - 1].strftime("%Y-%m-%d"),
        "validation_end_date": dates.iloc[VAL_END - 1].strftime("%Y-%m-%d"),
        "test_start_date": dates.iloc[VAL_END].strftime("%Y-%m-%d"),
        "scalers_fit_end_index_exclusive": TRAIN_END,
        "shuffle": False,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_mae_scaled": float(min(history.history["val_loss"])),
        "elapsed_seconds": elapsed,
        "checkpoint": str(checkpoint),
        "forecast_files": split_predictions,
        "metrics": metric_rows,
        "test_used_for_training_or_selection": False,
    }
    (OUT / "runs" / f"{run_id}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    tf.keras.backend.clear_session()


def run_e4_missing_seeds() -> None:
    """Run the missing seed-2/seed-3 Keras baselines without a shell wrapper."""
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for model_name, seed in E4_MISSING_RUNS:
        run_id = f"{model_name}_seed{seed}"
        metadata_path = OUT / "runs" / f"{run_id}_metadata.json"
        if metadata_path.exists():
            print(f"SKIP completed {run_id}", flush=True)
            continue
        print(f"START {run_id}", flush=True)
        train_one(argparse.Namespace(model=model_name, seed=seed))
        print(f"COMPLETE {run_id}", flush=True)
    print("ALL_E4_KERAS_RUNS_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct execution runs the missing E4 LSTM/TCN seed experiments."
    )
    parser.add_argument("--model", choices=["lstm", "tcn"])
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    if args.model is None and args.seed is None:
        run_e4_missing_seeds()
        return
    if args.model is None or args.seed is None:
        parser.error("single-run mode requires both --model and --seed")
    train_one(args)


if __name__ == "__main__":
    main()
