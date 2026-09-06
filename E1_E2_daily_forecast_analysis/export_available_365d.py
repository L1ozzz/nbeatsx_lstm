"""Export the recoverable 365-day forecasts without training or tuning.

The retained NBeatsx_LSTM checkpoint was first verified point by point against
the 28 values in 222.py. Persistence is deterministic. The output is explicitly
named *partial* because the exact trained LSTM, TCN and NBEATSx instances have
not been recovered and must not be silently replaced by new training runs.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(r"D:\Research-Haozhi Liao\NBeatsx-LSTM\NBeatsx-LSTM")
CHECKPOINT = SOURCE_ROOT / "model" / "model_nbeats_lstm_model.pth"
DATA_DIR = Path(r"F:\nbeatsx-lstm\data")
DATA_FILE = "imputed_data_KNN_1.csv"


def metric_row(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    error = actual - forecast
    mse = float(np.mean(error**2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(np.mean(np.abs(error) / ((np.abs(actual) + np.abs(forecast)) / 2)) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - np.mean(actual)) ** 2)),
    }


def main() -> None:
    sys.path.insert(0, str(SOURCE_ROOT))
    from nbeatsx_lstm_dataset import TimeSeriesDataset
    from nbeatsx_lstm_epf import EPF
    from nbeatsx_lstm_loader import TimeSeriesLoader

    y_df, x_df, s_df = EPF.load(str(DATA_DIR), DATA_FILE)
    for column in x_df.select_dtypes(include=[object, "category"]).columns:
        x_df[column] = LabelEncoder().fit_transform(x_df[column].astype(str))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(CHECKPOINT, map_location=device)
    model.device = str(device)
    model.model.to(device)

    horizon = 365
    history_length = 365
    predictions: list[float] = []
    internal_log = io.StringIO()
    with contextlib.redirect_stdout(internal_log):
        for i in range(horizon):
            start = -(history_length + horizon) + i
            end_value = -horizon + i
            end = end_value if end_value != 0 else None
            current_y = y_df.iloc[start:end]
            current_x = x_df.iloc[start:end]
            mask = np.ones(len(current_y))
            mask[-1] = 0
            dataset = TimeSeriesDataset(Y_df=current_y, X_df=current_x, S_df=s_df, ts_train_mask=mask)
            loader = TimeSeriesLoader(
                model="nbeats",
                ts_dataset=dataset,
                window_sampling_limit=8,
                offset=0,
                input_size=7,
                output_size=1,
                idx_to_sample_freq=1,
                batch_size=1,
                is_train_loader=False,
                shuffle=False,
            )
            _, y_hat, *_ = model.predict(ts_loader=loader, return_decomposition=False)
            predictions.append(float(np.asarray(y_hat).reshape(-1)[-1]))

    dates = pd.to_datetime(y_df["ds"].iloc[-horizon:]).dt.strftime("%Y-%m-%d").to_numpy()
    actual = y_df["y"].iloc[-horizon:].to_numpy(dtype=float)
    preceding = float(y_df["y"].iloc[-horizon - 1])
    persistence = np.r_[preceding, actual[:-1]]
    output = pd.concat(
        [
            pd.DataFrame({"date": dates, "model": "persistence", "actual": actual, "forecast": persistence}),
            pd.DataFrame(
                {"date": dates, "model": "nbeatsx_lstm", "actual": actual, "forecast": np.asarray(predictions)}
            ),
        ],
        ignore_index=True,
    )
    output.to_csv(HERE / "forecasts_365d_partial.csv", index=False, float_format="%.8f")
    metric_rows = []
    print_lines = []
    for model_name, group in output.groupby("model", sort=False):
        row = metric_row(group["actual"].to_numpy(), group["forecast"].to_numpy())
        metric_rows.append({"model": model_name, **row})
        print_lines.append(
            f"{model_name:14s} MAE={row['MAE']:.4f} MSE={row['MSE']:.4f} "
            f"RMSE={row['RMSE']:.4f} sMAPE={row['sMAPE']:.4f} R2={row['R2']:.4f}"
        )
    pd.DataFrame(metric_rows).to_csv(HERE / "metrics_365d_partial.csv", index=False, float_format="%.10f")
    (HERE / "metrics_365d_partial_printout.txt").write_text("\n".join(print_lines) + "\n", encoding="utf-8")
    (HERE / "export_available_365d_internal_log.txt").write_text(internal_log.getvalue(), encoding="utf-8")
    status = {
        "status": "partial",
        "models_present": ["persistence", "nbeatsx_lstm"],
        "models_missing": ["nbeatsx", "lstm", "tcn"],
        "rows": int(len(output)),
        "dates": int(horizon),
        "training_or_tuning_performed": False,
        "reason_not_named_forecasts_365d_csv": (
            "A canonical five-model forecasts_365d.csv would be scientifically misleading until the "
            "exact trained NBEATSx, LSTM and TCN instances are recovered and validated against 222.py."
        ),
    }
    (HERE / "forecasts_365d_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
