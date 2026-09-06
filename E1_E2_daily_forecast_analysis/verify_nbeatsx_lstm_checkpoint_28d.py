"""Verify whether the retained proposed-model checkpoint reproduces 222.py.

No fitting or tuning is performed. The comparison is point by point over the
paper's final 28-day window using true history at every step.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = Path(r"D:\Research-Haozhi Liao\NBeatsx-LSTM\NBeatsx-LSTM")
CHECKPOINT = SOURCE_ROOT / "model" / "model_nbeats_lstm_model.pth"
SOURCE_222 = Path(r"D:\NBeatsx-LSTM\222.py")
DATA_DIR = Path(r"F:\nbeatsx-lstm\data")
DATA_FILE = "imputed_data_KNN_1.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def retained_predictions() -> np.ndarray:
    tree = ast.parse(SOURCE_222.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == "nbeatsx_lstm":
            return np.asarray(ast.literal_eval(node.value), dtype=float)
    raise RuntimeError("nbeatsx_lstm array not found in 222.py")


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

    forecast_length = 28
    history_length = 365
    predictions: list[float] = []
    for i in range(forecast_length):
        start = -(history_length + forecast_length) + i
        end_value = -forecast_length + i
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

    retained = retained_predictions()
    predicted = np.asarray(predictions)
    difference = predicted - retained
    dates = pd.to_datetime(y_df["ds"].iloc[-forecast_length:]).dt.strftime("%Y-%m-%d").to_numpy()
    comparison = pd.DataFrame(
        {
            "date": dates,
            "checkpoint_prediction": predicted,
            "retained_222_prediction": retained,
            "difference": difference,
            "absolute_difference": np.abs(difference),
        }
    )
    comparison.to_csv(HERE / "nbeatsx_lstm_checkpoint_pointwise_check_28d.csv", index=False, float_format="%.10f")
    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "source_root": str(SOURCE_ROOT),
        "device": str(device),
        "history_protocol": "true observed history supplied separately for each forecast date",
        "training_or_tuning_performed": False,
        "n": int(len(predicted)),
        "max_absolute_difference": float(np.max(np.abs(difference))),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "exact_to_1e_6": bool(np.allclose(predicted, retained, rtol=0, atol=1e-6)),
        "exact_to_1e_4": bool(np.allclose(predicted, retained, rtol=0, atol=1e-4)),
    }
    (HERE / "nbeatsx_lstm_checkpoint_verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    main()
