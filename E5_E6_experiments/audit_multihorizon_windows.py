"""Pre-training audit of complete-window counts and date alignment for E5/E6."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import train_e5_e6_nbeatsx_lstm as experiment


def main() -> None:
    _, dataset_class, loader_class, y_df, x_df, s_df = experiment.import_model_family()
    for column in x_df.select_dtypes(include=[object, "category"]).columns:
        encoder = LabelEncoder().fit(x_df[column].iloc[: experiment.TRAIN_END].astype(str))
        x_df[column] = encoder.transform(x_df[column].astype(str))
    target = y_df["y"].to_numpy(dtype=float)
    report = {"status": "PASS", "input_window_days": experiment.INPUT_SIZE, "horizons": {}}
    for horizon in experiment.HORIZONS:
        pre_y, pre_x = y_df.iloc[: experiment.VAL_END].copy(), x_df.iloc[: experiment.VAL_END].copy()
        pre_mask = np.ones(len(pre_y))
        pre_mask[experiment.TRAIN_END :] = 0
        _, train_loader = experiment.make_full_split_loader(
            dataset_class, loader_class, pre_y, pre_x, s_df, pre_mask, horizon, train=True, shuffle=False
        )
        _, validation_loader = experiment.make_full_split_loader(
            dataset_class, loader_class, pre_y, pre_x, s_df, pre_mask, horizon, train=False, shuffle=False
        )
        test_mask = np.ones(len(y_df))
        test_mask[experiment.VAL_END :] = 0
        _, test_loader = experiment.make_full_split_loader(
            dataset_class, loader_class, y_df, x_df, s_df, test_mask, horizon, train=False, shuffle=False
        )
        first = test_loader._nbeats_batch([test_loader.windows_sampling_idx[0]])
        expected = target[experiment.VAL_END : experiment.VAL_END + horizon]
        observed = first["outsample_y"].numpy().reshape(-1)
        if not np.allclose(observed, expected, atol=2e-6, rtol=0):
            raise RuntimeError(f"h={horizon}: first test target is not date-aligned")
        expected_test_origins = 365 - horizon + 1
        if len(test_loader.windows_sampling_idx) != expected_test_origins:
            raise RuntimeError(
                f"h={horizon}: expected {expected_test_origins} complete test origins, "
                f"found {len(test_loader.windows_sampling_idx)}"
            )
        report["horizons"][str(horizon)] = {
            "complete_training_windows": len(train_loader.windows_sampling_idx),
            "complete_validation_origins": len(validation_loader.windows_sampling_idx),
            "complete_test_origins": len(test_loader.windows_sampling_idx),
            "test_forecast_cells": len(test_loader.windows_sampling_idx) * horizon,
            "first_test_target_date": pd.to_datetime(y_df["ds"].iloc[experiment.VAL_END]).strftime("%Y-%m-%d"),
            "last_test_target_date": pd.to_datetime(y_df["ds"].iloc[experiment.TEST_END - 1]).strftime("%Y-%m-%d"),
            "first_target_alignment_max_abs_error": float(np.max(np.abs(observed - expected))),
        }
    output = Path(__file__).with_name("multihorizon_window_audit.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
