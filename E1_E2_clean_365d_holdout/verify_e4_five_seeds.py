"""Independent structural and metric check for the E4 five-seed package."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(r"F:\nbeatsx-lstm\E1_E2_clean_365d_holdout")
OUT = ROOT / "E4_five_seed_results"
METRICS = ["MAE", "MSE", "RMSE", "sMAPE", "R2"]


def metric_values(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame.actual.to_numpy(dtype=float)
    forecast = frame.forecast.to_numpy(dtype=float)
    error = actual - forecast
    mse = float(np.mean(error**2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(np.mean(np.abs(error) / ((np.abs(actual) + np.abs(forecast)) / 2)) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def main() -> None:
    forecasts = pd.read_csv(OUT / "forecasts_365d_all_five_seeds.csv", dtype={"date": str})
    stored = pd.read_csv(OUT / "metrics_by_seed.csv")
    audit = pd.read_csv(OUT / "checkpoint_and_forecast_audit.csv")
    expected_models = {"LSTM", "TCN", "NBEATSx", "NBeatsx_LSTM", "Persistence"}
    if set(forecasts.model) != expected_models:
        raise RuntimeError("Unexpected model set")
    if len(forecasts) != 7665 or len(stored) != 21 or len(audit) != 20:
        raise RuntimeError("Unexpected evidence row count")
    if forecasts[forecasts.model != "Persistence"].groupby(["model", "seed"]).size().ne(365).any():
        raise RuntimeError("A learned-model run does not contain 365 rows")
    if len(forecasts[forecasts.model == "Persistence"]) != 365:
        raise RuntimeError("Persistence does not contain 365 rows")

    recalculated = []
    for (model, variant, seed), frame in forecasts.groupby(["model", "variant", "seed"], sort=False):
        row = {"model": model, "variant": variant, "seed": seed}
        row.update(metric_values(frame))
        recalculated.append(row)
    recalculated = pd.DataFrame(recalculated)
    merged = stored.merge(recalculated, on=["model", "variant", "seed"], suffixes=("_stored", "_recalc"))
    differences = []
    for metric in METRICS:
        differences.extend(np.abs(merged[f"{metric}_stored"] - merged[f"{metric}_recalc"]).tolist())
    max_difference = float(max(differences))
    if max_difference > 1e-8:
        raise RuntimeError(f"Metric reproduction failed: {max_difference}")
    if not audit.checkpoint_exists.all() or (audit.checkpoint_bytes <= 0).any():
        raise RuntimeError("Missing or empty checkpoint")

    report = {
        "status": "PASS",
        "forecast_rows": int(len(forecasts)),
        "metric_rows": int(len(stored)),
        "learned_run_audit_rows": int(len(audit)),
        "max_metric_absolute_difference": max_difference,
        "date_start": forecasts.date.min(),
        "date_end": forecasts.date.max(),
        "all_checkpoints_nonempty": True,
        "all_model_seed_forecasts_complete": True,
    }
    (OUT / "independent_verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
