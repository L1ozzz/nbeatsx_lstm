"""Verify checkpoints, date coverage, and metric reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(r"F:\nbeatsx-lstm\E1_E2_clean_365d_holdout")
RUNS = HERE / "runs"
SEEDS = [1, 42, 2024]


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest().upper()


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame.actual.to_numpy(float)
    forecast = frame.forecast.to_numpy(float)
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
    run_ids = []
    for variant in ["original", "lower_regularization", "low_decay_low_lr"]:
        run_ids.extend(f"nbeatsx_lstm_{variant}_seed{seed}" for seed in SEEDS)
    run_ids.extend(f"nbeatsx_released_fixed_config_seed{seed}" for seed in SEEDS)
    run_ids.extend(f"lstm_seed{seed}" for seed in SEEDS)
    run_ids.extend(f"tcn_seed{seed}" for seed in SEEDS)

    checkpoint_rows = []
    for run_id in run_ids:
        meta_path = RUNS / f"{run_id}_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        checkpoint = Path(meta["checkpoint"])
        boundary_ok = (
            meta["train_end_date"] == "2021-05-24"
            and meta["validation_end_date"] == "2023-05-24"
            and meta["test_start_date"] == "2023-05-25"
            and meta["test_rows"] == 365
            and meta["test_used_for_training_or_selection"] is False
        )
        checkpoint_rows.append(
            {
                "run_id": run_id,
                "model": meta["model"],
                "variant": meta["variant"],
                "seed": meta["seed"],
                "checkpoint": str(checkpoint),
                "exists": checkpoint.exists(),
                "bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
                "sha256": digest(checkpoint) if checkpoint.exists() else "",
                "boundary_ok": boundary_ok,
            }
        )
    checkpoint_audit = pd.DataFrame(checkpoint_rows)
    checkpoint_audit.to_csv(HERE / "checkpoint_and_output_audit.csv", index=False)

    tidy365 = pd.read_csv(HERE / "forecasts_365d_selected_config.csv", dtype={"date": str})
    tidy28 = pd.read_csv(HERE / "forecasts_28d_selected_config_clean_holdout.csv", dtype={"date": str})
    model_counts_365 = tidy365.groupby("model").size().to_dict()
    model_counts_28 = tidy28.groupby("model").size().to_dict()

    all_seeds = pd.read_csv(HERE / "forecasts_365d_selected_config_all_seeds.csv", dtype={"date": str})
    recorded = pd.read_csv(HERE / "metrics_365d_selected_config_by_seed.csv")
    maximum_metric_difference = 0.0
    for (model, variant, seed), frame in all_seeds.groupby(["model", "variant", "seed"]):
        observed = metrics(frame)
        expected = recorded[
            (recorded.model == model) & (recorded.variant == variant) & (recorded.seed == seed)
        ].iloc[0]
        for name, value in observed.items():
            maximum_metric_difference = max(maximum_metric_difference, abs(value - float(expected[name])))

    report = {
        "status": "PASS",
        "checkpoint_count": len(checkpoint_audit),
        "all_checkpoints_exist_and_nonempty": bool(
            checkpoint_audit.exists.all() and (checkpoint_audit.bytes > 0).all()
        ),
        "all_training_boundaries_pass": bool(checkpoint_audit.boundary_ok.all()),
        "forecasts_365d_rows": len(tidy365),
        "forecasts_365d_rows_per_model": model_counts_365,
        "forecasts_365d_date_range": [tidy365.date.min(), tidy365.date.max()],
        "forecasts_28d_rows": len(tidy28),
        "forecasts_28d_rows_per_model": model_counts_28,
        "forecasts_28d_date_range": [tidy28.date.min(), tidy28.date.max()],
        "all_seed_forecast_rows": len(all_seeds),
        "maximum_absolute_metric_recalculation_difference": maximum_metric_difference,
        "metrics_reproduced_from_csv": maximum_metric_difference < 1e-9,
    }
    if not (
        report["all_checkpoints_exist_and_nonempty"]
        and report["all_training_boundaries_pass"]
        and set(model_counts_365.values()) == {365}
        and set(model_counts_28.values()) == {28}
        and len(tidy365) == 1825
        and len(tidy28) == 140
        and len(all_seeds) == 4745
        and report["metrics_reproduced_from_csv"]
    ):
        report["status"] = "FAIL"
    (HERE / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
