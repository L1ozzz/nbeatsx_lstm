"""Build the final five-seed evidence package for Experiment Protocol E4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(r"F:\nbeatsx-lstm\E1_E2_clean_365d_holdout")
PROJECT = HERE.parent
OUT = HERE / "E4_five_seed_results"
SEEDS = [1, 2, 3, 42, 2024]
TEST_START = 8604
TEST_END = 8969
METRICS = ["MAE", "MSE", "RMSE", "sMAPE", "R2"]
RUNS = {
    "LSTM": ("released_fixed_config", "lstm_seed{seed}"),
    "TCN": ("released_fixed_config", "tcn_seed{seed}"),
    "NBEATSx": ("released_fixed_config", "nbeatsx_released_fixed_config_seed{seed}"),
    "NBeatsx_LSTM": ("lower_regularization", "nbeatsx_lstm_lower_regularization_seed{seed}"),
}


def calculate_metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    error = actual - forecast
    mse = float(np.mean(error**2))
    denominator = (np.abs(actual) + np.abs(forecast)) / 2
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(np.mean(np.abs(error) / denominator) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(PROJECT / "data" / "imputed_data_KNN_1.csv")
    dates = pd.to_datetime(data["Day(Local_Date)"])
    actual = data["SoilM(%)"].to_numpy(dtype=float)[TEST_START:TEST_END]
    expected_dates = dates.iloc[TEST_START:TEST_END].dt.strftime("%Y-%m-%d").to_numpy()

    forecast_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, float | int | str]] = []
    audit_rows: list[dict[str, str | int | bool]] = []

    for model, (variant, pattern) in RUNS.items():
        for seed in SEEDS:
            run_id = pattern.format(seed=seed)
            forecast_path = HERE / "runs" / f"{run_id}_test_forecasts.csv"
            metadata_path = HERE / "runs" / f"{run_id}_metadata.json"
            if not forecast_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Incomplete E4 run: {run_id}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frame = pd.read_csv(forecast_path, dtype={"date": str})
            if len(frame) != 365:
                raise RuntimeError(f"{run_id}: expected 365 forecasts, found {len(frame)}")
            if not np.array_equal(frame["date"].to_numpy(), expected_dates):
                raise RuntimeError(f"{run_id}: test dates are not aligned")
            if not np.allclose(frame["actual"].to_numpy(dtype=float), actual, atol=2e-6, rtol=0):
                raise RuntimeError(f"{run_id}: actual values do not match the fixed test period")
            if metadata.get("test_used_for_training_or_selection") is not False:
                raise RuntimeError(f"{run_id}: metadata does not confirm test isolation")
            if metadata.get("train_end_date") != "2021-05-24" or metadata.get("test_start_date") != "2023-05-25":
                raise RuntimeError(f"{run_id}: split metadata mismatch")

            clean = frame[["date", "actual", "forecast"]].copy()
            clean.insert(1, "model", model)
            clean.insert(2, "variant", variant)
            clean.insert(3, "seed", seed)
            forecast_parts.append(clean)
            row = {"model": model, "variant": variant, "seed": seed}
            row.update(calculate_metrics(clean.actual.to_numpy(), clean.forecast.to_numpy()))
            metric_rows.append(row)

            checkpoint_path = Path(metadata["checkpoint"])
            audit_rows.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "variant": variant,
                    "seed": seed,
                    "forecast_rows": len(frame),
                    "test_start": frame.date.iloc[0],
                    "test_end": frame.date.iloc[-1],
                    "test_isolated": True,
                    "checkpoint_exists": checkpoint_path.exists(),
                    "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
                    "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path.exists() else "",
                    "forecast_sha256": sha256(forecast_path),
                }
            )

    persistence_forecast = data["SoilM(%)"].to_numpy(dtype=float)[TEST_START - 1 : TEST_END - 1]
    persistence = pd.DataFrame(
        {
            "date": expected_dates,
            "model": "Persistence",
            "variant": "deterministic",
            "seed": 0,
            "actual": actual,
            "forecast": persistence_forecast,
        }
    )
    forecast_parts.append(persistence)
    persistence_row: dict[str, float | int | str] = {
        "model": "Persistence",
        "variant": "deterministic",
        "seed": 0,
    }
    persistence_row.update(calculate_metrics(actual, persistence_forecast))
    metric_rows.append(persistence_row)

    forecasts = pd.concat(forecast_parts, ignore_index=True)
    forecasts.to_csv(OUT / "forecasts_365d_all_five_seeds.csv", index=False, float_format="%.10f")
    by_seed = pd.DataFrame(metric_rows)
    by_seed.to_csv(OUT / "metrics_by_seed.csv", index=False, float_format="%.10f")

    learned = by_seed[by_seed.model != "Persistence"]
    summary = learned.groupby(["model", "variant"], sort=False)[METRICS].agg(["mean", "std"]).reset_index()
    summary.columns = ["model", "variant"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[2:]]
    p = by_seed[by_seed.model == "Persistence"].iloc[0]
    p_summary = {"model": "Persistence", "variant": "deterministic"}
    for metric in METRICS:
        p_summary[f"{metric}_mean"] = float(p[metric])
        p_summary[f"{metric}_std"] = 0.0
    summary = pd.concat([summary, pd.DataFrame([p_summary])], ignore_index=True)
    summary["RMSE_change_vs_persistence_percent"] = (
        (float(p["RMSE"]) - summary["RMSE_mean"]) / float(p["RMSE"]) * 100
    )
    summary.to_csv(OUT / "metrics_five_seed_summary.csv", index=False, float_format="%.10f")

    long_rows = []
    for _, row in by_seed.iterrows():
        for metric in METRICS:
            long_rows.append(
                {
                    "model": row["model"],
                    "variant": row["variant"],
                    "seed": int(row["seed"]),
                    "metric": metric,
                    "value": float(row[metric]),
                }
            )
    pd.DataFrame(long_rows).to_csv(OUT / "results_clean_split.csv", index=False, float_format="%.10f")
    pd.DataFrame(audit_rows).to_csv(OUT / "checkpoint_and_forecast_audit.csv", index=False)

    learned_counts = learned.groupby("model").seed.nunique().to_dict()
    report = {
        "status": "PASS",
        "fixed_seeds": SEEDS,
        "learned_model_runs": int(len(learned)),
        "expected_learned_model_runs": 20,
        "runs_per_model": learned_counts,
        "test_dates": [expected_dates[0], expected_dates[-1]],
        "test_rows_per_run": 365,
        "forecast_rows_total": int(len(forecasts)),
        "all_checkpoints_present": bool(pd.DataFrame(audit_rows).checkpoint_exists.all()),
        "all_outputs_retained": True,
        "best_seed_selected": False,
        "metrics": METRICS,
        "proposed_model_variant_frozen": "lower_regularization",
        "test_used_for_tuning": False,
    }
    if len(learned) != 20 or any(value != 5 for value in learned_counts.values()):
        raise RuntimeError(f"Incomplete five-seed matrix: {learned_counts}")
    (OUT / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "E4 clean chronological split: five-seed summary",
        f"Test period: {expected_dates[0]} to {expected_dates[-1]} (365 days)",
        "Seeds: 1, 2, 3, 42, 2024",
        "All values are mean +/- sample standard deviation across five seeds.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row.model:16s} MAE={row.MAE_mean:.4f}+/-{row.MAE_std:.4f} "
            f"MSE={row.MSE_mean:.4f}+/-{row.MSE_std:.4f} "
            f"RMSE={row.RMSE_mean:.4f}+/-{row.RMSE_std:.4f} "
            f"sMAPE={row.sMAPE_mean:.4f}+/-{row.sMAPE_std:.4f} "
            f"R2={row.R2_mean:.4f}+/-{row.R2_std:.4f} "
            f"RMSE_change_vs_Persistence={row.RMSE_change_vs_persistence_percent:.4f}%"
        )
    (OUT / "metrics_printout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
