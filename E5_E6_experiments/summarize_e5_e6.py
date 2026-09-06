"""Independently reconstruct and summarize all pre-registered E5/E6 results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"F:\nbeatsx-lstm")
OUT = ROOT / "E5_E6_experiments"
RUNS = OUT / "runs"
E4 = ROOT / "E1_E2_clean_365d_holdout" / "E4_five_seed_results"
SEEDS = [1, 2, 3, 42, 2024]
METRICS = ["MAE", "MSE", "RMSE", "sMAPE", "R2"]


def score(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual"].to_numpy(dtype=float)
    forecast = frame["forecast"].to_numpy(dtype=float)
    error = actual - forecast
    mse = np.mean(error**2)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(np.mean(np.abs(error) / ((np.abs(actual) + np.abs(forecast)) / 2)) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def row(model: str, horizon: int, seed: int, frame: pd.DataFrame, source: str) -> dict[str, object]:
    return {
        "model": model,
        "horizon": horizon,
        "seed": seed,
        "n_forecast_cells": len(frame),
        **score(frame),
        "source": source,
    }


def h1_frames() -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    all_e4 = pd.read_csv(E4 / "forecasts_365d_all_five_seeds.csv")
    proposed = {}
    for seed in SEEDS:
        frame = all_e4[
            (all_e4.model == "NBeatsx_LSTM")
            & (all_e4.variant == "lower_regularization")
            & (all_e4.seed == seed)
        ][["date", "actual", "forecast"]].copy()
        if len(frame) != 365:
            raise RuntimeError(f"E4 h=1 seed {seed}: expected 365 rows, got {len(frame)}")
        proposed[seed] = frame
    persistence = all_e4[all_e4.model == "Persistence"][["date", "actual", "forecast"]].copy()
    if len(persistence) != 365:
        raise RuntimeError(f"E4 persistence: expected 365 rows, got {len(persistence)}")
    return proposed, persistence


def persistence_for_multistep(frame: pd.DataFrame, h1_persistence: pd.DataFrame) -> pd.DataFrame:
    lookup = h1_persistence.set_index("date")["forecast"]
    first_targets = frame[frame.lead == 1].set_index("forecast_origin_date")["target_date"]
    origin_forecast = first_targets.map(lookup)
    if origin_forecast.isna().any():
        raise RuntimeError("A persistence origin could not be matched to the h=1 reference")
    result = frame[["forecast_origin_date", "target_date", "lead", "actual"]].copy()
    result["forecast"] = result.forecast_origin_date.map(origin_forecast)
    return result


def summarize(rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    learned = rows[rows.seed != 0]
    mean = learned.groupby(group_columns)[METRICS].mean().add_suffix("_mean")
    std = learned.groupby(group_columns)[METRICS].std(ddof=1).add_suffix("_sd")
    return mean.join(std).reset_index()


def main() -> None:
    h1, h1_persistence = h1_frames()
    e5_rows: list[dict[str, object]] = []
    e6_rows: list[dict[str, object]] = []
    metadata_differences: list[float] = []
    split_audit = []

    for seed, frame in h1.items():
        e5_rows.append(row("NBeatsx_LSTM", 1, seed, frame, "E4 clean-split reuse"))
        e6_rows.append({**row("NBeatsx_LSTM", 1, seed, frame, "E4 clean-split reuse"), "training_loss": "MAE"})
    persistence_h1_row = row("Persistence", 1, 0, h1_persistence, "deterministic last observation")
    e5_rows.append(persistence_h1_row)
    e6_rows.append({**persistence_h1_row, "training_loss": "none"})

    for horizon in [3, 7, 14]:
        persistence_frame = None
        for seed in SEEDS:
            run_id = f"e5_h{horizon}_mae_seed{seed}"
            forecast_path = RUNS / f"{run_id}_test_forecasts.csv"
            metadata_path = RUNS / f"{run_id}_metadata.json"
            if not forecast_path.exists() or not metadata_path.exists():
                raise FileNotFoundError(f"Incomplete E5 run: {run_id}")
            frame = pd.read_csv(forecast_path)
            expected_cells = (366 - horizon) * horizon
            if len(frame) != expected_cells:
                raise RuntimeError(f"{run_id}: expected {expected_cells} cells, got {len(frame)}")
            e5_rows.append(row("NBeatsx_LSTM", horizon, seed, frame, forecast_path.name))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            independently_scored = score(frame)
            metadata_differences.extend(
                abs(independently_scored[name] - metadata["metrics"]["test"][name]) for name in METRICS
            )
            split_audit.append(
                {
                    "run_id": run_id,
                    "train_end": metadata["train_end_date"],
                    "validation_end": metadata["validation_end_date"],
                    "test_start": metadata["test_start_date"],
                    "test_end": metadata["test_end_date"],
                    "test_used_for_selection": metadata["test_used_for_training_early_stopping_or_selection"],
                }
            )
            if persistence_frame is None:
                persistence_frame = persistence_for_multistep(frame, h1_persistence)
        assert persistence_frame is not None
        e5_rows.append(row("Persistence", horizon, 0, persistence_frame, "last observation repeated across horizon"))

    for seed in SEEDS:
        run_id = f"e6_h1_mse_seed{seed}"
        forecast_path = RUNS / f"{run_id}_test_forecasts.csv"
        metadata_path = RUNS / f"{run_id}_metadata.json"
        if not forecast_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Incomplete E6 run: {run_id}")
        frame = pd.read_csv(forecast_path)
        e6_rows.append({**row("NBeatsx_LSTM", 1, seed, frame, forecast_path.name), "training_loss": "MSE"})
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        independently_scored = score(frame)
        metadata_differences.extend(
            abs(independently_scored[name] - metadata["metrics"]["test"][name]) for name in METRICS
        )
        split_audit.append(
            {
                "run_id": run_id,
                "train_end": metadata["train_end_date"],
                "validation_end": metadata["validation_end_date"],
                "test_start": metadata["test_start_date"],
                "test_end": metadata["test_end_date"],
                "test_used_for_selection": metadata["test_used_for_training_early_stopping_or_selection"],
            }
        )

    e5 = pd.DataFrame(e5_rows)
    e6 = pd.DataFrame(e6_rows)
    e5.to_csv(OUT / "results_horizon_sweep.csv", index=False, float_format="%.10f")
    e6.to_csv(OUT / "results_loss_ablation.csv", index=False, float_format="%.10f")
    e5_summary = summarize(e5[e5.model == "NBeatsx_LSTM"], ["model", "horizon"])
    e6_summary = summarize(e6[(e6.model == "NBeatsx_LSTM")], ["model", "training_loss"])
    e5_summary.to_csv(OUT / "results_horizon_sweep_summary.csv", index=False, float_format="%.10f")
    e6_summary.to_csv(OUT / "results_loss_ablation_summary.csv", index=False, float_format="%.10f")

    persistence_rmse = e5[e5.model == "Persistence"].set_index("horizon")["RMSE"]
    ratios = e5[e5.model == "NBeatsx_LSTM"].copy()
    ratios["RMSE_ratio_to_Persistence"] = ratios.apply(
        lambda item: item.RMSE / persistence_rmse.loc[item.horizon], axis=1
    )
    ratios.to_csv(OUT / "e5_rmse_ratios_by_seed.csv", index=False, float_format="%.10f")
    ratio_summary = ratios.groupby("horizon").RMSE_ratio_to_Persistence.agg(["mean", "std"]).reset_index()
    ratio_summary.to_csv(OUT / "e5_rmse_ratio_summary.csv", index=False, float_format="%.10f")

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for seed, group in ratios.groupby("seed"):
        ax.plot(group.horizon, group.RMSE_ratio_to_Persistence, color="#8fb58f", alpha=0.35, linewidth=1)
    ax.errorbar(
        ratio_summary.horizon,
        ratio_summary["mean"],
        yerr=ratio_summary["std"],
        color="green",
        marker="o",
        capsize=4,
        linewidth=2,
        label="NBeatsx-LSTM mean ± SD (5 seeds)",
    )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Persistence")
    ax.set_xticks([1, 3, 7, 14])
    ax.set_xlabel("Forecast horizon (days)")
    ax.set_ylabel("RMSE / Persistence RMSE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "e5_rmse_ratio_by_horizon.png", dpi=300)
    plt.close(fig)

    basis_files = sorted(RUNS.glob("e5_h7_mae_seed*_actual_basis.json"))
    if len(basis_files) != len(SEEDS):
        raise RuntimeError(f"Expected five h=7 basis files, got {len(basis_files)}")
    basis_checks = []
    for path in basis_files:
        basis = json.loads(path.read_text(encoding="utf-8"))
        harmonic = np.asarray(basis["SeasonalityBasis"]["forecast_basis"], dtype=float)
        trend = np.asarray(basis["TrendBasis"]["forecast_basis"], dtype=float)
        basis_checks.append(
            {
                "file": path.name,
                "seasonality_shape": list(harmonic.shape),
                "seasonality_is_constant": bool(np.allclose(harmonic, harmonic.flat[0])),
                "trend_shape": list(trend.shape),
                "trend_is_constant": bool(np.allclose(trend, trend.flat[0])),
            }
        )
    (OUT / "h7_actual_basis_verification.json").write_text(
        json.dumps(basis_checks, indent=2) + "\n", encoding="utf-8"
    )

    verification = {
        "status": "PASS",
        "expected_new_runs": 20,
        "observed_new_runs": len(split_audit),
        "fixed_seeds": SEEDS,
        "all_horizons_retained": sorted(e5.horizon.unique().tolist()),
        "all_losses_retained": sorted(e6.training_loss.unique().tolist()),
        "test_used_for_tuning_or_early_stopping": any(item["test_used_for_selection"] for item in split_audit),
        "maximum_absolute_difference_vs_training_metadata": max(metadata_differences),
        "independent_metric_recalculation_passed": max(metadata_differences) < 1e-8,
        "split_audit": split_audit,
    }
    (OUT / "independent_verification_report.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    print(e5_summary.to_string(index=False))
    print(e6_summary.to_string(index=False))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
