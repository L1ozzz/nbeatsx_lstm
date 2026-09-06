"""Aggregate fixed-seed clean-holdout forecasts and run E1/E2.

Primary scientific summaries are mean and standard deviation across the three
pre-registered seeds. A one-row-per-(date, model) CSV is also produced using
the mean forecast across seeds for the requested tidy interchange format.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(r"F:\nbeatsx-lstm")
HERE = PROJECT / "E1_E2_clean_365d_holdout"
RUNS = HERE / "runs"
DATA = PROJECT / "data" / "imputed_data_KNN_1.csv"
SEEDS = [1, 42, 2024]
RUN_SPECS = {
    "nbeatsx_lstm": ("original", "nbeatsx_lstm_original_seed{seed}"),
    "nbeatsx": ("released_fixed_config", "nbeatsx_released_fixed_config_seed{seed}"),
    "lstm": ("released_fixed_config", "lstm_seed{seed}"),
    "tcn": ("released_fixed_config", "tcn_seed{seed}"),
}
DISPLAY = {
    "persistence": "Persistence",
    "lstm": "LSTM",
    "nbeatsx_lstm": "NBeatsx_LSTM",
    "nbeatsx": "NBEATSx",
    "tcn": "TCN",
}
COLORS = {
    "persistence": "black",
    "lstm": "blue",
    "nbeatsx_lstm": "orange",
    "nbeatsx": "green",
    "tcn": "#888888",
}
MARKERS = {"persistence": "s", "lstm": "v", "nbeatsx_lstm": "^", "nbeatsx": "o", "tcn": "D"}
EXPECTED_TABLE_VII = {
    "persistence": {"MAE": 0.7286, "MSE": 1.4307, "RMSE": 1.1961, "sMAPE": 1.6472, "R2": 0.7757},
    "nbeatsx_lstm": {"MAE": 0.7753, "MSE": 1.3824, "RMSE": 1.1757, "sMAPE": 1.7917, "R2": 0.7833},
    "nbeatsx": {"MAE": 0.8904, "MSE": 2.0121, "RMSE": 1.4185, "sMAPE": 2.0588, "R2": 0.6845},
    "lstm": {"MAE": 1.6038, "MSE": 4.6593, "RMSE": 2.1585, "sMAPE": 3.7530, "R2": 0.2695},
    "tcn": {"MAE": 2.0171, "MSE": 6.1091, "RMSE": 2.4717, "sMAPE": 4.6162, "R2": 0.0421},
}


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


def validate_metadata(run_id: str) -> dict:
    path = RUNS / f"{run_id}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "train_end_date": "2021-05-24",
        "validation_end_date": "2023-05-24",
        "test_start_date": "2023-05-25",
        "test_rows": 365,
        "test_used_for_training_or_selection": False,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"{run_id}: metadata {key}={meta.get(key)!r}, expected {expected!r}")
    return meta


def main() -> None:
    raw = pd.read_csv(DATA, parse_dates=["Day(Local_Date)"])
    test_raw = raw.iloc[-365:].copy()
    test_dates = test_raw["Day(Local_Date)"].dt.strftime("%Y-%m-%d").to_numpy()
    test_actual = test_raw["SoilM(%)"].to_numpy(dtype=float)
    previous_actual = float(raw.iloc[-366]["SoilM(%)"])
    expected_persistence = np.r_[previous_actual, test_actual[:-1]]

    parts = []
    metadata_audit = []
    for model, (variant, pattern) in RUN_SPECS.items():
        for seed in SEEDS:
            run_id = pattern.format(seed=seed)
            meta = validate_metadata(run_id)
            metadata_audit.append(
                {
                    "run_id": run_id,
                    "model": model,
                    "variant": variant,
                    "seed": seed,
                    "train_end_date": meta["train_end_date"],
                    "validation_end_date": meta["validation_end_date"],
                    "test_start_date": meta["test_start_date"],
                    "test_rows": meta["test_rows"],
                    "test_used_for_training_or_selection": meta["test_used_for_training_or_selection"],
                    "checkpoint": meta["checkpoint"],
                }
            )
            path = RUNS / f"{run_id}_test_forecasts.csv"
            frame = pd.read_csv(path, dtype={"date": str})
            if len(frame) != 365 or not np.array_equal(frame["date"].to_numpy(), test_dates):
                raise RuntimeError(f"{run_id}: test dates are not the expected 365-day range")
            if not np.allclose(frame["actual"], test_actual, rtol=0, atol=2e-6):
                raise RuntimeError(f"{run_id}: actual values do not match the dataset")
            parts.append(frame[["date", "model", "variant", "seed", "actual", "forecast", "split"]])

    all_seeds = pd.concat(parts, ignore_index=True)
    persistence = pd.DataFrame(
        {
            "date": test_dates,
            "model": "persistence",
            "variant": "deterministic",
            "seed": 0,
            "actual": test_actual,
            "forecast": expected_persistence,
            "split": "test",
        }
    )
    all_with_persistence = pd.concat([persistence, all_seeds], ignore_index=True)
    all_with_persistence.to_csv(HERE / "forecasts_365d_all_seeds.csv", index=False, float_format="%.10f")
    pd.DataFrame(metadata_audit).to_csv(HERE / "training_boundary_audit.csv", index=False)

    mean_forecasts = (
        all_with_persistence.groupby(["date", "model", "actual"], as_index=False)["forecast"].mean()
        .sort_values(["date", "model"])
        .reset_index(drop=True)
    )
    mean_forecasts[["date", "model", "actual", "forecast"]].to_csv(
        HERE / "forecasts_365d.csv", index=False, float_format="%.10f"
    )
    mean_forecasts[mean_forecasts["date"] >= "2024-04-26"][["date", "model", "actual", "forecast"]].to_csv(
        HERE / "forecasts_28d_clean_holdout.csv", index=False, float_format="%.10f"
    )

    metric_rows = []
    for (model, variant, seed), group in all_with_persistence.groupby(["model", "variant", "seed"], sort=False):
        metric_rows.append({"model": model, "variant": variant, "seed": seed, **metrics(group.actual.to_numpy(), group.forecast.to_numpy())})
    by_seed = pd.DataFrame(metric_rows)
    by_seed.to_csv(HERE / "metrics_365d_by_seed.csv", index=False, float_format="%.10f")
    summary = (
        by_seed.groupby(["model", "variant"])[["MAE", "MSE", "RMSE", "sMAPE", "R2"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["model", "variant"] + [f"{metric}_{stat}" for metric, stat in summary.columns.tolist()[2:]]
    summary = summary.fillna(0.0)
    summary.to_csv(HERE / "metrics_365d_seed_summary.csv", index=False, float_format="%.10f")

    mean_metric_rows = []
    for model, group in mean_forecasts.groupby("model", sort=False):
        mean_metric_rows.append({"model": model, **metrics(group.actual.to_numpy(), group.forecast.to_numpy())})
    mean_metric_frame = pd.DataFrame(mean_metric_rows)
    mean_metric_frame.to_csv(HERE / "metrics_365d_from_mean_forecast.csv", index=False, float_format="%.10f")
    print_lines = []
    for _, row in summary.sort_values("model").iterrows():
        print_lines.append(
            f"{row['model']:14s} MAE={row['MAE_mean']:.4f}±{row['MAE_std']:.4f} "
            f"MSE={row['MSE_mean']:.4f}±{row['MSE_std']:.4f} "
            f"RMSE={row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f} "
            f"sMAPE={row['sMAPE_mean']:.4f}±{row['sMAPE_std']:.4f} "
            f"R2={row['R2_mean']:.4f}±{row['R2_std']:.4f}"
        )
    (HERE / "metrics_365d_seed_summary_printout.txt").write_text("\n".join(print_lines) + "\n", encoding="utf-8")

    # The clean holdout is a new experiment and is expected not to reproduce
    # the old Table VII values. Preserve the explicit comparison.
    clean28 = mean_forecasts[mean_forecasts["date"] >= "2024-04-26"]
    comparison_rows = []
    clean28_metrics = []
    for model, group in clean28.groupby("model"):
        observed = metrics(group.actual.to_numpy(), group.forecast.to_numpy())
        clean28_metrics.append({"model": model, **observed})
        for metric, expected in EXPECTED_TABLE_VII[model].items():
            comparison_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "clean_holdout_value": observed[metric],
                    "clean_holdout_4dp": round(observed[metric], 4),
                    "legacy_table_vii_4dp": expected,
                    "match": round(observed[metric], 4) == expected,
                }
            )
    pd.DataFrame(clean28_metrics).to_csv(HERE / "metrics_28d_clean_holdout.csv", index=False, float_format="%.10f")
    table_check = pd.DataFrame(comparison_rows)
    table_check.to_csv(HERE / "table_vii_clean_holdout_comparison.csv", index=False)
    (HERE / "table_vii_clean_holdout_status.json").write_text(
        json.dumps(
            {
                "reproduces_legacy_table_vii": bool(table_check["match"].all()),
                "matches": int(table_check["match"].sum()),
                "checks": int(len(table_check)),
                "interpretation": "Expected non-reproduction: clean retraining, corrected date alignment and leakage-controlled preprocessing define a new experiment. It must not replace or be labelled as the legacy Table VII run without revising the paper.",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # E2: define delta directly from the deterministic persistence forecast so
    # the first test date also has its true preceding observation.
    change = persistence[["date", "actual", "forecast"]].copy()
    change["delta"] = (change["actual"] - change["forecast"]).abs()
    labels = ["quiet", "medium", "volatile"]
    change["change_group"] = pd.qcut(change["delta"], 3, labels=labels, duplicates="drop")
    e2 = all_with_persistence.merge(change[["date", "delta", "change_group"]], on="date", validate="many_to_one")
    e2["absolute_error"] = (e2["actual"] - e2["forecast"]).abs()
    e2["squared_error"] = (e2["actual"] - e2["forecast"]) ** 2
    grouped = (
        e2.groupby(["change_group", "model", "variant", "seed"], observed=True)
        .agg(n=("absolute_error", "size"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"))
        .reset_index()
    )
    grouped["RMSE"] = np.sqrt(grouped["MSE"])
    persistence_ref = grouped[grouped["model"] == "persistence"][["change_group", "MAE", "RMSE"]].rename(
        columns={"MAE": "persistence_MAE", "RMSE": "persistence_RMSE"}
    )
    grouped = grouped.merge(persistence_ref, on="change_group", validate="many_to_one")
    grouped["MAE_ratio_to_persistence"] = grouped["MAE"] / grouped["persistence_MAE"]
    grouped["RMSE_ratio_to_persistence"] = grouped["RMSE"] / grouped["persistence_RMSE"]
    grouped.to_csv(HERE / "error_by_change_tercile_all_seeds.csv", index=False, float_format="%.10f")

    e2_summary = (
        grouped.groupby(["change_group", "model", "variant"], observed=True)[
            ["n", "MAE", "RMSE", "MAE_ratio_to_persistence", "RMSE_ratio_to_persistence"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    e2_summary.columns = ["change_group", "model", "variant"] + [
        f"{metric}_{stat}" for metric, stat in e2_summary.columns.tolist()[3:]
    ]
    numeric_summary_columns = e2_summary.select_dtypes(include=[np.number]).columns
    e2_summary[numeric_summary_columns] = e2_summary[numeric_summary_columns].fillna(0.0)
    e2_summary.to_csv(HERE / "error_by_change_tercile_seed_summary.csv", index=False, float_format="%.10f")

    mean_e2 = mean_forecasts.merge(change[["date", "delta", "change_group"]], on="date", validate="many_to_one")
    mean_e2["absolute_error"] = (mean_e2["actual"] - mean_e2["forecast"]).abs()
    mean_e2["squared_error"] = (mean_e2["actual"] - mean_e2["forecast"]) ** 2
    mean_grouped = (
        mean_e2.groupby(["change_group", "model"], observed=True)
        .agg(n=("absolute_error", "size"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"))
        .reset_index()
    )
    mean_grouped["RMSE"] = np.sqrt(mean_grouped["MSE"])
    mean_ref = mean_grouped[mean_grouped.model == "persistence"][["change_group", "MAE", "RMSE"]].rename(
        columns={"MAE": "persistence_MAE", "RMSE": "persistence_RMSE"}
    )
    mean_grouped = mean_grouped.merge(mean_ref, on="change_group", validate="many_to_one")
    mean_grouped["MAE_ratio_to_persistence"] = mean_grouped.MAE / mean_grouped.persistence_MAE
    mean_grouped["RMSE_ratio_to_persistence"] = mean_grouped.RMSE / mean_grouped.persistence_RMSE
    mean_grouped.to_csv(HERE / "error_by_change_tercile.csv", index=False, float_format="%.10f")

    fig, ax = plt.subplots(figsize=(9.7, 5.8))
    x = np.arange(len(labels))
    for model in ["persistence", "lstm", "nbeatsx_lstm", "nbeatsx", "tcn"]:
        values = e2_summary[e2_summary.model == model].copy()
        values["change_group"] = pd.Categorical(values["change_group"], categories=labels, ordered=True)
        values = values.sort_values("change_group")
        mean_values = values["RMSE_ratio_to_persistence_mean"].to_numpy()
        std_values = values["RMSE_ratio_to_persistence_std"].to_numpy()
        ax.plot(x, mean_values, color=COLORS[model], marker=MARKERS[model], linewidth=2, markersize=7, label=DISPLAY[model])
        if model != "persistence":
            ax.fill_between(x, mean_values - std_values, mean_values + std_values, color=COLORS[model], alpha=0.12)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Actual day-to-day change group")
    ax.set_ylabel("RMSE ratio to Persistence (mean across seeds)")
    ax.set_title("Clean 365-day holdout: error by actual change magnitude")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "rmse_ratio_by_change_tercile.png", dpi=300)
    fig.savefig(HERE / "figures" / "rmse_ratio_by_change_tercile.svg")
    plt.close(fig)

    outliers = []
    for model, group in mean_e2.groupby("model"):
        top = group.assign(squared_error=(group.actual - group.forecast) ** 2).nlargest(10, "squared_error")
        top = top.assign(model=model, rank=np.arange(1, len(top) + 1))
        outliers.append(top[["model", "rank", "date", "actual", "forecast", "delta", "change_group", "squared_error"]])
    pd.concat(outliers, ignore_index=True).to_csv(HERE / "top_squared_error_dates.csv", index=False, float_format="%.10f")

    result = {
        "status": "complete_for_fixed_baselines_and_original_proposed_config",
        "test_dates": [test_dates[0], test_dates[-1]],
        "models": ["persistence", "nbeatsx_lstm", "nbeatsx", "lstm", "tcn"],
        "fixed_seeds": SEEDS,
        "forecasts_365d_rows": int(len(mean_forecasts)),
        "forecasts_365d_all_seeds_rows": int(len(all_with_persistence)),
        "legacy_table_vii_reproduced_by_clean_retraining": bool(table_check["match"].all()),
        "e2_complete": True,
        "test_used_for_training_or_selection": False,
    }
    (HERE / "baseline_e1_e2_status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
