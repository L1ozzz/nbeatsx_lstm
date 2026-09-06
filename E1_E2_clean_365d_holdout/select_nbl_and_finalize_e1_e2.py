"""Select NBeatsx-LSTM on validation RMSE, then finalize clean E1/E2.

The selection decision is written before any test forecast is read.  All three
pre-registered variants and all three fixed seeds remain in the result files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(r"F:\nbeatsx-lstm\E1_E2_clean_365d_holdout")
RUNS = HERE / "runs"
SEEDS = [1, 42, 2024]
VARIANTS = ["original", "lower_regularization", "low_decay_low_lr"]
METRIC_NAMES = ["MAE", "MSE", "RMSE", "sMAPE", "R2"]
COLORS = {
    "persistence": "black",
    "lstm": "blue",
    "nbeatsx_lstm": "orange",
    "nbeatsx": "green",
    "tcn": "#888888",
}
MARKERS = {"persistence": "s", "lstm": "v", "nbeatsx_lstm": "^", "nbeatsx": "o", "tcn": "D"}
DISPLAY = {
    "persistence": "Persistence",
    "lstm": "LSTM",
    "nbeatsx_lstm": "NBeatsx_LSTM",
    "nbeatsx": "NBEATSx",
    "tcn": "TCN",
}


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual"].to_numpy(dtype=float)
    forecast = frame["forecast"].to_numpy(dtype=float)
    error = actual - forecast
    mse = float(np.mean(error**2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(np.mean(np.abs(error) / ((np.abs(actual) + np.abs(forecast)) / 2)) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def read_metadata(run_id: str) -> dict:
    path = RUNS / f"{run_id}_metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "train_end_date": "2021-05-24",
        "validation_end_date": "2023-05-24",
        "test_start_date": "2023-05-25",
        "test_rows": 365,
        "test_used_for_training_or_selection": False,
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise RuntimeError(f"{run_id}: {key}={meta.get(key)!r}; expected {value!r}")
    return meta


def read_forecast(run_id: str, split: str, expected_rows: int) -> pd.DataFrame:
    path = RUNS / f"{run_id}_{split}_forecasts.csv"
    frame = pd.read_csv(path, dtype={"date": str})
    if len(frame) != expected_rows or frame["date"].duplicated().any():
        raise RuntimeError(f"{run_id}: invalid {split} row count or duplicate date")
    if frame[["actual", "forecast"]].isna().any().any():
        raise RuntimeError(f"{run_id}: missing {split} actual or forecast")
    return frame


def flatten_summary(frame: pd.DataFrame, key_count: int) -> pd.DataFrame:
    frame.columns = [
        column if isinstance(column, str) else column[0]
        for column in frame.columns[:key_count]
    ] + [f"{metric}_{stat}" for metric, stat in frame.columns[key_count:]]
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].fillna(0.0)
    return frame


def main() -> None:
    protocol = HERE / "protocol_pre_registration.json"
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest().upper()
    expected_hash = "CEEE9D746DBA991BD96C76880DFC1944B629ECFAC65D6702A94EBC86A508D40E"
    if protocol_hash != expected_hash:
        raise RuntimeError("The pre-registration file hash changed after training began")

    # Phase 1: validation-only configuration selection.
    validation_parts = []
    validation_metrics = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_id = f"nbeatsx_lstm_{variant}_seed{seed}"
            read_metadata(run_id)
            frame = read_forecast(run_id, "validation", 730)
            validation_parts.append(frame)
            validation_metrics.append({"variant": variant, "seed": seed, **metrics(frame)})
    validation_frame = pd.concat(validation_parts, ignore_index=True)
    validation_frame.to_csv(HERE / "nbl_validation_forecasts_all_variants_all_seeds.csv", index=False, float_format="%.10f")
    validation_by_seed = pd.DataFrame(validation_metrics)
    validation_by_seed.to_csv(HERE / "nbl_validation_metrics_by_seed.csv", index=False, float_format="%.10f")
    validation_summary = (
        validation_by_seed.groupby("variant", sort=False)[METRIC_NAMES].agg(["mean", "std"]).reset_index()
    )
    validation_summary = flatten_summary(validation_summary, 1)
    validation_summary.to_csv(HERE / "nbl_validation_metrics_seed_summary.csv", index=False, float_format="%.10f")
    selected_variant = validation_summary.sort_values("RMSE_mean", kind="stable").iloc[0]["variant"]
    selection = {
        "selected_variant": selected_variant,
        "selection_metric": "mean validation RMSE across seeds 1, 42 and 2024",
        "selected_validation_RMSE_mean": float(
            validation_summary.loc[validation_summary.variant == selected_variant, "RMSE_mean"].iloc[0]
        ),
        "test_metrics_read_for_selection": False,
        "protocol_sha256_verified": protocol_hash,
    }
    (HERE / "nbl_selection_decision_validation_only.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Phase 2: after the decision is fixed, evaluate every pre-registered variant.
    test_metric_rows = []
    all_variant_test_parts = []
    for variant in VARIANTS:
        for seed in SEEDS:
            run_id = f"nbeatsx_lstm_{variant}_seed{seed}"
            frame = read_forecast(run_id, "test", 365)
            all_variant_test_parts.append(frame)
            test_metric_rows.append({"variant": variant, "seed": seed, **metrics(frame)})
    pd.concat(all_variant_test_parts, ignore_index=True).to_csv(
        HERE / "nbl_test_forecasts_all_variants_all_seeds.csv", index=False, float_format="%.10f"
    )
    test_by_seed = pd.DataFrame(test_metric_rows)
    test_by_seed.to_csv(HERE / "nbl_test_metrics_by_seed.csv", index=False, float_format="%.10f")
    test_summary = test_by_seed.groupby("variant", sort=False)[METRIC_NAMES].agg(["mean", "std"]).reset_index()
    test_summary = flatten_summary(test_summary, 1)
    test_summary.to_csv(HERE / "nbl_test_metrics_seed_summary.csv", index=False, float_format="%.10f")

    base_all = pd.read_csv(HERE / "forecasts_365d_all_seeds.csv", dtype={"date": str})
    base_all = base_all[base_all.model != "nbeatsx_lstm"].copy()
    selected_parts = [
        read_forecast(f"nbeatsx_lstm_{selected_variant}_seed{seed}", "test", 365) for seed in SEEDS
    ]
    boundary_audit = pd.read_csv(HERE / "training_boundary_audit.csv")
    boundary_audit = boundary_audit[boundary_audit.model != "nbeatsx_lstm"].copy()
    selected_boundary_rows = []
    for seed in SEEDS:
        run_id = f"nbeatsx_lstm_{selected_variant}_seed{seed}"
        meta = read_metadata(run_id)
        selected_boundary_rows.append(
            {
                "run_id": run_id,
                "model": "nbeatsx_lstm",
                "variant": selected_variant,
                "seed": seed,
                "train_end_date": meta["train_end_date"],
                "validation_end_date": meta["validation_end_date"],
                "test_start_date": meta["test_start_date"],
                "test_rows": meta["test_rows"],
                "test_used_for_training_or_selection": meta["test_used_for_training_or_selection"],
                "checkpoint": meta["checkpoint"],
            }
        )
    boundary_audit = pd.concat([boundary_audit, pd.DataFrame(selected_boundary_rows)], ignore_index=True)
    boundary_audit.to_csv(HERE / "training_boundary_audit_selected_config.csv", index=False)
    selected_all = pd.concat([base_all, *selected_parts], ignore_index=True)
    selected_all = selected_all.sort_values(["date", "model", "seed"]).reset_index(drop=True)
    if len(selected_all) != 4745:
        raise RuntimeError(f"Expected 4,745 all-seed rows, found {len(selected_all)}")
    selected_all.to_csv(HERE / "forecasts_365d_selected_config_all_seeds.csv", index=False, float_format="%.10f")

    selected_mean = (
        selected_all.groupby(["date", "model", "actual"], as_index=False)["forecast"].mean()
        .sort_values(["date", "model"])
        .reset_index(drop=True)
    )
    if len(selected_mean) != 1825:
        raise RuntimeError(f"Expected 1,825 tidy rows, found {len(selected_mean)}")
    selected_mean.to_csv(HERE / "forecasts_365d_selected_config.csv", index=False, float_format="%.10f")
    selected_mean[selected_mean.date >= "2024-04-26"].to_csv(
        HERE / "forecasts_28d_selected_config_clean_holdout.csv", index=False, float_format="%.10f"
    )

    final_metric_rows = []
    for (model, variant, seed), frame in selected_all.groupby(["model", "variant", "seed"]):
        final_metric_rows.append({"model": model, "variant": variant, "seed": seed, **metrics(frame)})
    final_by_seed = pd.DataFrame(final_metric_rows)
    final_by_seed.to_csv(HERE / "metrics_365d_selected_config_by_seed.csv", index=False, float_format="%.10f")
    final_summary = (
        final_by_seed.groupby(["model", "variant"], sort=False)[METRIC_NAMES].agg(["mean", "std"]).reset_index()
    )
    final_summary = flatten_summary(final_summary, 2)
    final_summary.to_csv(HERE / "metrics_365d_selected_config_seed_summary.csv", index=False, float_format="%.10f")
    mean_metric_rows = [{"model": model, **metrics(frame)} for model, frame in selected_mean.groupby("model")]
    pd.DataFrame(mean_metric_rows).to_csv(
        HERE / "metrics_365d_selected_config_from_mean_forecast.csv", index=False, float_format="%.10f"
    )

    persistence_rmse = float(final_summary.loc[final_summary.model == "persistence", "RMSE_mean"].iloc[0])
    selected_rmse = float(final_summary.loc[final_summary.model == "nbeatsx_lstm", "RMSE_mean"].iloc[0])
    variant_comparison = test_summary[["variant", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std"]].copy()
    variant_comparison["persistence_RMSE"] = persistence_rmse
    variant_comparison["RMSE_improvement_vs_persistence_percent"] = (
        (persistence_rmse - variant_comparison["RMSE_mean"]) / persistence_rmse * 100
    )
    variant_comparison["selected_by_validation"] = variant_comparison.variant == selected_variant
    variant_comparison.to_csv(HERE / "nbl_variants_vs_persistence_365d.csv", index=False, float_format="%.10f")

    # E2 on the validation-selected configuration.
    persistence = selected_all[selected_all.model == "persistence"].copy()
    change = persistence[["date", "actual", "forecast"]].copy()
    change["delta"] = (change.actual - change.forecast).abs()
    labels = ["quiet", "medium", "volatile"]
    change["change_group"] = pd.qcut(change.delta, 3, labels=labels, duplicates="drop")
    e2 = selected_all.merge(change[["date", "delta", "change_group"]], on="date", validate="many_to_one")
    e2["absolute_error"] = (e2.actual - e2.forecast).abs()
    e2["squared_error"] = (e2.actual - e2.forecast) ** 2
    grouped = (
        e2.groupby(["change_group", "model", "variant", "seed"], observed=True)
        .agg(n=("absolute_error", "size"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"))
        .reset_index()
    )
    grouped["RMSE"] = np.sqrt(grouped.MSE)
    reference = grouped[grouped.model == "persistence"][["change_group", "MAE", "RMSE"]].rename(
        columns={"MAE": "persistence_MAE", "RMSE": "persistence_RMSE"}
    )
    grouped = grouped.merge(reference, on="change_group", validate="many_to_one")
    grouped["MAE_ratio_to_persistence"] = grouped.MAE / grouped.persistence_MAE
    grouped["RMSE_ratio_to_persistence"] = grouped.RMSE / grouped.persistence_RMSE
    grouped.to_csv(HERE / "error_by_change_tercile_selected_config_all_seeds.csv", index=False, float_format="%.10f")
    e2_summary = (
        grouped.groupby(["change_group", "model", "variant"], observed=True)[
            ["n", "MAE", "RMSE", "MAE_ratio_to_persistence", "RMSE_ratio_to_persistence"]
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    e2_summary = flatten_summary(e2_summary, 3)
    e2_summary.to_csv(HERE / "error_by_change_tercile_selected_config_seed_summary.csv", index=False, float_format="%.10f")

    mean_e2 = selected_mean.merge(change[["date", "delta", "change_group"]], on="date", validate="many_to_one")
    mean_e2["absolute_error"] = (mean_e2.actual - mean_e2.forecast).abs()
    mean_e2["squared_error"] = (mean_e2.actual - mean_e2.forecast) ** 2
    mean_grouped = (
        mean_e2.groupby(["change_group", "model"], observed=True)
        .agg(n=("absolute_error", "size"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"))
        .reset_index()
    )
    mean_grouped["RMSE"] = np.sqrt(mean_grouped.MSE)
    mean_reference = mean_grouped[mean_grouped.model == "persistence"][["change_group", "MAE", "RMSE"]].rename(
        columns={"MAE": "persistence_MAE", "RMSE": "persistence_RMSE"}
    )
    mean_grouped = mean_grouped.merge(mean_reference, on="change_group", validate="many_to_one")
    mean_grouped["MAE_ratio_to_persistence"] = mean_grouped.MAE / mean_grouped.persistence_MAE
    mean_grouped["RMSE_ratio_to_persistence"] = mean_grouped.RMSE / mean_grouped.persistence_RMSE
    mean_grouped.to_csv(HERE / "error_by_change_tercile_selected_config.csv", index=False, float_format="%.10f")

    fig, ax = plt.subplots(figsize=(9.7, 5.8))
    x = np.arange(3)
    for model in ["persistence", "lstm", "nbeatsx_lstm", "nbeatsx", "tcn"]:
        values = e2_summary[e2_summary.model == model].copy()
        values["change_group"] = pd.Categorical(values.change_group, categories=labels, ordered=True)
        values = values.sort_values("change_group")
        means = values.RMSE_ratio_to_persistence_mean.to_numpy()
        stds = values.RMSE_ratio_to_persistence_std.to_numpy()
        ax.plot(x, means, color=COLORS[model], marker=MARKERS[model], linewidth=2, markersize=7, label=DISPLAY[model])
        if model != "persistence":
            ax.fill_between(x, means - stds, means + stds, color=COLORS[model], alpha=0.12)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Actual day-to-day change group")
    ax.set_ylabel("RMSE ratio to Persistence (mean across seeds)")
    ax.set_title(f"Clean 365-day holdout: validation-selected NBeatsx_LSTM ({selected_variant})")
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "rmse_ratio_by_change_tercile_selected_config.png", dpi=300)
    fig.savefig(HERE / "figures" / "rmse_ratio_by_change_tercile_selected_config.svg")
    plt.close(fig)

    result = {
        **selection,
        "test_dates": [selected_mean.date.min(), selected_mean.date.max()],
        "selected_test_RMSE_mean": selected_rmse,
        "selected_test_RMSE_std": float(
            final_summary.loc[final_summary.model == "nbeatsx_lstm", "RMSE_std"].iloc[0]
        ),
        "persistence_test_RMSE": persistence_rmse,
        "selected_RMSE_improvement_vs_persistence_percent": (persistence_rmse - selected_rmse) / persistence_rmse * 100,
        "selected_config_beats_persistence_on_mean_RMSE": selected_rmse < persistence_rmse,
        "any_preregistered_variant_beats_persistence_on_mean_RMSE": bool(
            (variant_comparison.RMSE_mean < persistence_rmse).any()
        ),
        "all_variants_and_seeds_reported": True,
        "test_used_for_training_or_selection": False,
        "final_forecasts_rows": len(selected_mean),
        "final_all_seed_forecasts_rows": len(selected_all),
    }
    (HERE / "final_e1_e2_status.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "final_metrics_printout.txt").write_text(
        final_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}") + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
