"""Reconstruct the paper's 28-day forecasts and run the E1/E2 audit.

This script does not train or tune any model. It reads the exact per-day arrays
retained in D:/NBeatsx-LSTM/222.py, verifies the actual observations against the
released dataset, adds the persistence forecast, reproduces Table VII, and only
then performs the pre-specified change-tercile analysis.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE_222 = Path(r"D:\NBeatsx-LSTM\222.py")
DATASET = Path(r"F:\nbeatsx-lstm\data\imputed_data_KNN_1.csv")

EXPECTED_TABLE_VII = {
    "persistence": {"MAE": 0.7286, "MSE": 1.4307, "RMSE": 1.1961, "sMAPE": 1.6472, "R2": 0.7757},
    "nbeatsx_lstm": {"MAE": 0.7753, "MSE": 1.3824, "RMSE": 1.1757, "sMAPE": 1.7917, "R2": 0.7833},
    "nbeatsx": {"MAE": 0.8904, "MSE": 2.0121, "RMSE": 1.4185, "sMAPE": 2.0588, "R2": 0.6845},
    "lstm": {"MAE": 1.6038, "MSE": 4.6593, "RMSE": 2.1585, "sMAPE": 3.7530, "R2": 0.2695},
    "tcn": {"MAE": 2.0171, "MSE": 6.1091, "RMSE": 2.4717, "sMAPE": 4.6162, "R2": 0.0421},
}

DISPLAY_NAMES = {
    "persistence": "Persistence",
    "nbeatsx_lstm": "NBeatsx_LSTM",
    "nbeatsx": "NBEATSx",
    "lstm": "LSTM",
    "tcn": "TCN",
}

COLORS = {
    "persistence": "black",
    "lstm": "blue",
    "nbeatsx_lstm": "orange",
    "nbeatsx": "green",
    "tcn": "#888888",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            pass
    return values


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
    values = literal_assignments(SOURCE_222)
    required = ["actual", "nbeatsx", "nbeatsx_lstm", "lstm", "tcn"]
    missing = [name for name in required if name not in values]
    if missing:
        raise RuntimeError(f"Missing arrays in {SOURCE_222}: {missing}")

    actual = np.asarray(values["actual"], dtype=float)
    if len(actual) != 28:
        raise RuntimeError(f"Expected 28 actual values, found {len(actual)}")

    raw = pd.read_csv(DATASET)
    date_column = next(name for name in ["ds", "DATE", "Day(Local_Date)"] if name in raw.columns)
    actual_column = next(name for name in ["y", "SM", "SoilM(%)"] if name in raw.columns)
    raw[date_column] = pd.to_datetime(raw[date_column])
    evaluation = raw.tail(28).copy()
    preceding_actual = float(raw.iloc[-29][actual_column])
    dates = evaluation[date_column].dt.strftime("%Y-%m-%d").to_numpy()
    dataset_actual = evaluation[actual_column].to_numpy(dtype=float)
    if not np.allclose(dataset_actual, actual, rtol=0, atol=1e-12):
        raise RuntimeError("The 28 actual values in 222.py do not match the dataset tail")

    forecasts = {
        "persistence": np.r_[preceding_actual, actual[:-1]],
        "nbeatsx_lstm": np.asarray(values["nbeatsx_lstm"], dtype=float),
        "nbeatsx": np.asarray(values["nbeatsx"], dtype=float),
        "lstm": np.asarray(values["lstm"], dtype=float),
        "tcn": np.asarray(values["tcn"], dtype=float),
    }
    if any(len(series) != 28 for series in forecasts.values()):
        lengths = {name: len(series) for name, series in forecasts.items()}
        raise RuntimeError(f"Every forecast series must have 28 values: {lengths}")

    tidy_parts = []
    for model, forecast in forecasts.items():
        tidy_parts.append(
            pd.DataFrame({"date": dates, "model": model, "actual": actual, "forecast": forecast})
        )
    tidy = pd.concat(tidy_parts, ignore_index=True)
    tidy.to_csv(HERE / "forecasts_28d.csv", index=False, float_format="%.8f")

    metrics = []
    print_lines = []
    gate_checks = []
    for model, group in tidy.groupby("model", sort=False):
        row = metric_row(group["actual"].to_numpy(), group["forecast"].to_numpy())
        metrics.append({"model": model, **row})
        print_lines.append(
            f"{model:14s} MAE={row['MAE']:.4f} MSE={row['MSE']:.4f} "
            f"RMSE={row['RMSE']:.4f} sMAPE={row['sMAPE']:.4f} R2={row['R2']:.4f}"
        )
        for metric, expected in EXPECTED_TABLE_VII[model].items():
            observed_4dp = round(row[metric], 4)
            gate_checks.append(
                {
                    "model": model,
                    "metric": metric,
                    "observed_full_precision": row[metric],
                    "observed_4dp": observed_4dp,
                    "expected_table_vii_4dp": expected,
                    "pass": observed_4dp == expected,
                }
            )

    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(HERE / "metrics_28d.csv", index=False, float_format="%.10f")
    (HERE / "metrics_28d_printout.txt").write_text("\n".join(print_lines) + "\n", encoding="utf-8")
    gate_frame = pd.DataFrame(gate_checks)
    gate_frame.to_csv(HERE / "table_vii_reproduction_checks.csv", index=False)
    gate_passed = bool(gate_frame["pass"].all())
    gate_summary = {
        "gate_passed": gate_passed,
        "criterion": "Every metric rounded to four decimal places equals Table VII",
        "checks_passed": int(gate_frame["pass"].sum()),
        "checks_total": int(len(gate_frame)),
        "source_222_sha256": sha256(SOURCE_222),
        "dataset_sha256": sha256(DATASET),
        "preceding_actual_2024_04_25": preceding_actual,
        "training_or_tuning_performed": False,
    }
    (HERE / "table_vii_gate.json").write_text(
        json.dumps(gate_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if not gate_passed:
        raise RuntimeError("Table VII reproduction gate failed; E2 was not run")

    # E2 on the complete, faithful 28-day record. Persistence forecast[t] is
    # actual[t-1], so it provides the preceding actual even for the first date.
    change = tidy.loc[tidy["model"] == "persistence", ["date", "actual", "forecast"]].copy()
    change["delta"] = (change["actual"] - change["forecast"]).abs()
    change["change_group"] = pd.qcut(
        change["delta"], q=3, labels=["quiet", "medium", "volatile"], duplicates="drop"
    )
    analysed = tidy.merge(change[["date", "delta", "change_group"]], on="date", validate="many_to_one")
    analysed["absolute_error"] = (analysed["actual"] - analysed["forecast"]).abs()
    analysed["squared_error"] = (analysed["actual"] - analysed["forecast"]) ** 2
    order = ["quiet", "medium", "volatile"]
    grouped = (
        analysed.groupby(["change_group", "model"], observed=True)
        .agg(n=("absolute_error", "size"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"))
        .reset_index()
    )
    grouped["RMSE"] = np.sqrt(grouped["MSE"])
    grouped["scope"] = "faithful_table_vii_28d"
    grouped["change_group"] = pd.Categorical(grouped["change_group"], categories=order, ordered=True)
    grouped = grouped.sort_values(["change_group", "model"]).reset_index(drop=True)

    reference = grouped.loc[grouped["model"] == "persistence", ["change_group", "MAE", "RMSE"]].rename(
        columns={"MAE": "persistence_MAE", "RMSE": "persistence_RMSE"}
    )
    grouped = grouped.merge(reference, on="change_group", validate="many_to_one")
    grouped["MAE_ratio_to_persistence"] = grouped["MAE"] / grouped["persistence_MAE"]
    grouped["RMSE_ratio_to_persistence"] = grouped["RMSE"] / grouped["persistence_RMSE"]
    grouped.to_csv(HERE / "error_by_change_tercile.csv", index=False, float_format="%.8f")

    ratios = grouped[
        ["scope", "change_group", "model", "n", "MAE_ratio_to_persistence", "RMSE_ratio_to_persistence"]
    ].copy()
    ratios.to_csv(HERE / "error_ratios_to_persistence_28d.csv", index=False, float_format="%.8f")
    analysed[["date", "model", "actual", "forecast", "delta", "change_group", "absolute_error", "squared_error"]].to_csv(
        HERE / "daily_errors_with_change_group_28d.csv", index=False, float_format="%.8f"
    )

    pivot = grouped.pivot(index="change_group", columns="model", values="RMSE_ratio_to_persistence").reindex(order)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    markers = {"persistence": "s", "nbeatsx_lstm": "^", "nbeatsx": "o", "lstm": "v", "tcn": "D"}
    for model in ["persistence", "lstm", "nbeatsx_lstm", "nbeatsx", "tcn"]:
        ax.plot(
            order,
            pivot[model].to_numpy(),
            marker=markers[model],
            linewidth=2,
            markersize=7,
            color=COLORS[model],
            label=DISPLAY_NAMES[model],
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Actual day-to-day change group")
    ax.set_ylabel("RMSE ratio to Persistence")
    ax.set_title("Error by actual change magnitude (faithful 28-day Table VII window)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(HERE / "rmse_ratio_by_change_tercile_28d.png", dpi=300)
    fig.savefig(HERE / "rmse_ratio_by_change_tercile_28d.svg")
    plt.close(fig)

    proposed = grouped.loc[grouped["model"] == "nbeatsx_lstm"].set_index("change_group")
    interpretation = {
        "scope": "faithful_table_vii_28d",
        "quiet_rmse_ratio": float(proposed.loc["quiet", "RMSE_ratio_to_persistence"]),
        "medium_rmse_ratio": float(proposed.loc["medium", "RMSE_ratio_to_persistence"]),
        "volatile_rmse_ratio": float(proposed.loc["volatile", "RMSE_ratio_to_persistence"]),
        "rule": (
            "Current mechanism is supported in this window only if volatile RMSE ratio < 1 "
            "and quiet RMSE ratio > 1."
        ),
    }
    interpretation["mechanism_supported_in_28d"] = bool(
        interpretation["volatile_rmse_ratio"] < 1 and interpretation["quiet_rmse_ratio"] > 1
    )
    (HERE / "e2_interpretation_28d.json").write_text(
        json.dumps(interpretation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
