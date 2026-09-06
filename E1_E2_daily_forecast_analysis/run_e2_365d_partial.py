"""Run the pre-specified E2 grouping on the two recoverable 365-day series.

This is not the requested five-model E2 result. It is retained as transparent
partial evidence because the proposed checkpoint and Persistence were run and
negative results must not be discarded.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def main() -> None:
    data = pd.read_csv(HERE / "forecasts_365d_partial.csv")
    change = data.loc[data["model"] == "persistence", ["date", "actual", "forecast"]].copy()
    change["delta"] = (change["actual"] - change["forecast"]).abs()
    labels = ["quiet", "medium", "volatile"]
    change["change_group"] = pd.qcut(change["delta"], 3, labels=labels, duplicates="drop")
    analysed = data.merge(change[["date", "delta", "change_group"]], on="date", validate="many_to_one")
    analysed["absolute_error"] = (analysed["actual"] - analysed["forecast"]).abs()
    analysed["squared_error"] = (analysed["actual"] - analysed["forecast"]) ** 2
    grouped = (
        analysed.groupby(["change_group", "model"], observed=True)
        .agg(n=("absolute_error", "size"), MAE=("absolute_error", "mean"), MSE=("squared_error", "mean"))
        .reset_index()
    )
    grouped["RMSE"] = np.sqrt(grouped["MSE"])
    ref = grouped.loc[grouped["model"] == "persistence", ["change_group", "MAE", "RMSE"]].rename(
        columns={"MAE": "persistence_MAE", "RMSE": "persistence_RMSE"}
    )
    grouped = grouped.merge(ref, on="change_group", validate="many_to_one")
    grouped["MAE_ratio_to_persistence"] = grouped["MAE"] / grouped["persistence_MAE"]
    grouped["RMSE_ratio_to_persistence"] = grouped["RMSE"] / grouped["persistence_RMSE"]
    grouped["scope"] = "partial_365d_only_persistence_and_nbeatsx_lstm"
    grouped.to_csv(HERE / "error_by_change_tercile_365d_partial.csv", index=False, float_format="%.8f")

    pivot = grouped.pivot(index="change_group", columns="model", values="RMSE_ratio_to_persistence").reindex(labels)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.plot(labels, pivot["persistence"], color="black", marker="s", linewidth=2, label="Persistence")
    ax.plot(labels, pivot["nbeatsx_lstm"], color="orange", marker="^", linewidth=2, label="NBeatsx_LSTM")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Actual day-to-day change group")
    ax.set_ylabel("RMSE ratio to Persistence")
    ax.set_title("Partial 365-day analysis (three baseline checkpoints unavailable)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "rmse_ratio_by_change_tercile_365d_partial.png", dpi=300)
    fig.savefig(HERE / "rmse_ratio_by_change_tercile_365d_partial.svg")
    plt.close(fig)

    proposed = grouped.loc[grouped["model"] == "nbeatsx_lstm"].set_index("change_group")
    ratios = {group: float(proposed.loc[group, "RMSE_ratio_to_persistence"]) for group in labels}
    result = {
        "scope": "partial_365d_only_persistence_and_nbeatsx_lstm",
        "rmse_ratios": ratios,
        "five_model_e2_complete": False,
        "missing_models": ["nbeatsx", "lstm", "tcn"],
        "predeclared_mechanism_supported_for_available_pair": bool(
            ratios["volatile"] < 1 and ratios["quiet"] > 1
        ),
        "note": "This partial result must not be presented as the requested five-model E2 experiment.",
    }
    (HERE / "e2_interpretation_365d_partial.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
