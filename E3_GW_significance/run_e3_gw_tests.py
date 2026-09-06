"""Run the pre-specified E3 Giacomini-White tests without retraining.

The test uses the same conditional-instrument construction as the released
``nbeatsx_lstm_metrics.py`` implementation for horizon one: a constant and
the lagged loss differential.  The repository implementation multiplies the
non-negative chi-square statistic by the direction of the mean loss
differential.  That convention makes the p-value asymmetric under model-order
reversal.  Here the standard non-negative statistic is used for the equal-
predictive-ability p-value, while direction is reported separately.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = Path(
    r"F:\nbeatsx-lstm\E1_E2_clean_365d_holdout"
    r"\forecasts_365d_selected_config_all_seeds.csv"
)
SEEDS = (1, 42, 2024)
WINDOWS = (28, 365)
LOSSES = ("absolute_error", "squared_error")
COMPARATORS = ("persistence", "nbeatsx")
ALPHA = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm step-down adjusted p-values in original row order."""

    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values, kind="stable")
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def repository_compatible_gw(loss_model: np.ndarray, loss_comparator: np.ndarray) -> dict:
    """Compute the horizon-one conditional GW statistic used by the repository."""

    if loss_model.shape != loss_comparator.shape:
        raise ValueError("Loss arrays do not have identical shapes")
    if loss_model.ndim != 1 or len(loss_model) < 4:
        raise ValueError("GW test requires one-dimensional loss arrays with at least four rows")

    differential = loss_model - loss_comparator
    current = differential[1:]
    instruments = np.column_stack([np.ones(len(current)), differential[:-1]])
    moments = instruments * current[:, None]
    effective_n = len(current)

    rank = int(np.linalg.matrix_rank(moments))
    if rank < 2:
        raise ValueError(f"GW moment matrix is rank deficient: rank={rank}")

    coefficients = np.linalg.lstsq(moments, np.ones(effective_n), rcond=None)[0]
    residual = np.ones(effective_n) - moments @ coefficients
    r_squared = 1.0 - float(np.mean(residual**2))
    statistic = max(0.0, effective_n * r_squared)
    # For two instruments the reference law is chi-square(2), whose exact
    # survival function is exp(-x/2).  This avoids an unnecessary SciPy
    # dependency while remaining mathematically identical.
    p_value = float(math.exp(-statistic / 2.0))
    mean_difference = float(np.mean(differential))

    return {
        "statistic": statistic,
        "p_value": p_value,
        "mean_loss_difference": mean_difference,
        "signed_repository_statistic": statistic * float(np.sign(mean_difference)),
        "n_effective": effective_n,
        "moment_rank": rank,
        "moment_condition_number": float(np.linalg.cond(moments)),
    }


def select_series(data: pd.DataFrame, model: str, seed: int | None) -> pd.DataFrame:
    selected = data[data["model"].eq(model)].copy()
    if seed is not None:
        selected = selected[selected["seed"].eq(seed)]
    if selected["date"].duplicated().any():
        raise ValueError(f"Duplicate dates for model={model}, seed={seed}")
    return selected.set_index("date").sort_index()


def loss_values(actual: np.ndarray, forecast: np.ndarray, loss_name: str) -> np.ndarray:
    error = actual - forecast
    if loss_name == "absolute_error":
        return np.abs(error)
    if loss_name == "squared_error":
        return np.square(error)
    raise ValueError(f"Unknown loss: {loss_name}")


def main() -> None:
    data = pd.read_csv(INPUT, parse_dates=["date"])
    required = {"date", "model", "variant", "seed", "actual", "forecast", "split"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing input columns: {sorted(required - set(data.columns))}")
    if data[list(required)].isna().any().any():
        raise ValueError("Input contains missing values")
    if set(data["split"].unique()) != {"test"}:
        raise ValueError("E3 input must contain test rows only")

    persistence = select_series(data, "persistence", seed=None)
    if len(persistence) != 365:
        raise ValueError(f"Expected 365 Persistence rows, found {len(persistence)}")
    canonical_dates = persistence.index
    canonical_actual = persistence["actual"].to_numpy(dtype=float)

    rows: list[dict] = []
    for window in WINDOWS:
        dates = canonical_dates[-window:]
        actual = canonical_actual[-window:]
        persistence_forecast = persistence.loc[dates, "forecast"].to_numpy(dtype=float)

        for seed in SEEDS:
            proposed = select_series(data, "nbeatsx_lstm", seed=seed).loc[dates]
            nbeatsx = select_series(data, "nbeatsx", seed=seed).loc[dates]
            if not proposed.index.equals(dates) or not nbeatsx.index.equals(dates):
                raise ValueError(f"Date mismatch in window={window}, seed={seed}")

            model_forecast = proposed["forecast"].to_numpy(dtype=float)
            comparator_forecasts = {
                "persistence": persistence_forecast,
                "nbeatsx": nbeatsx["forecast"].to_numpy(dtype=float),
            }

            for comparator in COMPARATORS:
                comparator_forecast = comparator_forecasts[comparator]
                for loss_name in LOSSES:
                    model_loss = loss_values(actual, model_forecast, loss_name)
                    comparator_loss = loss_values(actual, comparator_forecast, loss_name)
                    result = repository_compatible_gw(model_loss, comparator_loss)
                    mean_model = float(model_loss.mean())
                    mean_comparator = float(comparator_loss.mean())
                    improvement = (
                        float((mean_comparator - mean_model) / mean_comparator * 100.0)
                        if mean_comparator != 0
                        else np.nan
                    )
                    if result["mean_loss_difference"] < 0:
                        better_model = "nbeatsx_lstm"
                    elif result["mean_loss_difference"] > 0:
                        better_model = comparator
                    else:
                        better_model = "tie"

                    rows.append(
                        {
                            "window_days": window,
                            "start_date": dates.min().date().isoformat(),
                            "end_date": dates.max().date().isoformat(),
                            "seed": seed,
                            "pair": f"nbeatsx_lstm_vs_{comparator}",
                            "model": "nbeatsx_lstm",
                            "comparator": comparator,
                            "loss": loss_name,
                            "n_observations": window,
                            "n_effective": result["n_effective"],
                            "tau": 1,
                            "conditional_instruments": "constant+lag1_loss_difference",
                            "chi_square_df": 2,
                            "mean_loss_model": mean_model,
                            "mean_loss_comparator": mean_comparator,
                            "mean_loss_difference": result["mean_loss_difference"],
                            "relative_loss_improvement_percent": improvement,
                            "better_model_by_mean_loss": better_model,
                            "statistic": result["statistic"],
                            "signed_repository_statistic": result[
                                "signed_repository_statistic"
                            ],
                            "p_value": result["p_value"],
                            "moment_rank": result["moment_rank"],
                            "moment_condition_number": result[
                                "moment_condition_number"
                            ],
                        }
                    )

    results = pd.DataFrame(rows)
    results["holm_family"] = (
        results["window_days"].astype(str) + "d_seed" + results["seed"].astype(str)
    )
    results["p_holm"] = np.nan
    for _, indexes in results.groupby("holm_family", sort=False).groups.items():
        index_list = list(indexes)
        results.loc[index_list, "p_holm"] = holm_adjust(
            results.loc[index_list, "p_value"].to_numpy()
        )
    results["p_holm_global"] = holm_adjust(results["p_value"].to_numpy())
    results["reject_0_05_raw"] = results["p_value"] < ALPHA
    results["reject_0_05_holm"] = results["p_holm"] < ALPHA
    results["reject_0_05_holm_global"] = results["p_holm_global"] < ALPHA

    results = results.sort_values(
        ["window_days", "seed", "pair", "loss"], kind="stable"
    ).reset_index(drop=True)
    output_csv = HERE / "gw_tests.csv"
    results.to_csv(output_csv, index=False, float_format="%.12g")

    maximum_actual_spread = float(
        data.groupby("date")["actual"].agg(lambda values: values.max() - values.min()).max()
    )
    audit = {
        "status": "PASS",
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "input_rows": int(len(data)),
        "input_date_range": [
            data["date"].min().date().isoformat(),
            data["date"].max().date().isoformat(),
        ],
        "fixed_seeds": list(SEEDS),
        "windows_days": list(WINDOWS),
        "model_pairs": [
            "nbeatsx_lstm_vs_persistence",
            "nbeatsx_lstm_vs_nbeatsx",
        ],
        "losses": list(LOSSES),
        "expected_tests": 24,
        "actual_tests": int(len(results)),
        "tests_per_holm_family": 4,
        "holm_family_definition": "one family per window_days and seed",
        "additional_sensitivity_adjustment": "Holm across all 24 tests",
        "alpha": ALPHA,
        "actual_value_policy": "Use the deterministic Persistence row as the common actual value for every model comparison.",
        "maximum_framework_actual_value_spread": maximum_actual_spread,
        "retraining_or_tuning_performed": False,
        "test_definition": {
            "forecast_horizon": 1,
            "conditional_instruments": ["constant", "lag1_loss_difference"],
            "effective_sample_size": "window_days - 1",
            "reference_distribution": "chi-square with 2 degrees of freedom",
            "statistic": "repository-compatible T*R2 without the directional sign",
            "direction": "reported separately using mean_loss_difference",
        },
    }
    (HERE / "input_and_method_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary_rows = []
    for window in WINDOWS:
        subset = results[results["window_days"].eq(window)]
        summary_rows.append(
            {
                "window_days": window,
                "tests": int(len(subset)),
                "raw_rejections": int(subset["reject_0_05_raw"].sum()),
                "holm_family_rejections": int(subset["reject_0_05_holm"].sum()),
                "holm_global_rejections": int(
                    subset["reject_0_05_holm_global"].sum()
                ),
            }
        )
    summary = {
        "status": "PASS",
        "total_tests": int(len(results)),
        "summary_by_window": summary_rows,
        "all_moment_matrices_full_rank": bool(results["moment_rank"].eq(2).all()),
        "result_csv": str(output_csv),
    }
    (HERE / "e3_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    columns = [
        "window_days",
        "seed",
        "pair",
        "loss",
        "mean_loss_difference",
        "relative_loss_improvement_percent",
        "better_model_by_mean_loss",
        "statistic",
        "p_value",
        "p_holm",
        "p_holm_global",
        "reject_0_05_holm",
    ]
    printout = results[columns].to_string(
        index=False, float_format=lambda value: f"{value:.6g}"
    )
    (HERE / "gw_tests_printout.txt").write_text(
        printout + "\n\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(printout)
    print("\n", json.dumps(summary, indent=2, ensure_ascii=False), sep="")


if __name__ == "__main__":
    main()
