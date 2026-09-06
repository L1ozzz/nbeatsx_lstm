"""Independently reconstruct and verify every saved E3 GW result."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = Path(
    r"F:\nbeatsx-lstm\E1_E2_clean_365d_holdout"
    r"\forecasts_365d_selected_config_all_seeds.csv"
)
RESULT = HERE / "gw_tests.csv"


def independent_holm(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sorted_positions = np.argsort(values)
    sorted_adjusted = np.maximum.accumulate(
        np.minimum(1.0, (len(values) - np.arange(len(values))) * values[sorted_positions])
    )
    output = np.empty(len(values), dtype=float)
    output[sorted_positions] = sorted_adjusted
    return output


def independent_statistic(model_loss: np.ndarray, comparator_loss: np.ndarray) -> tuple[float, float]:
    differential = model_loss - comparator_loss
    instruments = np.column_stack(
        [np.ones(len(differential) - 1), differential[:-1]]
    )
    moment_matrix = instruments * differential[1:, None]
    normal_matrix = moment_matrix.T @ moment_matrix
    right_hand_side = moment_matrix.T @ np.ones(len(moment_matrix))
    coefficients = np.linalg.solve(normal_matrix, right_hand_side)
    residual = np.ones(len(moment_matrix)) - moment_matrix @ coefficients
    statistic = max(0.0, len(moment_matrix) * (1.0 - np.mean(residual**2)))
    return float(statistic), float(math.exp(-statistic / 2.0))


def main() -> None:
    source = pd.read_csv(SOURCE, parse_dates=["date"])
    saved = pd.read_csv(RESULT)
    persistence = (
        source[source.model.eq("persistence")]
        .sort_values("date")
        .set_index("date")
    )

    rebuilt_rows = []
    for saved_row in saved.itertuples(index=False):
        dates = persistence.index[-int(saved_row.window_days) :]
        actual = persistence.loc[dates, "actual"].to_numpy(dtype=float)
        model = (
            source[
                source.model.eq("nbeatsx_lstm")
                & source.seed.eq(int(saved_row.seed))
            ]
            .set_index("date")
            .loc[dates]
        )
        if saved_row.comparator == "persistence":
            comparator_forecast = persistence.loc[dates, "forecast"].to_numpy(
                dtype=float
            )
        else:
            comparator_forecast = (
                source[
                    source.model.eq(saved_row.comparator)
                    & source.seed.eq(int(saved_row.seed))
                ]
                .set_index("date")
                .loc[dates, "forecast"]
                .to_numpy(dtype=float)
            )

        model_error = actual - model.forecast.to_numpy(dtype=float)
        comparator_error = actual - comparator_forecast
        if saved_row.loss == "absolute_error":
            model_loss = np.abs(model_error)
            comparator_loss = np.abs(comparator_error)
        else:
            model_loss = np.square(model_error)
            comparator_loss = np.square(comparator_error)
        statistic, p_value = independent_statistic(model_loss, comparator_loss)
        rebuilt_rows.append(
            {
                "statistic": statistic,
                "p_value": p_value,
                "mean_loss_difference": float((model_loss - comparator_loss).mean()),
            }
        )

    rebuilt = pd.DataFrame(rebuilt_rows)
    differences = {
        column: float(np.max(np.abs(saved[column].to_numpy() - rebuilt[column].to_numpy())))
        for column in ("statistic", "p_value", "mean_loss_difference")
    }

    family_adjusted = np.empty(len(saved), dtype=float)
    for _, indexes in saved.groupby("holm_family", sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        family_adjusted[positions] = independent_holm(
            saved.loc[positions, "p_value"].to_numpy()
        )
    global_adjusted = independent_holm(saved.p_value.to_numpy())

    checks = {
        "row_count_is_24": len(saved) == 24,
        "six_holm_families": saved.holm_family.nunique() == 6,
        "four_tests_per_family": bool(
            saved.groupby("holm_family").size().eq(4).all()
        ),
        "three_fixed_seeds": sorted(saved.seed.unique().tolist()) == [1, 42, 2024],
        "both_windows": sorted(saved.window_days.unique().tolist()) == [28, 365],
        "both_pairs": saved.pair.nunique() == 2,
        "both_losses": saved.loss.nunique() == 2,
        "no_missing_values": not saved.isna().any().any(),
        "all_p_values_in_unit_interval": bool(
            saved[["p_value", "p_holm", "p_holm_global"]]
            .ge(0)
            .all()
            .all()
            and saved[["p_value", "p_holm", "p_holm_global"]]
            .le(1)
            .all()
            .all()
        ),
        "statistics_match_independent_normal_equation_solution": differences[
            "statistic"
        ]
        < 1e-9,
        "p_values_match_exact_chi_square_2_survival": differences["p_value"]
        < 1e-12,
        "mean_loss_differences_match": differences["mean_loss_difference"] < 1e-9,
        "family_holm_matches_independent_recalculation": bool(
            np.allclose(saved.p_holm.to_numpy(), family_adjusted, atol=1e-12)
        ),
        "global_holm_matches_independent_recalculation": bool(
            np.allclose(saved.p_holm_global.to_numpy(), global_adjusted, atol=1e-12)
        ),
        "all_moment_matrices_full_rank": bool(saved.moment_rank.eq(2).all()),
        "28d_has_no_family_holm_rejections": int(
            saved[saved.window_days.eq(28)].reject_0_05_holm.sum()
        )
        == 0,
        "365d_has_two_family_holm_rejections": int(
            saved[saved.window_days.eq(365)].reject_0_05_holm.sum()
        )
        == 2,
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "maximum_absolute_recalculation_differences": differences,
        "largest_moment_condition_number": float(
            saved.moment_condition_number.max()
        ),
        "family_holm_rejections": saved.loc[
            saved.reject_0_05_holm,
            ["window_days", "seed", "pair", "loss", "p_value", "p_holm"],
        ].to_dict(orient="records"),
    }
    (HERE / "verification_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
