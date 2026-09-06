"""Train leakage-controlled NBEATSx or NBeatsx_LSTM runs.

The final 365 rows are isolated as the test period. The preceding 730 rows are
used for validation and early stopping. Test forecasts are correctly aligned to
their target dates; the legacy one-day-shifted loop is not used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


OUT = Path(__file__).resolve().parent
PROJECT = OUT.parent
TRAIN_END = 7874
VAL_END = 8604
TEST_END = 8969
SEQUENCE_LENGTH = 7
SEEDS = [1, 2, 3, 42, 2024]
VARIANTS = {
    "original": {"learning_rate": 0.0002, "dropout_theta": 0.3, "dropout_exogenous": 0.3, "weight_decay": 0.1},
    "lower_regularization": {"learning_rate": 0.0002, "dropout_theta": 0.2, "dropout_exogenous": 0.2, "weight_decay": 0.01},
    "low_decay_low_lr": {"learning_rate": 0.0001, "dropout_theta": 0.2, "dropout_exogenous": 0.2, "weight_decay": 0.0},
}

# Union of the original pre-registered runs and the seed-2/seed-3 E4 additions.
REGISTERED_RUNS = [
    *(("nbeatsx_lstm", variant, seed) for variant in VARIANTS for seed in (1, 42, 2024)),
    *(("nbeatsx", "released_fixed_config", seed) for seed in (1, 42, 2024)),
    *(("nbeatsx", "released_fixed_config", seed) for seed in (2, 3)),
    *(("nbeatsx_lstm", "lower_regularization", seed) for seed in (2, 3)),
]


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


def import_family(name: str):
    if name == "nbeatsx_lstm":
        sys.path.insert(0, str(PROJECT))
        from nbeatsx_lstm_dataset import TimeSeriesDataset
        from nbeatsx_lstm_epf import EPF
        from nbeatsx_lstm_loader import TimeSeriesLoader
        from nbeats_LSTM import Nbeats

        y_df, x_df, s_df = EPF.load(str(PROJECT / "data"), "imputed_data_KNN_1.csv")
    else:
        source = PROJECT / "Comparison model" / "NBeatsx"
        sys.path.insert(0, str(source))
        os.chdir(PROJECT)
        from epfnbeatsx import EPF
        from nbeats import Nbeats
        from ts_dataset_nbeatsx import TimeSeriesDataset
        from ts_loader_nbeatsx import TimeSeriesLoader

        y_df, x_df, s_df = EPF.load(str(PROJECT / "data"))
    return Nbeats, TimeSeriesDataset, TimeSeriesLoader, y_df, x_df, s_df


def make_loader(TimeSeriesDataset, TimeSeriesLoader, y_df, x_df, s_df, mask, *, train: bool, shuffle: bool):
    dataset = TimeSeriesDataset(Y_df=y_df, X_df=x_df, S_df=s_df, ts_train_mask=mask)
    loader = TimeSeriesLoader(
        model="nbeats",
        ts_dataset=dataset,
        window_sampling_limit=len(y_df),
        offset=0,
        input_size=SEQUENCE_LENGTH,
        output_size=1,
        idx_to_sample_freq=1,
        batch_size=1024,
        is_train_loader=train,
        shuffle=shuffle,
    )
    return dataset, loader


def train_one(args: argparse.Namespace) -> None:
    if args.model == "nbeatsx_lstm" and args.variant not in VARIANTS:
        raise ValueError(f"Unknown proposed-model variant: {args.variant}")
    if args.model == "nbeatsx" and args.variant != "released_fixed_config":
        raise ValueError("NBEATSx uses only released_fixed_config")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for directory in (OUT / "runs", OUT / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    # Compatibility with the released implementation under current NumPy.
    if not hasattr(np, "float"):
        np.float = float

    Nbeats, TimeSeriesDataset, TimeSeriesLoader, y_df, x_df, s_df = import_family(args.model)
    if len(y_df) != TEST_END:
        raise RuntimeError(f"Expected {TEST_END} rows, found {len(y_df)}")
    string_columns = list(x_df.select_dtypes(include=[object, "category"]).columns)
    string_column_count = len(string_columns)
    for column in string_columns:
        encoder = LabelEncoder().fit(x_df[column].iloc[:TRAIN_END].astype(str))
        x_df[column] = encoder.transform(x_df[column].astype(str))

    pre_y, pre_x = y_df.iloc[:VAL_END].copy(), x_df.iloc[:VAL_END].copy()
    pre_mask = np.ones(len(pre_y))
    pre_mask[TRAIN_END:] = 0
    pre_dataset, train_loader = make_loader(
        TimeSeriesDataset, TimeSeriesLoader, pre_y, pre_x, s_df, pre_mask, train=True, shuffle=(args.model == "nbeatsx")
    )
    _, val_loader = make_loader(
        TimeSeriesDataset, TimeSeriesLoader, pre_y, pre_x, s_df, pre_mask, train=False, shuffle=False
    )
    test_mask = np.ones(len(y_df))
    test_mask[VAL_END:] = 0
    _, test_loader = make_loader(
        TimeSeriesDataset, TimeSeriesLoader, y_df, x_df, s_df, test_mask, train=False, shuffle=False
    )

    include_var_dict = {
        "y": list(range(-7, 0)),
        "GustDir": list(range(-7, 0)),
        "GustSpd": list(range(-7, 0)),
        "WindRun": list(range(-7, 0)),
        "Rain": list(range(-7, 0)),
        "Tmean": list(range(-7, 0)),
        "Tmax": list(range(-7, 0)),
        "Tmin": list(range(-7, 0)),
        "Tgmin": list(range(-7, 0)),
        "VapPress": list(range(-7, 0)),
        "ET10": list(range(-7, 0)),
        "Rad": list(range(-7, 0)),
        "week_day": [-1],
    }

    if args.model == "nbeatsx":
        config = {"learning_rate": 0.0001, "dropout_theta": 0.2, "dropout_exogenous": 0.2, "weight_decay": 0.0}
        model = Nbeats(
            input_size_multiplier=7,
            output_size=1,
            shared_weights=False,
            initialization="he_normal",
            activation="relu",
            stack_types=["trend", "exogenous_tcn"],
            n_blocks=[2, 2],
            n_layers=[2, 2],
            n_hidden=[[512, 512], [512, 512]],
            n_harmonics=0,
            n_polynomials=0,
            x_s_n_hidden=0,
            exogenous_n_channels=9,
            include_var_dict=include_var_dict,
            t_cols=pre_dataset.t_cols,
            batch_normalization=True,
            dropout_prob_theta=config["dropout_theta"],
            dropout_prob_exogenous=config["dropout_exogenous"],
            learning_rate=config["learning_rate"],
            lr_decay=0.6,
            n_lr_decay_steps=5,
            early_stopping=10,
            weight_decay=config["weight_decay"],
            l1_theta=0,
            n_iterations=args.max_iterations,
            loss="MAE",
            loss_hypar=0.5,
            val_loss="MAE",
            seasonality=7,
            random_seed=args.seed,
        )
        eval_steps = 100
    else:
        config = VARIANTS[args.variant]
        model = Nbeats(
            input_size_multiplier=7,
            output_size=1,
            shared_weights=False,
            initialization="he_normal",
            activation="relu",
            stack_types=["trend", "exogenous_lstm", "seasonality"],
            n_blocks=[2, 2, 2],
            n_layers=[2, 2, 2],
            n_hidden=[[512, 512], [512, 512], [512, 512]],
            n_harmonics=1,
            n_polynomials=4,
            x_s_n_hidden=0,
            exogenous_n_channels=string_column_count,
            include_var_dict=include_var_dict,
            t_cols=pre_dataset.t_cols,
            batch_normalization=True,
            dropout_prob_theta=config["dropout_theta"],
            dropout_prob_exogenous=config["dropout_exogenous"],
            learning_rate=config["learning_rate"],
            lr_decay=0.9,
            n_lr_decay_steps=50,
            early_stopping=10,
            weight_decay=config["weight_decay"],
            l1_theta=0,
            n_iterations=args.max_iterations,
            loss="MAE",
            loss_hypar=0.5,
            val_loss="MAE",
            seasonality=7,
            random_seed=args.seed,
        )
        eval_steps = 50

    started = time.time()
    model.fit(train_ts_loader=train_loader, val_ts_loader=val_loader, eval_steps=eval_steps)
    elapsed = time.time() - started
    run_id = f"{args.model}_{args.variant}_seed{args.seed}"
    checkpoint = OUT / "checkpoints" / f"{run_id}.pth"
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "model": args.model,
            "variant": args.variant,
            "seed": args.seed,
            "config": config,
            "train_end": TRAIN_END,
            "validation_end": VAL_END,
            "test_start": VAL_END,
            "trajectories": model.trajectories,
        },
        checkpoint,
    )
    pd.DataFrame(model.trajectories).to_csv(OUT / "runs" / f"{run_id}_history.csv", index=False)

    metric_rows = []
    forecast_files = {}
    for split_name, loader, expected_dates in [
        ("validation", val_loader, y_df["ds"].iloc[TRAIN_END:VAL_END]),
        ("test", test_loader, y_df["ds"].iloc[VAL_END:TEST_END]),
    ]:
        y_true, y_hat, *_ = model.predict(ts_loader=loader, return_decomposition=False)
        actual = np.asarray(y_true).reshape(-1)
        forecast = np.asarray(y_hat).reshape(-1)
        if len(actual) != len(expected_dates):
            raise RuntimeError(f"{split_name}: expected {len(expected_dates)} outputs, got {len(actual)}")
        if not np.allclose(actual, y_df["y"].iloc[expected_dates.index].to_numpy(dtype=float), atol=2e-6, rtol=0):
            raise RuntimeError(f"{split_name}: loader targets do not align with expected dates")
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(expected_dates).dt.strftime("%Y-%m-%d").to_numpy(),
                "model": args.model,
                "variant": args.variant,
                "seed": args.seed,
                "actual": actual,
                "forecast": forecast,
                "split": split_name,
            }
        )
        output_path = OUT / "runs" / f"{run_id}_{split_name}_forecasts.csv"
        frame.to_csv(output_path, index=False, float_format="%.10f")
        forecast_files[split_name] = str(output_path)
        metric_rows.append({"split": split_name, **metrics(actual, forecast)})

    metadata = {
        "run_id": run_id,
        "model": args.model,
        "variant": args.variant,
        "seed": args.seed,
        "config": config,
        "train_rows_raw": TRAIN_END,
        "validation_rows": VAL_END - TRAIN_END,
        "test_rows": TEST_END - VAL_END,
        "train_end_date": y_df["ds"].iloc[TRAIN_END - 1].strftime("%Y-%m-%d"),
        "validation_end_date": y_df["ds"].iloc[VAL_END - 1].strftime("%Y-%m-%d"),
        "test_start_date": y_df["ds"].iloc[VAL_END].strftime("%Y-%m-%d"),
        "elapsed_seconds": elapsed,
        "iterations_recorded": model.trajectories.get("iteration", []),
        "checkpoint": str(checkpoint),
        "forecast_files": forecast_files,
        "metrics": metric_rows,
        "test_used_for_training_or_selection": False,
        "date_alignment": "loader target date equals CSV date; legacy one-day shift not used",
    }
    (OUT / "runs" / f"{run_id}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def run_registered_experiments(max_iterations: int = 5000) -> None:
    """Run the complete registered NBEATS family matrix without a shell wrapper."""
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    for model_name, variant, seed in REGISTERED_RUNS:
        run_id = f"{model_name}_{variant}_seed{seed}"
        metadata_path = OUT / "runs" / f"{run_id}_metadata.json"
        if metadata_path.exists():
            print(f"SKIP completed {run_id}", flush=True)
            continue
        print(f"START {run_id}", flush=True)
        train_one(
            argparse.Namespace(
                model=model_name,
                variant=variant,
                seed=seed,
                max_iterations=max_iterations,
            )
        )
        print(f"COMPLETE {run_id}", flush=True)
    print("ALL_REGISTERED_NBEATS_RUNS_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct execution runs the complete registered NBEATS experiment matrix."
    )
    parser.add_argument("--model", choices=["nbeatsx", "nbeatsx_lstm"])
    parser.add_argument("--variant")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--max-iterations", type=int, default=5000)
    args = parser.parse_args()

    if args.model is None and args.variant is None and args.seed is None:
        run_registered_experiments(args.max_iterations)
        return
    if args.model is None or args.seed is None:
        parser.error("single-run mode requires both --model and --seed")
    args.variant = args.variant or "released_fixed_config"
    train_one(args)


if __name__ == "__main__":
    main()
