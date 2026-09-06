"""Train the pre-registered E5 horizon and E6 loss-ablation runs."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
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
INPUT_SIZE = 7
SEEDS = [1, 2, 3, 42, 2024]
HORIZONS = [1, 3, 7, 14]
CONFIG = {
    "learning_rate": 0.0002,
    "dropout_theta": 0.2,
    "dropout_exogenous": 0.2,
    "weight_decay": 0.01,
}

REGISTERED_RUNS = [
    *(('e5', horizon, 'MAE', seed) for horizon in (3, 7, 14) for seed in SEEDS),
    *(('e6', 1, 'MSE', seed) for seed in SEEDS),
]


def run_registered_experiments(max_iterations: int) -> None:
    """Run every pre-registered E5/E6 configuration, skipping completed runs."""
    runs_dir = OUT / "runs"
    logs_dir = OUT / "logs"
    checkpoints_dir = OUT / "checkpoints"
    for directory in (runs_dir, logs_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    status_path = logs_dir / "batch_status.txt"

    def record(message: str) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        line = f"{timestamp} {message}"
        print(line, flush=True)
        with status_path.open("a", encoding="utf-8") as status_file:
            status_file.write(line + "\n")

    for experiment, horizon, loss_name, seed in REGISTERED_RUNS:
        run_id = f"{experiment}_h{horizon}_{loss_name.lower()}_seed{seed}"
        metadata_path = runs_dir / f"{run_id}_metadata.json"
        if metadata_path.exists():
            record(f"SKIP completed {run_id}")
            continue

        log_path = logs_dir / f"{run_id}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--experiment", experiment,
            "--horizon", str(horizon),
            "--loss", loss_name,
            "--seed", str(seed),
            "--max-iterations", str(max_iterations),
        ]
        record(f"START {run_id}")
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                command,
                cwd=PROJECT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            record(f"FAILED {run_id} exit={completed.returncode}; see {log_path}")
            raise SystemExit(completed.returncode)
        record(f"COMPLETE {run_id}")

    record("ALL_E5_E6_RUNS_COMPLETE")


def metrics(actual: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    forecast = np.asarray(forecast, dtype=float).reshape(-1)
    error = actual - forecast
    mse = float(np.mean(error**2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(np.mean(np.abs(error) / ((np.abs(actual) + np.abs(forecast)) / 2)) * 100),
        "R2": float(1 - np.sum(error**2) / np.sum((actual - actual.mean()) ** 2)),
    }


def import_model_family():
    sys.path.insert(0, str(PROJECT))
    from nbeatsx_lstm_dataset import TimeSeriesDataset
    from nbeatsx_lstm_epf import EPF
    from nbeatsx_lstm_loader import TimeSeriesLoader
    from nbeats_LSTM import Nbeats

    y_df, x_df, s_df = EPF.load(str(PROJECT / "data"), "imputed_data_KNN_1.csv")
    return Nbeats, TimeSeriesDataset, TimeSeriesLoader, y_df, x_df, s_df


def make_full_split_loader(
    TimeSeriesDataset,
    TimeSeriesLoader,
    y_df,
    x_df,
    s_df,
    mask,
    horizon: int,
    *,
    train: bool,
    shuffle: bool,
):
    dataset = TimeSeriesDataset(Y_df=y_df, X_df=x_df, S_df=s_df, ts_train_mask=mask)
    loader = TimeSeriesLoader(
        model="nbeats",
        ts_dataset=dataset,
        window_sampling_limit=len(y_df),
        offset=0,
        input_size=INPUT_SIZE,
        output_size=horizon,
        idx_to_sample_freq=1,
        batch_size=1024,
        is_train_loader=train,
        shuffle=shuffle,
    )
    outsample_index = loader.t_cols.index("outsample_mask")
    insample_index = loader.t_cols.index("insample_mask")
    outsample_count = loader.ts_windows[:, outsample_index, -horizon:].sum(axis=1)
    insample_count = loader.ts_windows[:, insample_index, :INPUT_SIZE].sum(axis=1)
    full = torch.nonzero((outsample_count == horizon) & (insample_count == INPUT_SIZE)).flatten().numpy().tolist()
    loader.windows_sampling_idx = full
    if not full:
        raise RuntimeError("No complete windows remain after split-boundary filtering")
    return dataset, loader


def expected_matrix(values: np.ndarray, start: int, end: int, horizon: int) -> np.ndarray:
    return np.stack([values[index : index + horizon] for index in range(start, end - horizon + 1)])


def save_forecasts(
    model,
    loader,
    split: str,
    start: int,
    end: int,
    horizon: int,
    seed: int,
    experiment: str,
    loss_name: str,
    dates: pd.Series,
    target_values: np.ndarray,
    run_id: str,
) -> tuple[Path, dict[str, float], list[dict[str, float]]]:
    y_true, y_hat, mask = model.predict(ts_loader=loader, return_decomposition=False)
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    mask = np.asarray(mask, dtype=float)
    expected = expected_matrix(target_values, start, end, horizon)
    if y_true.shape != expected.shape or y_hat.shape != expected.shape:
        raise RuntimeError(f"{run_id} {split}: expected {expected.shape}, got y={y_true.shape}, yhat={y_hat.shape}")
    if not np.allclose(mask, 1.0, atol=0, rtol=0):
        raise RuntimeError(f"{run_id} {split}: incomplete horizon mask survived filtering")
    if not np.allclose(y_true, expected, atol=2e-6, rtol=0):
        raise RuntimeError(f"{run_id} {split}: target/date alignment failed")

    rows = []
    for origin_number, target_start in enumerate(range(start, end - horizon + 1)):
        for lead in range(1, horizon + 1):
            target_index = target_start + lead - 1
            rows.append(
                {
                    "experiment": experiment,
                    "horizon": horizon,
                    "loss": loss_name,
                    "seed": seed,
                    "split": split,
                    "forecast_origin_date": dates.iloc[target_start - 1].strftime("%Y-%m-%d"),
                    "target_date": dates.iloc[target_index].strftime("%Y-%m-%d"),
                    "lead": lead,
                    "actual": y_true[origin_number, lead - 1],
                    "forecast": y_hat[origin_number, lead - 1],
                }
            )
    frame = pd.DataFrame(rows)
    path = OUT / "runs" / f"{run_id}_{split}_forecasts.csv"
    frame.to_csv(path, index=False, float_format="%.10f")
    aggregate = metrics(frame.actual.to_numpy(), frame.forecast.to_numpy())
    by_lead = []
    for lead, group in frame.groupby("lead", sort=True):
        by_lead.append({"lead": int(lead), **metrics(group.actual.to_numpy(), group.forecast.to_numpy())})
    return path, aggregate, by_lead


def basis_payload(model) -> dict[str, object]:
    payload: dict[str, object] = {}
    for block in model.model.blocks:
        basis = block.basis
        name = basis.__class__.__name__
        if name in {"TrendBasis", "SeasonalityBasis"} and name not in payload:
            payload[name] = {
                "backcast_basis": basis.backcast_basis.detach().cpu().numpy().tolist(),
                "forecast_basis": basis.forecast_basis.detach().cpu().numpy().tolist(),
            }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-registered",
        action="store_true",
        help="run all pre-registered E5/E6 configurations and skip completed runs",
    )
    parser.add_argument("--experiment", choices=["e5", "e6"])
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--loss", choices=["MAE", "MSE"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--shape-audit-only", action="store_true")
    args = parser.parse_args()

    if args.run_registered:
        if any(value is not None for value in (args.experiment, args.horizon, args.loss, args.seed)):
            parser.error("--run-registered cannot be combined with single-run arguments")
        if args.shape_audit_only:
            parser.error("--shape-audit-only requires a single-run configuration")
        run_registered_experiments(args.max_iterations)
        return

    if any(value is None for value in (args.experiment, args.horizon, args.loss, args.seed)):
        parser.error("single-run mode requires --experiment, --horizon, --loss and --seed")
    if args.experiment == "e5" and (args.horizon == 1 or args.loss != "MAE"):
        raise ValueError("New E5 training is only required for MAE horizons 3, 7 and 14; h=1 reuses E4")
    if args.experiment == "e6" and (args.horizon != 1 or args.loss != "MSE"):
        raise ValueError("New E6 training is fixed to h=1 with MSE; MAE reuses E4")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    for directory in (OUT / "runs", OUT / "logs", OUT / "checkpoints"):
        directory.mkdir(parents=True, exist_ok=True)
    if not hasattr(np, "float"):
        np.float = float

    Nbeats, TimeSeriesDataset, TimeSeriesLoader, y_df, x_df, s_df = import_model_family()
    if len(y_df) != TEST_END:
        raise RuntimeError(f"Expected {TEST_END} rows, found {len(y_df)}")
    dates = pd.to_datetime(y_df["ds"])
    target_values = y_df["y"].to_numpy(dtype=float)
    string_columns = list(x_df.select_dtypes(include=[object, "category"]).columns)
    for column in string_columns:
        encoder = LabelEncoder().fit(x_df[column].iloc[:TRAIN_END].astype(str))
        x_df[column] = encoder.transform(x_df[column].astype(str))

    pre_y, pre_x = y_df.iloc[:VAL_END].copy(), x_df.iloc[:VAL_END].copy()
    pre_mask = np.ones(len(pre_y))
    pre_mask[TRAIN_END:] = 0
    pre_dataset, train_loader = make_full_split_loader(
        TimeSeriesDataset,
        TimeSeriesLoader,
        pre_y,
        pre_x,
        s_df,
        pre_mask,
        args.horizon,
        train=True,
        shuffle=False,
    )
    _, validation_loader = make_full_split_loader(
        TimeSeriesDataset,
        TimeSeriesLoader,
        pre_y,
        pre_x,
        s_df,
        pre_mask,
        args.horizon,
        train=False,
        shuffle=False,
    )
    test_mask = np.ones(len(y_df))
    test_mask[VAL_END:] = 0
    _, test_loader = make_full_split_loader(
        TimeSeriesDataset,
        TimeSeriesLoader,
        y_df,
        x_df,
        s_df,
        test_mask,
        args.horizon,
        train=False,
        shuffle=False,
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
    model = Nbeats(
        input_size_multiplier=INPUT_SIZE / args.horizon,
        output_size=args.horizon,
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
        exogenous_n_channels=len(string_columns),
        include_var_dict=include_var_dict,
        t_cols=pre_dataset.t_cols,
        batch_normalization=True,
        dropout_prob_theta=CONFIG["dropout_theta"],
        dropout_prob_exogenous=CONFIG["dropout_exogenous"],
        learning_rate=CONFIG["learning_rate"],
        lr_decay=0.9,
        n_lr_decay_steps=50,
        early_stopping=10,
        weight_decay=CONFIG["weight_decay"],
        l1_theta=0,
        n_iterations=args.max_iterations,
        loss=args.loss,
        loss_hypar=0.5,
        val_loss=args.loss,
        seasonality=7,
        random_seed=args.seed,
    )
    if model.input_size != INPUT_SIZE:
        raise RuntimeError(f"Input window changed unexpectedly: {model.input_size}")

    if args.shape_audit_only:
        model.fit(train_ts_loader=train_loader, val_ts_loader=validation_loader, n_iterations=0, eval_steps=50)
        batch = next(iter(train_loader))
        with torch.no_grad():
            forecast = model.model(
                x_s=model.to_tensor(batch["s_matrix"]),
                insample_y=model.to_tensor(batch["insample_y"]),
                insample_x_t=model.to_tensor(batch["insample_x"]),
                outsample_x_t=model.to_tensor(batch["outsample_x"]),
                insample_mask=model.to_tensor(batch["insample_mask"]),
            )
        audit = {
            "horizon": args.horizon,
            "loss": args.loss,
            "batch_insample_y_shape": list(batch["insample_y"].shape),
            "batch_insample_x_shape": list(batch["insample_x"].shape),
            "batch_outsample_x_shape": list(batch["outsample_x"].shape),
            "forecast_shape": list(forecast.shape),
            "first_block_mlp_in_features": model.model.blocks[0].layers[0].in_features,
            "basis": basis_payload(model),
        }
        audit_path = OUT / f"forward_shape_audit_h{args.horizon}.json"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(json.dumps(audit, indent=2))
        return

    started = time.time()
    model.fit(train_ts_loader=train_loader, val_ts_loader=validation_loader, eval_steps=50)
    elapsed = time.time() - started
    run_id = f"{args.experiment}_h{args.horizon}_{args.loss.lower()}_seed{args.seed}"
    checkpoint = OUT / "checkpoints" / f"{run_id}.pth"
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "experiment": args.experiment,
            "horizon": args.horizon,
            "loss": args.loss,
            "seed": args.seed,
            "config": CONFIG,
            "input_size": INPUT_SIZE,
            "train_end": TRAIN_END,
            "validation_end": VAL_END,
            "test_start": VAL_END,
            "trajectories": model.trajectories,
        },
        checkpoint,
    )
    pd.DataFrame(model.trajectories).to_csv(OUT / "runs" / f"{run_id}_history.csv", index=False)

    validation_path, validation_metrics, validation_by_lead = save_forecasts(
        model,
        validation_loader,
        "validation",
        TRAIN_END,
        VAL_END,
        args.horizon,
        args.seed,
        args.experiment,
        args.loss,
        dates,
        target_values,
        run_id,
    )
    test_path, test_metrics, test_by_lead = save_forecasts(
        model,
        test_loader,
        "test",
        VAL_END,
        TEST_END,
        args.horizon,
        args.seed,
        args.experiment,
        args.loss,
        dates,
        target_values,
        run_id,
    )
    basis = basis_payload(model)
    if args.horizon == 7:
        basis_path = OUT / "runs" / f"{run_id}_actual_basis.json"
        basis_path.write_text(json.dumps(basis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        basis_path = None

    metadata = {
        "run_id": run_id,
        "experiment": args.experiment,
        "model": "NBeatsx_LSTM",
        "variant": "lower_regularization",
        "horizon": args.horizon,
        "input_window_days": INPUT_SIZE,
        "training_loss": args.loss,
        "validation_loss": args.loss,
        "seed": args.seed,
        "config": CONFIG,
        "train_end_date": dates.iloc[TRAIN_END - 1].strftime("%Y-%m-%d"),
        "validation_end_date": dates.iloc[VAL_END - 1].strftime("%Y-%m-%d"),
        "test_start_date": dates.iloc[VAL_END].strftime("%Y-%m-%d"),
        "test_end_date": dates.iloc[TEST_END - 1].strftime("%Y-%m-%d"),
        "complete_training_windows": len(train_loader.windows_sampling_idx),
        "complete_validation_origins": len(validation_loader.windows_sampling_idx),
        "complete_test_origins": len(test_loader.windows_sampling_idx),
        "test_forecast_cells": len(test_loader.windows_sampling_idx) * args.horizon,
        "elapsed_seconds": elapsed,
        "iterations_recorded": model.trajectories.get("iteration", []),
        "checkpoint": str(checkpoint),
        "forecast_files": {"validation": str(validation_path), "test": str(test_path)},
        "basis_file": str(basis_path) if basis_path else None,
        "metrics": {"validation": validation_metrics, "test": test_metrics},
        "test_metrics_by_lead": test_by_lead,
        "validation_metrics_by_lead": validation_by_lead,
        "test_used_for_training_early_stopping_or_selection": False,
        "full_horizon_within_split_required": True,
        "exogenous_information": "observed oracle weather, unchanged from E4",
    }
    (OUT / "runs" / f"{run_id}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
