"""Command line runner for the paper's complete experiment matrix."""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .fmdnn import BUILDERS
from .models.raaf_fmdnn import build_raaf_fmdnn
from .preprocess import ROOT, load_experiment, save_preprocessing_manifest
from .train import train_and_evaluate

PAPER_RUNS = {
    "tlm": [("rate", "none", "multiclass"), ("gps", "none", "multiclass"), ("imu", "none", "multiclass"), ("all", "ifhs", "multiclass")],
    "alfa": [("imu", "none", "binary_engine"), ("velocity", "none", "binary_engine"), ("pitch_roll_yaw", "none", "binary_engine"), ("all", "ifhs", "multiclass")],
    "uav": [("sensor_combined", "none", "binary_spoofing"), ("sensor_combined", "none", "binary_jamming"), ("sensor_combined", "none", "multiclass"), ("sensor_combined", "ifhs", "multiclass")],
}


def _split(experiment, seed, strategy="paper"):
    encoder = LabelEncoder(); y = encoder.fit_transform(experiment.y); indices = np.arange(len(y))
    if strategy == "paper":
        train_idx, holdout = train_test_split(indices, test_size=.4, random_state=seed, stratify=y)
        val_idx, test_idx = train_test_split(holdout, test_size=.5, random_state=seed, stratify=y[holdout])
    else:
        # Contiguous per-flight blocks prevent neighboring rows from the same
        # log appearing in both train and test. A class with one flight cannot
        # be flight-group held out; this is a time-block diagnostic only.
        train_idx, val_idx, test_idx = [], [], []
        for group in pd.unique(experiment.groups):
            block = indices[experiment.groups == group]
            block = block[np.argsort(experiment.timestamps[block])]
            n = len(block); a = int(n * .6); b = int(n * .8)
            train_idx.extend(block[:a]); val_idx.extend(block[a:b]); test_idx.extend(block[b:])
        train_idx, val_idx, test_idx = map(np.asarray, (train_idx, val_idx, test_idx))
    values = experiment.x.to_numpy(dtype="float32"); scaler = StandardScaler()
    return {"x_train": scaler.fit_transform(values[train_idx]), "y_train": y[train_idx], "x_val": scaler.transform(values[val_idx]), "y_val": y[val_idx], "x_test": scaler.transform(values[test_idx]), "y_test": y[test_idx], "test_timestamps": experiment.timestamps[test_idx], "test_groups": experiment.groups[test_idx]}, list(encoder.classes_)


def _models(value):
    selected = [part.strip().lower() for part in value.split(",") if part.strip()]
    invalid = sorted(set(selected) - (set(BUILDERS) | {"raaf_fmdnn"}))
    if invalid: raise ValueError(f"Unknown model(s): {invalid}; choose mlp, cnn, fmdnn, raaf_fmdnn")
    return selected


def _run(args, subset, fusion, task):
    experiment = load_experiment(args.dataset, subset, fusion, task, uav_source=args.uav_source)
    split, labels = _split(experiment, args.seed, args.split_strategy)
    descriptor = f"{args.dataset}_{subset}_{fusion}_{task}_{args.split_strategy}"
    output = Path(args.output_dir) / descriptor
    save_preprocessing_manifest(output / "preprocessing_manifest.json", experiment, args.dataset, subset, fusion, task)
    pd.DataFrame({"feature": experiment.feature_names}).to_csv(output / "features_used.csv", index=False)
    print(f"\n{'=' * 80}\nDATASET: {args.dataset.upper()} | TASK: {task} | SUBSET: {subset} | FUSION: {fusion}\nTOTAL DATASET SAMPLES: {len(experiment.x)}\nFEATURES ({len(experiment.feature_names)}): {experiment.feature_names}\nCLASSES: {labels}\nSPLIT: train={len(split['y_train'])} validation={len(split['y_val'])} test={len(split['y_test'])}\nEPOCH BUDGET: {args.epochs} | VALIDATION MACRO-F1 PATIENCE: {args.early_stopping_patience} | SEED: {args.seed}\nOUTPUT: {output}")
    results = []
    for name in _models(args.models):
        print(f"\nTRAINING MODEL: {name.upper()} | DATASET: {args.dataset.upper()} | TASK: {task} | SEED: {args.seed} | EPOCH BUDGET: {args.epochs}", flush=True)
        tf.keras.backend.clear_session()
        if name == "raaf_fmdnn":
            model = build_raaf_fmdnn(split["x_train"].shape[1], len(labels), args.learning_rate, args.fusion_head)
        else:
            model = BUILDERS[name](split["x_train"].shape[1], len(labels), args.learning_rate)
        print(model.name, "parameters:", model.count_params())
        results.append(train_and_evaluate(model, name, split, labels, output, args.epochs, args.batch_size, args.class_weights, args.early_stopping_patience, total_dataset_samples=len(experiment.x), report_attack_recall=args.dataset != "tlm"))
    for row in results:
        row.update({"dataset": args.dataset, "subset": subset, "fusion": fusion, "task": task, "seed": args.seed})
    pd.DataFrame(results).to_csv(output / "results.csv", index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description="Paper-protocol FMDNN reproduction suite")
    parser.add_argument("--dataset", choices=["alfa", "tlm", "uav"], default="uav")
    parser.add_argument("--uav-source", choices=["processed", "raw"], default="processed", help="UAV only: use supplied labeled Processed CSVs (default) or legacy raw logs")
    parser.add_argument("--subset", default="all", help="ALFA: imu/velocity/pitch_roll_yaw; TLM: rate/gps/imu")
    parser.add_argument("--fusion", choices=["none", "ifhs"], default="none")
    parser.add_argument("--task", choices=["multiclass", "binary_engine", "binary_spoofing", "binary_jamming"], default="multiclass")
    parser.add_argument("--models", default="fmdnn,raaf_fmdnn")
    parser.add_argument("--fusion-head", choices=["concat", "adaptive", "raaf"], default="raaf", help="RAAF-FMDNN ablation head")
    parser.add_argument("--paper-suite", action="store_true", help="Run only the non-noisy paper comparisons")
    parser.add_argument("--split-strategy", choices=["paper", "time_blocked"], default="paper", help="paper=random stratified rows; time_blocked=contiguous per-flight diagnostic (IFHS is currently prepared before splitting)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--class-weights", action="store_true", help="Optional training-only inverse-frequency weights; not part of the paper baseline")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Stop after this many non-improving validation macro-F1 epochs; 0 disables stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "experiments"))
    args = parser.parse_args(); random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    runs = PAPER_RUNS[args.dataset] if args.paper_suite else [(args.subset, args.fusion, args.task)]
    all_results = []
    for subset, fusion, task in runs: all_results.extend(_run(args, subset, fusion, task))
    if args.paper_suite:
        path = Path(args.output_dir) / f"{args.dataset}_paper_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"; pd.DataFrame(all_results).to_csv(path, index=False); print(f"\nPaper-suite summary: {path}")


if __name__ == "__main__": main()
