"""Training, plots, confusion matrices, and test predictions."""
from __future__ import annotations
from pathlib import Path
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from .models.raaf_fmdnn import main_output


class EpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        accuracy = logs.get("accuracy", logs.get("output_main_accuracy", float("nan")))
        val_accuracy = logs.get("val_accuracy", logs.get("val_output_main_accuracy", float("nan")))
        print(f"Epoch {epoch + 1:03d}: loss={logs.get('loss', float('nan')):.5f} accuracy={accuracy:.5f} val_loss={logs.get('val_loss', float('nan')):.5f} val_accuracy={val_accuracy:.5f}", flush=True)


class BestValidationMacroF1(tf.keras.callbacks.Callback):
    """Select the checkpoint using validation macro-F1, never the test set."""
    def __init__(self, x_val, y_val, path, patience=0):
        super().__init__(); self.x_val = x_val; self.y_val = y_val; self.path = str(path); self.best = -np.inf; self.best_epoch = 0; self.patience = patience; self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        pred = main_output(self.model.predict(self.x_val, verbose=0)).argmax(axis=1)
        score = f1_score(self.y_val, pred, average="macro", zero_division=0)
        if score > self.best:
            self.best = score; self.best_epoch = epoch + 1; self.wait = 0; self.model.save_weights(self.path)
        elif self.patience:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def _history_plot(history, path: Path, title: str):
    # Multi-output RAAF models expose output_main_accuracy instead of the
    # single-output accuracy key used by MLP/CNN/FMDNN.
    accuracy_key = next((key for key in ("accuracy", "output_main_accuracy") if key in history), None)
    val_accuracy_key = next((key for key in ("val_accuracy", "val_output_main_accuracy") if key in history), None)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if accuracy_key:
        axes[0].plot(history[accuracy_key], label="Training")
    if val_accuracy_key:
        axes[0].plot(history[val_accuracy_key], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy"); axes[0].legend(); axes[0].grid(alpha=.25)
    for key, label in (("loss", "Training"), ("val_loss", "Validation")):
        axes[1].plot(history.get(key, []), label=label)
    axes[1].set(xlabel="Epoch", ylabel="Cross entropy", title="Loss"); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _detection_table(labels, truth, predicted, total_dataset_samples=None):
    """Return per-class counts used in the final detection report."""
    rows = []
    for index, label in enumerate(labels):
        actual = truth == index
        detected = predicted == index
        correct = actual & detected
        rows.append({
            "class": str(label),
            "total_dataset_samples": int(total_dataset_samples if total_dataset_samples is not None else len(truth)),
            "test_count": int(len(truth)),
            "actual_count": int(actual.sum()),
            "detected_count": int(detected.sum()),
            "correct_detections": int(correct.sum()),
            "recall": float(correct.sum() / actual.sum()) if actual.sum() else 0.0,
            "precision": float(correct.sum() / detected.sum()) if detected.sum() else 0.0,
        })
    return pd.DataFrame(rows)


def _confusion_plot(matrix, labels, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
    image = ax.imshow(matrix, cmap="Blues"); fig.colorbar(image, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    cutoff = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > cutoff else "black")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def train_and_evaluate(model, model_name, split, labels, output_dir: Path, epochs=200, batch_size=256, class_weights=False, early_stopping_patience=0, total_dataset_samples=None, report_attack_recall=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint = output_dir / f"{model_name}_best.weights.h5"
    callbacks = [EpochLogger(), BestValidationMacroF1(split["x_val"], split["y_val"], checkpoint, early_stopping_patience)]
    weights = None
    if class_weights:
        counts = np.bincount(split["y_train"], minlength=len(labels)); total = len(split["y_train"])
        weights = {i: total / (len(labels) * max(1, count)) for i, count in enumerate(counts)}
    targets = ({"output_main": split["y_train"], "aux_m": split["y_train"], "aux_c": split["y_train"]} if getattr(model, "multitask_outputs", False) else split["y_train"])
    val_targets = ({"output_main": split["y_val"], "aux_m": split["y_val"], "aux_c": split["y_val"]} if getattr(model, "multitask_outputs", False) else split["y_val"])
    history = model.fit(split["x_train"], targets, validation_data=(split["x_val"], val_targets), epochs=epochs, batch_size=batch_size, class_weight=weights, callbacks=callbacks, verbose=0).history
    pd.DataFrame(history).to_csv(output_dir / f"{model_name}_history.csv", index_label="epoch")
    if checkpoint.exists(): model.load_weights(checkpoint)
    probabilities = main_output(model.predict(split["x_test"], batch_size=batch_size, verbose=0))
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(split["y_test"], predicted, labels=np.arange(len(labels)))
    majority = int(np.bincount(split["y_train"]).argmax())
    majority_pred = np.repeat(majority, len(split["y_test"]))
    attack_indices = [i for i, label in enumerate(labels) if label.lower() not in {"benign", "normal"}]
    attack_recall = float(recall_score(np.isin(split["y_test"], attack_indices), np.isin(predicted, attack_indices), zero_division=0)) if attack_indices and report_attack_recall else None
    result = {"model": model_name, "parameters": int(model.count_params()), "accuracy": float(accuracy_score(split["y_test"], predicted)), "majority_baseline": float(accuracy_score(split["y_test"], majority_pred)), "balanced_accuracy": float(balanced_accuracy_score(split["y_test"], predicted)), "precision_macro": float(precision_score(split["y_test"], predicted, average="macro", zero_division=0)), "recall_macro": float(recall_score(split["y_test"], predicted, average="macro", zero_division=0)), "f1_macro": float(f1_score(split["y_test"], predicted, average="macro", zero_division=0)), "attack_recall": attack_recall, "total_dataset_samples": int(total_dataset_samples if total_dataset_samples is not None else len(split["x_train"]) + len(split["x_val"]) + len(split["x_test"])), "train_samples": len(split["y_train"]), "validation_samples": len(split["y_val"]), "test_samples": len(split["y_test"]), "selected_epoch": int(callbacks[1].best_epoch), "epochs_ran": int(len(history.get("loss", []))), "best_validation_macro_f1": float(callbacks[1].best), "training_time_seconds": float(time.perf_counter() - started)}
    _history_plot(history, output_dir / f"{model_name}_history.png", f"{model_name}: training and validation")
    _confusion_plot(matrix, labels, output_dir / f"{model_name}_confusion_matrix.png", f"{model_name}: test confusion matrix")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_dir / f"{model_name}_confusion_matrix.csv")
    _detection_table(labels, split["y_test"], predicted, total_dataset_samples).to_csv(output_dir / f"{model_name}_detection_table.csv", index=False)
    pd.DataFrame({"timestamp": split["test_timestamps"], "group": split["test_groups"], "true": [labels[i] for i in split["y_test"]], "predicted": [labels[i] for i in predicted], "confidence": probabilities.max(axis=1)}).to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False)
    (output_dir / f"{model_name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n=== {model_name.upper()} METRICS ===")
    metric_keys = ("accuracy", "majority_baseline", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro") + (("attack_recall",) if report_attack_recall else ())
    for key in metric_keys: print(f"{key:18s}: {result[key]:.4f}" if result[key] is not None else f"{key:18s}: n/a")
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(split["y_test"], predicted, labels=np.arange(len(labels)), target_names=labels, zero_division=0))
    print("Confusion matrix:\n", matrix)
    print("\n=== DETECTION TABLE ===")
    print(_detection_table(labels, split["y_test"], predicted, total_dataset_samples).to_string(index=False))
    return result
