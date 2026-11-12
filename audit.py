"""Compact, reproducible data audit for the supplied UAVGPSAttack exports."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .preprocess import (ROOT, UAV_INTERVALS, _clock_seconds, _uav_labels, _uav_streams,
                         _find_timestamp, _algorithm5_features, filter_processed_uav_features,
                         load_experiment, load_uav_processed)
from sklearn.feature_selection import mutual_info_classif


def _clock_string(seconds: float) -> str:
    seconds = max(0, seconds); h = int(seconds // 3600); m = int((seconds % 3600) // 60); s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def audit_uav(root: Path = ROOT, output: Path | None = None) -> dict:
    base = root / "UAVAttackData" / "Live GPS Spoofing and Jamming"
    flights = {}
    examples = []
    for flight in ("Benign Flight", "GPS Jamming", "GPS Spoofing"):
        streams = _uav_streams(base / flight); sensor, angular = streams
        raw = sensor.frame
        t = raw["__time"].to_numpy(dtype=float)
        labels = _uav_labels(flight, t)
        interval = UAV_INTERVALS.get(flight)
        retained_duration = (t.max() - t.min()) / 1e6
        if interval:
            log_start = _clock_seconds(interval["flight_end"]) - retained_duration
            clock = np.array([log_start + (value - t.min()) / 1e6 for value in t])
        else:
            log_start = None; clock = np.array([(value - t.min()) for value in t])
        rows = {"sensor_combined": len(sensor.frame), "vehicle_angular_velocity": len(angular.frame)}
        row = {"flight": flight, "raw_timestamp_min": float(t.min()), "raw_timestamp_max": float(t.max()), "timestamp_unit": "microseconds (PX4 boot-relative)", "clock_origin_assumption": "mapped to Table-I flight end" if interval else "relative only", "features": [c for c in sensor.frame.columns if c != "__time"], "counts": pd.Series(labels).value_counts().to_dict(), "duplicate_timestamps": int(pd.Series(t).duplicated().sum()), "nonmonotonic_after_sort": bool(np.any(np.diff(t) < 0)), "sensor_rows": rows}
        flights[flight] = row
        if interval:
            attack_start = _clock_seconds(interval["attack_start"]); attack_end = _clock_seconds(interval["attack_end"])
            order = np.argsort(np.abs(clock - attack_start)); inside = np.flatnonzero((clock >= attack_start) & (clock <= attack_end))
            candidates = list(order[:2]) + (list(inside[[0, -1]]) if len(inside) else [])
            after = np.flatnonzero(clock > attack_end); candidates += list(after[:2])
            for index in sorted(set(candidates)):
                examples.append({"flight": flight, "raw_timestamp": float(t[index]), "converted_clock": _clock_string(clock[index]), "label": str(labels[index]), "position": "inside" if attack_start <= clock[index] <= attack_end else ("before" if clock[index] < attack_start else "after")})
    for fusion in ("none", "ifhs"):
        data = load_experiment("uav", "sensor_combined", fusion, "multiclass")
        within_dup = 0; within_nonmono = False
        for group in pd.unique(data.groups):
            group_t = data.timestamps[data.groups == group]
            within_dup += int(pd.Series(group_t).duplicated().sum())
            within_nonmono = within_nonmono or bool(np.any(np.diff(group_t) < 0))
        all_idx = np.arange(len(data.y)); train_idx, holdout = train_test_split(all_idx, test_size=.4, random_state=42, stratify=data.y); val_idx, test_idx = train_test_split(holdout, test_size=.5, random_state=42, stratify=data.y[holdout])
        split_counts = {name: pd.Series(data.y[idx]).value_counts().to_dict() for name, idx in (("train", train_idx), ("validation", val_idx), ("test", test_idx))}
        flights[f"dataset_{fusion}"] = {"rows": len(data.x), "features": data.feature_names, "counts": pd.Series(data.y).value_counts().to_dict(), "per_flight_counts": pd.crosstab(data.groups, data.y).to_dict(), "split_counts": split_counts, "duplicate_feature_rows": int(data.x.duplicated().sum()), "timestamp_duplicates_across_flights": int(pd.Series(data.timestamps).duplicated().sum()), "timestamp_duplicates_within_flight": within_dup, "timestamp_nonmonotonic_across_flights": bool(np.any(np.diff(data.timestamps) < 0)), "timestamp_nonmonotonic_within_flight": within_nonmono}
    report = {"source": str(base), "flights": flights, "attack_examples": examples, "confirmed": ["IFHS is called separately for each flight; no cross-flight concatenation occurs before fusion", "timestamps are excluded from classifier features", "sensor_combined is the largest stream and is the IFHS base for the supplied exports", "the prior evaluation reported overall accuracy without a majority baseline or attack recall; the runner now reports both", "noisy-data experiments are intentionally not implemented or run"], "confirmed_fixes": ["removed noisy-run entries from the paper suite", "added validation-macro-F1 checkpoint selection", "added training-only optional class weights", "added within-flight duplicate/nonmonotonic checks and IFHS unit tests", "removed non-universal accelerometer_clipping and ALFA/TLM time-like fields from model inputs"], "limitations": ["the CSV exports are boot-relative and do not carry an absolute wall-clock origin; Table-I clock mapping is therefore anchored to each documented flight end", "the paper does not define its selected row windows precisely, so the supplied exports produce more rows than the paper's plotted subset", "paper-style random row splitting allows temporal neighbors across partitions", "IFHS is currently materialized before the train/validation/test split; time_blocked prevents row-neighbor mixing but is not a fully leakage-free interpolation protocol. A strict leakage-free IFHS run requires splitting each raw stream before spline fitting"]}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        lines = ["# UAV data audit", "", f"Source: `{base}`", "", "## Per-flight audit", ""]
        for name, values in flights.items(): lines += [f"### {name}", "", "```json", json.dumps(values, indent=2, default=str), "```", ""]
        lines += ["## Attack-window examples", "", "| flight | raw timestamp | converted clock | position | assigned label |", "|---|---:|---|---|---|"]
        lines += [f"| {x['flight']} | {x['raw_timestamp']:.0f} | {x['converted_clock']} | {x['position']} | {x['label']} |" for x in examples]
        lines += ["", "## Remaining limitations", "", *[f"- {x}" for x in report["limitations"]]]
        output.write_text("\n".join(lines), encoding="utf-8")
    return report


def audit_other(root: Path = ROOT, output: Path | None = None) -> dict:
    """Audit feature names/labels for TLM and ALFA without training models."""
    report = {"tlm": {}, "alfa": {}}
    for subset in ("rate", "gps", "imu"):
        data = load_experiment("tlm", subset, "none", "multiclass", root=root)
        report["tlm"][subset] = {"rows": len(data.x), "features": data.feature_names, "classes": pd.Series(data.y).value_counts().to_dict(), "time_like_features": [c for c in data.feature_names if any(token in c.lower() for token in ("time", "timestamp", "line", "index"))]}
    for subset in ("imu", "velocity", "pitch_roll_yaw"):
        data = load_experiment("alfa", subset, "none", "binary_engine", root=root)
        report["alfa"][subset] = {"rows": len(data.x), "features": data.feature_names, "classes": pd.Series(data.y).value_counts().to_dict(), "time_like_features": [c for c in data.feature_names if any(token in c.lower() for token in ("time", "timestamp", "line", "index"))]}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True); output.write_text("# Non-UAV feature audit\n\n```json\n" + json.dumps(report, indent=2, default=str) + "\n```\n", encoding="utf-8")
    return report


def audit_processed_uav(root: Path = ROOT, output: Path | None = None) -> dict:
    """Inventory all UAV CSVs and reconstruct what the processed labels support."""
    base = root / "UAVAttackData" / "Live GPS Spoofing and Jamming"
    inventory = []
    for path in sorted(base.rglob("*.csv")):
        frame = pd.read_csv(path)
        timestamp_columns = [c for c in frame.columns if c.lower() == "timestamp" or "time" in c.lower()]
        timestamp = timestamp_columns[0] if timestamp_columns else None
        ts = pd.to_numeric(frame[timestamp], errors="coerce") if timestamp else pd.Series(dtype=float)
        labels = frame.iloc[:, -1].value_counts(dropna=False).to_dict() if path.parent.name == "Processed" else {}
        inventory.append({"path": str(path), "rows": len(frame), "columns": len(frame.columns), "dtypes": {str(k): str(v) for k, v in frame.dtypes.value_counts().items()}, "timestamp_column": timestamp, "timestamp_min": None if timestamp is None else float(ts.min()), "timestamp_max": None if timestamp is None else float(ts.max()), "timestamp_monotonic": None if timestamp is None else bool(ts.is_monotonic_increasing), "timestamp_duplicates": None if timestamp is None else int(ts.duplicated().sum()), "last_column": str(frame.columns[-1]), "label_values_if_processed": {str(k): int(v) for k, v in labels.items()}, "already_merged_features": path.parent.name == "Processed"})
    processed = [item for item in inventory if item["already_merged_features"]]
    table = []
    processed_examples = []
    for item in processed:
        path = Path(item["path"]); frame = pd.read_csv(path); labels = frame.iloc[:, -1].astype(str); transition = np.flatnonzero(labels.to_numpy()[1:] != labels.to_numpy()[:-1]); transition_index = int(transition[0] + 1) if len(transition) else None; raw_ts = pd.to_numeric(frame["timestamp"], errors="coerce")
        table.append({"type": path.parent.parent.name, "raw_min": float(raw_ts.min()), "raw_max": float(raw_ts.max()), "label_transition_timestamp": None if transition_index is None else float(raw_ts.iloc[transition_index]), "relative_transition_seconds": None if transition_index is None else float((raw_ts.iloc[transition_index] - raw_ts.iloc[0]) / 1e6), "label_after_transition": None if transition_index is None else str(labels.iloc[transition_index]), "paper_attack_start": UAV_INTERVALS.get(path.parent.parent.name, {}).get("attack_start"), "paper_attack_end": UAV_INTERVALS.get(path.parent.parent.name, {}).get("attack_end"), "evidence": "processed label transition; absolute wall-clock origin unavailable"})
        if transition_index is not None:
            sample_indices = sorted(set([max(0, transition_index - 1), transition_index, min(len(frame) - 1, transition_index + max(1, (len(frame) - transition_index) // 2)), len(frame) - 1]))
            for index in sample_indices:
                processed_examples.append({"type": path.parent.parent.name, "raw_timestamp": float(raw_ts.iloc[index]), "relative_seconds_from_file_start": float((raw_ts.iloc[index] - raw_ts.iloc[0]) / 1e6), "label": str(labels.iloc[index]), "position": "before" if index < transition_index else ("inside_processed_malicious" if index < len(frame) - 1 else "after_or_end_of_file"), "absolute_clock": None})
    report = {"source": str(base), "inventory": inventory, "processed_table_I_reconstruction": table, "processed_examples": processed_examples, "processed_files": processed, "decision": "use processed labeled files for the requested baseline; they are merged GPS/attitude/position tables, not raw sensor_combined+angular_velocity IFHS inputs", "limitations": ["processed files have boot-relative timestamp values and no recoverable absolute clock origin", "processed malicious labels continue from one transition to end-of-file and do not exactly equal the Table-I attack windows", "the processed files contain 3622 spoofing rows and 6445 jamming rows, not the paper's selected 4384 and 4597 rows"]}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True); output.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        lines = ["# UAV processed/raw file audit", "", "## Processed Table-I reconstruction", "", "| type | rows | raw range | label transition | relative transition | labels | paper window |", "|---|---:|---|---:|---:|---|---|"]
        for item in table:
            lines.append(f"| {item['type']} | {next(x['rows'] for x in processed if x['path'] == str(base / item['type'] / 'Processed' / ('jamming-merged-gps-only.csv' if item['type']=='GPS Jamming' else 'spoofing-merged-gps-only.csv')))} | {item['raw_min']:.0f}–{item['raw_max']:.0f} | {item['label_transition_timestamp']:.0f} | {item['relative_transition_seconds']:.3f}s | {item['label_after_transition']} to EOF | {item['paper_attack_start']}–{item['paper_attack_end']} |")
        lines += ["", "Absolute Table-I clock reconstruction is unavailable from these files; the values are boot-relative. The processed label transition is evidence, not a fabricated Table-I timestamp.", "", "## Before/inside/after processed-label examples", "", "| type | raw timestamp | relative seconds | label | position |", "|---|---:|---:|---|---|"]
        lines += [f"| {x['type']} | {x['raw_timestamp']:.0f} | {x['relative_seconds_from_file_start']:.3f} | {x['label']} | {x['position']} |" for x in processed_examples]
        lines += ["", "## Complete CSV inventory", "", "| path | rows | columns | timestamp | range | monotonic | duplicates | last column | processed labels |", "|---|---:|---:|---|---|---|---:|---|---|"]
        for item in inventory:
            lines.append(f"| `{item['path']}` | {item['rows']} | {item['columns']} | {item['timestamp_column']} | {item['timestamp_min']}–{item['timestamp_max']} | {item['timestamp_monotonic']} | {item['timestamp_duplicates']} | {item['last_column']} | {item['label_values_if_processed']} |")
        lines += ["", "## Decision", "", report["decision"], "", *[f"- {x}" for x in report["limitations"]]]
        output.write_text("\n".join(lines), encoding="utf-8")
    return report


def audit_processed_uav_leakage(root: Path = ROOT, output: Path | None = None) -> dict:
    """Print/save the exact 45/47 pre-filter lists and post-filter statistics."""
    base = root / "UAVAttackData" / "Live GPS Spoofing and Jamming"
    paths = {
        "binary_jamming": base / "GPS Jamming" / "Processed" / "jamming-merged-gps-only.csv",
        "binary_spoofing": base / "GPS Spoofing" / "Processed" / "spoofing-merged-gps-only.csv",
    }
    files = {}
    for task, path in paths.items():
        raw = pd.read_csv(path)
        time_col = _find_timestamp(raw.columns)
        pre = _algorithm5_features(raw, time_col, "scenario")
        labels = raw.iloc[:, -1].astype(str).to_numpy()
        pre.columns = [c.split("__", 1)[1] for c in pre.columns]
        post, flags = filter_processed_uav_features(pre, labels)
        transition = np.flatnonzero(labels[1:] != labels[:-1])
        first_mal = int(np.flatnonzero(labels == "malicious")[0]) if np.any(labels == "malicious") else None
        stats = []
        binary = (labels == "malicious").astype(int)
        for col in post.columns:
            values = pd.to_numeric(post[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            benign, attack = values[labels == "benign"], values[labels == "malicious"]
            corr = float(np.corrcoef(values, binary)[0, 1]) if np.std(values) and np.std(binary) else 0.0
            try:
                mi = float(mutual_info_classif(values.reshape(-1, 1), binary, random_state=42)[0])
            except Exception:
                mi = None
            stats.append({"name": col, "dtype": str(raw[col].dtype), "benign_mean": float(np.mean(benign)), "benign_std": float(np.std(benign)), "attack_mean": float(np.mean(attack)), "attack_std": float(np.std(attack)), "correlation": corr, "mutual_information": mi})
        files[task] = {
            "path": str(path), "rows": len(raw), "timestamp_column": time_col,
            "before_features": list(pre.columns), "after_features": list(post.columns),
            "leaked_columns": flags, "first_malicious_row_index": first_mal,
            "labels_malicious_until_eof": bool(first_mal is not None and np.all(labels[first_mal:] == "malicious")),
            "label_transition_index": int(transition[0] + 1) if len(transition) else None,
            "label_transition_timestamp": float(raw[time_col].iloc[transition[0] + 1]) if len(transition) else None,
            "paper_window": UAV_INTERVALS["GPS Jamming" if task == "binary_jamming" else "GPS Spoofing"],
            "feature_statistics": stats,
        }
    # The multiclass input is the common post-filter intersection.
    common = [c for c in files["binary_jamming"]["after_features"] if c in files["binary_spoofing"]["after_features"]]
    report = {"exact_pre_filter_lists": {k: v["before_features"] for k, v in files.items()}, "files": files, "multiclass_after_features": common, "decision": "GPS/status/position columns are excluded before processed-UAV training; this is a diagnostic processed baseline, not raw IFHS."}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        lines = ["# Processed UAV leakage audit", "", "This audit covers only the two supplied Processed CSVs; ALFA/TLM are untouched.", ""]
        for task, item in files.items():
            lines += [f"## {task}", "", f"Path: `{item['path']}`", "", f"First malicious row index: `{item['first_malicious_row_index']}`; malicious until EOF: `{item['labels_malicious_until_eof']}`", f"Label transition index/timestamp: `{item['label_transition_index']}` / `{item['label_transition_timestamp']}`", f"Paper window: `{item['paper_window']['attack_start']}–{item['paper_window']['attack_end']}` (absolute clock unavailable in boot-relative CSV)", "", "### Exact features before filtering", "", ", ".join(item["before_features"]), "", "### Features after filtering", "", ", ".join(item["after_features"]), "", "### Removed/leaking columns", "", *[f"- `{c}` — {r}" for c, r in item["leaked_columns"].items()], "", "### Remaining-feature statistics", "", "| name | dtype | benign mean | benign std | attack mean | attack std | correlation | mutual information |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            lines += [f"| {s['name']} | {s['dtype']} | {s['benign_mean']:.6g} | {s['benign_std']:.6g} | {s['attack_mean']:.6g} | {s['attack_std']:.6g} | {s['correlation']:.6g} | {s['mutual_information'] if s['mutual_information'] is not None else 'NA'} |" for s in item["feature_statistics"]]
            lines.append("")
        lines += ["## Multiclass post-filter features", "", ", ".join(common)]
        output.write_text("\n".join(lines), encoding="utf-8")
    return report


def make_table_v_processed(root: Path = ROOT, output: Path | None = None) -> None:
    """Write a compact Table-V-style comparison from the short processed runs."""
    import csv
    base = root / "artifacts" / "processed_baseline"
    expected = {"binary_spoofing": "4274 benign / 110 attack", "binary_jamming": "4445 benign / 152 attack", "multiclass": "4274 benign / 262 combined attack"}
    rows = []
    for task in ("binary_spoofing", "binary_jamming", "multiclass"):
        path = base / f"uav_all_none_{task}_paper" / "results.csv"
        if not path.exists(): continue
        source = load_uav_processed(task, root)
        our_counts = pd.Series(source.y).value_counts().to_dict()
        for record in pd.read_csv(path).to_dict("records"):
            rows.append({"task": task, "model": record["model"], "our_rows": len(source.y), "our_class_counts": json.dumps(our_counts, sort_keys=True), "paper_table_v_count": expected[task], "our_count_vs_paper_count": "different processed subset", "accuracy": record["accuracy"], "majority_baseline": record["majority_baseline"], "balanced_accuracy": record["balanced_accuracy"], "macro_f1": record["f1_macro"], "attack_recall": record["attack_recall"]})
    if output and rows:
        output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
        lines = ["# Table-V-style processed UAV comparison", "", "These are non-noisy short runs on the supplied labeled Processed CSVs. Counts are not forced to match the paper.", "", "| task | model | our rows | our class counts | paper count | count comparison | accuracy | majority baseline | balanced accuracy | macro F1 | attack recall |", "|---|---|---:|---|---|---|---:|---:|---:|---:|---:|"]
        for row in rows: lines.append(f"| {row['task']} | {row['model']} | {row['our_rows']} | `{row['our_class_counts']}` | {row['paper_table_v_count']} | {row['our_count_vs_paper_count']} | {row['accuracy']:.4f} | {row['majority_baseline']:.4f} | {row['balanced_accuracy']:.4f} | {row['macro_f1']:.4f} | {row['attack_recall']:.4f} |")
        output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = audit_uav(output=ROOT / "artifacts" / "uav_audit.md")
    audit_other(output=ROOT / "artifacts" / "other_dataset_audit.md")
    audit_processed_uav(output=ROOT / "artifacts" / "uav_processed_audit.md")
    audit_processed_uav_leakage(output=ROOT / "artifacts" / "uav_processed_leakage_audit.md")
    make_table_v_processed(output=ROOT / "artifacts" / "table_v_processed.md")
    print(json.dumps(result, indent=2, default=str))
