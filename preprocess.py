"""Paper-specific loading, Algorithm-5 filtering, and genuine IFHS alignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[1]
LABEL_COLUMNS = {"label", "labels", "class", "target", "__label"}
TIME_TOKENS = ("time", "timestamp", "timems", "timeus", "stamp", "header", "line", "index", "gms")
LOCATION_TOKENS = ("lat", "lng", "lon", "longitude", "latitude")

# The supplied ``*-merged-gps-only.csv`` files are useful for auditing the
# processed baseline, but contain many GPS observables and status fields that
# make the label almost directly observable.  Keep this rule local to the
# processed-UAV loader; raw UAV/ALFA/TLM loading is intentionally unchanged.
PROCESSED_LEAK_TOKENS = (
    "lat", "lng", "lon", "longitude", "latitude", "gps", "position",
    "alt", "eph", "epv", "evh", "evv", "hdop", "vdop", "satell",
    "fix", "status", "jamming", "spoof", "attack", "malicious", "benign",
    "timestamp", "time", "dt", "index", "sample", "file", "folder", "flight",
    "vel_m", "vel_n", "vel_e", "vel_d", "vel_ned", "cog", "heading",
    "terrain", "dist_bottom", "xy_valid", "v_xy", "noise_per", "variance",
    "ref_", "vx", "vy", "vz", "z_deriv",
)


@dataclass
class SensorStream:
    name: str
    frame: pd.DataFrame
    labels: np.ndarray | None


@dataclass
class ExperimentData:
    x: pd.DataFrame
    y: np.ndarray
    timestamps: np.ndarray
    groups: np.ndarray
    feature_names: list[str]
    source_description: str


def _as_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("--.--", "", regex=False), errors="coerce")


def _find_timestamp(columns: Iterable[str]) -> str:
    columns = list(columns)
    for candidate in ("timestamp", "TimeUS", "abTimeUS", "time"):
        if candidate in columns:
            return candidate
    found = [column for column in columns if "time" in column.lower()]
    if not found:
        raise ValueError(f"No timestamp column found: {columns}")
    return found[0]


def _algorithm5_features(raw: pd.DataFrame, time_column: str, prefix: str) -> pd.DataFrame:
    """Exclude timing, identifiers, non-universal columns, and latitude/longitude."""
    keep: dict[str, pd.Series] = {}
    for column in raw.columns:
        name = column.lower().replace("_", "")
        if column == time_column or column.lower() in LABEL_COLUMNS:
            continue
        if any(token in name for token in TIME_TOKENS + LOCATION_TOKENS):
            continue
        values = _as_number(raw[column]).replace([np.inf, -np.inf], np.nan)
        if values.notna().sum() and values.nunique(dropna=True) > 1:
            keep[f"{prefix}__{column}"] = values
    return pd.DataFrame(keep)


def processed_leak_reason(column: str) -> str | None:
    """Return the explicit reason a processed-UAV column is not admissible in X."""
    name = column.lower().replace("_", "")
    if name in {"x", "y", "z", "positionx", "positiony", "positionz"}:
        return "GPS/position coordinate"
    # q[] and accelerometer channels are retained; everything matching these
    # tokens is GPS/position/status/time/metadata or a known label proxy.
    for token in PROCESSED_LEAK_TOKENS:
        if token.replace("_", "") in name:
            return f"GPS/position/status/time/metadata token: {token}"
    return None


def _transition_proxy_reason(values: pd.Series, labels: np.ndarray | None) -> str | None:
    """Flag a feature that is constant before the label transition and changes at it."""
    if labels is None or len(labels) != len(values):
        return None
    labels = np.asarray(labels).astype(str)
    transitions = np.flatnonzero(labels[1:] != labels[:-1])
    if len(transitions) != 1:
        return None
    boundary = int(transitions[0] + 1)
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    before, after = numeric[:boundary], numeric[boundary:]
    if len(before) and np.isfinite(before).all() and np.nanmax(before) == np.nanmin(before):
        if len(after) and np.isfinite(after).any() and not np.allclose(after, before[0], equal_nan=True):
            return "transition proxy: constant benign prefix changes at first malicious row"
    return None


def filter_processed_uav_features(frame: pd.DataFrame, labels: np.ndarray | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply the requested processed-UAV leakage audit and return X plus flags."""
    flags: dict[str, str] = {}
    for column in frame.columns:
        reason = processed_leak_reason(column) or _transition_proxy_reason(frame[column], labels)
        if reason:
            flags[column] = reason
    kept = [column for column in frame.columns if column not in flags]
    return frame.loc[:, kept].copy(), flags


def _prepare_stream(name: str, path: Path) -> SensorStream:
    raw = pd.read_csv(path)
    time_column = _find_timestamp(raw.columns)
    time = _as_number(raw[time_column])
    label_column = next((column for column in raw.columns if column.lower() in LABEL_COLUMNS), None)
    features = _algorithm5_features(raw, time_column, name)
    prepared = pd.concat([time.rename("__time"), features], axis=1)
    if label_column:
        prepared["__label"] = raw[label_column]
    prepared = prepared.dropna(subset=["__time"]).sort_values("__time").drop_duplicates("__time", keep="last").reset_index(drop=True)
    labels = prepared.pop("__label").to_numpy() if "__label" in prepared else None
    return SensorStream(name, prepared, labels)


def _spline(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    valid = np.isfinite(source_time) & np.isfinite(values)
    x, y = source_time[valid], values[valid]
    if len(x) == 0:
        return np.zeros(len(target_time))
    if len(x) == 1:
        return np.repeat(y[0], len(target_time))
    if len(x) < 4:
        return np.interp(target_time, x, y)
    return CubicSpline(x, y, bc_type="natural", extrapolate=False)(target_time)


def ifhs_fuse(streams: list[SensorStream]) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray, str]:
    """Algorithm 2: natural cubic splines onto the true highest-sample timeline.

    Only the streams' common time interval is retained, so no spline is used
    outside observed data. Labels are nearest-time categorical values, never
    interpolated.
    """
    if len(streams) < 2:
        raise ValueError("IFHS requires at least two sensor streams")
    base = max(streams, key=lambda stream: len(stream.frame))
    low = max(float(stream.frame["__time"].min()) for stream in streams)
    high = min(float(stream.frame["__time"].max()) for stream in streams)
    base_frame = base.frame.loc[base.frame["__time"].between(low, high)].copy().reset_index(drop=True)
    if len(base_frame) < 2:
        raise ValueError("No common timestamp range for IFHS")
    target = base_frame["__time"].to_numpy(dtype=float)
    result = pd.DataFrame({"timestamp": target})
    for stream in streams:
        if stream is base:
            for column in stream.frame.columns:
                if column != "__time":
                    result[column] = base_frame[column].to_numpy()
            continue
        source_time = stream.frame["__time"].to_numpy(dtype=float)
        for column in stream.frame.columns:
            if column != "__time":
                result[column] = _spline(source_time, stream.frame[column].to_numpy(dtype=float), target)
    result = result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    labels = None
    if base.labels is not None:
        source_time = base.frame["__time"].to_numpy(dtype=float)
        indices = np.searchsorted(source_time, result["timestamp"].to_numpy(), side="left").clip(0, len(source_time) - 1)
        labels = np.asarray(base.labels)[indices]
    return result, labels, result["timestamp"].to_numpy(), base.name


def _alfa_binary(name: str) -> str | None:
    name = name.lower()
    if "engine_failure" in name:
        return "engine_failure"
    if "no_failure" in name:
        return "normal"
    return None


def _alfa_multiclass(name: str) -> str | None:
    name = name.lower()
    if "no_failure" in name: return "normal"
    if "engine_failure" in name: return "engine_failure"
    if "elevator_failure" in name: return "elevator_failure"
    if "aileron" in name: return "aileron_failure"
    if "rudder" in name: return "rudder_failure"
    return None


def _alfa_paths(flight: Path) -> dict[str, Path]:
    paths = {"imu": flight / "imu" / "imu_clean.csv", "velocity": flight / "velocity" / "velocity_clean.csv", "rpy": flight / "rpy" / "rpy_clean.csv"}
    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError(f"Missing prepared ALFA stream under {flight}")
    return paths


def load_alfa(subset: str, fusion: str, task: str, root: Path = ROOT) -> ExperimentData:
    source = root / "ALFA" / "implementation_output" / "all_47_flights"
    label_fn = _alfa_binary if task == "binary_engine" else _alfa_multiclass
    selected = {"pitch_roll_yaw": "rpy", "rpy": "rpy"}.get(subset, subset)
    xs: list[pd.DataFrame] = []; ys: list[str] = []; times: list[float] = []; groups: list[str] = []
    for flight in sorted(path for path in source.iterdir() if path.is_dir()):
        label = label_fn(flight.name)
        if label is None:
            continue
        paths = _alfa_paths(flight)
        if fusion == "ifhs":
            x, _, timestamp, _ = ifhs_fuse([_prepare_stream(name, path) for name, path in paths.items()])
            x = x.drop(columns="timestamp")
        else:
            if selected not in paths:
                raise ValueError("ALFA subset must be imu, velocity, or pitch_roll_yaw")
            stream = _prepare_stream(selected, paths[selected]); x = stream.frame.drop(columns="__time"); timestamp = stream.frame["__time"].to_numpy()
        xs.append(x); ys.extend([label] * len(x)); times.extend(timestamp); groups.extend([flight.name] * len(x))
    if not xs:
        raise FileNotFoundError("No labelled ALFA flights found")
    data = pd.concat(xs, ignore_index=True).fillna(0.0)
    description = f"ALFA {selected if fusion == 'none' else 'IMU+Velocity+Pitch_Roll_Yaw IFHS'} ({task})"
    return ExperimentData(data, np.asarray(ys), np.asarray(times), np.asarray(groups), list(data.columns), description)


def _tlm_paths(root: Path) -> dict[str, Path]:
    base = root / "TLM" / "dataset"
    return {"rate": next((base / "RATE").glob("*.csv")), "gps": next((base / "GPS").glob("*.csv")), "imu": next((base / "IMU").glob("*.csv"))}


def load_tlm(subset: str, fusion: str, root: Path = ROOT) -> ExperimentData:
    paths = _tlm_paths(root)
    if fusion == "ifhs":
        x, y, timestamp, base = ifhs_fuse([_prepare_stream(name, path) for name, path in paths.items()])
        x = x.drop(columns="timestamp"); description = f"TLM RATE+GPS+IMU IFHS (base={base})"
    else:
        if subset not in paths:
            raise ValueError("TLM subset must be rate, gps, or imu")
        stream = _prepare_stream(subset, paths[subset]); x = stream.frame.drop(columns="__time"); y = stream.labels; timestamp = stream.frame["__time"].to_numpy(); description = f"TLM_{subset.upper()}"
    return ExperimentData(x.fillna(0.0), np.asarray(y).astype(str), timestamp, np.repeat(description, len(x)), list(x.columns), description)


UAV_INTERVALS = {
    "GPS Jamming": {"flight_start": "15:29:06", "attack_start": "15:54:09", "attack_end": "15:54:20", "flight_end": "15:55:09", "class": "gps_jamming"},
    "GPS Spoofing": {"flight_start": "15:58:19", "attack_start": "16:24:14", "attack_end": "16:24:42", "flight_end": "16:26:25", "class": "gps_spoofing"},
}


def _clock_seconds(value: str) -> int:
    hour, minute, second = (int(part) for part in value.split(":"))
    return hour * 3600 + minute * 60 + second


def _uav_streams(directory: Path) -> list[SensorStream]:
    sensor = next(directory.glob("*sensor_combined*.csv"), None)
    angular = next(directory.glob("*vehicle_angular_velocity*.csv"), None)
    if not sensor or not angular:
        raise FileNotFoundError(f"Missing sensor_combined/vehicle_angular_velocity in {directory}")
    return [_prepare_stream("sensor_combined", sensor), _prepare_stream("vehicle_angular_velocity", angular)]


def _uav_labels(flight: str, timestamp: np.ndarray) -> np.ndarray:
    """Construct Table-I ground truth, not labels from model predictions.

    The provided PX4 exports use boot-relative microseconds and retain only the
    end of each flight. The retained log is consequently mapped to the Table-I
    flight end, then the known attack start/end clocks are projected onto it.
    """
    labels = np.repeat("benign", len(timestamp)).astype(object)
    if flight == "Benign Flight":
        return labels
    interval = UAV_INTERVALS[flight]
    retained_duration = (timestamp.max() - timestamp.min()) / 1_000_000.0
    log_start = _clock_seconds(interval["flight_end"]) - retained_duration
    start = _clock_seconds(interval["attack_start"]) - log_start
    duration = _clock_seconds(interval["attack_end"]) - _clock_seconds(interval["attack_start"])
    elapsed = (timestamp - timestamp.min()) / 1_000_000.0
    labels[(elapsed >= start) & (elapsed <= start + duration)] = interval["class"]
    return labels


def load_uav(task: str, fusion: str, root: Path = ROOT) -> ExperimentData:
    root = root / "UAVAttackData" / "Live GPS Spoofing and Jamming"
    xs: list[pd.DataFrame] = []; ys: list[str] = []; times: list[float] = []; groups: list[str] = []
    for flight in ("Benign Flight", "GPS Jamming", "GPS Spoofing"):
        streams = _uav_streams(root / flight)
        if fusion == "ifhs":
            x, _, timestamp, _ = ifhs_fuse(streams); x = x.drop(columns="timestamp")
        else:
            stream = streams[0]; x = stream.frame.drop(columns="__time"); timestamp = stream.frame["__time"].to_numpy()
        labels = _uav_labels(flight, timestamp)
        xs.append(x); ys.extend(labels); times.extend(timestamp); groups.extend([flight] * len(x))
    # Algorithm 5's "universal" rule is applied across flights too. A column
    # present only in one scenario (for example accelerometer_clipping in the
    # supplied spoofing export) is a flight/label proxy and must not enter X.
    common = set(xs[0].columns)
    for frame in xs[1:]: common.intersection_update(frame.columns)
    feature_order = [column for column in xs[0].columns if column in common]
    xs = [frame.loc[:, feature_order] for frame in xs]
    data = pd.concat(xs, ignore_index=True).fillna(0.0); y = np.asarray(ys); times_array = np.asarray(times); group_array = np.asarray(groups)
    if task == "binary_spoofing":
        keep = y != "gps_jamming"; data = data.loc[keep].reset_index(drop=True); y = np.where(y[keep] == "gps_spoofing", "gps_spoofing", "benign"); times_array = times_array[keep]; group_array = group_array[keep]
    if task == "binary_jamming":
        keep = y != "gps_spoofing"; data = data.loc[keep].reset_index(drop=True); y = np.where(y[keep] == "gps_jamming", "gps_jamming", "benign"); times_array = times_array[keep]; group_array = group_array[keep]
    description = f"UAVGPSAttack sensor_combined{'+vehicle_angular_velocity IFHS' if fusion == 'ifhs' else ''} ({task}, non-noisy)"
    return ExperimentData(data, y, times_array, group_array, list(data.columns), description)


def load_uav_processed(task: str, root: Path = ROOT) -> ExperimentData:
    """Load the supplied labeled `Processed/*-merged-gps-only.csv` subsets.

    These files are already merged GPS/attitude/position tables; they do not
    contain the raw sensor_combined/angular-velocity pair needed to reproduce
    IFHS. Their final-column labels are therefore authoritative for this
    processed-subset baseline and are not replaced with Table-I labels.
    """
    base = root / "UAVAttackData" / "Live GPS Spoofing and Jamming"
    files = {"gps_jamming": base / "GPS Jamming" / "Processed" / "jamming-merged-gps-only.csv", "gps_spoofing": base / "GPS Spoofing" / "Processed" / "spoofing-merged-gps-only.csv"}
    if task == "binary_jamming": selected = {"gps_jamming": files["gps_jamming"]}
    elif task == "binary_spoofing": selected = {"gps_spoofing": files["gps_spoofing"]}
    elif task == "multiclass": selected = files
    else: raise ValueError("Processed UAV supports multiclass, binary_spoofing, or binary_jamming")
    frames = []; labels = []; timestamps = []; groups = []
    for scenario, path in selected.items():
        if not path.exists(): raise FileNotFoundError(path)
        stream = _prepare_stream(scenario, path)
        frame = stream.frame.drop(columns="__time")
        frame = frame.rename(columns={column: column.split("__", 1)[1] if "__" in column else column for column in frame.columns})
        raw_labels = np.asarray(stream.labels).astype(str)
        # Audit/filter the exact pre-filter 45/47-column lists before any
        # scenario concatenation.  The final-column labels remain ground truth.
        frame, _ = filter_processed_uav_features(frame, raw_labels)
        if task == "multiclass": raw_labels = np.where(raw_labels == "malicious", scenario, "benign")
        frames.append(frame); labels.extend(raw_labels); timestamps.extend(stream.frame["__time"].to_numpy()); groups.extend([scenario] * len(frame))
    common = set(frames[0].columns)
    for frame in frames[1:]: common.intersection_update(frame.columns)
    order = [column for column in frames[0].columns if column in common]
    data = pd.concat([frame.loc[:, order] for frame in frames], ignore_index=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    description = f"UAVGPSAttack processed labeled subset ({task}; explicit GPS/status leakage filter; no IFHS)"
    return ExperimentData(data, np.asarray(labels), np.asarray(timestamps), np.asarray(groups), order, description)


def load_experiment(dataset: str, subset: str, fusion: str, task: str, root: Path = ROOT, uav_source: str = "raw") -> ExperimentData:
    if dataset == "alfa": return load_alfa(subset, fusion, task, root)
    if dataset == "tlm": return load_tlm(subset, fusion, root)
    if dataset == "uav":
        if uav_source == "processed":
            if fusion == "ifhs":
                raise ValueError("The supplied Processed CSVs are already merged labeled subsets and do not support a new raw-stream IFHS step; use --fusion none or --uav-source raw")
            return load_uav_processed(task, root)
        return load_uav(task, fusion, root)
    raise ValueError("dataset must be alfa, tlm, or uav")


def save_preprocessing_manifest(path: Path, data: ExperimentData, dataset: str, subset: str, fusion: str, task: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_files = []
    if dataset == "uav" and "processed" in data.source_description:
        base = ROOT / "UAVAttackData" / "Live GPS Spoofing and Jamming"
        source_files = []
        if task in {"binary_jamming", "multiclass"}: source_files.append(str(base / "GPS Jamming" / "Processed" / "jamming-merged-gps-only.csv"))
        if task in {"binary_spoofing", "multiclass"}: source_files.append(str(base / "GPS Spoofing" / "Processed" / "spoofing-merged-gps-only.csv"))
    processed = "processed" in data.source_description
    path.write_text(json.dumps({
        "dataset": dataset, "subset": subset, "fusion": fusion, "task": task, "noisy_experiments": "not implemented by design",
        "samples": len(data.x), "features": data.feature_names, "source": data.source_description, "source_files": source_files,
        "ifhs": "not applied: supplied Processed files are already merged labeled subsets" if processed else "natural cubic spline on the observed largest-stream timeline; common time overlap only; no extrapolation",
        "feature_filtering": "time/identifier/non-numeric/constant/latitude/longitude columns excluded",
        "uav_ground_truth": "processed CSV final-column labels preserved; Table-I windows are reported separately and not substituted" if processed else "Table I attack intervals, mapped to PX4 boot-relative timestamps; predictions are never ground truth",
    }, indent=2), encoding="utf-8")
