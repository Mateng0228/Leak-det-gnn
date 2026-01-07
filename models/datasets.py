"""
WDN datasets for:
1) Normal predictor (denoising + 1-step forecasting) trained only on no-leak data.
2) Leak detector trained on abrupt single-pipe leak scenarios + no-leak scenarios,
   using 4-bucket time sampling (early/late/pre-leak/no-leak) and softmax multi-class labels.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


# ----------------------------
# Utils
# ----------------------------
def _read_manifest_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _filter_ok(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if str(r.get("status", "ok")).lower() == "ok":
            out.append(r)
    return out


def _ensure_sensor_order(df: pd.DataFrame, sensor_ids: Sequence[str]) -> pd.DataFrame:
    missing = [c for c in sensor_ids if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensors in csv: {missing[:10]} (and {len(missing)-10} more)")
    return df.loc[:, list(sensor_ids)]


def make_time_features(dt_index: pd.DatetimeIndex) -> np.ndarray:
    """Return (N, 9) time features: hour_sin, hour_cos, day_of_week one-hot (7)."""
    hour = dt_index.hour.astype(np.float32) + (dt_index.minute.astype(np.float32) / 60.0) + (dt_index.second.astype(np.float32) / 3600.0)
    hour = np.asarray(hour, dtype=np.float32)
    angle = (2.0 * np.pi) * (hour / 24.0)
    hour_sin = np.sin(angle).astype(np.float32)
    hour_cos = np.cos(angle).astype(np.float32)
    dow = dt_index.dayofweek.to_numpy() # 0=Mon
    dow_oh = np.eye(7, dtype=np.float32)[dow]
    feats = np.concatenate([hour_sin[:, None], hour_cos[:, None], dow_oh], axis=1)
    return feats.astype(np.float32)


@dataclass(frozen=True)
class SensorStandardizer:
    mean: np.ndarray  # (S,)
    std: np.ndarray   # (S,)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std) + self.mean

def compute_sensor_stats_from_normal(
    normal_root: str | Path,
    sensor_ids: Optional[Sequence[str]] = None,
    use_gt: bool = True,
    max_scenes: Optional[int] = None,
) -> SensorStandardizer:
    """Compute per-sensor mean/std from NO-LEAK data (sensors_gt.csv)"""
    
    normal_root = Path(normal_root)
    manifest = _filter_ok(_read_manifest_jsonl(normal_root / "manifest.jsonl"))
    if max_scenes is not None:
        manifest = manifest[:max_scenes]

    # Determine sensor order
    if sensor_ids is None:
        first_id = manifest[0]["window_id"]
        df0 = pd.read_csv(normal_root / first_id / ("sensors_gt.csv" if use_gt else "sensors.csv"), index_col=0, parse_dates=True)
        sensor_ids = list(df0.columns)

    s = len(sensor_ids)
    sum_ = np.zeros((s,), dtype=np.float64)
    sumsq = np.zeros((s,), dtype=np.float64)
    count = 0

    for row in manifest:
        wid = row["window_id"]
        fpath = normal_root / wid / ("sensors_gt.csv" if use_gt else "sensors.csv")
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
        df = _ensure_sensor_order(df, sensor_ids)
        x = df.to_numpy(dtype=np.float64)
        sum_ += x.sum(axis=0)
        sumsq += (x * x).sum(axis=0)
        count += x.shape[0]

    mean = (sum_ / max(count, 1)).astype(np.float32)
    var = (sumsq / max(count, 1) - mean.astype(np.float64) ** 2).clip(min=1e-12)
    std = np.sqrt(var).astype(np.float32)
    std = np.maximum(std, 1e-3).astype(np.float32)  # avoid zero
    return SensorStandardizer(mean=mean, std=std)


# ----------------------------
# Caching loader
# ----------------------------
@dataclass
class _LoadedSeries:
    noisy: np.ndarray # (T, S) float32
    gt: np.ndarray # (T, S) float32
    time_feat: np.ndarray # (T, 9) float32
    timestamps: pd.DatetimeIndex
    sensor_ids: List[str]


class ScenarioStore:
    """Lazy loader with small in-memory cache. Keeps arrays in float32."""
    def __init__(
        self,
        root: Path,
        sensor_ids: Optional[Sequence[str]] = None,
        standardizer: Optional[SensorStandardizer] = None,
        cache_size: int = 32,
    ) -> None:
        self.root = Path(root)
        self.sensor_ids = list(sensor_ids) if sensor_ids is not None else None
        self.standardizer = standardizer
        self.cache_size = int(cache_size)
        self._cache: Dict[str, _LoadedSeries] = {}
        self._lru: List[str] = []

    def _cache_put(self, key: str, val: _LoadedSeries) -> None:
        if key in self._cache:
            return
        self._cache[key] = val
        self._lru.append(key)
        if len(self._lru) > self.cache_size:
            old = self._lru.pop(0)
            self._cache.pop(old, None)

    def get(self, scene_id: str) -> _LoadedSeries:
        if scene_id in self._cache:
            # refresh LRU
            try:
                self._lru.remove(scene_id)
            except ValueError:
                pass
            self._lru.append(scene_id)
            return self._cache[scene_id]

        noisy_path = self.root / scene_id / "sensors.csv"
        gt_path = self.root / scene_id / "sensors_gt.csv"
        df_noisy = pd.read_csv(noisy_path, index_col=0, parse_dates=True)
        df_gt = pd.read_csv(gt_path, index_col=0, parse_dates=True)

        if self.sensor_ids is None:
            self.sensor_ids = list(df_noisy.columns)

        df_noisy = _ensure_sensor_order(df_noisy, self.sensor_ids)
        df_gt = _ensure_sensor_order(df_gt, self.sensor_ids)

        ts = df_noisy.index
        if not ts.equals(df_gt.index):
            raise ValueError(f"Timestamps mismatch between sensors.csv and sensors_gt.csv in {scene_id}")

        x_noisy = df_noisy.to_numpy(dtype=np.float32)
        x_gt = df_gt.to_numpy(dtype=np.float32)

        if self.standardizer is not None:
            x_noisy = self.standardizer.transform(x_noisy)
            x_gt = self.standardizer.transform(x_gt)

        tf = make_time_features(ts)

        loaded = _LoadedSeries(
            noisy=x_noisy,
            gt=x_gt,
            time_feat=tf,
            timestamps=ts,
            sensor_ids=list(self.sensor_ids),
        )
        self._cache_put(scene_id, loaded)
        return loaded


# ----------------------------
# Dataset 1: Normal predictor
# ----------------------------
class NormalPredictorDataset(Dataset):
    """
    Samples (X, Y) windows from NO-LEAK scenarios, with no constraint on the temporal interval between adjacent samples.
    - X uses sensors.csv (noisy) + time features
    - Y uses sensors_gt.csv (true) at t+1 (or horizon H)
    """
    def __init__(
        self,
        normal_root: str | Path,
        l_in_steps: int = 36,
        horizon_steps: int = 1,
        steps_per_epoch: int = 50000,
        seed: int = 42,
        sensor_ids: Optional[Sequence[str]] = None,
        standardizer: Optional[SensorStandardizer] = None,
        cache_size: int = 32,
    ) -> None:
        super().__init__()
        self.normal_root = Path(normal_root)
        self.l_in = int(l_in_steps)
        self.h = int(horizon_steps)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)

        manifest = _filter_ok(_read_manifest_jsonl(self.normal_root / "manifest.jsonl"))
        self.scene_ids = [r["window_id"] for r in manifest]
        self.store = ScenarioStore(self.normal_root, sensor_ids=sensor_ids, standardizer=standardizer, cache_size=cache_size)
        
        self._sensor_ids = self.store.get(self.scene_ids[0]).sensor_ids # Preload first scene to obtain sensor_ids

    def get_sensor_node_ids(self):
        return self._sensor_ids

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed + idx)

        scene_id = rng.choice(self.scene_ids)
        data = self.store.get(scene_id)
        T = data.noisy.shape[0]

        # choose t such that input uses [t-l_in+1..t] and target uses [t+1..t+h]
        t_min = self.l_in - 1
        t_max = T - self.h - 1
        if t_max < t_min:
            raise ValueError(f"Scene too short for l_in={self.l_in}, h={self.h}: {scene_id} (T={T})")

        t = int(rng.integers(t_min, t_max + 1))

        x = data.noisy[t - self.l_in + 1 : t + 1] # (L, S)
        x_tf = data.time_feat[t - self.l_in + 1 : t + 1] # (L, 9)
        y = data.gt[t + 1 : t + 1 + self.h] # (H, S)

        sample = {
            "scene_id": scene_id,
            "t": str(data.timestamps[t]),
            "x": torch.from_numpy(x),
            "x_time": torch.from_numpy(x_tf),
            "y": torch.from_numpy(y),
        }
        return sample


# ----------------------------
# Dataset 2: Leak detector (abrupt only)
# ----------------------------
@dataclass(frozen=True)
class DetectorSamplingConfig:
    # bucket probs (must sum to 1.0)
    p_early: float = 0.30
    p_late: float = 0.30
    p_pre: float = 0.20
    p_noleak: float = 0.20

    early_hours: float = 6.0
    pre_hours: float = 2.0

    q0: float = 1e-6  # leak flow threshold (m3/h) to define onset

    def validate(self) -> None:
        s = self.p_early + self.p_late + self.p_pre + self.p_noleak
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Bucket probs must sum to 1.0, got {s}")


class AbruptLeakDetectorDataset(Dataset):
    """
    Returns a contiguous sensor segment ending at time t for training a leak detector.

    Key idea for residual sequence (L_det steps) using a 1-step predictor with window L_pred:
    - we return segment of length (L_pred + L_det), covering times [t-(L_pred+L_det)+1 .. t].

    Your training loop can compute residuals by sliding the predictor over this segment.

    Label:
    - softmax multi-class: (num_pipes + 1)
      - class in [0..num_pipes-1] = leaking pipe
      - class == num_pipes        = no-leak
    Buckets (as requested):
    - Early positive:    t in [tau, tau+6h)
    - Late positive:     t >= tau+6h
    - Pre-leak negative: t in [tau-2h, tau)
    - No-leak negative:  any t from no-leak scenarios
    """
    def __init__(
        self,
        leak_root: str | Path,
        l_pred_steps: int = 36,
        l_det_steps: int = 36,
        steps_per_epoch: int = 80000,
        seed: int = 123,
        sensor_ids: Optional[Sequence[str]] = None,
        standardizer: Optional[SensorStandardizer] = None,
        step_min: int = 5, # default report time step in dataset
        sampling: DetectorSamplingConfig = DetectorSamplingConfig(),
        cache_size: int = 32,
        include_leak_types: Tuple[str, ...] = ("abrupt",),
    ) -> None:
        super().__init__()
        self.leak_root = Path(leak_root)
        self.l_pred = int(l_pred_steps)
        self.l_det = int(l_det_steps)
        self.seg_len = self.l_pred + self.l_det
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.include_leak_types = tuple(include_leak_types)

        sampling.validate()
        self.sampling = sampling

        manifest = _filter_ok(_read_manifest_jsonl(self.leak_root / "manifest.jsonl"))

        # Split scenarios into leak vs no-leak
        leak_rows: List[Dict[str, Any]] = []
        noleak_rows: List[Dict[str, Any]] = []
        for r in manifest:
            kind = str(r.get("kind", "")).lower()
            leak_type = str(r.get("leak_type", None)).lower()
            pipe_id = r.get("pipe_id", None)

            if kind == "leak" and leak_type is not None and pipe_id is not None:
                if leak_type in self.include_leak_types:
                    leak_rows.append(r)
                else:
                    continue
            else:
                noleak_rows.append(r)

        self.leak_scene_ids = [r["scenario_id"] for r in leak_rows]
        self.noleak_scene_ids = [r["scenario_id"] for r in noleak_rows]

        # Build pipe_id -> class index mapping from leak rows
        pipe_ids = sorted({str(r["pipe_id"]) for r in leak_rows})
        self.pipe_ids_in_order = pipe_ids
        self.pipe_to_idx = {pid: i for i, pid in enumerate(pipe_ids)}
        self.num_pipes = len(pipe_ids)
        self.no_leak_class = self.num_pipes

        self.store = ScenarioStore(self.leak_root, sensor_ids=sensor_ids, standardizer=standardizer, cache_size=cache_size)
        
        # Precompute eligible time indices for each bucket per scenario
        self._bucket_times: Dict[str, Dict[str, np.ndarray]] = {}
        self._pipe_idx: Dict[str, int] = {}

        # step per hour
        self.step_min = step_min
        steps_per_hour = 60.0 / self.step_min
        early_steps = int(round(self.sampling.early_hours * steps_per_hour))
        pre_steps = int(round(self.sampling.pre_hours * steps_per_hour))
        if early_steps <= 0 or pre_steps <= 0:
            raise ValueError(f"Unreasonable values of early_steps ({early_steps}) or pre_steps ({pre_steps}).")

        for r in leak_rows:
            sid = r["scenario_id"]
            pid = str(r["pipe_id"])
            self._pipe_idx[sid] = self.pipe_to_idx[pid]

            data = self.store.get(sid)
            T = data.noisy.shape[0]

            # find onset tau using leak_flow_m3h.csv
            flow_path = self.leak_root / sid / "leak_flow_m3h.csv"
            df_q = pd.read_csv(flow_path, index_col=0, parse_dates=True)
            q = df_q[pid].to_numpy(dtype=np.float32)
            if len(q) != T:
                raise ValueError(f"Length mismatch leak_flow vs sensors in {sid}: {len(q)} vs {T}")

            active = q > float(self.sampling.q0)
            if not np.any(active):
                # should not happen for leak scenario; fallback to treat as no-leak
                tau = T  # no active region
            else:
                tau = int(np.argmax(active))  # first True

            # eligible t must have enough history to form segment of length seg_len ending at t
            t_min = self.seg_len - 1
            t_max = T - 1
            if t_max < t_min:
                # scene too short, skip by leaving empty buckets
                self._bucket_times[sid] = {"early": np.array([], dtype=np.int64), "late": np.array([], dtype=np.int64), "pre": np.array([], dtype=np.int64)}
                continue

            # buckets (inclusive/exclusive ranges on indices)
            early_start = tau
            early_end = min(tau + early_steps, T)  # exclusive
            pre_start = max(tau - pre_steps, 0)
            pre_end = tau
            late_start = min(tau + early_steps, T)
            late_end = T

            def _range_to_idx(a: int, b: int) -> np.ndarray:
                # [a, b) intersect [t_min, t_max+1)
                a2 = max(a, t_min)
                b2 = min(b, t_max + 1)
                if b2 <= a2:
                    return np.array([], dtype=np.int64)
                return np.arange(a2, b2, dtype=np.int64)

            early_idx = _range_to_idx(early_start, early_end)
            late_idx = _range_to_idx(late_start, late_end)
            pre_idx = _range_to_idx(pre_start, pre_end)

            self._bucket_times[sid] = {"early": early_idx, "late": late_idx, "pre": pre_idx}

        # precompute eligible indices for no-leak scenarios
        self._noleak_times: Dict[str, np.ndarray] = {}
        for sid in self.noleak_scene_ids:
            data = self.store.get(sid)
            T = data.noisy.shape[0]
            t_min = self.seg_len - 1
            t_max = T - 1
            if t_max < t_min:
                self._noleak_times[sid] = np.array([], dtype=np.int64)
            else:
                self._noleak_times[sid] = np.arange(t_min, t_max + 1, dtype=np.int64)

        # Filter out leak scenarios that have no usable bucket times at all
        usable_leak = []
        for sid in self.leak_scene_ids:
            bt = self._bucket_times.get(sid, {})
            if bt and (bt["early"].size + bt["late"].size + bt["pre"].size) > 0:
                usable_leak.append(sid)
        self.leak_scene_ids = usable_leak

        usable_noleak = [sid for sid in self.noleak_scene_ids if self._noleak_times.get(sid, np.array([], dtype=np.int64)).size > 0]
        self.noleak_scene_ids = usable_noleak

        if not self.leak_scene_ids:
            raise RuntimeError("No usable abrupt leak scenarios found. Check manifest filters and scene lengths.")
        if not self.noleak_scene_ids:
            raise RuntimeError("No usable no-leak scenarios found. Check manifest or scene lengths.")

        # cumulative probs for bucket choice
        self._cum = np.array(
            [
                self.sampling.p_early,
                self.sampling.p_early + self.sampling.p_late,
                self.sampling.p_early + self.sampling.p_late + self.sampling.p_pre,
                1.0
            ],
            dtype=np.float64
        )
        
        # Preload first scene to obtain sensor_ids
        self._sensor_ids = self.store.get(self.leak_scene_ids[0]).sensor_ids

    def get_sensor_node_ids(self):
        return self._sensor_ids
    
    def get_pipe_ids_in_order(self):
        return list(self.pipe_ids_in_order)

    def get_pipe_to_idx(self):
        return dict(self.pipe_to_idx)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _choose_bucket(self, rng: np.random.Generator) -> str:
        u = float(rng.random())
        if u < self._cum[0]:
            return "early"
        if u < self._cum[1]:
            return "late"
        if u < self._cum[2]:
            return "pre"
        return "noleak"

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = np.random.default_rng(self.seed + idx)

        # Try a few times in case a bucket has no eligible indices (edge cases)
        for _ in range(10):
            bucket = self._choose_bucket(rng)

            if bucket == "noleak":
                sid = rng.choice(self.noleak_scene_ids)
                times = self._noleak_times[sid]
                if times.size == 0:
                    continue
                t = int(rng.choice(times))
                label = self.no_leak_class
                pipe_idx = -1
                pipe_id = None
            else:
                sid = rng.choice(self.leak_scene_ids)
                times = self._bucket_times[sid][bucket]
                if times.size == 0:
                    continue
                t = int(rng.choice(times))
                if bucket in ("early", "late"):
                    pipe_idx = int(self._pipe_idx[sid])
                    label = pipe_idx
                    # recover pipe_id if needed
                    pipe_id = next((pid for pid, j in self.pipe_to_idx.items() if j == pipe_idx), None)
                else:
                    # pre-leak negative
                    label = self.no_leak_class
                    pipe_idx = int(self._pipe_idx[sid])
                    pipe_id = next((pid for pid, j in self.pipe_to_idx.items() if j == pipe_idx), None)

            data = self.store.get(sid)
            # segment ending at t (inclusive): [t-seg_len+1 .. t]
            seg_noisy = data.noisy[t - self.seg_len + 1 : t + 1] # (seg_len, S)
            seg_time = data.time_feat[t - self.seg_len + 1 : t + 1] # (seg_len, 9)
            assert seg_noisy.shape[0] == self.seg_len

            sample = {
                "scenario_id": sid,
                "bucket": bucket,
                "t": str(data.timestamps[t]),
                "noisy_seg": torch.from_numpy(seg_noisy),
                "time_seg": torch.from_numpy(seg_time),
                "label": int(label),               # softmax class in [0..num_pipes]
                "pipe_index": int(pipe_idx),       # -1 for noleak scenario; else underlying leak pipe
                "pipe_id": (str(pipe_id) if pipe_id is not None else "NOLEAK"), # may be None
                "num_classes": int(self.num_pipes + 1),
                "l_pred": int(self.l_pred),
                "l_det": int(self.l_det),
            }
            return sample

        raise RuntimeError("Failed to sample a valid (scenario, time) after multiple attempts. Check data lengths/buckets.")


if __name__ == "__main__":
    import time
    start = time.perf_counter()
    
    normal_root = "data/simulate/L-TOWN-A/normal/dataset_v1"
    leak_root = "data/simulate/L-TOWN-A/leakage/dataset_v1"
    
    stdzr = compute_sensor_stats_from_normal(normal_root)
    sampling = DetectorSamplingConfig(
        p_early=0.30, p_late=0.30, p_pre=0.20, p_noleak=0.20,
        early_hours=6.0, pre_hours=2.0, q0=1e-6
    )
    
    ds_norm = NormalPredictorDataset(normal_root,standardizer=stdzr)

    ds_det = AbruptLeakDetectorDataset(
        leak_root=leak_root,
        l_pred_steps=36,
        l_det_steps=36,
        steps_per_epoch=1000,
        sampling=sampling,
        standardizer=stdzr,
        cache_size = 1024
    )
    
    # n, warmup, seed = 20000, 2000, 0
    # rng = np.random.default_rng(0)
    # L = len(ds_det)
    # idxs = rng.integers(0, L, size=n)

    # for i in idxs[:warmup]:
    #     _ = ds_det[int(i)]

    # t0 = time.perf_counter()
    # for i in idxs[warmup:]:
    #     _ = ds_det[int(i)]
    # t1 = time.perf_counter()

    # m = n - warmup
    # avg_ms = (t1 - t0) / m * 1000
    # it_s = m / (t1 - t0)
    # print(f"dataset __getitem__: {avg_ms:.3f} ms/sample, {it_s:.1f} samples/s")
    
    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.2f}s.")

    