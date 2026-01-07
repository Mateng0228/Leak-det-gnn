'''
python eval/event_evaluator.py \
  --dataset_root data/simulate/L-TOWN-A/leakage/dataset_v1 \
  --inp_path data/raw/L-TOWN-A/L-TOWN_AreaA.inp \
  --predictor_ckpt results/L-TOWN-A/predictor_best.ckpt \
  --detector_ckpt results/L-TOWN-A/detector_best.ckpt \
  --l_pred_hours 3 --l_det_hours 3 --step_minutes 5 \
  --max_leak_scens 2 --max_noleak_scens 2
'''
from __future__ import annotations
import argparse
import os
import sys
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.predictor import NormalPredictorGRU, NormalPredictorTCN
from models.detector import LeakDetector
from eval.metrics import EventResult, compute_event_metrics


# ----------------------------
# Helpers: EPANET parser
# ----------------------------

def _read_inp_sections(inp_path: str | Path) -> Dict[str, List[str]]:
    """Minimal EPANET .inp parser: returns section_name -> list of non-comment raw lines."""
    inp_path = str(inp_path)
    sections: Dict[str, List[str]] = {}
    cur = None
    with open(inp_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                cur = line.strip("[]").upper()
                sections.setdefault(cur, [])
                continue
            if cur is None:
                continue
            # strip inline comments
            if ";" in line:
                line = line.split(";", 1)[0].strip()
            if line:
                sections[cur].append(line)
    return sections


def _parse_links(inp_path: str | Path, eps: float = 0.1) -> Dict[str, Tuple[str, str, float, str]]:
    """Return: link_id -> (node1, node2, length_m, kind), where 'kind' in {'PIPES','PUMPS','VALVES'}"""
    sections = _read_inp_sections(inp_path)
    out = {}

    # PIPES: ID Node1 Node2 Length ...
    for line in sections.get("PIPES", []):
        toks = line.split()
        if len(toks) < 4: 
            continue
        lid, n1, n2 = toks[0], toks[1], toks[2]
        try:
            length = float(toks[3])
        except Exception:
            length = float("nan")
        out[lid] = (n1, n2, length, "PIPES")

    # PUMPS: ID Node1 Node2 ...
    for line in sections.get("PUMPS", []):
        toks = line.split()
        if len(toks) < 3:
            continue
        lid, n1, n2 = toks[0], toks[1], toks[2]
        out[lid] = (n1, n2, float(eps), "PUMPS")

    # VALVES: ID Node1 Node2 ...
    for line in sections.get("VALVES", []):
        toks = line.split()
        if len(toks) < 3:
            continue
        lid, n1, n2 = toks[0], toks[1], toks[2]
        out[lid] = (n1, n2, float(eps), "VALVES")

    return out


class PipeDistanceOracle:
    """Computes pipe-to-pipe network distance (meters)"""
    def __init__(self, inp_path: str | Path, pipe_ids_in_order: Sequence[str]):
        self.inp_path = str(inp_path)
        self.pipe_ids = list(pipe_ids_in_order)

        links_all = _parse_links(self.inp_path)
        if not links_all:
            raise ValueError("No [PIPES/PUMPS/VALVES] parsed from inp; cannot compute distances.")

        # node indexing from ALL links for connectivity
        node_set = set()
        for _, (n1, n2, _, _) in links_all.items():
            node_set.add(n1); node_set.add(n2)
        self.node_names = sorted(node_set)
        self.node_to_idx = {n: i for i, n in enumerate(self.node_names)}

        # adjacency list weighted by length
        n_nodes = len(self.node_names)
        self.adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_nodes)]
        for _, (n1, n2, length_m, _) in links_all.items():
            u = self.node_to_idx[n1]
            v = self.node_to_idx[n2]
            w = float(length_m) if (length_m is not None and not math.isnan(length_m)) else 1.0
            self.adj[u].append((v, w))
            self.adj[v].append((u, w))

        # map pipe ids to ends/length (only those in model label space)
        P = len(self.pipe_ids)
        self.pipe_ends = np.zeros((P, 2), dtype=np.int32)
        self.pipe_len = np.zeros((P,), dtype=np.float32)
        for i, pid in enumerate(self.pipe_ids):
            if pid not in links_all:
                raise ValueError(f"pipe_id '{pid}' not found in inp [PIPES/PUMPS/VALVES].")
            n1, n2, length_m, _ = links_all[pid]
            self.pipe_ends[i, 0] = self.node_to_idx[n1]
            self.pipe_ends[i, 1] = self.node_to_idx[n2]
            self.pipe_len[i] = float(length_m) if (length_m is not None and not math.isnan(length_m)) else 1.0

        self._node_dist_cache: Dict[int, np.ndarray] = {}

    def _dijkstra(self, start: int) -> np.ndarray:
        import heapq
        n = len(self.adj)
        dist = np.full(n, np.inf, dtype=np.float64)
        dist[start] = 0.0
        pq: List[Tuple[float, int]] = [(0.0, start)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in self.adj[u]:
                nd = d + float(w)
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (float(nd), v))
        return dist.astype(np.float32, copy=False)

    def _node_dists(self, start: int) -> np.ndarray:
        if start not in self._node_dist_cache:
            self._node_dist_cache[start] = self._dijkstra(start)
        return self._node_dist_cache[start]

    def pipe_distance(self, true_idx: int, pred_idx: int) -> float:
        if true_idx == pred_idx:
            return 0.0
        up, vp = int(self.pipe_ends[true_idx, 0]), int(self.pipe_ends[true_idx, 1])
        uq, vq = int(self.pipe_ends[pred_idx, 0]), int(self.pipe_ends[pred_idx, 1])
        d_up = self._node_dists(up)
        d_vp = self._node_dists(vp)
        dmin = min(float(d_up[uq]), float(d_up[vq]), float(d_vp[uq]), float(d_vp[vq]))
        return dmin + 0.5 * float(self.pipe_len[true_idx]) + 0.5 * float(self.pipe_len[pred_idx])


# ----------------------------
# Scenario reading
# ----------------------------
def read_manifest(manifest_path: str | Path) -> List[Dict]:
    manifest_path = str(manifest_path)
    rows: List[Dict] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_sensors_csv(path: str | Path, sensor_ids: Sequence[str]) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    missing = [c for c in sensor_ids if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensors in csv: {missing[:10]} ...")
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

def load_leak_flow_csv(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # assume single column; otherwise sum
    if df.shape[1] == 1:
        s = df.iloc[:, 0]
    else:
        s = df.sum(axis=1)
    return s


def find_tau_from_leak_flow(leak_flow: pd.Series, eps: float = 1e-9) -> Optional[pd.Timestamp]:
    mask = leak_flow.values.astype(np.float64) > float(eps)
    if not mask.any():
        return None
    return leak_flow.index[np.argmax(mask)]


# ----------------------------
# Model helpers
# ----------------------------

@dataclass(frozen=True)
class SensorStandardizer:
    mean: np.ndarray  # (S,)
    std: np.ndarray   # (S,)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std) + self.mean

def _load_predictor(ckpt_path: str | Path, device: torch.device, args) -> Tuple[nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = ckpt.get("arch", "tcn")
    sensor_ids = ckpt["sensor_ids"]
    S = len(sensor_ids)
    if arch == "gru":
        model = NormalPredictorGRU(num_sensors=S, time_dim=9)
    else:
        model = NormalPredictorTCN(num_sensors=S, time_dim=9)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt

def _load_detector(ckpt_path: str | Path, device: torch.device, args) -> Tuple[nn.Module, Dict]:
    inp_path = args.inp_path
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sensor_ids = ckpt["sensor_ids"]
    pipe_ids_in_order = ckpt["pipe_ids_in_order"]
    
    model = LeakDetector(inp_path, sensor_ids, pipe_ids_in_order, sensor_hidden=64, node_hidden=64, gnn_layers=2, dropout=0.1, use_time=True)
    model.load_state_dict(ckpt["detector_state"])
    model.to(device).eval()
    return model, ckpt

@torch.no_grad()
def build_residual_segment(predictor: nn.Module, noisy_seg: torch.Tensor, time_seg: torch.Tensor, l_pred, l_det, device) -> torch.Tensor:
    """Compute residual sequence for detector input from a (l_pred + l_det) segment."""
    # Ensure batch dim
    if noisy_seg.dim() == 2:
        noisy_seg = noisy_seg.unsqueeze(0)
        time_seg = time_seg.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    B, seg_len, S = noisy_seg.shape
    assert seg_len == l_pred + l_det, (seg_len, l_pred, l_det)

    if device is not None:
        noisy_seg = noisy_seg.to(device)
        time_seg = time_seg.to(device)

    xs = []
    xt = []
    targets = []
    for kk in range(l_det):
        k = l_pred + kk
        xs.append(noisy_seg[:, k - l_pred : k, :])
        xt.append(time_seg[:, k - l_pred : k, :])
        targets.append(noisy_seg[:, k, :])

    X = torch.cat(xs, dim=0)
    Tfeat = torch.cat(xt, dim=0)
    Y_noisy = torch.cat(targets, dim=0)

    y_hat = predictor(X, Tfeat)
    if y_hat.dim() == 3:
        y_hat = y_hat[:, 0, :] # 取第一步预测

    residual = Y_noisy - y_hat
    residual = residual.reshape(l_det, B, S).transpose(0, 1).contiguous()

    if squeeze_back:
        residual = residual.squeeze(0)
    return residual


@torch.no_grad()
def detector_logits(detector: nn.Module, res_seg: torch.Tensor, tf_seg: torch.Tensor, device: torch.device) -> torch.Tensor:
    if res_seg.dim() == 2:
        res_seg = res_seg.unsqueeze(0)
        tf_seg = tf_seg.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    res_seg = res_seg.to(device)
    tf_seg = tf_seg.to(device)
    logits = detector(res_seg, tf_seg)
    logits = logits.detach().cpu()

    if squeeze_back:
        logits = logits.squeeze(0)
    return logits


# ----------------------------
# Trigger + aggregation
# ----------------------------

def trigger_argmax(records: List[Tuple[pd.Timestamp, torch.Tensor]], noleak_class: int) -> Optional[int]:
    for i, (_, lg) in enumerate(records):
        if int(torch.argmax(lg).item()) != int(noleak_class):
            return i
    return None


def aggregate_sum_logits(records: List[Tuple[pd.Timestamp, torch.Tensor]], start_idx: int, agg_window: pd.Timedelta) -> torch.Tensor:
    t0 = records[start_idx][0]
    s = None
    for t, lg in records[start_idx:]:
        if t - t0 >= agg_window:
            break
        s = lg if s is None else (s + lg)
    return s if s is not None else records[start_idx][1]


# ----------------------------
# Main evaluation
# ----------------------------

def evaluate_dataset_event_level(
    dataset_root: str | Path, inp_path: str | Path, device: torch.device,
    predictor: nn.Module, detector: nn.Module, l_pred_steps: int, l_det_steps: int,
    standardizer: SensorStandardizer, sensor_ids: Sequence[str], pipe_ids_in_order: List[str],
    stride_steps: int = 1, agg_window_hours: float = 12.0,
    include_noleak: bool = True, max_leak_scens: int = 0, max_noleak_scens: int = 0, sample_seed: float = 42,
    eps_tau: float = 1e-9,
    success_radii_m: Sequence[float] = (50.0, 100.0, 300.0),
    out_dir: Optional[str | Path] = None,
) -> Dict[str, float]:
    """Evaluate a dataset at event-level. Returns summary metrics dict."""
    
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.jsonl not found under {dataset_root}")
    rows = read_manifest(manifest_path)

    pipe_to_idx = {pid: i for i, pid in enumerate(pipe_ids_in_order)}
    noleak_class = len(pipe_ids_in_order)
    num_classes = noleak_class + 1

    print(f"[event-eval] dataset_root={dataset_root}")
    print(f"[event-eval] pipes={len(pipe_ids_in_order)} num_classes={num_classes} include_noleak={include_noleak}")
    print(f"[event-eval] l_pred_steps={l_pred_steps} l_det_steps={l_det_steps} stride_steps={stride_steps} agg_window={agg_window_hours}h")

    oracle = PipeDistanceOracle(inp_path, pipe_ids_in_order)

    agg_window = pd.Timedelta(hours=float(agg_window_hours))
    events: List[EventResult] = []
    
    # optional outputs
    out_path = Path(out_dir) if out_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
        per_event_path = out_path / "per_event.jsonl"
        if per_event_path.exists():
            per_event_path.unlink()

    # select scenarios
    rng = random.Random(int(sample_seed))
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    leak_rows = [r for r in ok_rows if r.get("kind") == "leak"]
    noleak_rows = [r for r in ok_rows if r.get("kind") == "no_leak"]

    by_pipe = defaultdict(list)
    for r in leak_rows:
        pid = r.get("pipe_id")
        if pid is not None:
            by_pipe[pid].append(r)
    for pid in by_pipe:
        rng.shuffle(by_pipe[pid])

    selected_leak = []
    pipe_cycle = [pid for pid in pipe_ids_in_order if pid in by_pipe]
    rng.shuffle(pipe_cycle)
    if int(max_leak_scens) >= 0: # round-robin across pipes
        target = int(max_leak_scens)
        while len(selected_leak) < target:
            added_any = False
            for pid in pipe_cycle:
                if by_pipe[pid]:
                    selected_leak.append(by_pipe[pid].pop())
                    added_any = True
                    if len(selected_leak) >= target:
                        break
            if not added_any:
                break
    elif int(max_leak_scens) == -1: # take all leaks
        for pid in pipe_cycle:
            selected_leak.extend(by_pipe[pid])
    else:
        raise ValueError("Invalid max_leak_scens value.")

    selected_noleak = []
    if include_noleak and noleak_rows:
        max_noleak_scens = min(int(max_noleak_scens), len(noleak_rows))
        if max_noleak_scens >= 0:
            selected_noleak = rng.sample(noleak_rows, max_noleak_scens)
        elif max_noleak_scens == -1:
            selected_noleak = noleak_rows
        else:
            raise ValueError("Invalid max_noleak_scens value.")

    ok_rows = selected_leak + selected_noleak
    rng.shuffle(ok_rows)
    total_rows = len(ok_rows)
    for idx, r in enumerate(ok_rows, start=1):
        sid = r.get("scenario_id")
        kind = r.get("kind")
        if (not include_noleak) and (kind == "no_leak"):
            continue

        scen_dir = dataset_root / str(sid)
        sensors_path = scen_dir / "sensors.csv"
        if not sensors_path.exists():
            print(f"[event-eval][warn] missing sensors.csv for {sid}, skip.")
            continue
        df_sensor = load_sensors_csv(sensors_path, sensor_ids)  # (T,S)
        if len(df_sensor) < (l_pred_steps + l_det_steps):
            print(f"[event-eval][warn] too short ({len(df_sensor)} rows) for {sid}, skip.")
            continue

        # time features
        tfeat_all = make_time_features(pd.to_datetime(df_sensor.index))
        pressure_all = df_sensor.values.astype(np.float32)
        pressure_all = standardizer.transform(pressure_all)

        # leak truth
        is_leak_true = (kind == "leak")
        true_pipe_id = r.get("pipe_id") if is_leak_true else None
        if is_leak_true and (true_pipe_id not in pipe_to_idx):
            print(f"[event-eval][warn] pipe_id {true_pipe_id} not in label space, skip {sid}.")
            continue

        # tau from leak_flow
        tau = None
        if is_leak_true:
            lf_path = scen_dir / "leak_flow_m3h.csv"
            if lf_path.exists():
                leak_flow = load_leak_flow_csv(lf_path)
                tau = find_tau_from_leak_flow(leak_flow, eps=eps_tau)
            else:
                print(f"[event-eval][warn] missing leak_flow_m3h.csv for leak scenario {sid}.")
        tau_iso = tau.isoformat() if tau is not None else None

        # generate window logits records: list of (time_end, logits)
        records: List[Tuple[pd.Timestamp, torch.Tensor]] = []
        T = len(df_sensor)
        # t0 is prediction-start index; residual window ends at t0 + l_det_steps - 1
        for t0 in range(l_pred_steps, T - l_det_steps + 1, stride_steps):
            in_start = t0 - l_pred_steps
            out_end = t0 + l_det_steps
            seg_pressure = torch.from_numpy(pressure_all[in_start:out_end, :]) # (seg_len, S)
            seg_tf = torch.from_numpy(tfeat_all[in_start:out_end, :])          # (seg_len, F)

            residual = build_residual_segment(predictor, seg_pressure, seg_tf, l_pred_steps, l_det_steps, device) # (l_det, S)
            tf_out = torch.from_numpy(tfeat_all[t0:out_end, :])  # (l_det, F)
            lg = detector_logits(detector, residual, tf_out, device)  # (C,)
            # sanity: ensure correct class dimension
            if lg.numel() != num_classes:
                raise RuntimeError(f"Detector logits dim {lg.numel()} != expected num_classes {num_classes}")
            
            time_end = df_sensor.index[out_end - 1]
            records.append((time_end, lg))
            
        # trigger
        trig_idx = trigger_argmax(records, noleak_class=noleak_class)
        if trig_idx is None:
            # no alarm
            ev = EventResult(
                scenario_id=str(sid),
                is_leak_true=bool(is_leak_true),
                is_leak_pred=False,
                true_pipe_id=true_pipe_id,
                pred_pipe_id=None,
                tau_iso=tau_iso,
                alarm_time_iso=None,
                atd_m=None,
            )
            events.append(ev)
        else:
            alarm_time = records[trig_idx][0]
            sum_logits = aggregate_sum_logits(records, trig_idx, agg_window=agg_window)
            pred_idx = int(torch.argmax(sum_logits).item())
            is_leak_pred = (pred_idx != noleak_class)

            pred_pipe_id = None
            atd_m = None
            if is_leak_pred:
                pred_pipe_id = pipe_ids_in_order[pred_idx]
                if is_leak_true:
                    true_idx = pipe_to_idx[true_pipe_id]
                    atd_m = float(oracle.pipe_distance(true_idx, pred_idx))

            ev = EventResult(
                scenario_id=str(sid),
                is_leak_true=bool(is_leak_true),
                is_leak_pred=bool(is_leak_pred),
                true_pipe_id=true_pipe_id,
                pred_pipe_id=pred_pipe_id,
                tau_iso=tau_iso,
                alarm_time_iso=alarm_time.isoformat(),
                atd_m=atd_m,
            )
            events.append(ev)
            
        if (idx % 25) == 0 or idx == total_rows:
            print(f"[event-eval] processed {idx}/{total_rows} scenarios...")

        if out_path is not None:
            # append per-event jsonl
            with open(out_path / "per_event.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.__dict__, ensure_ascii=False) + "\n")

    # compute metrics
    summary = compute_event_metrics(
        events,
        success_radii_m=success_radii_m,
        localization_mode="detected",  # per user decision
    )

    print("[event-eval] SUMMARY")
    keys_show = [
        "n_events", "n_leak_events", "n_noleak_events",
        "det_precision", "det_recall", "det_f1", "det_fp_rate", "det_fn_rate",
        "loc_n_events", "loc_n_detected", "loc_accuracy_exact",
        "loc_atd_mean_m", "loc_atd_median_m",
        "loc_success_at_50m", "loc_success_at_100m", "loc_success_at_300m",
    ]
    for k in keys_show:
        if k in summary:
            v = summary[k]
            if isinstance(v, float) and (abs(v) < 1e6):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")

    if out_path is not None:
        with open(out_path / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[event-eval] saved: {out_path / 'summary.json'} , {out_path / 'per_event.jsonl'}")
    
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, required=True, help="Leak dataset root (manifest.jsonl + scenario dirs).")
    ap.add_argument("--inp_path", type=str, required=True, help="EPANET .inp path.")
    ap.add_argument("--predictor_ckpt", type=str, required=True, help="Predictor checkpoint.")
    ap.add_argument("--detector_ckpt", type=str, required=True, help="Detector checkpoint.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--l_pred_hours", type=float, default=3.0)
    ap.add_argument("--l_det_hours", type=float, default=3.0)
    ap.add_argument("--step_minutes", type=float, default=5.0)
    ap.add_argument("--stride_minutes", type=float, default=5.0)

    ap.add_argument("--agg_window_hours", type=float, default=12.0)
    ap.add_argument("--include_noleak", action="store_true", help="Include no-leak scenarios to compute detection metrics.")
    ap.add_argument("--max_leak_scens", type=int, default=0, help="最多读取多少个leak场景(-1表示全读).")
    ap.add_argument("--max_noleak_scens", type=int, default=0, help="最多读取多少个no_leak场景(-1表示全读).")
    ap.add_argument("--sample_seed", type=int, default=42, help="scenario sampling seed.")

    ap.add_argument("--out_dir", type=str, default=None, help="If set, save per_event.jsonl and summary.json here.")
    args = ap.parse_args()

    device = torch.device(args.device)
    
    # convert hours -> steps
    steps_per_hour = int(round(60.0 / float(args.step_minutes)))
    l_pred_steps = int(round(float(args.l_pred_hours) * steps_per_hour))
    l_det_steps = int(round(float(args.l_det_hours) * steps_per_hour))
    stride_steps = int(round(float(args.stride_minutes) / float(args.step_minutes)))
    stride_steps = max(1, stride_steps)
    
    # load models
    predictor, pred_ckpt = _load_predictor(args.predictor_ckpt, device, args)
    detector, det_ckpt = _load_detector(args.detector_ckpt, device, args)
    
    std_mean = pred_ckpt.get("standardizer_mean", None)
    std_std = pred_ckpt.get("standardizer_std", None)
    stdzr = SensorStandardizer(mean=np.asarray(std_mean, dtype=np.float32), std=np.asarray(std_std, dtype=np.float32))
    
    assert pred_ckpt["sensor_ids"] == det_ckpt["sensor_ids"]
    sensor_ids = det_ckpt["sensor_ids"]
    pipe_ids_in_order = det_ckpt["pipe_ids_in_order"]
    
    evaluate_dataset_event_level(
        dataset_root=args.dataset_root, inp_path=args.inp_path, device=device,
        predictor=predictor, detector=detector, l_pred_steps=l_pred_steps, l_det_steps=l_det_steps,
        standardizer=stdzr, sensor_ids=sensor_ids, pipe_ids_in_order=pipe_ids_in_order,
        stride_steps=stride_steps, agg_window_hours=args.agg_window_hours,
        include_noleak=bool(args.include_noleak),
        max_leak_scens=int(args.max_leak_scens), max_noleak_scens=int(args.max_noleak_scens), sample_seed=float(args.sample_seed),
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
