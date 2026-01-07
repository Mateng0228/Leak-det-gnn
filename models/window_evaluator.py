"""
A configurable evaluator for the WDN leak detector. 
Supports: basic, binary, bucket, atd, success, accuracy_i
"""
from __future__ import annotations
import heapq
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .utils import now, parse_epanet_inp, build_residual_sequence_from_segment

def _parse_pipes_with_length(inp_path: str | Path) -> Dict[str, Tuple[str, str, float]]:
    """Parse [PIPES] as: pipe_id -> (node1, node2, length_m)."""
    sections = parse_epanet_inp(inp_path)
    lines = sections.get("PIPES", [])
    out: Dict[str, Tuple[str, str, float]] = {}
    for line in lines:
        toks = line.split()
        if len(toks) < 4:
            continue
        pid, n1, n2 = toks[0], toks[1], toks[2]
        try:
            length_m = float(toks[3])
        except Exception:
            length_m = float("nan")
        out[pid] = (n1, n2, length_m)
    return out


def _parse_coordinates(inp_path: str | Path) -> Dict[str, Tuple[float, float]]:
    """
    Parse [COORDINATES] as: node_id -> (x, y).
    EPANET format: Node  X-Coord  Y-Coord
    """
    sections = parse_epanet_inp(inp_path)
    lines = sections.get("COORDINATES", [])
    coords: Dict[str, Tuple[float, float]] = {}
    for line in lines:
        toks = line.split()
        if len(toks) < 3:
            continue
        nid = toks[0]
        try:
            x = float(toks[1])
            y = float(toks[2])
        except Exception:
            continue
        coords[nid] = (x, y)
    return coords

def _dijkstra(adj: List[List[Tuple[int, float]]], start: int) -> np.ndarray:
    """Single-source shortest paths on a sparse graph (non-negative weights)."""
    n = len(adj)
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[start] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + float(w)
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (float(nd), v))
    return dist.astype(np.float32, copy=False)


def _parse_links_with_length(inp_path: str | Path, eps: float) -> Dict[str, Tuple[str, str, float, str]]:
    """Return: link_id -> (node1, node2, length_m, kind), where 'kind' in {'PIPES','PUMPS','VALVES'}"""
    sections = parse_epanet_inp(inp_path)
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
            length = 1.0
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


@dataclass
class PipeDistanceOracle:
    """Precomputes distances needed for ATD/Success@X/Accuracy_i under a pipe-class setting."""
    inp_path: str
    pipe_ids_in_order: List[str]

    node_names: List[str]
    node_to_idx: Dict[str, int]
    pipe_ends: np.ndarray
    pipe_len: np.ndarray
    node_adj: List[List[Tuple[int, float]]]
    node_dist_cache: Dict[int, np.ndarray]
    pipe_dist: Optional[np.ndarray]
    pipe_rank: Optional[np.ndarray]

    @staticmethod
    def build(inp_path: str | Path, pipe_ids_in_order: Sequence[str], eps: float = 0.1) -> "PipeDistanceOracle":
        inp_path = str(inp_path)
        pipe_ids = list(pipe_ids_in_order)

        links_all = _parse_links_with_length(inp_path, eps)
        if not links_all:
            raise ValueError("No links parsed from inp; cannot compute distance metrics.")
        
        node_set = set()
        for _, (n1, n2, _, _) in links_all.items():
            node_set.add(n1)
            node_set.add(n2)
        node_names = sorted(node_set)
        node_to_idx = {n: i for i, n in enumerate(node_names)}

        adj: List[List[Tuple[int, float]]] = [[] for _ in range(len(node_names))]
        for _, (n1, n2, length_m, _) in links_all.items():
            u = node_to_idx[n1]
            v = node_to_idx[n2]
            w = float(length_m) if (length_m is not None and not math.isnan(length_m)) else 1.0
            adj[u].append((v, w))
            adj[v].append((u, w))

        P = len(pipe_ids)
        pipe_ends = np.zeros((P, 2), dtype=np.int32)
        pipe_len = np.zeros((P,), dtype=np.float32)
        for i, pid in enumerate(pipe_ids):
            if pid not in links_all:
                raise ValueError(f"pipe_id '{pid}' not found in inp [PIPES]/[PUMPS]/[VALVES].")
            n1, n2, length_m, _ = links_all[pid]
            pipe_ends[i, 0] = node_to_idx[n1]
            pipe_ends[i, 1] = node_to_idx[n2]
            pipe_len[i] = float(length_m) if (length_m is not None and not math.isnan(length_m)) else 1.0

        return PipeDistanceOracle(
            inp_path=inp_path,
            pipe_ids_in_order=pipe_ids,
            node_names=node_names,
            node_to_idx=node_to_idx,
            pipe_ends=pipe_ends,
            pipe_len=pipe_len,
            node_adj=adj,
            node_dist_cache={},
            pipe_dist=None,
            pipe_rank=None,
        )

    def _node_dists(self, start: int) -> np.ndarray:
        if start not in self.node_dist_cache:
            self.node_dist_cache[start] = _dijkstra(self.node_adj, start)
        return self.node_dist_cache[start]

    def pipe_distance(self, p: int, q: int) -> float:
        """Distance between pipe p and q (midpoint approximation)."""
        if p == q:
            return 0.0
        up, vp = int(self.pipe_ends[p, 0]), int(self.pipe_ends[p, 1])
        uq, vq = int(self.pipe_ends[q, 0]), int(self.pipe_ends[q, 1])

        d_up = self._node_dists(up)
        d_vp = self._node_dists(vp)
        dmin = min(float(d_up[uq]), float(d_up[vq]), float(d_vp[uq]), float(d_vp[vq]))
        return dmin + 0.5 * float(self.pipe_len[p]) + 0.5 * float(self.pipe_len[q])

    def ensure_pipe_matrix(self) -> None:
        if self.pipe_dist is not None and self.pipe_rank is not None:
            return

        P = len(self.pipe_ids_in_order)
        ends = self.pipe_ends
        L = self.pipe_len

        unique_starts = np.unique(ends.reshape(-1))
        for s in unique_starts.tolist():
            _ = self._node_dists(int(s))

        dist_mat = np.zeros((P, P), dtype=np.float32)
        u_list = ends[:, 0].astype(np.int32)
        v_list = ends[:, 1].astype(np.int32)

        for p in range(P):
            up, vp = int(ends[p, 0]), int(ends[p, 1])
            d_up = self.node_dist_cache[up]
            d_vp = self.node_dist_cache[vp]

            dmin = np.minimum.reduce([
                d_up[u_list],
                d_up[v_list],
                d_vp[u_list],
                d_vp[v_list],
            ]).astype(np.float32)

            dist_mat[p, :] = dmin + 0.5 * L[p] + 0.5 * L
            dist_mat[p, p] = 0.0

        self.pipe_dist = dist_mat
        self.pipe_rank = np.argsort(dist_mat, axis=1).astype(np.int32)


class DetectorEvaluator:
    """Evaluator for (predictor + detector) on a dataloader."""
    
    def __init__(
        self,
        predictor: nn.Module,
        detector: nn.Module,
        device: torch.device,
        *,
        l_pred: int,
        l_det: int,
        topk: int = 5,
        metric_groups: Sequence[str] = ("basic", "binary", "bucket"),
        # distance metrics config
        inp_path: Optional[str | Path] = None,
        pipe_ids_in_order: Optional[Sequence[str]] = None,
        success_radii_m: Sequence[float] = (50.0, 100.0, 300.0),
        accuracy_is: Sequence[int] = (1, 5, 10, 20),
    ) -> None:
        self.predictor = predictor
        self.detector = detector
        self.device = device
        self.l_pred = int(l_pred)
        self.l_det = int(l_det)
        self.topk = int(topk)
        self.metric_groups = set(metric_groups) | {"basic"}
        self.inp_path = inp_path
        self.pipe_ids_in_order = pipe_ids_in_order
        self.success_radii_m = [float(x) for x in success_radii_m]
        self.accuracy_is = [int(i) for i in accuracy_is]
        self.residual_builder = build_residual_sequence_from_segment
        print(f"{now()} [metric_evaluator] building evaluator for metrics: {self.metric_groups}.")

        self.oracle: Optional[PipeDistanceOracle] = None
        if ({"atd", "success", "accuracy_i"} & self.metric_groups):
            if self.inp_path is None or self.pipe_ids_in_order is None:
                raise ValueError("Distance metrics requested but inp_path/pipe_ids_in_order not provided.")
            self.oracle = PipeDistanceOracle.build(self.inp_path, self.pipe_ids_in_order)
            if "accuracy_i" in self.metric_groups:
                self.oracle.ensure_pipe_matrix()

    @torch.no_grad()
    def evaluate(self, loader: Iterable[Dict[str, Any]]) -> Dict[str, float]:
        self.predictor.eval()
        self.detector.eval()

        total = 0
        correct1 = 0
        correctk = 0

        nl_total = 0
        nl_correct = 0

        leak_total = 0
        leak_correct1 = 0
        leak_correctk = 0

        # Binary detection counts
        tp = fp = fn = tn = 0
        leak_pred_as_noleak = 0
        pre_false_alarm = 0
        noleak_false_alarm = 0
        pre_total = 0
        noleak_only_total = 0

        # Bucket stats
        buckets = ["early", "late", "pre", "noleak"]
        b_total = {b: 0 for b in buckets}
        b_correct1 = {b: 0 for b in buckets}
        b_correctk = {b: 0 for b in buckets}
        b_pred_as_noleak = {b: 0 for b in buckets}

        # Distance metrics
        atd_vals: List[float] = []
        success_hits = {r: 0 for r in self.success_radii_m}
        acc_i_hits = {i: 0 for i in self.accuracy_is}

        # AR (Average Rank) for leak localization (rank of true pipe among pipe logits, 1=best)
        ar_ranks: List[int] = []

        for batch in loader:
            noisy_seg = batch["noisy_seg"].to(self.device)
            time_seg = batch["time_seg"].to(self.device)
            label = torch.as_tensor(batch["label"], device=self.device, dtype=torch.long)

            bucket_list = batch.get("bucket", None)
            if bucket_list is None:
                bucket_list = ["unknown"] * label.numel()

            num_classes = batch.get("num_classes", None)
            if isinstance(num_classes, (list, tuple)):
                num_classes = int(num_classes[0])
            elif torch.is_tensor(num_classes):
                num_classes = int(num_classes[0].item())
            else:
                num_classes = None

            residual = self.residual_builder(self.predictor, noisy_seg, time_seg, l_pred=self.l_pred, l_det=self.l_det, device=self.device)
            tfeat = time_seg[:, self.l_pred:, :]
            logits = self.detector(residual, tfeat)

            if num_classes is None:
                num_classes = int(logits.size(-1))
            no_leak_class = num_classes - 1

            pred1 = logits.argmax(dim=-1)
            total += label.numel()
            correct1 += int((pred1 == label).sum().item())

            k = min(self.topk, logits.size(-1))
            topk_idx = logits.topk(k=k, dim=-1).indices
            correctk += int((topk_idx == label.unsqueeze(1)).any(dim=1).sum().item())

            # no-leak acc
            mask_nl = (label == no_leak_class)
            if mask_nl.any():
                nl_total += int(mask_nl.sum().item())
                nl_correct += int((pred1[mask_nl] == label[mask_nl]).sum().item())

            # leak-only
            mask_leak = ~mask_nl
            if mask_leak.any():
                leak_total += int(mask_leak.sum().item())
                leak_correct1 += int((pred1[mask_leak] == label[mask_leak]).sum().item())
                leak_correctk += int((topk_idx[mask_leak] == label[mask_leak].unsqueeze(1)).any(dim=1).sum().item())
            
            # AR: rank of the true pipe among pipe logits (exclude no-leak class)
            C = no_leak_class
            pipe_logits = logits[:, :C]
            order = torch.argsort(pipe_logits, dim=1, descending=True)
            inv = torch.empty_like(order)
            inv.scatter_(1, order, torch.arange(C, device=pipe_logits.device).unsqueeze(0).expand(order.size(0), -1))
            if mask_leak.any():
                ranks_1based = inv.gather(1, label.clamp_max(C-1).unsqueeze(1)).squeeze(1) + 1
                ar_ranks.extend(ranks_1based[mask_leak].detach().cpu().to(torch.int32).tolist())

            # binary detection
            pred_is_leak = (pred1 != no_leak_class)
            true_is_leak = (label != no_leak_class)
            tp += int((pred_is_leak & true_is_leak).sum().item())
            fp += int((pred_is_leak & ~true_is_leak).sum().item())
            fn += int((~pred_is_leak & true_is_leak).sum().item())
            tn += int((~pred_is_leak & ~true_is_leak).sum().item())

            if true_is_leak.any():
                leak_pred_as_noleak += int((~pred_is_leak & true_is_leak).sum().item())

            # buckets
            if "bucket" in self.metric_groups:
                for i in range(label.numel()):
                    b = str(bucket_list[i])
                    if b in b_total:
                        b_total[b] += 1
                        b_correct1[b] += int(pred1[i].item() == label[i].item())
                        b_correctk[b] += int((topk_idx[i] == label[i]).any().item())
                        b_pred_as_noleak[b] += int(pred1[i].item() == no_leak_class)

                for i in range(label.numel()):
                    b = str(bucket_list[i])
                    if label[i].item() == no_leak_class and pred1[i].item() != no_leak_class:
                        if b == "pre":
                            pre_false_alarm += 1
                        elif b == "noleak":
                            noleak_false_alarm += 1
                pre_total += int(sum(1 for b in bucket_list if str(b) == "pre"))
                noleak_only_total += int(sum(1 for b in bucket_list if str(b) == "noleak"))

            # distance metrics (leak samples only)
            if self.oracle is not None and ({"atd", "success", "accuracy_i"} & self.metric_groups):
                for i in range(label.numel()):
                    y = int(label[i].item())
                    if y == no_leak_class:
                        continue
                    p_pred = int(pred1[i].item())
                    if p_pred == no_leak_class:
                        d = float("inf")
                    else:
                        d = float(self.oracle.pipe_distance(y, p_pred))

                    if "atd" in self.metric_groups:
                        atd_vals.append(d)

                    if "success" in self.metric_groups:
                        for r in self.success_radii_m:
                            if d <= r:
                                success_hits[r] += 1

                    if "accuracy_i" in self.metric_groups and p_pred != no_leak_class:
                        ranks = self.oracle.pipe_rank[p_pred]
                        for ii in self.accuracy_is:
                            if ii > 0 and y in ranks[:ii]:
                                acc_i_hits[ii] += 1

        def safe_div(a: float, b: float) -> float:
            return float(a / b) if b > 0 else 0.0

        out: Dict[str, float] = {}

        if "basic" in self.metric_groups:
            out.update({
                "acc_top1": safe_div(correct1, total),
                f"acc_top{self.topk}": safe_div(correctk, total),
                "noleak_acc": safe_div(nl_correct, nl_total),
                "leak_acc_top1": safe_div(leak_correct1, leak_total),
                f"leak_acc_top{self.topk}": safe_div(leak_correctk, leak_total),
                "ar_mean": float(np.mean(ar_ranks)) if ar_ranks else float('inf'),
                "ar_median": float(np.median(ar_ranks)) if ar_ranks else float('inf'),
                "ar_n": float(len(ar_ranks)),
                "n_total": float(total),
                "n_leak": float(leak_total),
                "n_noleak": float(nl_total),
            })

        if "binary" in self.metric_groups:
            prec = safe_div(tp, tp + fp)
            rec = safe_div(tp, tp + fn)
            f1 = safe_div(2 * prec * rec, prec + rec) if (prec + rec) > 0 else 0.0
            out.update({
                "det_precision": float(prec),
                "det_recall": float(rec),
                "det_f1": float(f1),
                "det_tp": float(tp),
                "det_fp": float(fp),
                "det_fn": float(fn),
                "det_tn": float(tn),
                "leak_pred_as_noleak_rate": safe_div(leak_pred_as_noleak, leak_total),
            })
            if pre_total > 0:
                out["pre_false_alarm_rate"] = safe_div(pre_false_alarm, pre_total)
            if noleak_only_total > 0:
                out["noleak_false_alarm_rate"] = safe_div(noleak_false_alarm, noleak_only_total)

        if "bucket" in self.metric_groups:
            for b in buckets:
                out[f"{b}_n"] = float(b_total[b])
                out[f"{b}_acc_top1"] = safe_div(b_correct1[b], b_total[b])
                out[f"{b}_acc_top{self.topk}"] = safe_div(b_correctk[b], b_total[b])
                out[f"{b}_pred_as_noleak_rate"] = safe_div(b_pred_as_noleak[b], b_total[b])

        if "atd" in self.metric_groups:
            finite = [x for x in atd_vals if math.isfinite(x)]
            out["atd_mean_m"] = float(np.mean(finite)) if finite else float("inf")
            out["atd_median_m"] = float(np.median(finite)) if finite else float("inf")
            out["atd_n"] = float(len(atd_vals))
            out["atd_missed_rate"] = safe_div(sum(1 for x in atd_vals if not math.isfinite(x)), len(atd_vals))

        if "success" in self.metric_groups:
            leak_detected = leak_total - leak_pred_as_noleak
            for r in self.success_radii_m:
                out[f"success_at_{int(r)}"] = safe_div(success_hits[r], leak_detected)
                out[f"success_at_{int(r)}_e2e"] = safe_div(success_hits[r], leak_total)

        if "accuracy_i" in self.metric_groups:
            for ii in self.accuracy_is:
                out[f"accuracy_{int(ii)}"] = safe_div(acc_i_hits[ii], leak_total)

        return out

    @torch.no_grad()
    def visualize_random_sample(
        self,
        loader: Iterable[Dict[str, Any]],
        *,
        seed: Optional[int] = None,
        save_path: Optional[str | Path] = None,
        viz_num: int = 1,
        max_batches: int = 200,
        show_sensors: bool = True,
        figsize: Tuple[int, int] = (9, 7),
        atd_rank_range: Tuple[float, float] = (0.0, 1.0),
        include_pred_noleak_in_rank: bool = False,
        # Drawing style
        background_color: str = "0.75",
        background_alpha: float = 0.5,
        background_lw: float = 0.8,
        pred_color: str = "#1f77b4",
        true_color: str = "#d62728",
        highlight_lw: float = 4.0,
    ):
        """Randomly select one *leak* sample from a loader and visualize the WDN topology, highlighting the true and predicted pipes."""
        
        if self.inp_path is None or self.pipe_ids_in_order is None:
            raise ValueError("visualize_random_sample requires inp_path and pipe_ids_in_order (pass them when constructing DetectorEvaluator).")
        
        rng = random.Random(seed)

        # Ensure we have a distance oracle for ATD-based ranking / distance display.
        oracle = self.oracle
        if oracle is None:
            oracle = PipeDistanceOracle.build(self.inp_path, self.pipe_ids_in_order)

        self.predictor.eval()
        self.detector.eval()

        # Collect candidates: (dist, y_idx, p_idx, bucket, scenario_id, no_leak_class)
        candidates: List[Tuple[float, int, int, str, Optional[str], int]] = []

        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break

            noisy_seg = batch["noisy_seg"].to(self.device)
            time_seg = batch["time_seg"].to(self.device)
            label = torch.as_tensor(batch["label"], device=self.device, dtype=torch.long)

            bucket_list = batch.get("bucket", None)
            scenario_id_list = batch.get("scenario_id", None)

            # infer no-leak class for this loader
            num_classes = batch.get("num_classes", None)
            if isinstance(num_classes, (list, tuple)):
                num_classes = int(num_classes[0])
            elif torch.is_tensor(num_classes):
                num_classes = int(num_classes[0].item())
            else:
                num_classes = None

            residual = self.residual_builder(self.predictor, noisy_seg, time_seg, self.l_pred, self.l_det, self.device)
            tfeat = time_seg[:, self.l_pred:, :]
            logits = self.detector(residual, tfeat)

            if num_classes is None:
                num_classes = int(logits.size(-1))
            no_leak_class = num_classes - 1

            pred1 = logits.argmax(dim=-1)

            B = label.numel()
            for i in range(B):
                y = int(label[i].item())
                if y == no_leak_class:
                    continue  # rank only leak samples

                p = int(pred1[i].item())
                if p == no_leak_class:
                    if not include_pred_noleak_in_rank:
                        continue
                    dist = float("inf")
                else:
                    dist = float(oracle.pipe_distance(y, p))

                b = str(bucket_list[i]) if bucket_list is not None else "unknown"
                sid = str(scenario_id_list[i]) if scenario_id_list is not None else None
                candidates.append((dist, y, p, b, sid, no_leak_class))

        if not candidates:
            raise RuntimeError("No leak samples found in the scanned subset (max_batches too small?).")

        # Rank by distance (inf = worst)
        order = sorted(range(len(candidates)), key=lambda idx: candidates[idx][0])
        ranked = [candidates[idx] for idx in order]
        N = len(ranked)
        
        lo, hi = atd_rank_range
        is_fractional = (0.0 <= float(lo) <= 1.0) and (0.0 <= float(hi) <= 1.0)
        if is_fractional:
            lo_i = int(math.floor(float(lo) * N))
            hi_i = int(math.ceil(float(hi) * N)) - 1
        else:
            lo_i = int(lo)
            hi_i = int(hi)

        lo_i = max(0, min(lo_i, N - 1))
        hi_i = max(0, min(hi_i, N - 1))
        if hi_i < lo_i:
            lo_i, hi_i = hi_i, lo_i

        # Build topology for drawing from inp [PIPES] only
        pipes_all = _parse_pipes_with_length(self.inp_path)
        coords = _parse_coordinates(self.inp_path)

        node_set = set()
        for _, (n1, n2, _) in pipes_all.items():
            node_set.add(n1)
            node_set.add(n2)

        pos: Dict[str, Tuple[float, float]] = {n: coords[n] for n in node_set if n in coords}

        if len(pos) < max(3, int(0.6 * len(node_set))):
            try:
                import networkx as nx
                G = nx.Graph()
                for pid, (n1, n2, _) in pipes_all.items():
                    G.add_edge(n1, n2, key=pid)
                init = {n: pos[n] for n in pos}
                layout = nx.spring_layout(G, seed=seed, pos=init if init else None)
                pos = {n: (float(layout[n][0]), float(layout[n][1])) for n in G.nodes()}
            except Exception:
                nodes = sorted(list(node_set))
                for i, n in enumerate(nodes):
                    ang = 2 * math.pi * i / max(1, len(nodes))
                    pos[n] = (math.cos(ang), math.sin(ang))

        # draw pictures
        for fig_idx in range(viz_num):
            fig = plt.figure(figsize=figsize)
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="datalim")
            
            # draw all pipes in gray (clearly de-emphasized)
            for pid, (n1, n2, _) in pipes_all.items():
                x1, y1 = pos.get(n1, (0.0, 0.0))
                x2, y2 = pos.get(n2, (0.0, 0.0))
                ax.plot(
                    [x1, x2], [y1, y2],
                    linewidth=background_lw,
                    alpha=background_alpha,
                    color=background_color,
                    zorder=1,
                )

            picked_rank = rng.randint(lo_i, hi_i)
            dist_m, y, p, bucket, scenario_id, no_leak_class = ranked[picked_rank]
            true_pipe_id = self.pipe_ids_in_order[y] if y != no_leak_class else "NOLEAK"
            pred_pipe_id = self.pipe_ids_in_order[p] if p != no_leak_class else "NOLEAK"

            def draw_pipe(pid: str, *, lw: float, alpha: float, color: str, z: int, label: Optional[str] = None):
                if pid not in pipes_all:
                    return
                n1, n2, _ = pipes_all[pid]
                x1, y1 = pos.get(n1, (0.0, 0.0))
                x2, y2 = pos.get(n2, (0.0, 0.0))
                ax.plot(
                    [x1, x2], [y1, y2],
                    linewidth=lw,
                    alpha=alpha,
                    color=color,
                    zorder=z,
                    label=label,
                )

            # highlight pred and true with thick colored lines
            if pred_pipe_id != "NOLEAK":
                draw_pipe(pred_pipe_id, lw=highlight_lw, alpha=0.95, color=pred_color, z=3, label="pred")
            if true_pipe_id != "NOLEAK":
                draw_pipe(true_pipe_id, lw=highlight_lw, alpha=0.98, color=true_color, z=4, label="true")

            # optionally mark sensors
            if show_sensors and hasattr(self.detector, "sensor_node_ids"):
                sns = getattr(self.detector, "sensor_node_ids")
                xs, ys = [], []
                for n in sns:
                    if n in pos:
                        xs.append(pos[n][0])
                        ys.append(pos[n][1])
                if xs:
                    ax.scatter(xs, ys, s=22, alpha=0.9, color="black", zorder=5, label="sensors")

            # Legend (avoid duplicates)
            handles, labels = ax.get_legend_handles_labels()
            uniq = {}
            for h, l in zip(handles, labels):
                if l and l not in uniq:
                    uniq[l] = h
            if uniq:
                ax.legend(list(uniq.values()), list(uniq.keys()), loc="upper right", frameon=True)

            title = f"WDN leak localization | bucket={bucket} | true={true_pipe_id} pred={pred_pipe_id}"
            if math.isfinite(dist_m):
                title += f" | dist={dist_m:.1f} m"
            else:
                title += " | dist=inf (pred NOLEAK)"
            title += f" | rank={picked_rank}/{N-1}"
            ax.set_title(title)
            ax.axis("off")

            if save_path is not None:
                save_path = Path(save_path)
                fig_path = save_path.with_stem(f"{save_path.stem}_{fig_idx}")
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path, dpi=240, bbox_inches="tight")