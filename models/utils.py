"""
Utilities for WDN leak detection project.
"""
from __future__ import annotations
import re
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch

_SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def parse_epanet_inp(inp_path: str | Path) -> Dict[str, List[str]]:
    """
    Parse an EPANET .inp file into sections -> lines.

    - Strips comments after ';'
    - Keeps token lines (not split), one per list item.
    """
    inp_path = Path(inp_path)
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None

    with inp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            m = _SECTION_RE.match(line)
            if m:
                current = m.group(1).strip().upper()
                sections.setdefault(current, [])
                continue

            if current is None:
                continue

            if ";" in line:
                line = line.split(";", 1)[0].strip()
            if not line:
                continue

            sections[current].append(line)

    return sections


def _parse_nodes(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        toks = line.split()
        if toks:
            out.append(toks[0])
    return out


def _parse_links(lines: List[str]) -> Dict[str, Tuple[str, str]]:
    out: Dict[str, Tuple[str, str]] = {}
    for line in lines:
        toks = line.split()
        if len(toks) >= 3:
            out[toks[0]] = (toks[1], toks[2])
    return out


@dataclass(frozen=True)
class WDNGraph:
    node_names: List[str]
    node_to_idx: Dict[str, int]
    # Classification pipes (trainable/detectable)
    pipe_ids: List[str]
    pipe_to_idx: Dict[str, int]
    pipe_ends: np.ndarray  # (P, 2) int64
    # Full topology for message passing
    edge_index: Any        # torch.LongTensor (2, E)


def build_wdn_graph_from_inp(
    inp_path: str | Path,
    sensor_node_ids: Sequence[str],
    pipe_ids_in_order: Sequence[str],
    *,
    include_all_nodes: bool = True,
    include_links: Sequence[str] = ("PIPES", "PUMPS", "VALVES"),
    add_self_loops: bool = True,
    make_undirected: bool = True,
) -> WDNGraph:
    """
    Build a WDN graph from an EPANET .inp file.
    - Node set:
        sensors + ([JUNCTIONS]/[RESERVOIRS]/[TANKS] if include_all_nodes=True) + any link endpoints
    - edge_index:
        built from include_links (default: PIPES+PUMPS+VALVES), bidirectional if make_undirected=True
    - pipe_ends:
        built ONLY for pipe_ids_in_order
    """
    sections = parse_epanet_inp(inp_path)

    # ---- full topology links for message passing ----
    link_endpoints: Dict[str, Tuple[str, str]] = {}
    for sec in include_links:
        link_endpoints.update(_parse_links(sections.get(sec.upper(), [])))
    if not link_endpoints:
        raise ValueError(f"No link endpoints found from sections {tuple(include_links)} in inp file.")

    # ---- node set ----
    node_set = set(sensor_node_ids)
    if include_all_nodes:
        node_set.update(_parse_nodes(sections.get("JUNCTIONS", [])))
        node_set.update(_parse_nodes(sections.get("RESERVOIRS", [])))
        node_set.update(_parse_nodes(sections.get("TANKS", [])))
    
    # Always include endpoints from topology links
    for n1, n2 in link_endpoints.values():
        node_set.add(n1)
        node_set.add(n2)

    node_names = sorted(node_set)
    node_to_idx = {n: i for i, n in enumerate(node_names)}

    # ---- pipe endpoints for classification labels ----
    pipe_endpoints = _parse_links(sections.get("PIPES", []))
    if not pipe_endpoints:
        raise ValueError("No [PIPES] section found or empty; cannot map pipe_ids to endpoints.")

    pipe_ids = list(pipe_ids_in_order)
    pipe_to_idx = {pid: i for i, pid in enumerate(pipe_ids)}

    pipe_ends = np.zeros((len(pipe_ids), 2), dtype=np.int64)
    for i, pid in enumerate(pipe_ids):
        if pid not in pipe_endpoints:
            raise ValueError(f"Pipe id {pid} not found in inp [PIPES].")
        n1, n2 = pipe_endpoints[pid]
        pipe_ends[i, 0] = node_to_idx[n1]
        pipe_ends[i, 1] = node_to_idx[n2]

    # ---- edge_index (full topology) ----
    src: List[int] = []
    dst: List[int] = []
    for (n1, n2) in link_endpoints.values():
        u = node_to_idx[n1]
        v = node_to_idx[n2]
        src.append(u); dst.append(v)
        if make_undirected:
            src.append(v); dst.append(u)

    if add_self_loops:
        for i in range(len(node_names)):
            src.append(i); dst.append(i)

    edge_index = torch.tensor([src, dst], dtype=torch.long)

    return WDNGraph(
        node_names=node_names,
        node_to_idx=node_to_idx,
        pipe_ids=pipe_ids,
        pipe_to_idx=pipe_to_idx,
        pipe_ends=pipe_ends,
        edge_index=edge_index,
    )


def build_residual_sequence_from_segment(
    predictor: Any,
    noisy_seg: Any,
    time_seg: Any,
    l_pred: int,
    l_det: int,
    device: Optional[Any] = None,
) -> Any:
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
        y_hat = y_hat[:, -1, :]

    residual = Y_noisy - y_hat
    residual = residual.view(l_det, B, S).transpose(0, 1).contiguous()

    if squeeze_back:
        residual = residual.squeeze(0)
    return residual

