"""
Event-level metrics for WDN leak detection & localization.

This module is intentionally standalone: it does NOT depend on window-level evaluators
(e.g., evaluate_detector.py). It consumes per-scenario results and produces dataset-level
metrics, all computed with the SCENARIO as the statistical unit.

Default localization protocol:
- Localization@Detected: localization metrics are computed only on scenarios that are
  true-leak AND predicted-leak (i.e., successfully detected).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class EventResult:
    scenario_id: str
    is_leak_true: bool
    is_leak_pred: bool
    true_pipe_id: Optional[str] = None
    pred_pipe_id: Optional[str] = None
    # Optional timestamps (ISO strings) kept as strings so this module stays pandas-free.
    tau_iso: Optional[str] = None
    alarm_time_iso: Optional[str] = None
    # Localization distance in meters (only meaningful when is_leak_true & is_leak_pred)
    atd_m: Optional[float] = None


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def compute_event_metrics(
    events: Iterable[EventResult],
    *,
    success_radii_m: Sequence[float] = (50.0, 100.0, 300.0),
    localization_mode: str = "detected",  # "detected" or "all_leaks"
) -> Dict[str, float]:
    """
    Compute event-level (scenario-level) metrics.

    Parameters
    ----------
    events:
        Iterable of per-scenario results.
    success_radii_m:
        Radii thresholds for Success@X (meters).
    localization_mode:
        - "detected": localization metrics computed on true-leak AND predicted-leak only.
        - "all_leaks": localization metrics computed on all true-leak; missed detections are failures
          (Success@X counts as 0; ATD is excluded from mean/median but tracked via miss rate).

    Returns
    -------
    dict of metrics (floats).
    """
    if localization_mode not in ("detected", "all_leaks"):
        raise ValueError("localization_mode must be 'detected' or 'all_leaks'.")

    events = list(events)

    # --- Detection counts (scenario as unit) ---
    tp = fp = fn = tn = 0
    n_leak = 0
    n_noleak = 0

    for e in events:
        if e.is_leak_true:
            n_leak += 1
            if e.is_leak_pred:
                tp += 1
            else:
                fn += 1
        else:
            n_noleak += 1
            if e.is_leak_pred:
                fp += 1
            else:
                tn += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0

    out: Dict[str, float] = {
        "n_events": float(len(events)),
        "n_leak_events": float(n_leak),
        "n_noleak_events": float(n_noleak),
        "det_tp": float(tp),
        "det_fp": float(fp),
        "det_fn": float(fn),
        "det_tn": float(tn),
        "det_precision": float(precision),
        "det_recall": float(recall),
        "det_f1": float(f1),
        "det_fp_rate": _safe_div(fp, n_noleak),
        "det_fn_rate": _safe_div(fn, n_leak),
    }

    # --- Localization metrics (scenario as unit) ---
    # Choose evaluation set
    loc_events: List[EventResult] = []
    if localization_mode == "detected":
        loc_events = [e for e in events if (e.is_leak_true and e.is_leak_pred)]
        miss_count = 0
    else:
        # all true leaks included; missed detections are counted as "misses" for Success@X
        loc_events = [e for e in events if e.is_leak_true]
        miss_count = sum(1 for e in loc_events if not e.is_leak_pred)

    # Exact match accuracy among localization-evaluated events with a predicted pipe
    exact_ok = 0
    exact_total = 0

    atd_vals: List[float] = []
    # success counters
    succ_hits = {float(r): 0 for r in success_radii_m}
    succ_total = len(loc_events) if localization_mode == "all_leaks" else sum(1 for e in loc_events if e.is_leak_pred)

    for e in loc_events:
        if localization_mode == "all_leaks" and (not e.is_leak_pred):
            # missed detection => fail Success@X; no ATD; no exact
            continue

        # e is leak_true and leak_pred
        if e.true_pipe_id is not None and e.pred_pipe_id is not None:
            exact_total += 1
            if e.true_pipe_id == e.pred_pipe_id:
                exact_ok += 1

        if e.atd_m is not None:
            atd_vals.append(float(e.atd_m))
            for r in success_radii_m:
                if e.atd_m <= float(r):
                    succ_hits[float(r)] += 1

    out["loc_mode"] = 1.0 if localization_mode == "detected" else 2.0  # numeric tag for logging
    out["loc_n_events"] = float(len(loc_events))
    out["loc_n_detected"] = float(sum(1 for e in loc_events if e.is_leak_pred))
    out["loc_miss_rate"] = _safe_div(miss_count, len(loc_events)) if len(loc_events) > 0 else 0.0

    out["loc_accuracy_exact"] = _safe_div(exact_ok, exact_total)

    if atd_vals:
        out["loc_atd_mean_m"] = float(np.mean(atd_vals))
        out["loc_atd_median_m"] = float(np.median(atd_vals))
    else:
        out["loc_atd_mean_m"] = float("inf")
        out["loc_atd_median_m"] = float("inf")

    for r in success_radii_m:
        out[f"loc_success_at_{int(r)}m"] = _safe_div(succ_hits[float(r)], succ_total)

    return out
