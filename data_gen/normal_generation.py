"""
normal_generation.py

用途：
- 基于 normal_simulator.py（NormalSimExecutor）批量生成“无泄漏”正常窗口数据，
  用于训练状态估计/正常预测器（推荐：sensor -> sensor）。
- 输出：manifest.jsonl + 每个窗口一个目录。

输出目录结构：
<output_root>/
  manifest.jsonl
  <window_id>/
    sensors.csv          # 带噪传感器压力（输入）
    sensors_gt.csv       # 无噪传感器压力（标签，从 nodal_pressure.csv 子集得到）
    meta.json
"""

from __future__ import annotations
import argparse
import copy
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import wntr

from config_io import SimConfig
from normal_simulator import NormalSimExecutor


# -------------------------- utils --------------------------

def _format_float_4(x):
    if pd.isna(x):
        return ""
    s = "{:.4f}".format(float(x))
    return s.rstrip("0").rstrip(".")


def parse_duration(s: str) -> pd.Timedelta:
    s = str(s).strip().lower()
    if s.endswith("h"):
        return pd.Timedelta(hours=float(s[:-1]))
    if s.endswith("d"):
        return pd.Timedelta(days=float(s[:-1]))
    if s.endswith("m") and s[:-1].replace(".", "", 1).isdigit():
        return pd.Timedelta(minutes=float(s[:-1]))
    if s.endswith("min"):
        return pd.Timedelta(minutes=float(s[:-3]))
    return pd.Timedelta(s)


def ceil_days(td: pd.Timedelta) -> int:
    return int(math.ceil(td.total_seconds() / 86400.0))


def sanitize_id(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def random_window_start(anchor: pd.Timestamp, jitter_days: int, step_min: int, rng: np.random.Generator) -> pd.Timestamp:
    d = int(rng.integers(0, max(1, jitter_days + 1)))
    steps_per_day = int(24 * 60 // step_min)
    k = int(rng.integers(0, max(1, steps_per_day)))
    return pd.Timestamp(anchor) + pd.Timedelta(days=d, minutes=k * step_min)


def build_sensors_gt_from_nodal_pressure(nodal_pressure_csv: str, sensors_csv: str, out_csv: str) -> None:
    sens = pd.read_csv(sensors_csv, index_col=0, parse_dates=True)
    nodal = pd.read_csv(nodal_pressure_csv, index_col=0, parse_dates=True)

    sensor_cols = list(sens.columns)
    # 取交集，保持 sensors.csv 列顺序
    kept = [c for c in sensor_cols if c in nodal.columns]
    if not kept:
        raise RuntimeError("No sensor columns found in nodal_pressure.csv; check sensors.csv columns and nodal_pressure.csv header.")

    gt = nodal[kept].copy()
    # 对齐 index（以 sensors.csv 为准）
    if len(gt) == len(sens):
        gt.index = sens.index

    gt.map(_format_float_4).to_csv(out_csv, index_label="datetime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="sim_LTA.yaml", help="SimConfig YAML path")
    ap.add_argument("--output_root", type=str, required=True, help="输出根目录（会在其下创建 normal_windows/ 与 manifest.jsonl）")
    ap.add_argument("--seed", type=int, default=42, help="全局随机种子")

    ap.add_argument("--n_windows", type=int, default=1000, help="生成多少个正常窗口")
    ap.add_argument("--duration_list", type=str, nargs="+", default=["7d", "14d", "21d"], help="窗口长度候选列表（将随机抽取）")
    ap.add_argument("--jitter_days", type=int, default=365 * 2, help="从cfg仿真起点日期随机偏移的最大天数")

    ap.add_argument("--warmup_days", type=int, default=None, help="覆盖 cfg.simulation.warmup_days")
    ap.add_argument("--fix_keep_nodal_pressure", action="store_true", default=False, help="保留 nodal_pressure.csv（默认删除）")
    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--continue_on_error", action="store_true", default=True)
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    base_cfg = SimConfig.from_yaml(args.config)
    out_root = Path(args.output_root)
    windows_root = out_root
    windows_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"

    durations = [parse_duration(x) for x in args.duration_list]
    if not durations:
        raise ValueError("duration_list is empty")
    anchor = pd.Timestamp(base_cfg.simulation.start_datetime)
    step_min = int(base_cfg.simulation.timestep_min)

    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    print(f"[INFO] output_root={out_root}")
    print(f"[INFO] n_windows={args.n_windows}, duration_list={args.duration_list}, step_min={step_min}")
    print(f"[INFO] anchor={anchor}, jitter_days={args.jitter_days}, seed={args.seed}")
    print(f"[INFO] keep_nodal_pressure={bool(args.fix_keep_nodal_pressure)}, overwrite={args.overwrite}")

    ok = failed = skipped = 0
    t0 = time.time()
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for i in range(1, int(args.n_windows) + 1):
            # 每个窗口独立 seed：可复现
            window_seed = int(args.seed) * 1_000_000 + i
            np.random.seed(window_seed)
            random.seed(window_seed)

            start_dt = random_window_start(anchor, int(args.jitter_days), step_min, rng)
            dur_td = rng.choice(durations)
            dur_days = max(1, ceil_days(dur_td))

            window_id = sanitize_id(f"{i:06d}_normal_d{dur_days}")
            window_dir = windows_root / window_id
            if window_dir.exists() and not args.overwrite:
                skipped += 1
                mf.write(json.dumps({"window_id": window_id, "status": "skipped_exists", "window_dir": str(window_dir)}, ensure_ascii=False) + "\n")
                mf.flush()
                continue
            window_dir.mkdir(parents=True, exist_ok=True)
            
            # 为这个窗口准备 cfg（深拷贝避免污染）
            cfg = base_cfg.model_copy(deep=True)
            cfg.paths.output_dir = str(window_dir)
            cfg.simulation.start_datetime = pd.Timestamp(start_dt)
            cfg.simulation.duration_days = int(dur_days)
            cfg.meta.random_seed = int(window_seed)
            if args.warmup_days is not None:
                try:
                    cfg.simulation.warmup_days = int(args.warmup_days)
                except Exception:
                    pass

            try:
                # 跑仿真：会在 window_dir 下写 sensors.csv / nodal_pressure.csv / meta.json
                ex = NormalSimExecutor(cfg)
                ex.run_once()
                sensors_csv = str(window_dir / "sensors.csv")
                nodal_csv = str(window_dir / "nodal_pressure.csv")
                meta_json = str(window_dir / "meta.json")
                sensors_gt_csv = str(window_dir / "sensors_gt.csv")

                # 生成 sensors_gt.csv（从 nodal_pressure 取传感器列）
                build_sensors_gt_from_nodal_pressure(nodal_csv, sensors_csv, sensors_gt_csv)

                # 默认删除 nodal_pressure.csv，避免占空间
                if not args.fix_keep_nodal_pressure:
                    try:
                        Path(nodal_csv).unlink()
                    except Exception:
                        pass

                    # 同时把 meta.json 里关于 nodal_pressure 的路径信息更新一下（可选）
                    try:
                        meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
                        meta.setdefault("paths", {})
                        meta["paths"]["sensors_gt_path"] = sensors_gt_csv
                        meta["paths"]["nodal_pressure_path"] = None
                        Path(meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass
                else:
                    try:
                        meta = json.loads(Path(meta_json).read_text(encoding="utf-8"))
                        meta.setdefault("paths", {})
                        meta["paths"]["sensors_gt_path"] = sensors_gt_csv
                        Path(meta_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                mf.write(json.dumps({
                    "window_id": window_id,
                    "status": "ok",
                    "window_dir": str(window_dir),
                    "sensors_csv": sensors_csv,
                    "sensors_gt_csv": sensors_gt_csv,
                    "meta_json": meta_json,
                    "start_datetime": pd.Timestamp(start_dt).isoformat(),
                    "duration_days": int(dur_days),
                    "window_seed": int(window_seed),
                }, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1

            except Exception as e:
                failed += 1
                mf.write(json.dumps({
                    "window_id": window_id,
                    "status": "failed",
                    "error": repr(e),
                    "window_dir": str(window_dir),
                    "start_datetime": pd.Timestamp(start_dt).isoformat(),
                    "duration_days": int(dur_days),
                    "window_seed": int(window_seed),
                }, ensure_ascii=False) + "\n")
                mf.flush()
                print(f"[ERROR] {window_id} failed: {e}")
                if not args.continue_on_error:
                    raise

            # 进度
            if i == 1 or i % 20 == 0 or i == args.n_windows:
                elapsed = time.time() - t0
                rate = elapsed / max(1, i)
                eta_min = rate * (args.n_windows - i) / 60.0
                print(f"[PROGRESS] {i}/{args.n_windows}  ok={ok}  skipped={skipped}  failed={failed} | avg={rate:.2f}s/win  ETA={eta_min:.1f}min")

    print("\nDone.")
    print(f"Output root: {out_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Windows attempted: {args.n_windows}  ok={ok}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
