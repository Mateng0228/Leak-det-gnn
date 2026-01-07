"""
批量生成单泄漏的短时间轴泄漏场景。
 - 每个场景的前段时间无泄漏，后段时间有泄漏。
 - 特别的，我们把泄漏分为 abrupt 型与 incipient 型：abrupt 型泄漏流量瞬间达到稳态最大值，incipient 型泄漏流量则经历一段时间爬升后达到稳态值
 - 目前模型仅处理单管段的 abrupt 型泄漏。

默认输出目录结构：
<output_root>/
  manifest.jsonl
  <scenario_id>/
    sensors.csv # 传感器准确仿真值
    sensors_gt.csv # 传感器加噪仿真值（加噪是为了模仿真实采集情形）
    leak_flow_m3h.csv # 泄漏管段的流量
    meta.json # 配置元数据
"""

from __future__ import annotations
import argparse
import time
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import wntr

from config_io import SimConfig
from leak_simulator import LeakSimExecutor, LeakScenario, LeakRecord


# -------------------------- utils--------------------------

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

SID_PREFIX_RE = re.compile(r"^(?P<sid>\d{6})_")

def find_next_sid(out_root: Path, manifest_path: Optional[Path] = None) -> int:
    """
    Find the maximum 6-digit scenario id prefix in an existing dataset folder,
    It scans both existing scenario directories and (optionally) manifest.jsonl.
    """
    max_sid = 0
    if out_root.exists():
        for child in out_root.iterdir():
            if child.is_dir():
                m = SID_PREFIX_RE.match(child.name)
                if m:
                    max_sid = max(max_sid, int(m.group("sid")))
    if manifest_path is not None and manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    m = re.search(r'"scenario_id"\s*:\s*"(\d{6})_', line)
                    if m:
                        max_sid = max(max_sid, int(m.group(1)))
        except Exception:
            pass
    return int(max_sid)


def load_all_pipe_ids(inp_path: str) -> List[str]:
    wn = wntr.network.WaterNetworkModel(inp_path)
    return list(wn.pipe_name_list)


def pick_pipes(pipe_ids: List[str], ratio: float, max_pipes: Optional[int], rng: np.random.Generator) -> List[str]:
    ratio = float(ratio)
    ratio = min(max(ratio, 0.0), 1.0)
    if ratio <= 0 or not pipe_ids:
        return []
    n = max(1, int(round(len(pipe_ids) * ratio)))
    if max_pipes is not None:
        n = min(n, int(max_pipes))
    idx = rng.choice(len(pipe_ids), size=n, replace=False)
    return [pipe_ids[i] for i in idx]


def random_scenario_start(anchor: pd.Timestamp, jitter_days: int, step_min: int, rng: np.random.Generator) -> pd.Timestamp:
    d = int(rng.integers(0, max(1, jitter_days + 1)))
    steps_per_day = int(24 * 60 // step_min)
    k = int(rng.integers(0, max(1, steps_per_day)))
    return pd.Timestamp(anchor) + pd.Timedelta(days=d, minutes=k * step_min)


def parse_tiers(tiers_str: List[str]) -> List[Tuple[float, float]]:
    tiers: List[Tuple[float, float]] = []
    for t in tiers_str:
        lo, hi = t.split(",")
        tiers.append((float(lo), float(hi)))
    return tiers


def sample_diameter_from_random_tier(tiers: List[Tuple[float, float]], rng: np.random.Generator) -> Tuple[int, float]:
    tier_idx = int(rng.integers(0, len(tiers)))
    lo, hi = tiers[tier_idx]
    d = float(rng.uniform(lo, hi))
    return tier_idx, d


# -------------------------- scenario builders --------------------------

def build_single_leak_scenario(
    *,
    scenario_id: str,
    scenario_start: pd.Timestamp,
    pre: pd.Timedelta,
    growth: pd.Timedelta,
    post: pd.Timedelta,
    pipe_id: str,
    leak_type: str,
    leak_diameter_m: float,
) -> LeakScenario:
    scenario_start = pd.Timestamp(scenario_start)

    if leak_type == "abrupt":
        leak_start = scenario_start + pre
        leak_peak = None
        leak_end = leak_start + post
        total = pre + post
    elif leak_type == "incipient":
        leak_start = scenario_start + pre
        leak_peak = leak_start + growth
        leak_end = leak_peak + post
        total = pre + growth + post
    else:
        raise ValueError(f"unknown leak_type={leak_type}")

    duration_days = ceil_days(total)
    # 确保 leak_end 被覆盖
    if leak_end > scenario_start + pd.Timedelta(days=duration_days):
        duration_days = ceil_days(leak_end - scenario_start)

    rec = LeakRecord(
        leak_id="1",
        pipe_id=pipe_id,
        leak_type=leak_type,
        start_datetime=leak_start,
        peak_datetime=leak_peak,
        end_datetime=leak_end,
        leak_diameter_m=float(leak_diameter_m),
        discharge_coeff=None,
    )
    return LeakScenario(
        scenario_id=scenario_id,
        start_datetime=scenario_start,
        duration_days=int(duration_days),
        leak_records=[rec],
    )


def build_no_leak_scenario(*, scenario_id: str, scenario_start: pd.Timestamp, duration: pd.Timedelta) -> LeakScenario:
    duration_days = max(1, ceil_days(duration))
    return LeakScenario(
        scenario_id=scenario_id,
        start_datetime=pd.Timestamp(scenario_start),
        duration_days=int(duration_days),
        leak_records=[],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="sim_LTA.yaml", help="仿真配置文件")
    ap.add_argument("--output_root", type=str, default=None, help="数据集输出目录，覆盖配置文件中的output_dir" )
    ap.add_argument("--seed", type=int, default=42, help="随机种子" )
    # 管段采样
    ap.add_argument("--pipe_sample_ratio", type=float, default=1.0, help="采样多少比例的管段参与泄漏，例如0.5代表随机为一半数目的管段分别生成泄漏场景")
    ap.add_argument("--max_pipes", type=int, default=None, help="最多选多少个管段（用于快速测试）" )
    # 管段生成多少个leak场景
    ap.add_argument("--abrupt_per_pipe", type=int, default=1, help="每个管段生成多少个 abrupt 泄漏场景")
    ap.add_argument("--incipient_per_pipe", type=int, default=1, help="每个管段生成多少个 incipient 泄漏场景")
    # no-leak 占总场景比例
    ap.add_argument("--no_leak_ratio", type=float, default=0.2, help="完全无泄漏场景占比（占总场景）" )
    # 泄漏相关的三段时长列表：growth / pre / post
    ap.add_argument(
        "--pre_list",
        type=str,
        nargs="+",
        default=["6h","12h","24h"],
        help="重要：用于指定泄漏前时长列表（对于我们目前要的abrupt型泄漏，总仿真时长 = pre + post）",
    )
    ap.add_argument(
        "--growth_list",
        type=str,
        nargs="+",
        default=["12h", "1d", "2d", "3d"],
        help="incipient growth 时长列表，仅在incipient型泄漏有意义",
    )
    ap.add_argument(
        "--post_list",
        type=str,
        nargs="+",
        default=["24h","36h","48h","60h"],
        help="重要：用于指定泄漏后时长列表（对于我们目前要的abrupt型泄漏，总仿真时长 = pre + post）",
    )

    # 泄漏强度三档（孔口直径 m）
    ap.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        default=["0.008,0.012", "0.012,0.017", "0.017,0.023"],
        help="三档泄漏直径范围（m），格式 'lo,hi'"
    )
    # incipient ramp 控制
    ap.add_argument("--ramp_update_minutes", type=int, default=15, help="incipient ramp 更新频率（分钟）")
    ap.add_argument("--ramp_shape", type=str, default="quadratic", choices=["linear", "quadratic"])
    # warmup
    ap.add_argument("--warmup_days", type=int, default=0, help="覆盖 cfg.simulation.warmup_days" )

    ap.add_argument("--overwrite", action="store_true", default=False)
    ap.add_argument("--continue_on_error", action="store_true", default=True)

    args = ap.parse_args()
    rng = np.random.default_rng(int(args.seed))

    # ---- 1) 读 config ----
    cfg = SimConfig.from_yaml(args.config)
    try:
        cfg.simulation.warmup_days = int(args.warmup_days)
    except Exception:
        print("[WARN] cfg.simulation.warmup_days is not settable; please set warmup_days=0 in yaml." )

    if args.output_root is not None:
        try:
            cfg.paths.output_dir = str(args.output_root)
        except Exception:
            print("[WARN] cfg.paths.output_dir is not settable; please set output_dir in yaml or pass a mutable SimConfig." )

    out_root = Path(cfg.paths.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"

    # ---- 2) pipe 列表并采样 ----
    pipe_ids_all = load_all_pipe_ids(cfg.paths.inp_path)
    pipe_ids = pick_pipes(pipe_ids_all, args.pipe_sample_ratio, args.max_pipes, rng)
    if not pipe_ids:
        raise RuntimeError("No pipes selected. Check pipe_sample_ratio / max_pipes." )

    # ---- 3) 三档强度 ----
    tiers: List[Tuple[float, float]] = []
    for t in args.tiers:
        lo, hi = t.split(",")
        tiers.append((float(lo), float(hi)))

    # ---- 4) 时长列表 ----
    growth_list = [parse_duration(x) for x in args.growth_list]
    pre_list = [parse_duration(x) for x in (args.pre_list or args.growth_list)]
    post_list = [parse_duration(x) for x in (args.post_list or args.growth_list)]

    # start anchor：用 cfg.simulation.start_datetime 作为锚点，在两年内抖动
    anchor = pd.Timestamp(cfg.simulation.start_datetime)
    jitter_days = 365 * 2
    
    # ---- 5) 生成场景 ----
    scenarios: List[LeakScenario] = []
    meta_list: List[Dict] = []

    # 设置sid以支持追加
    if args.overwrite:
        sid = 0
    else:
        sid = find_next_sid(out_root, manifest_path)
        if sid > 0:
            print(f"[INFO] append mode: found existing max sid={sid:06d}; new scenarios will start from {(sid+1):06d}")

    # 每个管段：生成多份 abrupt / incipient 场景（seen-pipe）
    abrupt_k = max(0, int(args.abrupt_per_pipe))
    incipient_k = max(0, int(args.incipient_per_pipe))
    for pipe_id in pipe_ids:
        # abrupt replicates
        for rep in range(abrupt_k):
            sid += 1
            pre = rng.choice(pre_list)
            post = rng.choice(post_list)
            tier, leak_d = sample_diameter_from_random_tier(tiers, rng)
            start = random_scenario_start(anchor, jitter_days, int(cfg.simulation.timestep_min), rng)
            scen_id = sanitize_id(f"{sid:06d}_{pipe_id}_abrupt_r{rep+1}")
            scen = build_single_leak_scenario(
                scenario_id=scen_id,
                scenario_start=start,
                pre=pre,
                growth=pd.Timedelta(0),
                post=post,
                pipe_id=pipe_id,
                leak_type="abrupt",
                leak_diameter_m=leak_d,
            )
            scenarios.append(scen)
            meta_list.append({
                "kind": "leak",
                "pipe_id": pipe_id,
                "leak_type": "abrupt",
                "replicate": int(rep + 1),
                "pre": str(pre),
                "growth": "0s",
                "post": str(post),
                "tier": tier,
                "leak_diameter_m": leak_d,
            })

        # incipient replicates
        for rep in range(incipient_k):
            sid += 1
            pre = rng.choice(pre_list)
            growth = rng.choice(growth_list)
            post = rng.choice(post_list)
            tier, leak_d = sample_diameter_from_random_tier(tiers, rng)
            start = random_scenario_start(anchor, jitter_days, int(cfg.simulation.timestep_min), rng)
            scen_id = sanitize_id(f"{sid:06d}_{pipe_id}_incipient_r{rep+1}")
            scen = build_single_leak_scenario(
                scenario_id=scen_id,
                scenario_start=start,
                pre=pre,
                growth=growth,
                post=post,
                pipe_id=pipe_id,
                leak_type="incipient",
                leak_diameter_m=leak_d,
            )
            scenarios.append(scen)
            meta_list.append({
                "kind": "leak",
                "pipe_id": pipe_id,
                "leak_type": "incipient",
                "replicate": int(rep + 1),
                "pre": str(pre),
                "growth": str(growth),
                "post": str(post),
                "tier": tier,
                "leak_diameter_m": leak_d,
            })
    n_leak = len(scenarios)
    r = float(args.no_leak_ratio)
    r = min(max(r, 0.0), 0.95)
    n_no_leak = int(round((r / (1.0 - r)) * n_leak)) if r > 0 else 0
    # 无泄漏：时长也用 pre+growth+post 的组合随机挑
    for _ in range(n_no_leak):
        sid += 1
        pre = rng.choice(pre_list)
        growth = rng.choice(growth_list)
        post = rng.choice(post_list)
        duration = pre + growth + post
        start = random_scenario_start(anchor, jitter_days, int(cfg.simulation.timestep_min), rng)
        scen_id = sanitize_id(f"{sid:06d}_no_leak")
        scen = build_no_leak_scenario(scenario_id=scen_id, scenario_start=start, duration=duration)
        scenarios.append(scen)
        meta_list.append({
            "kind": "no_leak",
            "pipe_id": None,
            "leak_type": None,
            "pre": str(pre),
            "growth": str(growth),
            "post": str(post),
            "tier": None,
            "leak_diameter_m": None,
            "duration": str(duration),
        })

    # 进度提示
    n_total = len(scenarios)
    print(f"[INFO] scenarios: leak={n_leak}  no_leak={n_no_leak}  total={n_total}")
    print(f"[INFO] output_root={out_root}")
    print(f"[INFO] manifest={manifest_path}")

    ex = LeakSimExecutor(cfg)

    if args.overwrite and manifest_path.exists():
        manifest_path.unlink()

    t0 = time.time()
    ok = failed = skipped = 0
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for i, (scen, meta) in enumerate(zip(scenarios, meta_list), start=1):
            scen_dir = out_root / scen.scenario_id
            if scen_dir.exists() and not args.overwrite:
                skipped += 1
                print(f"[INFO] Warning: scenario_id {scen.scenario_id} alraedy exists in {str(scen_dir)}, skipped.")
                continue

            m = SID_PREFIX_RE.match(scen.scenario_id)
            sid_num = int(m.group("sid")) if m else i
            scenario_seed = int(args.seed) * 1_000_000 + sid_num
            np.random.seed(scenario_seed)
            random.seed(scenario_seed)

            try:
                try:
                    cfg.meta.random_seed = scenario_seed
                except Exception:
                    pass

                res = ex.run_scenario(
                    scen,
                    ramp_update_minutes=int(args.ramp_update_minutes),
                    ramp_shape=str(args.ramp_shape),
                )

                mf.write(json.dumps({
                    "scenario_id": scen.scenario_id,
                    "status": "ok",
                    "scenario_dir": res.get("scenario_dir"),
                    "sensors_csv": res.get("sensors_path"),
                    "sensors_gt_csv": res.get("sensors_gt_path"),
                    "leak_flow_m3h_csv": res.get("leak_path"),
                    "meta_json": res.get("meta_path"),
                    "scenario_seed": scenario_seed,
                    **meta,
                }, ensure_ascii=False) + "\n")
                mf.flush()
                ok += 1

                if i % 20 == 0 or i == 1:
                    elapsed = time.time() - t0
                    rate = elapsed / max(1, i)
                    remain = rate * (len(scenarios) - i)
                    print(f"[PROGRESS] {i}/{len(scenarios)}  ok={ok}  skipped={skipped}  failed={failed} | avg={rate:.2f}s/scen  ETA={remain/60:.1f}min")

            except Exception as e:
                failed += 1
                print(f"[ERROR] scenario {scen.scenario_id} failed: {e}")
                if not args.continue_on_error:
                    raise

    print("\nDone.")
    print(f"Output root: {out_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Scenarios attempted: {len(scenarios)}  ok={ok}  skipped={skipped}  failed={failed}")


if __name__ == "__main__":
    main()
