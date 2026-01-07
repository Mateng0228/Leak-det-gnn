# sim_executor.py
from __future__ import annotations
import os
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import networkx as nx
import wntr

from config_io import SimConfig, AnyDist

# 工具函数：随机分布采样
def sample_anydist(dist: AnyDist, size=None):
    """支持 normal / trunc_normal / uniform / constant 的标量或数组采样。"""
    name = dist.name
    if name == "normal":
        mu, sigma = dist.mu, dist.sigma
        return np.random.normal(mu, sigma, size)
    elif name == "trunc_normal":
        mu, sigma = dist.mu, dist.sigma
        lo, hi = dist.clip
        x = np.random.normal(mu, sigma, size)
        return np.clip(x, lo, hi)
    elif name == "uniform":
        return np.random.uniform(dist.low, dist.high, size)
    elif name == "constant":
        if size is None:
            return dist.value
        else:
            return np.full(size, dist.value)
    else:
        raise ValueError(f"Unknown dist: {name}")

# 工具函数：图中1-hop平滑
def one_hop_smooth(values: np.ndarray, G: nx.Graph, nodes: List[str], mix_weight: float) -> np.ndarray:
    idx = {n: i for i, n in enumerate(nodes)}
    out = values.copy()
    for u in nodes:
        i = idx[u]
        nbrs = list(G.neighbors(u))
        if not nbrs:
            continue
        out[i] = (1 - mix_weight) * values[i] + mix_weight * np.mean([values[idx[v]] for v in nbrs])
    return out

# 工具函数：生成时间轴index
def build_time_index(start_ts: pd.Timestamp, duration_days: int, step_min: int) -> pd.DatetimeIndex:
    n_steps = (duration_days * 24 * 60 // step_min) + 1
    return pd.date_range(start=start_ts, periods=n_steps, freq=f"{step_min}min")

def make_piecewise_constant_series(length: int, refresh_every_steps: int, dist: AnyDist) -> np.ndarray:
    """生成分段常数序列（长度=length），每refresh_every_steps重新采样一次。"""
    if refresh_every_steps <= 0:
        refresh_every_steps = length
    n_chunks = math.ceil(length / refresh_every_steps)
    chunk_vals = sample_anydist(dist, size=n_chunks)
    seq = np.repeat(chunk_vals, refresh_every_steps)[:length]
    return seq

def expand_base_pattern_to_report_steps(pattern_vals, report_steps, pattern_step_min, report_step_min, start_datetime: pd.Timestamp):
    """将pattern（间隔=pattern_step_min）扩展为report_steps长度、步长=report_step_min的序列"""
    if pattern_step_min < report_step_min:
        raise ValueError("pattern_step_min 必须大于等于 report_step_min")
    if pattern_vals is None or len(pattern_vals) == 0:
        return np.ones(report_steps, dtype=float)

    pattern_vals = np.asarray(pattern_vals, dtype=float)
    
    base_dt = start_datetime.normalize() - pd.Timedelta(days=start_datetime.weekday())
    offset_min = int((start_datetime - base_dt).total_seconds() // 60)
    t_report = offset_min + (np.arange(report_steps) * report_step_min)
    
    idx = (t_report // pattern_step_min).astype(int)
    idx = idx % len(pattern_vals)
    full = pattern_vals[idx]
    return full

def compose_node_pattern(
    base_pattern_vals: List[float] | None,
    mult_series: np.ndarray,
    report_steps: int,
    pattern_step_min: int,
    report_step_min: int,
    start_datetime: pd.Timestamp,
) -> np.ndarray:
    base = expand_base_pattern_to_report_steps(base_pattern_vals, report_steps, pattern_step_min, report_step_min, start_datetime)
    return base * mult_series

# 工具函数：获得每个传感器的历史噪声频率
def get_sensor_sigma_rel(cfg: SimConfig):
    target_sensors = cfg.sensors.pressure_node_ids
    if cfg.sensors_noise.sensor_sigma_path is not None:
        sigma_rel = pd.read_csv(cfg.sensors_noise.sensor_sigma_path, index_col=0).squeeze("columns")
        sigma_rel = sigma_rel.reindex(target_sensors)
        if sigma_rel.notna().sum() == 0:
            raise ValueError("Loaded sensor_sigma file has no valid entries for cfg.sensors")

        sigma_rel = sigma_rel.fillna(sigma_rel.median())
        return sigma_rel
    
    path = cfg.sensors_noise.historical_data_path
    dt_col = cfg.sensors_noise.datetime_col
    sep = cfg.sensors_noise.sep
    win = cfg.sensors_noise.smooth_window_steps
    
    df = pd.read_csv(path, sep=sep, decimal=",", parse_dates=[dt_col])
    df = df.set_index(dt_col)
    df = df[df.columns.intersection(target_sensors)]
    if df.shape[1] == 0:
        raise ValueError("No overlapping sensors between historical data and cfg.sensors")
    
    # 平滑趋势
    trend = df.rolling(window=win, center=True, min_periods=1).mean()
    residual = df - trend
    sigma_abs = residual.std(axis=0)
    mean_level = df.mean(axis=0).abs()
    sigma_rel = (sigma_abs / mean_level).replace([np.inf, np.nan], 0.0)
    
    # 保存并返回与当前 sensor 列表对齐的 Series
    sigma_rel = sigma_rel.reindex(target_sensors).fillna(sigma_rel.median())
    sigma_rel.to_csv("data/temp/sigma_rel.csv", header=["sigma_rel"])
    return sigma_rel


# 主执行器
class NormalSimExecutor:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        np.random.seed(cfg.meta.random_seed)

    def run_once(self):
        cfg = self.cfg
        # ---- 1) 读网络 ----
        wn = wntr.network.WaterNetworkModel(cfg.paths.inp_path)
        # ---- 2) 结构噪声 ----
        if cfg.noise_switches.structure:
            self._apply_structure_noise(wn)
        # ---- 3) 设置时间步与求解器 ----
        start_dt = pd.Timestamp(cfg.simulation.start_datetime)
        dur_days = int(cfg.simulation.duration_days)
        warmup_days = int(cfg.simulation.warmup_days)
        step_min = int(cfg.simulation.timestep_min)

        total_seconds = dur_days * 24 * 60 * 60
        step_seconds = step_min * 60
        
        wn.options.time.duration = total_seconds
        wn.options.time.report_timestep = step_seconds
        wn.options.time.hydraulic_timestep = step_seconds
        # 水头损失与收敛设置
        headloss = cfg.simulation.solver.headloss
        if headloss == "Hazen-Williams":
            wn.options.hydraulic.headloss = "H-W"
        elif headloss == "Darcy-Weisbach":
            wn.options.hydraulic.headloss = "D-W"
        elif headloss == "Chezy-Manning":
            wn.options.hydraulic.headloss = "C-M"
        wn.options.hydraulic.accuracy = cfg.simulation.solver.accuracy
        wn.options.hydraulic.trials = cfg.simulation.solver.trials
        wn.options.hydraulic.unbalanced = cfg.simulation.solver.unbalanced.mode
        wn.options.hydraulic.unbalanced_value = cfg.simulation.solver.unbalanced.trials

        # ---- 4) 预烘焙需求乘子序列，并与INP中的pattern合成 ----
        time_index = build_time_index(start_dt, dur_days, step_min)
        report_steps = len(time_index)
        
        # 4.1 pattern_profile 噪声（global_scale + weekend_factor）
        if cfg.noise_switches.pattern_profile:
            gevery = cfg.pattern_profile.global_scale.refresh_every_minutes
            gs_steps = max(1, gevery // step_min)
            global_scale = make_piecewise_constant_series(report_steps, gs_steps, cfg.pattern_profile.global_scale.dist)
        
            weekend_factor_val = float(sample_anydist(cfg.pattern_profile.weekend_factor.dist))
            is_weekend = time_index.weekday >= 5  # Sat=5, Sun=6
            weekend_factor = np.where(is_weekend, weekend_factor_val, 1.0)
        else:
            global_scale = np.ones(report_steps, dtype=float)
            weekend_factor = np.ones(report_steps, dtype=float)
        
        # 4.2 节点需求噪声（node_multiplier + 空间平滑）
        junctions = list(wn.junction_name_list)
        num_junctions = len(junctions)
        
        G = wn.to_graph().to_undirected() # 基于管道图的无向邻接
        G = G.subgraph(junctions).copy()
        node_mult = np.ones((report_steps, num_junctions), dtype=float)
        if cfg.noise_switches.demand:
            nevery = cfg.demand_noise.node_multiplier.refresh_every_minutes
            nm_steps = max(1, nevery // step_min)
        
            # 每个刷新区间独立采样长度为num_junctions的向量，并做一次1-hop平滑
            mix_w = cfg.demand_noise.spatial_smoothing.mix_weight
            hops = cfg.demand_noise.spatial_smoothing.hops
            assert hops == 1, "当前实现仅支持1-hop平滑"

            n_chunks = math.ceil(report_steps / nm_steps)
            for k in range(n_chunks):
                base = sample_anydist(cfg.demand_noise.node_multiplier.dist, size=num_junctions).astype(float)
                base = one_hop_smooth(base, G, junctions, mix_w)
                start = k * nm_steps
                end = min((k + 1) * nm_steps, report_steps)
                node_mult[start:end, :] = base[None, :]

        # 4.3 汇总：最终节点乘子 = global_scale(t) × weekend_factor(t) × node_mult_j(t)
        full_mult = node_mult * global_scale[:, None] * weekend_factor[:, None]
        
        # 4.5 与INP自带pattern合成
        pattern_step_sec = getattr(wn.options.time, "pattern_timestep", 3600) # Epanet 中 pattern 的“步长”默认60min
        pattern_step_min = max(1, int(pattern_step_sec // 60))
        for j_idx, j_name in enumerate(junctions):
            j = wn.get_node(j_name)
            ts_list = j.demand_timeseries_list

            for k, ts in enumerate(ts_list):
                # 取该 timeseries 对应的原始 pattern 值；没有 pattern 则视为常数 1.0
                base_vals = None
                pat = ts.pattern
                if pat is not None:
                    base_vals = list(pat.multipliers)

                seq = compose_node_pattern(base_vals, full_mult[:, j_idx], report_steps, pattern_step_min, step_min, start_dt)
                new_pat_name = f"pat_{j_name}_{k}"
                if new_pat_name in wn.pattern_name_list:
                    wn.remove_pattern(new_pat_name)
                wn.add_pattern(new_pat_name, seq.tolist())
                ts.pattern_name = new_pat_name
        
        # ---- 5) 仿真 ----
        prefix = os.path.join("data", "temp", "epanet_run")
        os.makedirs(os.path.dirname(prefix), exist_ok=True)
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim(file_prefix=prefix)

        # ---- 6) 取出 junction head，并去掉 warmup 天 ----
        head = results.node["head"].loc[:, junctions]  # DataFrame: time × junctions
        elev_series = pd.Series(
            {j_name: wn.get_node(j_name).elevation for j_name in junctions},
            index=junctions, dtype=float,
        )
        pressure = head.subtract(elev_series, axis=1)
        
        if warmup_days > 0:
            cut = warmup_days * 24 * 60 // step_min
            pressure = pressure.iloc[cut:, :]
            time_index = time_index[cut:]
        
        pressure.index = time_index

        # ---- 7) 构造标签（全节点真值）和传感器观测（带噪） ----
        label_df = pressure.copy()

        all_sensor_nodes = cfg.sensors.pressure_node_ids
        sensor_nodes = [nid for nid in all_sensor_nodes if nid in label_df.columns]
        if not sensor_nodes:
            raise ValueError(
                f"No valid pressure_node_ids found in head results. Config sensors: {all_sensor_nodes}, "
                f"available junctions: {list(label_df.columns)}"
            )
        sensor_df = label_df[sensor_nodes].copy()

        # ---- 8) 对传感器观测加噪声 / 量化 / 缺失 ----
        if cfg.noise_switches.sensors:
            sn_cfg = cfg.sensors_noise
            sensor_vals = sensor_df.values.astype(float)
            T, N = sensor_vals.shape

            # 应用乘性噪声
            dist = sn_cfg.multiplicative_noise_rel.dist
            factors = sample_anydist(dist, size=sensor_vals.shape).astype(float)
            sensor_vals = sensor_vals * factors
            
            # 量化
            q = float(sn_cfg.quantization_head_m)
            if q > 0.0:
                sensor_vals = np.round(sensor_vals / q) * q

            # 缺失
            missing_rate = float(sn_cfg.missing_rate)
            if missing_rate > 0.0:
                miss_mask = np.random.rand(*sensor_vals.shape) < missing_rate
                sensor_vals = sensor_vals.astype(float)
                sensor_vals[miss_mask] = np.nan
            else:
                miss_mask = None

            # 写回 DataFrame
            sensor_df = pd.DataFrame(sensor_vals, index=sensor_df.index, columns=sensor_df.columns)

        # ---- 9) 写出 CSV 和 meta ----
        out_dir = cfg.paths.output_dir
        os.makedirs(out_dir, exist_ok=True)

        sensors_path = os.path.join(out_dir, "sensors.csv")
        labels_path = os.path.join(out_dir, "nodal_pressure.csv")
        meta_path = os.path.join(out_dir, "meta.json")

        # 宽表：时间索引 + 每个节点一列
        sensor_df_fmt = sensor_df.map(self._format_float_4)
        label_df_fmt = label_df.map(self._format_float_4)
        sensor_df_fmt.to_csv(sensors_path, index_label="datetime")
        label_df_fmt.to_csv(labels_path, index_label="datetime")

        # meta 信息：方便后续检查和复现实验
        meta = {
            "config_meta": {
                "version": cfg.meta.version,
                "random_seed": cfg.meta.random_seed,
            },
            "paths": {
                "inp_path": cfg.paths.inp_path,
                "output_dir": cfg.paths.output_dir,
            },
            "simulation": {
                "start_datetime": cfg.simulation.start_datetime.isoformat(),
                "duration_days": cfg.simulation.duration_days,
                "warmup_days": cfg.simulation.warmup_days,
                "timestep_min": cfg.simulation.timestep_min,
                "n_time_steps_after_warmup": len(time_index),
            },
            "nodes": {
                "junctions": junctions,
                "pressure_node_ids": all_sensor_nodes,
                "pressure_node_ids_effective": sensor_nodes,
            },
            "noise_switches": {
                "structure": cfg.noise_switches.structure,
                "pattern_profile": cfg.noise_switches.pattern_profile,
                "demand": cfg.noise_switches.demand,
                "sensors": cfg.noise_switches.sensors,
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {
            "sensors_path": sensors_path,
            "labels_path": labels_path,
            "meta_path": meta_path,
        }

    # -------------------- 结构噪声注入 --------------------
    def _apply_structure_noise(self, wn: wntr.network.WaterNetworkModel) -> None:
        sn = self.cfg.structure_noise
        
        # 直径（×比例）
        scale = float(sample_anydist(sn.diameters["dist"]))
        for name, pipe in wn.pipes():
            pipe.diameter *= scale
        # 糙率（×比例）
        scale = float(sample_anydist(sn.roughness["dist"]))
        for name, pipe in wn.pipes():
            pipe.roughness *= scale
        # 次损（×比例）
        scale = float(sample_anydist(sn.minor_loss["dist"]))
        for name, pipe in wn.pipes():
            pipe.minor_loss *= scale
        # PRV设点（×比例）
        scale = float(sample_anydist(sn.prv_setpoint["dist"]))
        for name, valve in wn.valves():
            if valve.valve_type.upper() == "PRV":
                valve.initial_setting *= scale
        
    def _format_float_4(self, x):
        if pd.isna(x):
            return ""  # 或者返回 np.nan
        s = "{:.4f}".format(float(x))
        s = s.rstrip('0').rstrip('.')  # 去掉多余的0和小数点
        return s

if __name__ == "__main__":
    import time
    cfg = SimConfig.from_yaml("configs/sim_LTA.yaml")
    cfg.paths.output_dir = str(Path(cfg.paths.output_dir) / "normal_test")
    simulator = NormalSimExecutor(cfg)
    start = time.time()
    simulator.run_once()
    end = time.time()
    print("运行时间：", end - start, "秒")