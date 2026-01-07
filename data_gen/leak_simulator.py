from __future__ import annotations
import os
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx
import wntr

from config_io import SimConfig, AnyDist

def enable_wntr_leak_area_setter():
    # 让 ControlAction(leak_node, "leak_area", ...) 可写
    from wntr.network.elements import Junction, Tank, Reservoir

    def _set_leak_area(self, value):
        self._leak_area = float(value)

    for cls in (Junction, Tank, Reservoir):
        prop = getattr(cls, "leak_area", None)
        if isinstance(prop, property) and prop.fset is None:
            cls.leak_area = prop.setter(_set_leak_area)


# -------------------- 通用工具 --------------------
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


def one_hop_smooth(values: np.ndarray, G: nx.Graph, nodes: List[str], mix_weight: float) -> np.ndarray:
    """对节点属性做 1-hop 平滑：out[i] = (1-w)*x[i] + w*mean(x[nbrs])"""
    idx = {n: i for i, n in enumerate(nodes)}
    out = values.copy()
    for u in nodes:
        i = idx[u]
        nbrs = list(G.neighbors(u))
        if not nbrs:
            continue
        out[i] = (1 - mix_weight) * values[i] + mix_weight * np.mean([values[idx[v]] for v in nbrs])
    return out


def build_time_index(start_ts: pd.Timestamp, duration_days: int, step_min: int) -> pd.DatetimeIndex:
    n_steps = (duration_days * 24 * 60 // step_min) + 1
    return pd.date_range(start=start_ts, periods=n_steps, freq=f"{step_min}min")


def make_piecewise_constant_series(length: int, refresh_every_steps: int, dist: AnyDist) -> np.ndarray:
    """生成分段常数序列（长度=length），每 refresh_every_steps 重新采样一次。"""
    if refresh_every_steps <= 0:
        refresh_every_steps = length
    n_chunks = math.ceil(length / refresh_every_steps)
    chunk_vals = sample_anydist(dist, size=n_chunks)
    seq = np.repeat(chunk_vals, refresh_every_steps)[:length]
    return seq


def expand_base_pattern_to_report_steps(
    pattern_vals,
    report_steps: int,
    pattern_step_min: int,
    report_step_min: int,
    offset_min: int = 0
) -> np.ndarray:
    """将 pattern（间隔=pattern_step_min）扩展为 report_steps 长度、步长=report_step_min 的序列。"""
    if pattern_step_min < report_step_min:
        raise ValueError("pattern_step_min 必须大于等于 report_step_min")
    if pattern_vals is None or len(pattern_vals) == 0:
        return np.ones(report_steps, dtype=float)

    pattern_vals = np.asarray(pattern_vals, dtype=float)
    t_report = int(offset_min) + (np.arange(report_steps) * report_step_min)
    idx = (t_report // pattern_step_min).astype(int)
    idx = idx % len(pattern_vals)
    return pattern_vals[idx]


def compose_node_pattern(
    base_pattern_vals: Optional[List[float]],
    mult_series: np.ndarray,
    report_steps: int,
    pattern_step_min: int,
    report_step_min: int,
    offset_min: int = 0
) -> np.ndarray:
    base = expand_base_pattern_to_report_steps(base_pattern_vals, report_steps, pattern_step_min, report_step_min, offset_min)
    return base * mult_series


def _format_float_4(x):
    if pd.isna(x):
        return ""
    s = "{:.4f}".format(float(x))
    s = s.rstrip("0").rstrip(".")
    return s


# -------------------- 泄漏场景数据结构 --------------------

@dataclass
class LeakRecord:
    """一条泄漏记录（对应一个管段的泄漏）"""
    leak_id: str
    pipe_id: str
    leak_type: str  # 'abrupt' or 'incipient'
    start_datetime: pd.Timestamp
    peak_datetime: Optional[pd.Timestamp]
    end_datetime: pd.Timestamp
    leak_diameter_m: float
    discharge_coeff: Optional[float] = None # 若 None 则用 WNTR 0.75

    def __post_init__(self):
        self.start_datetime = pd.Timestamp(self.start_datetime)
        self.end_datetime = pd.Timestamp(self.end_datetime)
        self.peak_datetime = pd.Timestamp(self.peak_datetime) if self.peak_datetime is not None else None
        self.leak_type = str(self.leak_type).lower().strip()
        if self.leak_type not in {"abrupt", "incipient"}:
            raise ValueError(f"Unknown leak_type={self.leak_type}, expected 'abrupt' or 'incipient'")


@dataclass
class LeakScenario:
    """一个泄漏场景（可包含多个管段同时泄漏）"""
    scenario_id: str
    start_datetime: pd.Timestamp
    duration_days: int
    leak_records: List[LeakRecord] = field(default_factory=list)

    def __post_init__(self):
        self.start_datetime = pd.Timestamp(self.start_datetime)


# -------------------- 主执行器：LeakSimExecutor --------------------

class LeakSimExecutor:
    """
    生成泄漏场景数据：
    - 输入：SimConfig（与 normal_simulator 相同） + LeakScenario（场景信息在代码里给）
    - 输出：每个场景一个目录，保存：
        - sensors.csv: 传感器压力宽表（含传感器噪声）
        - leak_flow_m3h.csv: 泄漏量宽表（列=发生过泄漏的 pipe_id；单位 m^3/h）
        - meta.json: 元数据 + 场景配置
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        np.random.seed(cfg.meta.random_seed)

    # -------------------- 结构噪声 --------------------
    def _apply_structure_noise(self, wn: wntr.network.WaterNetworkModel) -> None:
        sn = self.cfg.structure_noise
        # 直径（×比例）
        scale = float(sample_anydist(sn.diameters["dist"]))
        for _, pipe in wn.pipes():
            pipe.diameter *= scale
        # 糙率（×比例）
        scale = float(sample_anydist(sn.roughness["dist"]))
        for _, pipe in wn.pipes():
            pipe.roughness *= scale
        # 次损（×比例）
        scale = float(sample_anydist(sn.minor_loss["dist"]))
        for _, pipe in wn.pipes():
            pipe.minor_loss *= scale
        # PRV 设点（×比例）
        scale = float(sample_anydist(sn.prv_setpoint["dist"]))
        for _, valve in wn.valves():
            if str(valve.valve_type).upper() == "PRV":
                valve.initial_setting *= scale

    # -------------------- 需求/模式噪声 --------------------
    def _apply_pattern_and_demand_noise(self, wn: wntr.network.WaterNetworkModel, time_index: pd.DatetimeIndex, step_min: int) -> None:
        cfg = self.cfg
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
        if num_junctions == 0:
            return

        G = wn.to_graph().to_undirected()  # 基于管道图的无向邻接
        G = G.subgraph(junctions).copy()
        node_mult = np.ones((report_steps, num_junctions), dtype=float)
        if cfg.noise_switches.demand:
            nevery = cfg.demand_noise.node_multiplier.refresh_every_minutes
            nm_steps = max(1, nevery // step_min)

            mix_w = cfg.demand_noise.spatial_smoothing.mix_weight
            hops = cfg.demand_noise.spatial_smoothing.hops
            assert hops == 1, "当前实现仅支持1-hop平滑（与 normal_simulator 对齐）"

            n_chunks = math.ceil(report_steps / nm_steps)
            for k in range(n_chunks):
                base = sample_anydist(cfg.demand_noise.node_multiplier.dist, size=num_junctions).astype(float)
                base = one_hop_smooth(base, G, junctions, mix_w)
                start = k * nm_steps
                end = min((k + 1) * nm_steps, report_steps)
                node_mult[start:end, :] = base[None, :]

        # 4.3 汇总：最终节点乘子 = global_scale(t) × weekend_factor(t) × node_mult_j(t)
        full_mult = node_mult * global_scale[:, None] * weekend_factor[:, None]

        # 4.4 与 INP 自带 pattern 合成
        pattern_step_sec = getattr(wn.options.time, "pattern_timestep", 3600)  # EPANET pattern 步长默认 60min
        pattern_step_min = max(1, int(pattern_step_sec // 60))
        try:
            ref_dt = pd.Timestamp(cfg.simulation.start_datetime)
            scen_dt = pd.Timestamp(time_index[0])
            offset_min = int((scen_dt - ref_dt).total_seconds() // 60)
        except Exception:
            offset_min = 0
        
        for j_idx, j_name in enumerate(junctions):
            j = wn.get_node(j_name)
            ts_list = j.demand_timeseries_list

            for k, ts in enumerate(ts_list):
                base_vals = None
                pat = ts.pattern
                if pat is not None:
                    base_vals = list(pat.multipliers)

                seq = compose_node_pattern(base_vals, full_mult[:, j_idx], report_steps, pattern_step_min, step_min, offset_min)
                new_pat_name = f"pat_{j_name}_{k}"
                if new_pat_name in wn.pattern_name_list:
                    wn.remove_pattern(new_pat_name)
                wn.add_pattern(new_pat_name, seq.tolist())
                ts.pattern_name = new_pat_name

    
    # -------------------- 泄漏注入：pipe -> split -> new junction -> add_leak + controls --------------------
    @staticmethod
    def _leak_area_from_diameter(d_m: float) -> float:
        return math.pi * (float(d_m) / 2.0) ** 2

    def _inject_leaks_with_ramp(
        self,
        wn: wntr.network.WaterNetworkModel,
        scenario: LeakScenario,
        *,
        ramp_update_minutes: int = 60,   # incipient ramp 更新频率：默认 1 小时
        ramp_shape: str = "quadratic",   # 'linear' or 'quadratic'
        default_discharge_coeff: float = 0.75,  # WNTR add_leak 默认值
        split_at_point: float = 0.5,
        add_pipe_at_end: bool = True,
    ) -> Dict[str, str]:
        """返回：pipe_id -> leak_junction_name"""
        pipe_to_leaknode: Dict[str, str] = {}

        C = wntr.network.controls
        step_sec = ramp_update_minutes * 60
        hyd_step = int(wn.options.time.hydraulic_timestep)  # 秒
        def snap_to_grid(sec: float) -> int:
            # 向下对齐到仿真网格，避免控制触发不到
            s = int(sec)
            return (s // hyd_step) * hyd_step

        for rec in scenario.leak_records:
            pipe_id = rec.pipe_id

            # 1) split pipe：在中点加一个 junction
            leak_node_name = f"leak_{rec.leak_id}_{pipe_id}"
            new_pipe_name = f"{pipe_id}_split_{rec.leak_id}"
            wntr.morph.link.split_pipe(
                wn,
                pipe_name_to_split=pipe_id,
                new_pipe_name=new_pipe_name,
                new_junction_name=leak_node_name,
                add_pipe_at_end=add_pipe_at_end,
                split_at_point=split_at_point,
                return_copy=False,
            )
            leak_node = wn.get_node(leak_node_name)

            # 2) 计算 start/end/peak（秒，基于 scenario.start_datetime）
            start_sec = (rec.start_datetime - scenario.start_datetime).total_seconds()
            end_sec = (rec.end_datetime - scenario.start_datetime).total_seconds()
            if start_sec < 0:
                raise ValueError(f"Leak {rec.leak_id} starts before scenario.start_datetime.")
            if end_sec <= start_sec:
                raise ValueError(f"Leak {rec.leak_id} end<=start.")
            start_sec = float(start_sec)
            end_sec = float(end_sec)

            # 3) 在 leak_node 上启用 WNTR leak（Q = Cd * A * sqrt(2 g h)）
            cd = float(rec.discharge_coeff) if rec.discharge_coeff is not None else float(default_discharge_coeff)
            a_peak = self._leak_area_from_diameter(rec.leak_diameter_m)  # m^2

            t_start = snap_to_grid(start_sec)
            t_end = snap_to_grid(end_sec)

            # abrupt: 直接给峰值面积；incipient: 初始面积为 0，通过控制逐步增大
            a0 = float(a_peak) if rec.leak_type == "abrupt" else 0.0
            leak_node.add_leak(
                wn,
                area=float(a0),
                discharge_coeff=float(cd),
                start_time=int(t_start),
                end_time=int(t_end),
            )

            # 4) incipient：用 time controls 对 leak_area 做 ramp（piecewise constant）
            if rec.leak_type == "incipient":
                if rec.peak_datetime is None:
                    raise ValueError(f"incipient leak {rec.leak_id} requires peak_datetime.")

                peak_sec = (rec.peak_datetime - scenario.start_datetime).total_seconds()
                peak_sec = max(float(start_sec), min(float(peak_sec), float(end_sec)))
                t_peak = snap_to_grid(peak_sec)

                # 生成 ramp 控制时间点（包含 peak）
                if t_peak == t_start:
                    t_list = [t_start]
                else:
                    t_list = list(range(int(t_start), int(t_peak) + 1, int(step_sec)))
                    if t_list[-1] != int(t_peak):
                        t_list.append(int(t_peak))

                denom = max(1.0, float(t_peak - t_start))
                for k, t in enumerate(t_list):
                    s = (float(t) - float(t_start)) / denom
                    s = max(0.0, min(1.0, float(s)))
                    if ramp_shape == "linear":
                        a_t = float(a_peak) * s
                    else:
                        a_t = float(a_peak) * (s ** 2)

                    act = C.ControlAction(leak_node, "leak_area", float(a_t))
                    ctrl = C.Control._time_control(wn, int(t), "SIM_TIME", False, act)
                    wn.add_control(f"ctrl_leak_{rec.leak_id}_area_{k}", ctrl)

                # peak 时刻保险：强制设为峰值面积
                actp = C.ControlAction(leak_node, "leak_area", float(a_peak))
                ctrlp = C.Control._time_control(wn, int(t_peak), "SIM_TIME", False, actp)
                wn.add_control(f"ctrl_leak_{rec.leak_id}_area_peak", ctrlp)

            # 5) end 时刻把 leak_area 归零（即使 leak_status 已经被 add_leak 的 end_control 关掉，也更直观）
            act_end = C.ControlAction(leak_node, "leak_area", 0.0)
            ctrl_end = C.Control._time_control(wn, int(t_end), "SIM_TIME", False, act_end)
            wn.add_control(f"ctrl_leak_{rec.leak_id}_area_end", ctrl_end)

            pipe_to_leaknode[pipe_id] = leak_node_name

        return pipe_to_leaknode

    # -------------------- 传感器噪声 --------------------
    def _apply_sensor_noise(self, sensor_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.cfg
        if not cfg.noise_switches.sensors:
            return sensor_df

        sn_cfg = cfg.sensors_noise
        vals = sensor_df.values.astype(float)
        # 乘性噪声
        dist = sn_cfg.multiplicative_noise_rel.dist
        factors = sample_anydist(dist, size=vals.shape).astype(float)
        vals = vals * factors

        # 量化
        q = float(sn_cfg.quantization_head_m)
        if q > 0.0:
            vals = np.round(vals / q) * q

        # 缺失
        missing_rate = float(sn_cfg.missing_rate)
        if missing_rate > 0.0:
            miss_mask = np.random.rand(*vals.shape) < missing_rate
            vals = vals.astype(float)
            vals[miss_mask] = np.nan

        return pd.DataFrame(vals, index=sensor_df.index, columns=sensor_df.columns)

    # -------------------- 主接口：run_scenario --------------------
    def run_scenario(
        self,
        scenario: LeakScenario,
        *,
        ramp_update_minutes: int = 60,
        ramp_shape: str = "quadratic",
        default_discharge_coeff: float = 0.75,
    ) -> Dict[str, str]:
        cfg = self.cfg

        # ---- 1) 读网络 ----
        wn = wntr.network.WaterNetworkModel(cfg.paths.inp_path)

        # ---- 2) 结构噪声 ----
        if cfg.noise_switches.structure:
            self._apply_structure_noise(wn)

        # ---- 3) 设置时间步与求解器 ----
        dur_days = int(scenario.duration_days) # from scenario
        start_dt = pd.Timestamp(scenario.start_datetime) # from scenario
        step_min = int(cfg.simulation.timestep_min)
        warmup_days = int(cfg.simulation.warmup_days)

        total_seconds = dur_days * 24 * 60 * 60
        step_seconds = step_min * 60
        wn.options.time.duration = int(total_seconds)
        wn.options.time.report_timestep = int(step_seconds)
        wn.options.time.hydraulic_timestep = int(step_seconds)
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
        # unbalanced
        if hasattr(cfg.simulation.solver, "unbalanced") and cfg.simulation.solver.unbalanced is not None:
            wn.options.hydraulic.unbalanced = cfg.simulation.solver.unbalanced.mode
            wn.options.hydraulic.unbalanced_value = cfg.simulation.solver.unbalanced.trials
        
        # ---- 4) 需求/模式噪声 ----
        time_index_all = build_time_index(start_dt, duration_days=dur_days, step_min=step_min)
        self._apply_pattern_and_demand_noise(wn, time_index=time_index_all, step_min=step_min)

        # ---- 5) 注入泄漏（pipe -> split -> junction -> add_leak + ramp controls） ----
        enable_wntr_leak_area_setter()
        pipe_to_leaknode = self._inject_leaks_with_ramp(
            wn, scenario,
            ramp_update_minutes=ramp_update_minutes,
            ramp_shape=ramp_shape,
            default_discharge_coeff=default_discharge_coeff,
        )

        # ---- 6) 仿真 ----
        sim = wntr.sim.WNTRSimulator(wn)
        results = sim.run_sim()
        
        # ---- 7) 生成时间轴 + 去 warmup ----
        warmup_steps = warmup_days * 24 * 60 // step_min
        if warmup_steps >= len(time_index_all):
            raise ValueError("warmup_steps >= total steps; reduce warmup_days or increase duration_days.")
        time_index = time_index_all[warmup_steps:]

        # ---- 8) 取 pressure 结果（节点）----
        pressure_all = results.node["pressure"]
        pressure_all = pressure_all.iloc[warmup_steps:, :]
        pressure_all.index = time_index

        # ---- 9) 取传感器压力并加噪 ----
        all_sensor_nodes = list(cfg.sensors.pressure_node_ids)
        # 过滤掉不存在的 node
        sensor_nodes = [nid for nid in all_sensor_nodes if nid in pressure_all.columns]
        if not sensor_nodes:
            raise ValueError(f"No valid pressure_node_ids found in pressure results. ")
        label_df = pressure_all[sensor_nodes].copy()
        sensor_df = self._apply_sensor_noise(label_df)

        # ---- 10) 取 leak_demand（m^3/s）并映射回 pipe_id，输出 m^3/h ----
        if "leak_demand" not in results.node:
            raise RuntimeError(
                "results.node does not contain 'leak_demand'. "
                "Please ensure you are using WNTRSimulator and leaks are added via node.add_leak()."
            )

        leak_raw = results.node["leak_demand"].iloc[warmup_steps:, :]
        leak_raw.index = time_index

        # 初始化：列为场景中出现过的 pipe_id
        pipe_cols = [rec.pipe_id for rec in scenario.leak_records]
        pipe_cols = list(dict.fromkeys(pipe_cols))
        leak_df = pd.DataFrame(0.0, index=time_index, columns=pipe_cols)

        for pipe_id, leak_node_name in pipe_to_leaknode.items():
            if leak_node_name not in leak_raw.columns:
                continue
            q_m3h = leak_raw[leak_node_name].astype(float).clip(lower=0.0) * 3600.0
            leak_df[pipe_id] += q_m3h
        
        # ---- 11) 保存 ----
        out_base = Path(cfg.paths.output_dir) / scenario.scenario_id
        out_base.mkdir(parents=True, exist_ok=True)
        sensors_path = str(out_base / "sensors.csv")
        sensors_gt_path = str(out_base / "sensors_gt.csv")
        leak_path = str(out_base / "leak_flow_m3h.csv")
        meta_path = str(out_base / "meta.json")

        sensor_df.map(_format_float_4).to_csv(sensors_path, index_label="datetime")
        label_df.map(_format_float_4).to_csv(sensors_gt_path, index_label="datetime")
        if len(scenario.leak_records) != 0:
            leak_df.map(_format_float_4).to_csv(leak_path, index_label="datetime")
        # meta
        meta = {
            "config_meta": {
                "version": cfg.meta.version,
                "random_seed": cfg.meta.random_seed,
            },
            "paths": {
                "inp_path": cfg.paths.inp_path,
                "output_dir": cfg.paths.output_dir,
                "scenario_dir": str(out_base),
            },
            "simulation": {
                "start_datetime": str(start_dt),
                "duration_days": dur_days,
                "warmup_days": warmup_days,
                "timestep_min": step_min,
                "n_time_steps_after_warmup": int(len(time_index)),
            },
            "scenario": {
                "scenario_id": scenario.scenario_id,
                "n_leaks": len(scenario.leak_records),
                "ramp_update_minutes": int(ramp_update_minutes),
                "ramp_shape": ramp_shape,
                "default_discharge_coeff": float(default_discharge_coeff),
                "leaks": [
                    {
                        "leak_id": r.leak_id,
                        "pipe_id": r.pipe_id,
                        "leak_node_name": pipe_to_leaknode.get(r.pipe_id, None),
                        "type": r.leak_type,
                        "start_datetime": str(r.start_datetime),
                        "peak_datetime": str(r.peak_datetime) if r.peak_datetime is not None else None,
                        "end_datetime": str(r.end_datetime),
                        "leak_diameter_m": float(r.leak_diameter_m),
                        "peak_area_m2": float(self._leak_area_from_diameter(r.leak_diameter_m)),
                        "discharge_coeff": float(r.discharge_coeff) if r.discharge_coeff is not None else None,
                    }
                    for r in scenario.leak_records
                ],
            },
            "outputs": {
                "sensors_csv": sensors_path,
                "sensors_gt_csv": sensors_gt_path,
                "leak_flow_m3h_csv": leak_path,
            },
            "notes": {
                "leak_demand_unit_in_results": "m^3/s",
                "saved_leak_flow_unit": "m^3/h",
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return {
            "scenario_dir": str(out_base),
            "sensors_path": sensors_path,
            "sensors_gt_path": sensors_gt_path,
            "leak_path": leak_path,
            "meta_path": meta_path,
        }


if __name__ == "__main__":
    import time
    from datetime import datetime, timedelta
    
    cfg = SimConfig.from_yaml("configs/sim_LTA.yaml")
    start_datetime, peak_datetime, end_datetime = datetime(2018, 1, 2), datetime(2018, 1, 2), datetime(2018, 1, 9)
    
    scenario = LeakScenario(
        scenario_id="p193_single",
        start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M"),
        duration_days=(end_datetime - start_datetime).days,
        leak_records=[
            LeakRecord(
                leak_id="1",
                pipe_id="p193",
                leak_type="incipient",
                start_datetime=start_datetime.strftime("%Y-%m-%d %H:%M"),
                peak_datetime=peak_datetime.strftime("%Y-%m-%d %H:%M"),
                end_datetime=end_datetime.strftime("%Y-%m-%d %H:%M"),
                leak_diameter_m=0.01239,
            )
        ],
    )
    
    start = time.time()
    ex = LeakSimExecutor(cfg)
    ex.run_scenario(scenario, ramp_update_minutes=15)
    end = time.time()
    print("运行时间：", end - start, "秒")
    
    # import matplotlib.pyplot as plt
    # import matplotlib.dates as mdates
    # leak_df = pd.read_csv("data/simulate/L-TOWN-A/p193_single/leak_flow_m3h.csv", parse_dates=["datetime"], index_col=["datetime"], dtype=float)
    # vals = leak_df.loc[
    #     "2019-05-18 00:00:00":"2019-06-09 00:00:00",
    #     "p193"
    # ]
    # fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    # ax.plot(vals.index, vals.values, linewidth=2)
    # ax.set_title("Leak flow p193")
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Flow")

    # # 关键：更智能的日期刻度（自动选取合适的间隔）
    # locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
    # formatter = mdates.ConciseDateFormatter(locator)
    # ax.xaxis.set_major_locator(locator)
    # ax.xaxis.set_major_formatter(formatter)

    # ax.grid(True, alpha=0.3)
    # fig.tight_layout()

    # fig.savefig("test.png", dpi=300, bbox_inches="tight")
    # plt.close(fig)
    
    
    