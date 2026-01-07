from __future__ import annotations
from typing import List, Dict, Tuple, Literal, Optional, Union
from datetime import datetime
import os, re, pathlib, yaml
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# ---------- dist 定义 ----------
class NormalDist(BaseModel):
    name: Literal["normal"]
    mu: float
    sigma: float

class TruncNormalDist(BaseModel):
    name: Literal["trunc_normal"]
    mu: float
    sigma: float
    clip: Tuple[float, float]

class UniformDist(BaseModel):
    name: Literal["uniform"]
    low: float
    high: float

class ConstantDist(BaseModel):
    name: Literal["constant"]
    value: float

AnyDist = Union[NormalDist, TruncNormalDist, UniformDist, ConstantDist]

# ---------- 元数据 ----------
class Meta(BaseModel):
    version: int
    random_seed: int

class Paths(BaseModel):
    inp_path: str
    output_dir: str

# ---------- 仿真参数设置 ----------
class Unbalanced(BaseModel):
    mode: Literal["Continue", "Stop", "Terminate"]
    trials: int

class Solver(BaseModel):
    headloss: Literal["Hazen-Williams", "Darcy-Weisbach", "Chezy-Manning"]
    accuracy: float
    trials: int
    unbalanced: Unbalanced

class Simulation(BaseModel):
    start_datetime: datetime
    duration_days: int
    warmup_days: int
    timestep_min: int
    solver: Solver

# ---------- 传感器列表 ----------
class Sensors(BaseModel):
    pressure_node_ids: List[str]

class NoiseSwitches(BaseModel):
    structure: bool
    pattern_profile: bool
    demand: bool
    sensors: bool

# ---------- 结构噪声 ----------
class StructureNoise(BaseModel):
    diameters: Dict[str, AnyDist]
    roughness: Dict[str, AnyDist]
    minor_loss: Dict[str, AnyDist]
    prv_setpoint: Dict[str, AnyDist]

# ---------- 用水模式 ----------
class WeekendFactor(BaseModel):
    dist: ConstantDist

class GlobalScale(BaseModel):
    dist: UniformDist
    refresh_every_minutes: int

class PatternProfile(BaseModel):
    weekend_factor: WeekendFactor
    global_scale: GlobalScale

# ---------- 节点级需求噪声 ----------
class NodeMultiplier(BaseModel):
    dist: TruncNormalDist
    refresh_every_minutes: int

class SpatialSmoothing(BaseModel):
    hops: int
    mix_weight: float

class DemandNoise(BaseModel):
    node_multiplier: NodeMultiplier
    spatial_smoothing: SpatialSmoothing

# ---------- 传感器测量噪声 ----------
class MultiNoiseRel(BaseModel):
    dist: TruncNormalDist

class SensorsNoise(BaseModel):
    multiplicative_noise_rel: MultiNoiseRel
    quantization_head_m: float = Field(..., ge=0.0)
    missing_rate: float = Field(..., ge=0.0, le=1.0)


# ---------- 顶层 Config ----------
class SimConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    meta: Meta
    paths: Paths
    simulation: Simulation
    sensors: Sensors
    noise_switches: NoiseSwitches
    structure_noise: StructureNoise
    pattern_profile: PatternProfile
    demand_noise: DemandNoise
    sensors_noise: SensorsNoise


    @classmethod
    def from_yaml(cls, path: str | os.PathLike) -> "SimConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw = _expand_env_strings(raw)
        return cls.model_validate(raw)

    def to_yaml(self, path: str | os.PathLike, *, create_parents: bool = True) -> None:
        path = pathlib.Path(path)
        if create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(mode="python"),
                           f, sort_keys=False, allow_unicode=True)

def _expand_env_strings(obj):
    if isinstance(obj, dict):
        return {k: _expand_env_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_strings(x) for x in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj
