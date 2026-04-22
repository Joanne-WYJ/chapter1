from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np


COMM_DYNAMIC = "dynamic"
COMM_STATIC = "static"
COMM_PERIODIC = "periodic"
MISSION_SPIRAL = "spiral"
MISSION_LOW_ALT = "low_altitude"


@dataclass
class PhaseConfig:
    takeoff_duration: float = 6.0
    formation_duration: float = 8.0
    mission_duration: float = 26.0
    z_safe: float = 6.0


@dataclass
class VehicleConfig:
    mass: float = 1.2
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.02, 0.02, 0.035]))
    gravity: float = 9.81
    max_thrust: float = 25.0
    min_thrust: float = 0.5
    max_torque: float = 2.5


@dataclass
class ControlConfig:
    kp_outer: float = 2.5
    kv_outer: float = 2.2
    keps_outer: float = 0.8
    kR: float = 7.0
    kOmega: float = 1.8
    psi_d: float = 0.0


@dataclass
class WindConfig:
    enable_mean: bool = True
    enable_shear: bool = True
    enable_turbulence: bool = True
    enable_building: bool = True
    mean_wind_0: np.ndarray = field(default_factory=lambda: np.array([2.5, 0.3, 0.0]))
    mean_wind_tau: float = 120.0
    mean_wind_drive: np.ndarray = field(default_factory=lambda: np.array([0.3, 0.08, 0.0]))
    shear_ref_height: float = 6.0
    shear_power_ref_speed: float = 1.2
    shear_power_alpha: float = 0.18
    shear_ref_speed_vertical: float = 0.06
    shear_slope_vertical: float = 0.015
    shear_time_scale: float = 80.0
    turbulence_lambda: np.ndarray = field(default_factory=lambda: np.array([1.2, 1.0, 1.8]))
    turbulence_gamma: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 1.2]))
    turbulence_c: np.ndarray = field(default_factory=lambda: np.eye(3))
    turbulence_noise_std: float = 1.0
    drag_trans: np.ndarray = field(default_factory=lambda: np.diag([0.7, 0.7, 0.6]))
    drag_rot: np.ndarray = field(default_factory=lambda: np.diag([0.08, 0.08, 0.06]))
    buildings: tuple[dict, ...] = field(default_factory=lambda: (
        {
            "center": np.array([6.0, 0.0, 5.0]),
            "radius": 4.0,
            "height": 8.0,
            "deflect_gain": 0.55,
            "shear_gain": 0.9,
            "wake_gain": 0.7,
            "wake_freq": 1.2,
        },
        {
            "center": np.array([12.0, -3.0, 5.0]),
            "radius": 3.5,
            "height": 7.0,
            "deflect_gain": 0.4,
            "shear_gain": 0.7,
            "wake_gain": 0.5,
            "wake_freq": 1.6,
        },
    ))


@dataclass
class ObserverConfig:
    alpha_c: float = 2.5
    beta_c: float = 2.0
    F_c_gain_p: float = 0.12
    F_c_gain_v: float = 0.18
    omega_turb: float = 3.5
    omega_h: float = 8.0
    omega_att_obs: float = 10.0


@dataclass
class TriggerConfig:
    strategy: Literal["dynamic", "static", "periodic"] = COMM_DYNAMIC
    sigma0: float = 0.18
    sigma_static: float = 0.18
    delta0: float = 5e-4
    periodic_interval: float = 0.25
    dbar_l: float = 4.0


@dataclass
class SimConfig:
    n_agents: int = 5
    dt: float = 0.01
    total_time: float = 40.0
    seed: int = 7
    quick_mode: bool = False
    mission_mode: Literal["spiral", "low_altitude"] = MISSION_LOW_ALT
    results_dir: str = "results"
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)

    def __post_init__(self) -> None:
        if self.quick_mode:
            self.dt = 0.02
            self.total_time = 16.0
            self.phase.takeoff_duration = 3.0
            self.phase.formation_duration = 4.0
            self.phase.mission_duration = self.total_time - self.phase.takeoff_duration - self.phase.formation_duration
            self.trigger.periodic_interval = 0.4
        else:
            self.phase.mission_duration = self.total_time - self.phase.takeoff_duration - self.phase.formation_duration
        self.results_path = Path(self.results_dir)
        self.results_path.mkdir(parents=True, exist_ok=True)


def adjacency_matrix_five() -> np.ndarray:
    return np.array(
        [
            [0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0],
        ],
        dtype=float,
    )


def laplacian_from_adjacency(adj: np.ndarray) -> np.ndarray:
    deg = np.diag(np.sum(adj, axis=1))
    return deg - adj


def formation_offsets(n_agents: int, radius: float = 2.0, shape: str = "pentagon") -> np.ndarray:
    if shape == "pentagon":
        angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
        offsets = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], axis=1)
        return offsets
    if shape == "v":
        offsets = []
        for i in range(n_agents):
            branch = -1 if i % 2 == 0 else 1
            level = i // 2
            offsets.append(np.array([1.5 * level, branch * 1.2 * (level + 1), 0.0]))
        offsets = np.array(offsets)
        offsets -= offsets.mean(axis=0)
        return offsets
    if shape == "star":
        angles = np.linspace(0, 2 * np.pi, n_agents, endpoint=False)
        radii = np.array([radius, 0.6 * radius, radius, 0.6 * radius, radius])[:n_agents]
        offsets = np.stack([radii * np.cos(angles), radii * np.sin(angles), np.zeros_like(angles)], axis=1)
        return offsets
    raise ValueError(f"Unsupported formation shape: {shape}")


def build_default_config(
    *,
    quick_mode: bool = False,
    strategy: str = COMM_DYNAMIC,
    mission_mode: str = MISSION_LOW_ALT,
    seed: int = 7,
) -> SimConfig:
    cfg = SimConfig(quick_mode=quick_mode, seed=seed, mission_mode=mission_mode)
    cfg.trigger.strategy = strategy
    return cfg
