from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimConfig


@dataclass
class WindState:
    mean_wind: np.ndarray
    turbulence_states: np.ndarray  # (N, 3)


class CompositeWindField:
    def __init__(self, cfg: SimConfig, n_agents: int, rng: np.random.Generator):
        self.cfg = cfg
        self.n_agents = n_agents
        self.rng = rng
        self.state = WindState(
            mean_wind=cfg.wind.mean_wind_0.astype(float).copy(),
            turbulence_states=np.zeros((n_agents, 3), dtype=float),
        )

    def reset(self) -> None:
        self.state.mean_wind[:] = self.cfg.wind.mean_wind_0
        self.state.turbulence_states[:] = 0.0

    def step(self, dt: float, t: float) -> None:
        wcfg = self.cfg.wind
        if wcfg.enable_mean:
            drive = wcfg.mean_wind_drive * np.array([
                np.sin(0.01 * t),
                np.cos(0.013 * t),
                0.15 * np.sin(0.017 * t),
            ])
            self.state.mean_wind += dt * (-(self.state.mean_wind - wcfg.mean_wind_0) / wcfg.mean_wind_tau + drive)
        else:
            self.state.mean_wind[:] = 0.0

        if wcfg.enable_turbulence:
            lam = wcfg.turbulence_lambda
            gam = wcfg.turbulence_gamma
            noise = self.rng.normal(scale=wcfg.turbulence_noise_std, size=(self.n_agents, 3))
            self.state.turbulence_states += dt * (-self.state.turbulence_states * lam + noise * gam)
        else:
            self.state.turbulence_states[:] = 0.0

    def mean_wind(self) -> np.ndarray:
        return self.state.mean_wind.copy()

    def shear_components(self, z: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        wcfg = self.cfg.wind
        zr = wcfg.shear_ref_height
        alpha_t = wcfg.shear_power_alpha * (1.0 + 0.04 * np.sin(2 * np.pi * t / wcfg.shear_time_scale))
        u_ref = wcfg.shear_power_ref_speed * (1.0 + 0.08 * np.cos(2 * np.pi * t / (1.3 * wcfg.shear_time_scale)))
        wx_r = np.array([u_ref * (zr / max(zr, 1.0)) ** alpha_t, 0.0, wcfg.shear_ref_speed_vertical])
        S_r = np.array([
            alpha_t * max(u_ref, 1e-6) / max(zr, 1.0),
            0.0,
            wcfg.shear_slope_vertical,
        ])
        w_sh = wx_r + S_r * (z - zr)
        return wx_r, S_r, w_sh

    def turbulence(self, agent_idx: int) -> np.ndarray:
        if not self.cfg.wind.enable_turbulence:
            return np.zeros(3)
        return self.cfg.wind.turbulence_c @ self.state.turbulence_states[agent_idx]

    def building_flow(self, p: np.ndarray, t: float) -> np.ndarray:
        if not self.cfg.wind.enable_building:
            return np.zeros(3)
        w_mean = self.state.mean_wind
        mean_dir = w_mean.copy()
        norm = np.linalg.norm(mean_dir[:2])
        if norm < 1e-6:
            mean_dir[:2] = np.array([1.0, 0.0])
            norm = 1.0
        mean_dir_xy = mean_dir[:2] / norm
        perp_xy = np.array([-mean_dir_xy[1], mean_dir_xy[0]])
        total = np.zeros(3)
        for b in self.cfg.wind.buildings:
            c = b["center"]
            r = b["radius"]
            rel = p - c
            rho = np.exp(-np.dot(rel, rel) / (r**2))
            forward = np.dot(rel[:2], mean_dir_xy)
            lateral = np.dot(rel[:2], perp_xy)
            z_rel = p[2] - b["height"]
            # windward lateral deflection
            windward = np.exp(-((forward + 0.7 * r) ** 2 + lateral**2) / (0.8 * r**2))
            deflect = b["deflect_gain"] * windward * np.array([perp_xy[0], perp_xy[1], 0.0])
            # rooftop shear near building top
            rooftop = np.exp(-(forward**2 + lateral**2) / (r**2)) * np.exp(-(z_rel**2) / (1.2**2))
            roof_shear = b["shear_gain"] * rooftop * np.array([0.2, 0.0, np.sign(z_rel + 1e-6)])
            # lee wake behind building
            wake = np.exp(-((forward - r) ** 2) / (1.8 * r**2)) * np.exp(-(lateral**2) / (1.2 * r**2))
            vortex = 0.25 * np.sin(2 * np.pi * b["wake_freq"] * t)
            wake_flow = b["wake_gain"] * wake * np.array([-mean_dir_xy[0], -mean_dir_xy[1] + vortex, 0.08 * vortex])
            total += rho * (deflect + roof_shear + wake_flow)
        return total

    def total_wind(self, p: np.ndarray, agent_idx: int, t: float) -> dict[str, np.ndarray]:
        w_mean = self.mean_wind() if self.cfg.wind.enable_mean else np.zeros(3)
        _, _, w_sh = self.shear_components(p[2], t) if self.cfg.wind.enable_shear else (np.zeros(3), np.zeros(3), np.zeros(3))
        w_t = self.turbulence(agent_idx) if self.cfg.wind.enable_turbulence else np.zeros(3)
        w_b = self.building_flow(p, t) if self.cfg.wind.enable_building else np.zeros(3)
        total = w_mean + w_sh + w_t + w_b
        return {
            "mean": w_mean,
            "shear": w_sh,
            "turbulence": w_t,
            "building": w_b,
            "total": total,
        }

    def nominal_common_disturbance(self, z_desired: float, t: float) -> np.ndarray:
        wcfg = self.cfg.wind
        mean_w = self.mean_wind() if wcfg.enable_mean else np.zeros(3)
        w_sh_r, S_r, _ = self.shear_components(wcfg.shear_ref_height, t)
        if not wcfg.enable_shear:
            w_sh_r = np.zeros(3)
            S_r = np.zeros(3)
        return -wcfg.drag_trans @ (mean_w + w_sh_r + S_r * (z_desired - wcfg.shear_ref_height))
