from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimConfig, formation_offsets, MISSION_LOW_ALT, MISSION_SPIRAL
from .dynamics import AgentState, attitude_error, skew


@dataclass
class ReferenceState:
    p_d: np.ndarray
    v_d: np.ndarray
    a_d: np.ndarray
    psi_d: float


@dataclass
class ControlOutput:
    a_c: np.ndarray
    f_d: np.ndarray
    R_d: np.ndarray
    T: float
    tau: np.ndarray
    e_p: np.ndarray
    e_v: np.ndarray
    eps_v_hat: np.ndarray
    psi_att: float


def smoothstep_segment(p0: np.ndarray, p1: np.ndarray, tau: float, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if duration <= 1e-9:
        return p1.copy(), np.zeros_like(p1), np.zeros_like(p1)
    s = np.clip(tau / duration, 0.0, 1.0)
    pos = p0 + (3 * s**2 - 2 * s**3) * (p1 - p0)
    vel = (6 * s - 6 * s**2) * (p1 - p0) / duration
    acc = (6 - 12 * s) * (p1 - p0) / (duration**2)
    return pos, vel, acc


def mission_reference_centroid(t_rel: float, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg.mission_mode == MISSION_SPIRAL:
        radius = 4.0
        omega = 0.18
        climb_rate = 0.08
        p = np.array([radius * np.cos(omega * t_rel), radius * np.sin(omega * t_rel), cfg.phase.z_safe + climb_rate * t_rel])
        v = np.array([-radius * omega * np.sin(omega * t_rel), radius * omega * np.cos(omega * t_rel), climb_rate])
        a = np.array([-radius * omega**2 * np.cos(omega * t_rel), -radius * omega**2 * np.sin(omega * t_rel), 0.0])
        return p, v, a
    # low altitude mission: vertical rise then translation through buildings
    if t_rel < 8.0:
        return smoothstep_segment(
            np.array([0.0, 0.0, cfg.phase.z_safe]),
            np.array([0.0, 0.0, cfg.phase.z_safe + 5.0]),
            t_rel,
            8.0,
        )
    t2 = t_rel - 8.0
    p = np.array([1.2 * t2, 1.5 * np.sin(0.18 * t2), cfg.phase.z_safe + 5.0 + 0.8 * np.sin(0.05 * t2)])
    v = np.array([1.2, 1.5 * 0.18 * np.cos(0.18 * t2), 0.8 * 0.05 * np.cos(0.05 * t2)])
    a = np.array([0.0, -1.5 * 0.18**2 * np.sin(0.18 * t2), -0.8 * 0.05**2 * np.sin(0.05 * t2)])
    return p, v, a


def reference_for_agent(
    agent_idx: int,
    t: float,
    cfg: SimConfig,
    init_positions: np.ndarray,
    offsets: np.ndarray,
    centroid0: np.ndarray,
) -> ReferenceState:
    phase = cfg.phase
    p_init = init_positions[agent_idx]
    takeoff_target = np.array([p_init[0], p_init[1], phase.z_safe])
    formation_target = centroid0 + offsets[agent_idx] + np.array([0.0, 0.0, phase.z_safe])
    if t < phase.takeoff_duration:
        p_d, v_d, a_d = smoothstep_segment(p_init, takeoff_target, t, phase.takeoff_duration)
        return ReferenceState(p_d, v_d, a_d, cfg.control.psi_d)
    if t < phase.takeoff_duration + phase.formation_duration:
        tau = t - phase.takeoff_duration
        p_d, v_d, a_d = smoothstep_segment(takeoff_target, formation_target, tau, phase.formation_duration)
        return ReferenceState(p_d, v_d, a_d, cfg.control.psi_d)
    tau = t - phase.takeoff_duration - phase.formation_duration
    centroid_p, centroid_v, centroid_a = mission_reference_centroid(tau, cfg)
    return ReferenceState(centroid_p + offsets[agent_idx], centroid_v, centroid_a, cfg.control.psi_d)


def outer_loop_control(
    i: int,
    state: AgentState,
    ref: ReferenceState,
    neighbor_vel_predictions: np.ndarray,
    adjacency: np.ndarray,
    d_hat_p: np.ndarray,
    wbar_hat: np.ndarray,
    cfg: SimConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_p = state.p - ref.p_d
    e_v = state.v - ref.v_d
    eps_v_hat = np.zeros(3)
    for j in range(adjacency.shape[0]):
        if adjacency[i, j] > 0:
            eps_v_hat += adjacency[i, j] * (state.v - neighbor_vel_predictions[j])
    c = cfg.control
    m = cfg.vehicle.mass
    a_ff = -(cfg.wind.drag_trans @ wbar_hat) / m
    a_c = ref.a_d - (c.kp_outer / m) * e_p - (c.kv_outer / m) * e_v - (c.keps_outer / m) * eps_v_hat - d_hat_p / m + a_ff
    a_norm = np.linalg.norm(a_c)
    if a_norm > 8.0:
        a_c = a_c * (8.0 / a_norm)
    return a_c, e_p, eps_v_hat


def desired_rotation_and_thrust(a_c: np.ndarray, d_hat_p: np.ndarray, psi_d: float, cfg: SimConfig) -> tuple[np.ndarray, np.ndarray, float]:
    e3 = np.array([0.0, 0.0, 1.0])
    m = cfg.vehicle.mass
    f_d = m * (a_c + cfg.vehicle.gravity * e3) - d_hat_p
    f_norm = np.linalg.norm(f_d)
    if f_norm < 1e-6:
        f_d = np.array([0.0, 0.0, m * cfg.vehicle.gravity])
        f_norm = np.linalg.norm(f_d)
    b3d = f_d / f_norm
    b1c = np.array([np.cos(psi_d), np.sin(psi_d), 0.0])
    cross = np.cross(b3d, b1c)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-6:
        b1c = np.array([1.0, 0.0, 0.0])
        cross = np.cross(b3d, b1c)
        cross_norm = max(np.linalg.norm(cross), 1e-6)
    b2d = cross / cross_norm
    b1d = np.cross(b2d, b3d)
    R_d = np.column_stack([b1d, b2d, b3d])
    return f_d, R_d, f_norm


def inner_loop_control(
    state: AgentState,
    R_d: np.ndarray,
    d_hat_omega: np.ndarray,
    cfg: SimConfig,
    Omega_d: np.ndarray | None = None,
    Omega_d_dot: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    if Omega_d is None:
        Omega_d = np.zeros(3)
    if Omega_d_dot is None:
        Omega_d_dot = np.zeros(3)
    psi_att, e_R = attitude_error(state.R, R_d)
    e_Omega = state.Omega - state.R.T @ R_d @ Omega_d
    feedforward = cfg.vehicle.inertia @ (
        skew(state.Omega) @ (state.R.T @ R_d @ Omega_d) + state.R.T @ R_d @ Omega_d_dot
    )
    tau = -cfg.control.kR * e_R - cfg.control.kOmega * e_Omega - d_hat_omega + feedforward
    tau = np.clip(tau, -cfg.vehicle.max_torque, cfg.vehicle.max_torque)
    return tau, psi_att


def full_control_step(
    i: int,
    state: AgentState,
    ref: ReferenceState,
    neighbor_vel_predictions: np.ndarray,
    adjacency: np.ndarray,
    d_hat_p: np.ndarray,
    d_hat_omega: np.ndarray,
    wbar_hat: np.ndarray,
    cfg: SimConfig,
) -> ControlOutput:
    a_c, e_p, eps_v_hat = outer_loop_control(i, state, ref, neighbor_vel_predictions, adjacency, d_hat_p, wbar_hat, cfg)
    f_d, R_d, _ = desired_rotation_and_thrust(a_c, d_hat_p, ref.psi_d, cfg)
    T = float(np.clip(f_d.T @ (state.R @ np.array([0.0, 0.0, 1.0])), cfg.vehicle.min_thrust, cfg.vehicle.max_thrust))
    tau, psi_att = inner_loop_control(state, R_d, d_hat_omega, cfg)
    e_v = state.v - ref.v_d
    return ControlOutput(a_c=a_c, f_d=f_d, R_d=R_d, T=T, tau=tau, e_p=e_p, e_v=e_v, eps_v_hat=eps_v_hat, psi_att=psi_att)
