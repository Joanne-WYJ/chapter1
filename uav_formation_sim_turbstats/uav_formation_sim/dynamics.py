from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm


@dataclass
class AgentState:
    p: np.ndarray
    v: np.ndarray
    R: np.ndarray
    Omega: np.ndarray

    def copy(self) -> "AgentState":
        return AgentState(self.p.copy(), self.v.copy(), self.R.copy(), self.Omega.copy())


def skew(w: np.ndarray) -> np.ndarray:
    return np.array(
        [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]],
        dtype=float,
    )


def vee(M: np.ndarray) -> np.ndarray:
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)


def so3_exp(w: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3) + skew(w)
    W = skew(w / theta)
    return np.eye(3) + np.sin(theta) * W + (1.0 - np.cos(theta)) * (W @ W)


def project_to_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R)
    Rproj = u @ vt
    if np.linalg.det(Rproj) < 0:
        u[:, -1] *= -1
        Rproj = u @ vt
    return Rproj


def axis_angle_to_rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    return so3_exp(axis * angle)


def random_rotation_small(rng: np.random.Generator, angle_std: float = 0.08) -> np.ndarray:
    axis = rng.normal(size=3)
    angle = rng.normal(scale=angle_std)
    return axis_angle_to_rot(axis, angle)


def attitude_error(R: np.ndarray, Rd: np.ndarray) -> tuple[float, np.ndarray]:
    psi = 0.5 * np.trace(np.eye(3) - Rd.T @ R)
    eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
    return float(psi), eR


def rk4_step_agent(
    state: AgentState,
    thrust: float,
    torque: np.ndarray,
    d_p: np.ndarray,
    d_omega: np.ndarray,
    mass: float,
    inertia: np.ndarray,
    gravity: float,
    dt: float,
) -> AgentState:
    invJ = np.linalg.inv(inertia)
    e3 = np.array([0.0, 0.0, 1.0])

    def f(x: AgentState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p_dot = x.v
        v_dot = -gravity * e3 + (thrust / mass) * (x.R @ e3) + d_p / mass
        R_dot = x.R @ skew(x.Omega)
        omega_dot = invJ @ (torque - np.cross(x.Omega, inertia @ x.Omega) + d_omega)
        return p_dot, v_dot, R_dot, omega_dot

    def add_state(x: AgentState, k, scale: float) -> AgentState:
        p_dot, v_dot, R_dot, omega_dot = k
        return AgentState(
            p=x.p + scale * p_dot,
            v=x.v + scale * v_dot,
            R=x.R + scale * R_dot,
            Omega=x.Omega + scale * omega_dot,
        )

    k1 = f(state)
    k2 = f(add_state(state, k1, 0.5 * dt))
    k3 = f(add_state(state, k2, 0.5 * dt))
    k4 = f(add_state(state, k3, dt))

    p = state.p + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    v = state.v + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    R = state.R + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    Omega = state.Omega + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
    v = np.clip(v, -25.0, 25.0)
    Omega = np.clip(Omega, -20.0, 20.0)
    R = project_to_so3(R)
    return AgentState(p=p, v=v, R=R, Omega=Omega)


def initial_states_random(n_agents: int, rng: np.random.Generator) -> list[AgentState]:
    states = []
    for _ in range(n_agents):
        p = np.array([
            rng.uniform(-3.0, 3.0),
            rng.uniform(-3.0, 3.0),
            rng.uniform(0.0, 0.5),
        ])
        v = rng.normal(scale=0.15, size=3)
        R = random_rotation_small(rng)
        Omega = rng.normal(scale=0.05, size=3)
        states.append(AgentState(p=p, v=v, R=R, Omega=Omega))
    return states


def block_matrix_A_nom(kp_over_m: float, kv_over_m: float, keps_over_m: float, lambda2: float) -> np.ndarray:
    A = np.zeros((6, 6))
    A[:3, 3:] = np.eye(3)
    A[3:, :3] = -kp_over_m * np.eye(3)
    A[3:, 3:] = -(kv_over_m + lambda2 * keps_over_m) * np.eye(3)
    return A


def predictor_step(z_hat: np.ndarray, A_nom: np.ndarray, dt: float) -> np.ndarray:
    Aexp = expm(A_nom * dt)
    return Aexp @ z_hat
