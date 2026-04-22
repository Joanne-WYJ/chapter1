from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .dynamics import AgentState, attitude_error, project_to_so3, so3_exp


@dataclass
class TranslationalObserverState:
    d_hat_bar: np.ndarray
    p_hat: np.ndarray
    v_hat: np.ndarray
    d_hat_turb: np.ndarray
    xi_turb_hat: np.ndarray
    d_hat_local: np.ndarray
    r_bar_lpf: np.ndarray
    r_local_lpf: np.ndarray


@dataclass
class RotationalObserverState:
    R_hat: np.ndarray
    Omega_hat: np.ndarray
    d_hat_omega: np.ndarray


@dataclass
class ObserverBundle:
    trans: list[TranslationalObserverState]
    rot: list[RotationalObserverState]


def init_observers(states: list[AgentState]) -> ObserverBundle:
    trans = []
    rot = []
    for st in states:
        trans.append(
            TranslationalObserverState(
                d_hat_bar=np.zeros(3),
                p_hat=st.p.copy(),
                v_hat=st.v.copy(),
                d_hat_turb=np.zeros(3),
                xi_turb_hat=np.zeros(3),
                d_hat_local=np.zeros(3),
                r_bar_lpf=np.zeros(3),
                r_local_lpf=np.zeros(3),
            )
        )
        rot.append(
            RotationalObserverState(
                R_hat=st.R.copy(),
                Omega_hat=st.Omega.copy(),
                d_hat_omega=np.zeros(3),
            )
        )
    return ObserverBundle(trans=trans, rot=rot)


def update_distributed_low_frequency(
    obs: list[TranslationalObserverState],
    common_signal: np.ndarray,
    adjacency: np.ndarray,
    dt: float,
    alpha_c: float,
    beta_c: float,
    gain_p: float | None = None,
    gain_v: float | None = None,
) -> None:
    """Track the common low-frequency mode using a consensus-coupled tracker.

    gain_p/gain_v are retained for backward compatibility with older callers.
    """
    del gain_p, gain_v
    n = len(obs)
    target = np.asarray(common_signal, dtype=float)

    for o in obs:
        o.r_bar_lpf = o.r_bar_lpf + dt * alpha_c * (target - o.r_bar_lpf)

    current = np.stack([o.d_hat_bar for o in obs], axis=0)
    lp_targets = np.stack([o.r_bar_lpf for o in obs], axis=0)
    next_vals = np.zeros_like(current)
    for i in range(n):
        coupling = np.zeros(3)
        for j in range(n):
            if adjacency[i, j] > 0:
                coupling += adjacency[i, j] * (current[i] - current[j])
        next_vals[i] = current[i] + dt * (-alpha_c * (current[i] - lp_targets[i]) - beta_c * coupling)
    for i in range(n):
        obs[i].d_hat_bar = np.clip(next_vals[i], -20.0, 20.0)


def update_turbulence_modeled_observer(
    obs: TranslationalObserverState,
    local_signal: np.ndarray,
    dt: float,
    turbulence_lambda: np.ndarray,
    turbulence_c: np.ndarray,
    drag_trans: np.ndarray,
    omega_turb: float,
) -> None:
    """Partially model turbulence instead of leaving it entirely to the lumped ESO.

    The paper models turbulence via xi_dot = -Lambda xi + Gamma nu, w_tur = C xi.
    The corresponding translational turbulence disturbance is approximately
        d_turb = -D C xi.
    Here we estimate xi_hat using a Luenberger-like correction driven by the local
    residual channel. Any mismatch from building flow / shear-error remains for the
    local ESO to absorb.
    """
    local_signal = np.asarray(local_signal, dtype=float)
    lam = np.asarray(turbulence_lambda, dtype=float)
    B_t = -(drag_trans @ turbulence_c)

    # Use a stable innovation on the disturbance residual directly.
    d_hat_turb = B_t @ obs.xi_turb_hat
    innovation = local_signal - d_hat_turb

    # Map disturbance innovation back to xi-coordinates when possible.
    try:
        B_inv = np.linalg.inv(B_t)
        xi_innovation = B_inv @ innovation
    except np.linalg.LinAlgError:
        xi_innovation = innovation.copy()

    Lt = np.diag(np.full(3, float(omega_turb)))
    xi_dot = -np.diag(lam) @ obs.xi_turb_hat + Lt @ xi_innovation
    obs.xi_turb_hat = np.clip(obs.xi_turb_hat + dt * xi_dot, -20.0, 20.0)
    obs.d_hat_turb = np.clip(B_t @ obs.xi_turb_hat, -20.0, 20.0)


def update_local_translational_eso(
    obs: TranslationalObserverState,
    state: AgentState,
    thrust: float,
    mass: float,
    gravity: float,
    dt: float,
    omega_h: float,
    local_signal: np.ndarray,
) -> None:
    """Track only the non-common, non-turbulence residual channel.

    local_signal should already have the modeled turbulence contribution removed.
    A lightweight high-pass structure is kept so that the lumped residual observer
    focuses on fast building-flow / shear-error leftovers rather than re-absorbing
    the modeled turbulence channel.
    """
    local_signal = np.asarray(local_signal, dtype=float)
    lp_bw = max(0.20 * omega_h, 0.4)
    obs.r_local_lpf = obs.r_local_lpf + dt * lp_bw * (local_signal - obs.r_local_lpf)
    local_hp = local_signal - obs.r_local_lpf

    l1 = 2.0 * omega_h
    l2 = 1.5 * omega_h**2
    p_err = state.p - obs.p_hat
    e3 = np.array([0.0, 0.0, 1.0])
    total_hat = obs.d_hat_bar + obs.d_hat_turb + obs.d_hat_local
    p_hat_dot = obs.v_hat + l1 * p_err
    v_hat_dot = -gravity * e3 + (thrust / mass) * (state.R @ e3) + total_hat / mass + l2 * p_err

    # Residual-lumped observer: only the leftover after removing common+turbulence.
    d_hat_local_dot = omega_h * (local_hp - obs.d_hat_local) + 0.10 * omega_h**2 * p_err

    obs.p_hat = obs.p_hat + dt * p_hat_dot
    obs.v_hat = np.clip(obs.v_hat + dt * v_hat_dot, -20.0, 20.0)
    obs.d_hat_local = np.clip(obs.d_hat_local + dt * d_hat_local_dot, -20.0, 20.0)


def enforce_zero_sum_local(obs: list[TranslationalObserverState]) -> None:
    if not obs:
        return
    mean_local = np.mean(np.stack([o.d_hat_local for o in obs], axis=0), axis=0)
    for o in obs:
        o.d_hat_local = o.d_hat_local - mean_local


def update_rotational_auxiliary_observer(
    obs: RotationalObserverState,
    state: AgentState,
    torque: np.ndarray,
    inertia: np.ndarray,
    dt: float,
    omega_obs: float,
) -> None:
    invJ = np.linalg.inv(inertia)
    l2 = 2.5 * omega_obs
    l3 = omega_obs**2
    _, eR_obs = attitude_error(state.R, obs.R_hat)
    eOmega = state.Omega - obs.Omega_hat
    Omega_hat_dot = invJ @ (
        torque
        - np.cross(obs.Omega_hat, inertia @ obs.Omega_hat)
        + obs.d_hat_omega
        + inertia @ (l2 * eR_obs)
        + 0.8 * eOmega
    )
    d_hat_dot = inertia @ (l3 * eR_obs)
    obs.Omega_hat = np.clip(obs.Omega_hat + dt * Omega_hat_dot, -15.0, 15.0)
    obs.R_hat = project_to_so3(obs.R_hat @ so3_exp(obs.Omega_hat * dt))
    obs.d_hat_omega = np.clip(obs.d_hat_omega + dt * d_hat_dot, -5.0, 5.0)


def translational_total_estimate(obs: TranslationalObserverState) -> np.ndarray:
    return obs.d_hat_bar + obs.d_hat_turb + obs.d_hat_local
