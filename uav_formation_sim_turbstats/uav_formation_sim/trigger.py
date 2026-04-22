from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import expm

from .config import COMM_DYNAMIC, COMM_PERIODIC, COMM_STATIC, TriggerConfig


@dataclass
class AgentCommState:
    last_broadcast_time: float = 0.0
    last_broadcast_z: np.ndarray = field(default_factory=lambda: np.zeros(6))
    predicted_z: np.ndarray = field(default_factory=lambda: np.zeros(6))
    event_times: list[float] = field(default_factory=list)


class CommunicationManager:
    def __init__(self, n_agents: int, A_nom: np.ndarray, cfg: TriggerConfig, dt: float):
        self.n_agents = n_agents
        self.A_nom = A_nom
        self.cfg = cfg
        self.dt = dt
        self.Aexp = expm(A_nom * dt)
        self.agents = [AgentCommState() for _ in range(n_agents)]

    def reset(self, z0_all: np.ndarray) -> None:
        for i, agent in enumerate(self.agents):
            agent.last_broadcast_time = 0.0
            agent.last_broadcast_z = z0_all[i].copy()
            agent.predicted_z = z0_all[i].copy()
            agent.event_times = [0.0]

    def step_predictors(self) -> None:
        for agent in self.agents:
            agent.predicted_z = self.Aexp @ agent.predicted_z

    def comm_view(self) -> np.ndarray:
        return np.stack([ag.predicted_z for ag in self.agents], axis=0)

    def maybe_trigger(self, i: int, t: float, z_actual: np.ndarray, d_hat_local_norm: float) -> bool:
        agent = self.agents[i]
        error = agent.predicted_z - z_actual
        err_norm_sq = float(error @ error)
        z_norm_sq = float(z_actual @ z_actual)
        if self.cfg.strategy == COMM_PERIODIC:
            if t - agent.last_broadcast_time >= self.cfg.periodic_interval - 1e-12:
                self._broadcast(agent, z_actual, t)
                return True
            return False
        if self.cfg.strategy == COMM_STATIC:
            threshold = self.cfg.sigma_static * z_norm_sq + self.cfg.delta0
            if err_norm_sq >= threshold:
                self._broadcast(agent, z_actual, t)
                return True
            return False
        sigma = self.cfg.sigma0 * np.exp(-np.linalg.norm(error)) * self.cfg.dbar_l / (self.cfg.dbar_l + d_hat_local_norm + 1e-9)
        threshold = sigma * z_norm_sq + self.cfg.delta0
        if err_norm_sq >= threshold:
            self._broadcast(agent, z_actual, t)
            return True
        return False

    def _broadcast(self, agent: AgentCommState, z_actual: np.ndarray, t: float) -> None:
        agent.last_broadcast_z = z_actual.copy()
        agent.predicted_z = z_actual.copy()
        agent.last_broadcast_time = t
        agent.event_times.append(float(t))

    def min_inter_event_time(self) -> float:
        mins = []
        for ag in self.agents:
            if len(ag.event_times) >= 2:
                diffs = np.diff(ag.event_times)
                mins.append(float(np.min(diffs)))
        return min(mins) if mins else np.nan

    def event_counts(self) -> list[int]:
        return [max(0, len(ag.event_times) - 1) for ag in self.agents]
