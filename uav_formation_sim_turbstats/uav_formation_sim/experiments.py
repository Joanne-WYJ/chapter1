from __future__ import annotations

from dataclasses import asdict
import copy
from pathlib import Path
from typing import Any
import json
import numpy as np
import pandas as pd

from .config import (
    COMM_DYNAMIC,
    COMM_PERIODIC,
    COMM_STATIC,
    MISSION_LOW_ALT,
    build_default_config,
    adjacency_matrix_five,
    formation_offsets,
    laplacian_from_adjacency,
    SimConfig,
)
from .controller import ControlOutput, full_control_step, reference_for_agent
from .dynamics import AgentState, block_matrix_A_nom, initial_states_random, rk4_step_agent
from .metrics import compute_summary, save_npz, compute_component_error_statistics, build_exp4_exp5_stats_table
from .observers import (
    enforce_zero_sum_local,
    init_observers,
    translational_total_estimate,
    update_distributed_low_frequency,
    update_turbulence_modeled_observer,
    update_local_translational_eso,
    update_rotational_auxiliary_observer,
)
from .plotting import plot_comm_comparison, plot_disturbance_and_observer, plot_errors, plot_trajectories, plot_triggering, plot_wind_field, plot_component_statistics, plot_exp4_exp5_statistics_table
from .trigger import CommunicationManager
from .wind import CompositeWindField


def _setup_experiment_flags(cfg: SimConfig, experiment_name: str) -> None:
    if experiment_name == "Exp1_no_wind":
        cfg.wind.enable_mean = False
        cfg.wind.enable_shear = False
        cfg.wind.enable_turbulence = False
        cfg.wind.enable_building = False
    elif experiment_name == "Exp2_mean":
        cfg.wind.enable_mean = True
        cfg.wind.enable_shear = False
        cfg.wind.enable_turbulence = False
        cfg.wind.enable_building = False
    elif experiment_name == "Exp3_mean_shear":
        cfg.wind.enable_mean = True
        cfg.wind.enable_shear = True
        cfg.wind.enable_turbulence = False
        cfg.wind.enable_building = False
    elif experiment_name == "Exp4_mean_shear_turb":
        cfg.wind.enable_mean = True
        cfg.wind.enable_shear = True
        cfg.wind.enable_turbulence = True
        cfg.wind.enable_building = False
    elif experiment_name in {"Exp5_full_wind", "Exp6_comm_compare"} or experiment_name.startswith("Scan_"):
        cfg.wind.enable_mean = True
        cfg.wind.enable_shear = True
        cfg.wind.enable_turbulence = True
        cfg.wind.enable_building = True
    else:
        raise ValueError(f"Unknown experiment {experiment_name}")


def _sample_wind_fields(wind: CompositeWindField, results_time: float) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    x = np.linspace(-2, 18, 20)
    z = np.linspace(0, 14, 15)
    X, Z = np.meshgrid(x, z)
    U_s, W_s = np.zeros_like(X), np.zeros_like(X)
    U_b, W_b = np.zeros_like(X), np.zeros_like(X)
    U_c, W_c = np.zeros_like(X), np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = np.array([X[i, j], 0.0, Z[i, j]])
            _, _, w_sh = wind.shear_components(p[2], results_time)
            w_b = wind.building_flow(p, results_time)
            w_all = wind.total_wind(p, 0, results_time)["total"]
            U_s[i, j], W_s[i, j] = w_sh[0], w_sh[2]
            U_b[i, j], W_b[i, j] = w_b[0], w_b[2]
            U_c[i, j], W_c[i, j] = w_all[0], w_all[2]
    return {
        "shear": (X, Z, U_s, W_s),
        "building": (X, Z, U_b, W_b),
        "composite": (X, Z, U_c, W_c),
    }


def run_single_experiment(cfg: SimConfig, experiment_name: str, strategy: str) -> tuple[dict[str, Any], pd.DataFrame]:
    cfg = copy.deepcopy(cfg)
    cfg.trigger.strategy = strategy
    _setup_experiment_flags(cfg, experiment_name)
    rng = np.random.default_rng(cfg.seed)

    adj = adjacency_matrix_five()
    lap = laplacian_from_adjacency(adj)
    eigvals = np.sort(np.real(np.linalg.eigvals(lap)))
    lambda2 = float(eigvals[1])
    offsets = formation_offsets(cfg.n_agents, shape="pentagon")

    states = initial_states_random(cfg.n_agents, rng)
    init_positions = np.stack([s.p.copy() for s in states], axis=0)
    centroid0 = init_positions[:, :2].mean(axis=0)
    centroid0 = np.array([centroid0[0], centroid0[1], 0.0])

    observers = init_observers(states)
    wind = CompositeWindField(cfg, cfg.n_agents, rng)
    A_nom = block_matrix_A_nom(
        cfg.control.kp_outer / cfg.vehicle.mass,
        cfg.control.kv_outer / cfg.vehicle.mass,
        cfg.control.keps_outer / cfg.vehicle.mass,
        lambda2,
    )
    comm = CommunicationManager(cfg.n_agents, A_nom, cfg.trigger, cfg.dt)

    refs0 = [reference_for_agent(i, 0.0, cfg, init_positions, offsets, centroid0) for i in range(cfg.n_agents)]
    z0 = np.stack([np.concatenate([states[i].p - refs0[i].p_d, states[i].v - refs0[i].v_d]) for i in range(cfg.n_agents)], axis=0)
    comm.reset(z0)

    n_steps = int(round(cfg.total_time / cfg.dt)) + 1
    times = np.linspace(0.0, cfg.total_time, n_steps)

    logs: dict[str, Any] = {
        "time": [],
        "positions": [],
        "desired_positions": [],
        "pos_err_norm": [],
        "vel_err_norm": [],
        "formation_err_norm": [],
        "psi_att": [],
        "d_p_true_norm": [],
        "d_p_hat_norm": [],
        "d_p_est_err_norm": [],
        "d_bar_err_norm": [],
        "d_turb_err_norm": [],
        "d_local_err_norm": [],
        "d_turb_true_norm": [],
        "d_local_true_norm": [],
        "d_omega_err_norm": [],
        "bar_share": [],
        "turb_share": [],
        "local_zero_sum_norm": [],
        "common_target_norm": [],
        "event_times": None,
    }

    prev_controls = [
        ControlOutput(
            a_c=np.zeros(3),
            f_d=np.array([0.0, 0.0, cfg.vehicle.mass * cfg.vehicle.gravity]),
            R_d=np.eye(3),
            T=cfg.vehicle.mass * cfg.vehicle.gravity,
            tau=np.zeros(3),
            e_p=np.zeros(3),
            e_v=np.zeros(3),
            eps_v_hat=np.zeros(3),
            psi_att=0.0,
        )
        for _ in range(cfg.n_agents)
    ]

    for t in times:
        wind.step(cfg.dt, t)
        refs = [reference_for_agent(i, t, cfg, init_positions, offsets, centroid0) for i in range(cfg.n_agents)]

        comm.step_predictors()
        comm_view = comm.comm_view()

        desired_positions = np.zeros((cfg.n_agents, 3))
        pos_err_norm = np.zeros(cfg.n_agents)
        vel_err_norm = np.zeros(cfg.n_agents)
        formation_err_norm = np.zeros(cfg.n_agents)
        psi_att = np.zeros(cfg.n_agents)
        d_p_true_norm = np.zeros(cfg.n_agents)
        d_p_hat_norm = np.zeros(cfg.n_agents)
        d_bar_err_norm = np.zeros(cfg.n_agents)
        d_turb_err_norm = np.zeros(cfg.n_agents)
        d_local_err_norm = np.zeros(cfg.n_agents)
        d_turb_true_norm = np.zeros(cfg.n_agents)
        d_local_true_norm = np.zeros(cfg.n_agents)
        d_omega_err_norm = np.zeros(cfg.n_agents)
        d_p_est_err_norm = np.zeros(cfg.n_agents)

        controls: list[ControlOutput] = []
        true_dp_all: list[np.ndarray] = []
        true_domega_all: list[np.ndarray] = []
        true_dp_turb_all: list[np.ndarray] = []
        wind_parts_all: list[dict[str, np.ndarray]] = []

        # Control computation using current observer states
        for i in range(cfg.n_agents):
            desired_positions[i] = refs[i].p_d
            neighbor_vel_predictions = np.zeros((cfg.n_agents, 3))
            for j in range(cfg.n_agents):
                refj = refs[j]
                neighbor_vel_predictions[j] = refj.v_d + comm_view[j, 3:]
            d_hat_p = translational_total_estimate(observers.trans[i])
            wbar_hat = wind.mean_wind()
            d_hat_omega = observers.rot[i].d_hat_omega.copy()
            control = full_control_step(i, states[i], refs[i], neighbor_vel_predictions, adj, d_hat_p, d_hat_omega, wbar_hat, cfg)
            controls.append(control)

            wind_parts = wind.total_wind(states[i].p, i, t)
            wind_parts_all.append(wind_parts)
            total_w = wind_parts["total"]
            d_p_true = -(cfg.wind.drag_trans @ total_w)
            d_p_turb_true = -(cfg.wind.drag_trans @ wind_parts["turbulence"])
            d_omega_true = -(cfg.wind.drag_rot @ (states[i].R.T @ total_w))
            true_dp_all.append(d_p_true)
            true_dp_turb_all.append(d_p_turb_true)
            true_domega_all.append(d_omega_true)

        # Use the paper's nominal common-mode disturbance target rather than the mean of the true total disturbance.
        common_target = np.mean(
            np.stack([wind.nominal_common_disturbance(refs[i].p_d[2], t) for i in range(cfg.n_agents)], axis=0),
            axis=0,
        )

        # Revised observer structure:
        # 1) distributed low-frequency common mode,
        # 2) turbulence-model observer using the known shaping-filter structure,
        # 3) local lumped ESO for the residual after removing common+turbulence.
        update_distributed_low_frequency(
            observers.trans,
            common_target,
            adj,
            cfg.dt,
            cfg.observer.alpha_c,
            cfg.observer.beta_c,
            cfg.observer.F_c_gain_p,
            cfg.observer.F_c_gain_v,
        )
        for i in range(cfg.n_agents):
            local_target = true_dp_all[i] - common_target
            update_turbulence_modeled_observer(
                observers.trans[i],
                local_target,
                cfg.dt,
                cfg.wind.turbulence_lambda,
                cfg.wind.turbulence_c,
                cfg.wind.drag_trans,
                cfg.observer.omega_turb,
            )
            local_residual_target = local_target - true_dp_turb_all[i]
            residual_signal = local_target - observers.trans[i].d_hat_turb
            update_local_translational_eso(
                observers.trans[i],
                states[i],
                controls[i].T,
                cfg.vehicle.mass,
                cfg.vehicle.gravity,
                cfg.dt,
                cfg.observer.omega_h,
                residual_signal,
            )
        enforce_zero_sum_local(observers.trans)

        for i in range(cfg.n_agents):
            update_rotational_auxiliary_observer(
                observers.rot[i],
                states[i],
                controls[i].tau,
                cfg.vehicle.inertia,
                cfg.dt,
                cfg.observer.omega_att_obs,
            )

        # Event triggering and diagnostics
        for i in range(cfg.n_agents):
            z_actual = np.concatenate([states[i].p - refs[i].p_d, states[i].v - refs[i].v_d])
            d_hat_local_norm = float(np.linalg.norm(observers.trans[i].d_hat_local))
            comm.maybe_trigger(i, t, z_actual, d_hat_local_norm)

            pos_err_norm[i] = float(np.linalg.norm(z_actual[:3]))
            vel_err_norm[i] = float(np.linalg.norm(z_actual[3:]))
            form_err = np.zeros(3)
            for j in range(cfg.n_agents):
                if adj[i, j] > 0:
                    form_err += (states[i].p - offsets[i]) - (states[j].p - offsets[j])
            formation_err_norm[i] = float(np.linalg.norm(form_err))
            psi_att[i] = controls[i].psi_att

            d_p_true_norm[i] = float(np.linalg.norm(true_dp_all[i]))
            d_hat_p = translational_total_estimate(observers.trans[i])
            d_p_hat_norm[i] = float(np.linalg.norm(d_hat_p))
            d_p_est_err_norm[i] = float(np.linalg.norm(d_hat_p - true_dp_all[i]))

            d_bar_err_norm[i] = float(np.linalg.norm(observers.trans[i].d_hat_bar - common_target))
            d_turb_true_norm[i] = float(np.linalg.norm(true_dp_turb_all[i]))
            d_turb_err_norm[i] = float(np.linalg.norm(observers.trans[i].d_hat_turb - true_dp_turb_all[i]))
            d_local_true = (true_dp_all[i] - common_target) - true_dp_turb_all[i]
            d_local_true_norm[i] = float(np.linalg.norm(d_local_true))
            d_local_err_norm[i] = float(np.linalg.norm(observers.trans[i].d_hat_local - d_local_true))
            d_omega_err_norm[i] = float(np.linalg.norm(observers.rot[i].d_hat_omega - true_domega_all[i]))

        # Dynamics propagation
        new_states = []
        for i in range(cfg.n_agents):
            new_state = rk4_step_agent(
                states[i],
                controls[i].T,
                controls[i].tau,
                true_dp_all[i],
                true_domega_all[i],
                cfg.vehicle.mass,
                cfg.vehicle.inertia,
                cfg.vehicle.gravity,
                cfg.dt,
            )
            new_states.append(new_state)
        states = new_states
        prev_controls = controls

        local_mean = np.mean(np.stack([o.d_hat_local for o in observers.trans], axis=0), axis=0)
        total_est = np.stack([translational_total_estimate(o) for o in observers.trans], axis=0)
        total_norm = np.linalg.norm(total_est, axis=1) + 1e-8
        bar_share = np.linalg.norm(np.stack([o.d_hat_bar for o in observers.trans], axis=0), axis=1) / total_norm
        turb_share = np.linalg.norm(np.stack([o.d_hat_turb for o in observers.trans], axis=0), axis=1) / total_norm

        logs["time"].append(t)
        logs["positions"].append(np.stack([s.p.copy() for s in states], axis=0))
        logs["desired_positions"].append(desired_positions.copy())
        logs["pos_err_norm"].append(pos_err_norm.copy())
        logs["vel_err_norm"].append(vel_err_norm.copy())
        logs["formation_err_norm"].append(formation_err_norm.copy())
        logs["psi_att"].append(psi_att.copy())
        logs["d_p_true_norm"].append(d_p_true_norm.copy())
        logs["d_p_hat_norm"].append(d_p_hat_norm.copy())
        logs["d_p_est_err_norm"].append(d_p_est_err_norm.copy())
        logs["d_bar_err_norm"].append(d_bar_err_norm.copy())
        logs["d_turb_err_norm"].append(d_turb_err_norm.copy())
        logs["d_local_err_norm"].append(d_local_err_norm.copy())
        logs["d_turb_true_norm"].append(d_turb_true_norm.copy())
        logs["d_local_true_norm"].append(d_local_true_norm.copy())
        logs["d_omega_err_norm"].append(d_omega_err_norm.copy())
        logs["bar_share"].append(bar_share.copy())
        logs["turb_share"].append(turb_share.copy())
        logs["local_zero_sum_norm"].append(float(np.linalg.norm(local_mean)))
        logs["common_target_norm"].append(float(np.linalg.norm(common_target)))

    logs["event_times"] = [ag.event_times for ag in comm.agents]
    logs["trigger_counts_final"] = comm.event_counts()
    logs["min_trigger_dt"] = comm.min_inter_event_time()
    logs["phase_takeoff_end"] = cfg.phase.takeoff_duration
    logs["phase_formation_end"] = cfg.phase.takeoff_duration + cfg.phase.formation_duration

    wind_grids = _sample_wind_fields(wind, results_time=float(times[-1]))
    logs["shear_grid"] = wind_grids["shear"]
    logs["building_grid"] = wind_grids["building"]
    logs["composite_grid"] = wind_grids["composite"]

    summary = compute_summary(logs, experiment_name, strategy, cfg.results_path)
    save_npz(logs, cfg.results_path / f"logs_{experiment_name}_{strategy}.npz")
    return logs, summary


def run_standard_suite(cfg: SimConfig) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    experiments = ["Exp1_no_wind", "Exp2_mean", "Exp3_mean_shear", "Exp4_mean_shear_turb", "Exp5_full_wind"]
    if cfg.quick_mode:
        experiments = ["Exp1_no_wind", "Exp5_full_wind"]
    for name in experiments:
        logs, summary = run_single_experiment(cfg, name, COMM_DYNAMIC)
        out_dir = cfg.results_path / f"{name}_{COMM_DYNAMIC}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_trajectories(logs, out_dir, f"{name}_{COMM_DYNAMIC}")
        plot_errors(logs, out_dir, f"{name}_{COMM_DYNAMIC}")
        plot_disturbance_and_observer(logs, out_dir, f"{name}_{COMM_DYNAMIC}")
        if name in {"Exp4_mean_shear_turb", "Exp5_full_wind"}:
            per_agent_stats, aggregate_stats = compute_component_error_statistics(logs, name, COMM_DYNAMIC, out_dir)
            plot_component_statistics(per_agent_stats, aggregate_stats, out_dir, f"{name}_{COMM_DYNAMIC}")
            logs["component_error_stats_aggregate"] = aggregate_stats.to_dict(orient="records")
        plot_triggering(logs, out_dir, f"{name}_{COMM_DYNAMIC}")
        plot_wind_field(logs, out_dir, f"{name}_{COMM_DYNAMIC}")
        all_rows.append(summary.iloc[0].to_dict())

    # Exp6 communication comparison on full wind
    comm_strategies = [COMM_PERIODIC, COMM_DYNAMIC] if cfg.quick_mode else [COMM_PERIODIC, COMM_DYNAMIC, COMM_STATIC]
    periodic_logs, periodic_summary = run_single_experiment(cfg, "Exp6_comm_compare", COMM_PERIODIC)
    baseline_comm = float(periodic_summary.iloc[0]["total_trigger_count"])

    periodic_summary.loc[:, "comm_saving_vs_periodic"] = 0.0
    periodic_summary.to_csv(cfg.results_path / f"summary_Exp6_comm_compare_{COMM_PERIODIC}.csv", index=False)
    out_dir = cfg.results_path / f"Exp6_comm_compare_{COMM_PERIODIC}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_trajectories(periodic_logs, out_dir, f"Exp6_comm_compare_{COMM_PERIODIC}")
    plot_errors(periodic_logs, out_dir, f"Exp6_comm_compare_{COMM_PERIODIC}")
    plot_disturbance_and_observer(periodic_logs, out_dir, f"Exp6_comm_compare_{COMM_PERIODIC}")
    plot_triggering(periodic_logs, out_dir, f"Exp6_comm_compare_{COMM_PERIODIC}")
    plot_wind_field(periodic_logs, out_dir, f"Exp6_comm_compare_{COMM_PERIODIC}")
    all_rows.append(periodic_summary.iloc[0].to_dict())

    for strategy in [s for s in comm_strategies if s != COMM_PERIODIC]:
        logs, summary = run_single_experiment(cfg, "Exp6_comm_compare", strategy)
        summary.loc[:, "comm_saving_vs_periodic"] = 1.0 - summary["total_trigger_count"] / baseline_comm
        summary.to_csv(cfg.results_path / f"summary_Exp6_comm_compare_{strategy}.csv", index=False)
        out_dir = cfg.results_path / f"Exp6_comm_compare_{strategy}"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_trajectories(logs, out_dir, f"Exp6_comm_compare_{strategy}")
        plot_errors(logs, out_dir, f"Exp6_comm_compare_{strategy}")
        plot_disturbance_and_observer(logs, out_dir, f"Exp6_comm_compare_{strategy}")
        plot_triggering(logs, out_dir, f"Exp6_comm_compare_{strategy}")
        plot_wind_field(logs, out_dir, f"Exp6_comm_compare_{strategy}")
        all_rows.append(summary.iloc[0].to_dict())

    # parameter scans
    omega_scan = [8.0] if cfg.quick_mode else [5.0, 8.0, 12.0]
    alpha_scan = [2.5] if cfg.quick_mode else [1.5, 2.5, 4.0]
    for omega_h in omega_scan:
        cfg_scan = copy.deepcopy(cfg)
        cfg_scan.observer.omega_h = omega_h
        logs, summary = run_single_experiment(cfg_scan, f"Scan_omega_h_{omega_h:g}", COMM_DYNAMIC)
        all_rows.append(summary.iloc[0].to_dict())
    for alpha_c in alpha_scan:
        cfg_scan = copy.deepcopy(cfg)
        cfg_scan.observer.alpha_c = alpha_c
        cfg_scan.observer.beta_c = 2.0 * alpha_c
        logs, summary = run_single_experiment(cfg_scan, f"Scan_alpha_c_{alpha_c:g}", COMM_DYNAMIC)
        all_rows.append(summary.iloc[0].to_dict())

    combined_rows = []
    for exp_name in ["Exp4_mean_shear_turb", "Exp5_full_wind"]:
        stats_path = cfg.results_path / f"{exp_name}_{COMM_DYNAMIC}" / f"component_error_stats_aggregate_{exp_name}_{COMM_DYNAMIC}.csv"
        if stats_path.exists():
            combined_rows.append(pd.read_csv(stats_path))
    if combined_rows:
        combined_df = pd.concat(combined_rows, ignore_index=True)
        build_exp4_exp5_stats_table(combined_df, cfg.results_path)
        plot_exp4_exp5_statistics_table(combined_df, cfg.results_path)

    df = pd.DataFrame(all_rows)
    df.to_csv(cfg.results_path / "summary_all.csv", index=False)
    plot_comm_comparison(all_rows, cfg.results_path)
    with open(cfg.results_path / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({"dt": cfg.dt, "total_time": cfg.total_time, "mission_mode": cfg.mission_mode}, f, indent=2)
    return df


def make_default_config(quick: bool = False) -> SimConfig:
    cfg = build_default_config(quick_mode=quick, strategy=COMM_DYNAMIC, mission_mode=MISSION_LOW_ALT)
    cfg.observer.beta_c = 2.0 * cfg.observer.alpha_c
    return cfg
