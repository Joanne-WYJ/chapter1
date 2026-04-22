from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.switch_backend("Agg")


LITERATURE_NRMSE_BAND = (17.0, 25.0)


def _ensure(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)


def plot_trajectories(logs: dict, results_dir: Path, title_suffix: str = "") -> None:
    _ensure(results_dir)
    pos = np.asarray(logs["positions"])  # (T,N,3)
    ref = np.asarray(logs["desired_positions"])
    times = np.asarray(logs["time"])
    phase_take = logs["phase_takeoff_end"]
    phase_form = logs["phase_formation_end"]
    idx_form = np.searchsorted(times, phase_form)
    n = pos.shape[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for i in range(n):
        ax.plot(pos[:idx_form, i, 0], pos[:idx_form, i, 1], pos[:idx_form, i, 2], alpha=0.5)
        ax.plot(pos[idx_form:, i, 0], pos[idx_form:, i, 1], pos[idx_form:, i, 2], linewidth=2)
        ax.plot(ref[:, i, 0], ref[:, i, 1], ref[:, i, 2], linestyle="--", alpha=0.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"3D trajectories {title_suffix}")
    fig.tight_layout()
    fig.savefig(results_dir / f"traj_3d_{title_suffix}.png", dpi=180)
    plt.close(fig)

    for plane, (a, b, labels) in {
        "xy": (0, 1, ("x [m]", "y [m]")),
        "xz": (0, 2, ("x [m]", "z [m]")),
    }.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in range(n):
            ax.plot(pos[:, i, a], pos[:, i, b], linewidth=1.8)
            ax.plot(ref[:, i, a], ref[:, i, b], linestyle="--", alpha=0.5)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(f"{plane.upper()} trajectories {title_suffix}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / f"traj_{plane}_{title_suffix}.png", dpi=180)
        plt.close(fig)


def plot_errors(logs: dict, results_dir: Path, title_suffix: str = "") -> None:
    _ensure(results_dir)
    t = np.asarray(logs["time"])
    metrics = [
        ("pos_err_norm", "Position error norm"),
        ("formation_err_norm", "Formation error norm"),
        ("psi_att", "Attitude error Psi"),
        ("vel_err_norm", "Velocity error norm"),
    ]
    for key, ylabel in metrics:
        arr = np.asarray(logs[key])
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(arr.shape[1]):
            ax.plot(t, arr[:, i], label=f"UAV {i+1}")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(results_dir / f"{key}_{title_suffix}.png", dpi=180)
        plt.close(fig)


def plot_disturbance_and_observer(logs: dict, results_dir: Path, title_suffix: str = "") -> None:
    _ensure(results_dir)
    t = np.asarray(logs["time"])
    keys = [
        ("d_p_true_norm", "True translational disturbance norm"),
        ("d_p_hat_norm", "Estimated translational disturbance norm"),
        ("d_bar_err_norm", "Low-frequency observer error norm"),
        ("d_turb_err_norm", "Turbulence-model observer error norm"),
        ("d_local_err_norm", "Local ESO residual error norm"),
        ("d_omega_err_norm", "Rotational observer error norm"),
    ]
    for key, ylabel in keys:
        if key not in logs:
            continue
        arr = np.asarray(logs[key])
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(arr.shape[1]):
            ax.plot(t, arr[:, i], label=f"UAV {i+1}")
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(results_dir / f"{key}_{title_suffix}.png", dpi=180)
        plt.close(fig)


def plot_triggering(logs: dict, results_dir: Path, title_suffix: str = "") -> None:
    _ensure(results_dir)
    event_times = logs["event_times"]
    n = len(event_times)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, events in enumerate(event_times):
        ax.scatter(events, np.full(len(events), i + 1), s=10)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("Agent index")
    ax.set_title(f"Trigger raster {title_suffix}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / f"trigger_raster_{title_suffix}.png", dpi=180)
    plt.close(fig)

    counts = np.asarray(logs["trigger_counts_final"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(1, n + 1), counts)
    ax.set_xlabel("Agent index")
    ax.set_ylabel("Trigger count")
    ax.set_title(f"Trigger counts {title_suffix}")
    fig.tight_layout()
    fig.savefig(results_dir / f"trigger_counts_{title_suffix}.png", dpi=180)
    plt.close(fig)


def plot_wind_field(logs: dict, results_dir: Path, title_suffix: str = "") -> None:
    _ensure(results_dir)
    shear_grid = logs["shear_grid"]
    build_grid = logs["building_grid"]
    composite_grid = logs["composite_grid"]

    for name, grid in [("shear", shear_grid), ("building", build_grid), ("composite", composite_grid)]:
        X, Z, U, W = grid
        fig, ax = plt.subplots(figsize=(8, 4))
        mag = np.sqrt(U**2 + W**2)
        scale = max(float(np.max(mag)), 1e-3) * 12.0
        ax.quiver(X, Z, U, W, angles="xy", scale_units="xy", scale=scale)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")
        ax.set_title(f"{name} wind field {title_suffix}")
        fig.tight_layout()
        fig.savefig(results_dir / f"wind_{name}_{title_suffix}.png", dpi=180)
        plt.close(fig)


def plot_comm_comparison(summary_rows: list[dict], results_dir: Path) -> None:
    _ensure(results_dir)
    if not summary_rows:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"{r['experiment']}\n{r['strategy']}" for r in summary_rows]
    vals = [r["total_trigger_count"] for r in summary_rows]
    ax.bar(labels, vals)
    ax.set_ylabel("Total communications")
    ax.set_title("Communication cost comparison")
    fig.tight_layout()
    fig.savefig(results_dir / "communication_comparison.png", dpi=180)
    plt.close(fig)


def plot_component_statistics(per_agent_df: pd.DataFrame, aggregate_df: pd.DataFrame, results_dir: Path, title_suffix: str = "") -> None:
    _ensure(results_dir)
    if aggregate_df.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    components = list(aggregate_df["component"])
    x = np.arange(len(components))

    axes[0].bar(x, aggregate_df["rmse"].values)
    axes[0].set_xticks(x, components, rotation=20)
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Aggregate RMSE")

    axes[1].bar(x, aggregate_df["mae"].values)
    axes[1].set_xticks(x, components, rotation=20)
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Aggregate MAE")

    nrmse_vals = np.nan_to_num(aggregate_df["nrmse_percent"].values, nan=0.0)
    axes[2].bar(x, nrmse_vals)
    axes[2].axhspan(LITERATURE_NRMSE_BAND[0], LITERATURE_NRMSE_BAND[1], alpha=0.15)
    axes[2].set_xticks(x, components, rotation=20)
    axes[2].set_ylabel("NRMSE [%]")
    axes[2].set_title("Aggregate NRMSE with proxy band")

    for idx, ax in enumerate(axes):
        ax.grid(True, axis="y", alpha=0.3)
    if aggregate_df.get("low_signal_for_nrmse", pd.Series([], dtype=bool)).any():
        axes[2].text(0.02, 0.95, "0-height bar on NRMSE means low-signal; see CSV flag.", transform=axes[2].transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(results_dir / f"component_error_stats_{title_suffix}.png", dpi=180)
    plt.close(fig)

    if per_agent_df.empty:
        return

    metrics = ["rmse", "mae", "nrmse_percent"]
    for component in per_agent_df["component"].unique():
        sub = per_agent_df[per_agent_df["component"] == component]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        agents = sub["agent"].tolist()
        xpos = np.arange(len(agents))
        for ax, metric in zip(axes, metrics):
            vals = np.nan_to_num(sub[metric].values, nan=0.0)
            ax.bar(xpos, vals)
            ax.set_xticks(xpos, agents, rotation=20)
            ax.set_title(f"{component} {metric}")
            ax.grid(True, axis="y", alpha=0.3)
            if metric == "nrmse_percent":
                ax.axhspan(LITERATURE_NRMSE_BAND[0], LITERATURE_NRMSE_BAND[1], alpha=0.15)
        fig.tight_layout()
        fig.savefig(results_dir / f"component_error_stats_{component}_{title_suffix}.png", dpi=180)
        plt.close(fig)


def plot_exp4_exp5_statistics_table(stats_df: pd.DataFrame, results_dir: Path) -> None:
    _ensure(results_dir)
    if stats_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(2.8, 0.45 * (len(stats_df) + 1))))
    ax.axis("off")
    disp = stats_df.copy()
    for c in ["rmse", "mae", "true_rms", "nrmse_percent"]:
        if c in disp.columns:
            disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    table = ax.table(cellText=disp.values, colLabels=disp.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    fig.tight_layout()
    fig.savefig(results_dir / "component_error_stats_Exp4_Exp5_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
