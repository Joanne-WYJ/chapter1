from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


EPS = 1e-8
NRMSE_SIGNAL_FLOOR = 5e-2


def _safe_rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x))))


def _safe_mean_abs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(x)))


def compute_summary(logs: dict, experiment_name: str, strategy: str, results_dir: Path) -> pd.DataFrame:
    pos_err = np.asarray(logs["pos_err_norm"])
    form_err = np.asarray(logs["formation_err_norm"])
    psi_err = np.asarray(logs["psi_att"])
    d_err = np.asarray(logs["d_p_est_err_norm"])
    trigger_counts = np.asarray(logs["trigger_counts_final"])
    min_trigger_dt = float(logs.get("min_trigger_dt", np.nan))
    baseline_periodic = logs.get("periodic_baseline_comm", np.nan)
    total_comm = float(np.sum(trigger_counts))
    savings = np.nan
    if baseline_periodic is not None and np.isfinite(baseline_periodic) and baseline_periodic > 0:
        savings = 1.0 - total_comm / baseline_periodic
    row = {
        "experiment": experiment_name,
        "strategy": strategy,
        "final_mean_position_error": float(np.mean(pos_err[-1])),
        "max_formation_error": float(np.max(form_err)),
        "mean_attitude_error": float(np.mean(psi_err)),
        "mean_trans_dist_est_error": float(np.mean(d_err)),
        "total_trigger_count": total_comm,
        "min_trigger_interval": min_trigger_dt,
        "comm_saving_vs_periodic": savings,
    }
    df = pd.DataFrame([row])
    out = results_dir / f"summary_{experiment_name}_{strategy}.csv"
    df.to_csv(out, index=False)
    return df


def compute_component_error_statistics(logs: dict, experiment_name: str, strategy: str, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Literature-aligned statistics for d_local_err_norm and d_turb_err_norm.

    Three metric classes are reported:
    1) RMSE
    2) MAE
    3) NRMSE [%], normalized by RMS of the corresponding true component norm.
    """
    metric_specs = [
        ("d_local_err_norm", "d_local_true_norm", "local_residual"),
        ("d_turb_err_norm", "d_turb_true_norm", "turbulence_component"),
    ]

    per_agent_rows: list[dict] = []
    aggregate_rows: list[dict] = []

    for err_key, true_key, component_name in metric_specs:
        if err_key not in logs or true_key not in logs:
            continue
        err = np.asarray(logs[err_key], dtype=float)  # (T,N)
        true = np.asarray(logs[true_key], dtype=float)  # (T,N)
        n_agents = err.shape[1]

        for i in range(n_agents):
            e_i = err[:, i]
            t_i = true[:, i]
            rmse = _safe_rms(e_i)
            mae = _safe_mean_abs(e_i)
            true_rms = _safe_rms(t_i)
            low_signal = true_rms < NRMSE_SIGNAL_FLOOR
            nrmse = np.nan if low_signal else 100.0 * rmse / max(true_rms, EPS)
            per_agent_rows.append(
                {
                    "experiment": experiment_name,
                    "strategy": strategy,
                    "component": component_name,
                    "agent": f"UAV_{i+1}",
                    "rmse": rmse,
                    "mae": mae,
                    "true_rms": true_rms,
                    "nrmse_percent": nrmse,
                "low_signal_for_nrmse": low_signal,
                    "low_signal_for_nrmse": low_signal,
                }
            )

        err_flat = err.reshape(-1)
        true_flat = true.reshape(-1)
        rmse = _safe_rms(err_flat)
        mae = _safe_mean_abs(err_flat)
        true_rms = _safe_rms(true_flat)
        low_signal = true_rms < NRMSE_SIGNAL_FLOOR
        nrmse = np.nan if low_signal else 100.0 * rmse / max(true_rms, EPS)
        aggregate_rows.append(
            {
                "experiment": experiment_name,
                "strategy": strategy,
                "component": component_name,
                "scope": "aggregate",
                "rmse": rmse,
                "mae": mae,
                "true_rms": true_rms,
                "nrmse_percent": nrmse,
                "low_signal_for_nrmse": low_signal,
                "literature_proxy_nrmse_lower": 17.0,
                "literature_proxy_nrmse_upper": 25.0,
            }
        )

    per_agent_df = pd.DataFrame(per_agent_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)

    if not per_agent_df.empty:
        per_agent_df.to_csv(results_dir / f"component_error_stats_agents_{experiment_name}_{strategy}.csv", index=False)
    if not aggregate_df.empty:
        aggregate_df.to_csv(results_dir / f"component_error_stats_aggregate_{experiment_name}_{strategy}.csv", index=False)

    return per_agent_df, aggregate_df


def build_exp4_exp5_stats_table(summary_df: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df
    out = results_dir / "component_error_stats_Exp4_Exp5_combined.csv"
    summary_df.to_csv(out, index=False)
    return summary_df


def save_npz(logs: dict, path: Path) -> None:
    to_save = {}
    for k, v in logs.items():
        if isinstance(v, list):
            try:
                to_save[k] = np.asarray(v)
            except Exception:
                pass
        elif isinstance(v, np.ndarray):
            to_save[k] = v
    np.savez(path, **to_save)
