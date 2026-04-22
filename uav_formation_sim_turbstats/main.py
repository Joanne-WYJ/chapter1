from __future__ import annotations

import argparse
from pathlib import Path

from uav_formation_sim.experiments import make_default_config, run_standard_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure Python multi-UAV formation simulation")
    parser.add_argument("--quick", action="store_true", help="run quick test mode")
    parser.add_argument("--results", type=str, default="results", help="results directory")
    args = parser.parse_args()

    cfg = make_default_config(quick=args.quick)
    cfg.results_dir = args.results
    cfg.results_path = Path(args.results)
    cfg.results_path.mkdir(parents=True, exist_ok=True)
    df = run_standard_suite(cfg)
    print(df)
    print(f"Saved results to {cfg.results_path.resolve()}")


if __name__ == "__main__":
    main()
