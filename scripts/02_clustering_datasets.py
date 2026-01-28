import argparse
import logging
import sys
from pathlib import Path

from dapinet.clustering import run_all_datasets_in_npz


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def main():
    root = Path(__file__).resolve().parents[1]
    npz_path = root / "datasets" / "test" / "CesarComin" / "cfg000" / "CesarComin_cfg000.npz"
    parser = argparse.ArgumentParser(description="Run DAPINet experiments on NPZ datasets.")
    parser.add_argument(
        "--npz-path", type=Path, default=npz_path, help="Path to the NPZ with <id>_X / <id>_y keys."
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per algorithm.")
    parser.add_argument("--seed", type=int, default=42, help="Master seed.")
    parser.add_argument(
        "--optuna-jobs",
        type=int,
        default=1,
        help="Parallel trials per dataset (n_jobs for Optuna).",
    )
    parser.add_argument(
        "--algo",
        dest="algorithms",
        action="append",
        help="Algorithm to include (default is all algorithms).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each algorithm optimization.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=1, help="Increase logging verbosity (-v, -vv)."
    )
    args = parser.parse_args()

    configure_logging(args.verbose)

    run_all_datasets_in_npz(
        npz_path=args.npz_path,
        algorithms=args.algorithms,
        n_trials_per_algo=args.n_trials,
        master_seed=args.seed,
        optuna_n_jobs=args.optuna_jobs,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
