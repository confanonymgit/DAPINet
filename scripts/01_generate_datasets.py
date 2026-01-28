import argparse
import logging
from pathlib import Path

from dapinet.synthesis import GenerationSettings, run_generation


def main():
    parser = argparse.ArgumentParser(description="Auto-Clust Synthetic Data Generation")

    # Output
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/synthetic"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed")
    parser.add_argument(
        "--n-repeats", type=int, default=10, help="Number of repetitions per config/strategy"
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0, help="Timeout in seconds per dataset (0 to disable)"
    )

    # Config Bounds
    parser.add_argument(
        "--n-configs", type=int, default=50, help="Number of LHS configs to generate"
    )
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=15)
    parser.add_argument("--n-low", type=int, default=100)
    parser.add_argument("--n-high", type=int, default=2500)
    parser.add_argument("--d-low", type=int, default=2)
    parser.add_argument("--d-high", type=int, default=200)

    args = parser.parse_args()

    # Ensure output directory exists for logging
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.output_dir / "generation.log"),
        ],
    )

    settings = GenerationSettings(
        n_repeats=args.n_repeats,
        master_seed=args.seed,
        output_dir=args.output_dir,
        timeout=args.timeout,
        n_configs=args.n_configs,
        k_min=args.k_min,
        k_max=args.k_max,
        n_low=args.n_low,
        n_high=args.n_high,
        d_low=args.d_low,
        d_high=args.d_high,
    )

    print("Starting Generation with settings:")
    print(settings)

    run_generation(settings)


if __name__ == "__main__":
    main()
