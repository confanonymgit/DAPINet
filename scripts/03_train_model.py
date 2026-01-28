import argparse
import logging
from pathlib import Path

from dapinet.architecture import Config, run_training_pipeline

logger = logging.getLogger(__name__)


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG

    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Train Model with K-Fold Cross-Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model Architecture Arguments

    parser.add_argument(
        "--d-model",
        type=int,
        default=Config.D_MODEL,
        help="Model embedding dimension",
    )

    parser.add_argument(
        "--n-head",
        type=int,
        default=Config.N_HEAD,
        help="Number of attention heads",
    )

    parser.add_argument(
        "--n-layers",
        type=int,
        default=Config.N_LAYERS,
        help="Number of transformer layers",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=Config.DROPOUT,
        help="Dropout rate",
    )

    parser.add_argument(
        "--num-bins",
        type=int,
        default=Config.NUM_BINS,
        help="Number of quantile bins for DACE column identification",
    )

    # Training Arguments

    parser.add_argument(
        "--lr",
        type=float,
        default=Config.LEARNING_RATE,
        help="Learning rate",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=Config.WEIGHT_DECAY,
        help="Weight decay for optimizer",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=Config.BATCH_SIZE,
        help="Training batch size",
    )

    parser.add_argument(
        "--accum-steps",
        type=int,
        default=Config.ACCUM_STEPS,
        help="Gradient accumulation steps",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=Config.EPOCHS,
        help="Number of training epochs",
    )

    parser.add_argument(
        "--k-folds",
        type=int,
        default=Config.K_FOLDS,
        help="Number of folds for cross-validation",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=Config.PATIENCE,
        help="Patience for early stopping",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=Config.NUM_WORKERS,
        help="Number of workers for data loading",
    )

    # Dataset Arguments

    parser.add_argument(
        "--data-dir",
        type=Path,
        default="datasets",
        help="Root directory of datasets",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default="results",
        help="Directory for output files",
    )

    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase logging verbosity (-v, -vv)."
    )

    args = parser.parse_args()

    configure_logging(args.verbose)

    Config.update_from_args(args)

    # Start training

    run_training_pipeline(
        data_root=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
