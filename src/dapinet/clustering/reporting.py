import json
import logging
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_logs(log_dir: Path, output_dir: Path) -> None:
    """Analyze Slurm logs and write reports."""
    logger.info(f"Analyzing logs in {log_dir}...")

    log_files = sorted(log_dir.glob("*.out")) + sorted(log_dir.glob("*.err"))

    failures = []
    stats = {"timeout": 0, "oom": 0, "python_error": 0, "success": 0, "unknown": 0}

    # Regex patterns for common slurm/python errors
    re_timeout = re.compile(r"(DUE TO TIME LIMIT|CANCELLED AT|step canceled)", re.IGNORECASE)
    re_oom = re.compile(r"(Out Of Memory|killed by the cgroup|oom-kill)", re.IGNORECASE)
    re_py_err = re.compile(r"(Traceback \(most recent call last\)|Exception:)", re.IGNORECASE)

    for log_file in log_files:
        try:
            content = log_file.read_text(errors="replace")

            # Simple classification
            if re_timeout.search(content):
                stats["timeout"] += 1
                failures.append((log_file, "TIMEOUT"))
            elif re_oom.search(content):
                stats["oom"] += 1
                failures.append((log_file, "OOM"))
            elif re_py_err.search(content):
                stats["python_error"] += 1
                failures.append((log_file, "PYTHON_ERROR"))
            else:
                # If it finished? Check for success marker if possible
                if "Successfully finished" in content or "Done" in content:  # heuristic
                    stats["success"] += 1
                else:
                    stats["unknown"] += 1
                    # Treat unknown as potential silent failure or running

        except Exception as e:
            logger.warning(f"Could not read {log_file}: {e}")

    # Write report
    report_lines = [
        "Log Analysis Report",
        "===================",
        f"Total Logs scanned: {len(log_files)}",
        f"Timeouts: {stats['timeout']}",
        f"OOMs: {stats['oom']}",
        f"Python Errors: {stats['python_error']}",
        f"Apparent Success: {stats['success']}",
        f"Unknown/Running: {stats['unknown']}",
    ]

    output_report = output_dir / "job_report.txt"
    output_report.write_text("\n".join(report_lines))
    logger.info("\n".join(report_lines))

    # Write failed log paths
    output_failures = output_dir / "failed_logs.txt"
    with open(output_failures, "w") as f:
        for log_f, reason in failures:
            f.write(f"{reason}: {log_f.resolve()}\n")

    logger.info(f"Detailed failures written to {output_failures}")


def load_results(root_path: Path) -> pd.DataFrame:
    """Aggregate result JSON files into a DataFrame."""
    records = []

    # Walk through all files
    for path in root_path.rglob("*.json"):
        # Basic check if it looks like a result file (has corresponding .npz or just check content)
        # The user mentioned .json files in the folder.
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            continue

        if "results" not in data:
            continue

        # Infer metadata from path relative to root
        # Expected: Strategy/Config/Filename.json
        try:
            rel_path = path.relative_to(root_path)
            parts = rel_path.parts
            if len(parts) >= 2:
                strategy = parts[0]
                config = parts[1]
            else:
                strategy = "Unknown"
                config = "Unknown"
        except ValueError:
            strategy = "Unknown"
            config = "Unknown"

        results = data["results"]
        for ds_id, algo_results in results.items():
            for algo_name, res in algo_results.items():
                if "best_ari" in res:
                    records.append(
                        {
                            "strategy": strategy,
                            "config": config,
                            "dataset_id": ds_id,
                            "algorithm": algo_name,
                            "ari": res["best_ari"],
                            "file_path": str(path),
                        }
                    )

    df = pd.DataFrame(records)
    return df


def find_missing_predictions(data_dir: Path) -> list[Path]:
    """Find datasets missing prediction files."""
    npz_files = sorted(data_dir.rglob("*.npz"))
    missing = []

    for npz in npz_files:
        # Ignore files that are themselves predictions or training data
        if "predictions.npz" in npz.name or "prediction.npz" in npz.name:
            continue

        if "_training.npz" in npz.name:
            continue

        stem = npz.stem
        parent = npz.parent

        # Check for <stem>*predictions.npz
        # This handles cases like <stem>_algorithm_predictions.npz
        candidates = list(parent.glob(f"{stem}*predictions.npz"))

        if not candidates:
            missing.append(npz)

    return missing


def generate_missing_report(input_dir: Path) -> list[Path]:
    """Write a report of missing predictions."""
    missing_files = find_missing_predictions(input_dir)
    if missing_files:
        out_file = input_dir / "missing_jobs.txt"
        with open(out_file, "w") as f:
            for p in missing_files:
                f.write(str(p.resolve()) + "\n")
        logger.info(f"Wrote missing paths to {out_file}")

    return missing_files


def _save_plot(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    data: pd.DataFrame | pd.Series | None = None,
):
    output_path = output_dir / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=600)
    plt.close(fig)
    logger.info(f"Saved plot to {output_path}")

    if data is not None:
        data_path = output_path.with_suffix(".csv")
        if isinstance(data, pd.Series):
            data.to_frame(name="value").to_csv(data_path, index=False)
        else:
            data.to_csv(data_path, index=False)
        logger.info(f"Saved plot data to {data_path}")


def _compute_stats(series: pd.Series) -> str:
    desc = series.describe()
    return (
        f"Count: {int(desc['count'])}\n"
        f"Mean:  {desc['mean']:.3f}\n"
        f"Std:   {desc['std']:.3f}\n"
        f"Min:   {desc['min']:.3f}\n"
        f"Max:   {desc['max']:.3f}"
    )


def _plot_histogram(
    ax: plt.Axes,
    data: pd.Series,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    color: str | None = None,
    stats: bool = True,
):
    ax.hist(data, bins=50, edgecolor="black", alpha=0.7, color=color)
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if stats and len(data) > 0:
        stats_text = _compute_stats(data)
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
        )


def _create_single_histogram(
    data: pd.Series,
    output_dir: Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
    color: str | None = None,
    save_data: pd.DataFrame | pd.Series | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_histogram(ax, data, title, xlabel=xlabel, ylabel=ylabel, color=color)
    if save_data is None:
        save_data = data
    _save_plot(fig, output_dir, filename, data=save_data)


def _create_multi_histogram(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    output_dir: Path,
    filename: str,
    title_template: str,
    xlabel: str,
    ylabel: str = "Frequency",
    color: str | None = None,
):
    groups = sorted(df[group_col].unique())
    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(10, 4 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]  # type: ignore

    for i, group in enumerate(groups):
        ax = axes[i]  # type: ignore
        data = df[df[group_col] == group][value_col]
        if len(data) > 0:
            _plot_histogram(ax, data, title_template.format(group), ylabel=ylabel, color=color)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(title_template.format(group))
            ax.set_ylabel(ylabel)

    axes[-1].set_xlabel(xlabel)  # type: ignore
    _save_plot(fig, output_dir, filename, data=df)


def _create_stacked_histogram(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    output_dir: Path,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str = "Frequency",
):
    groups = sorted(df[group_col].unique())
    bins = np.arange(-0.1, 1.2, 0.1)

    plt.figure(figsize=(12, 8))
    group_data = []
    labels = []
    for group in groups:
        data = df[df[group_col] == group][value_col]
        group_data.append(data)
        labels.append(group)

    plt.hist(group_data, bins=bins, stacked=True, label=labels, edgecolor="black")  # type: ignore
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    _save_plot(plt.gcf(), output_dir, filename, data=df)


def _generate_negative_ari_report(df: pd.DataFrame, output_dir: Path):
    negative_df = df[df["ari"] < 0].copy()
    log_path = output_dir / "negative_ari_report.txt"

    with open(log_path, "w") as f:
        f.write("=== Negative ARI Report ===\n\n")
        f.write(f"Total Negative ARI Counts: {len(negative_df)}\n")
        f.write(f"Total Data Points: {len(df)}\n")
        f.write(f"Percentage: {len(negative_df) / len(df) * 100:.2f}%\n\n")

        if not negative_df.empty:
            f.write("--- Summary by Algorithm ---\n")
            f.write(negative_df["algorithm"].value_counts().to_string())
            f.write("\n\n")

            f.write("--- Summary by Strategy ---\n")
            f.write(negative_df["strategy"].value_counts().to_string())
            f.write("\n\n")

            f.write("--- Detailed List ---\n")
            negative_df = negative_df.sort_values("ari")
            for _, row in negative_df.iterrows():
                f.write(
                    f"ARI: {row['ari']:.4f} | Algo: {row['algorithm']} | Strat: {row['strategy']} |"
                    f"DS: {row['dataset_id']} | File: {row['file_path']}\n"
                )

    logger.info(f"Saved negative ARI report to {log_path}")


def generate_report(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        logger.warning("No data found to visualize.")
        return

    # Pre-calculate Max ARI per dataset
    df["unique_ds_id"] = df["strategy"] + "_" + df["config"] + "_" + df["dataset_id"]
    max_ari_df = df.groupby("unique_ds_id")["ari"].max().reset_index()
    max_ari_df.rename(columns={"ari": "max_ari"}, inplace=True)

    winners_df = df.merge(
        max_ari_df, left_on=["unique_ds_id", "ari"], right_on=["unique_ds_id", "max_ari"]
    )

    ds_meta = df[["unique_ds_id", "strategy"]].drop_duplicates()
    max_ari_strat = max_ari_df.merge(ds_meta, on="unique_ds_id")

    # 1. All ARI Histogram
    _create_single_histogram(
        df["ari"],
        output_dir,
        "01_all_ari_hist.pdf",
        "Histogram of All ARI Values",
        "ARI",
        save_data=df,
    )

    # 2. ARI per Algorithm
    _create_multi_histogram(
        df,
        "algorithm",
        "ari",
        output_dir,
        "02_ari_per_algorithm.pdf",
        "ARI Distribution: {}",
        "ARI",
    )

    # 3. ARI per Strategy
    _create_multi_histogram(
        df,
        "strategy",
        "ari",
        output_dir,
        "03_ari_per_strategy.pdf",
        "ARI Distribution: {}",
        "ARI",
    )

    # 4. Max ARI Histogram
    _create_single_histogram(
        max_ari_df["max_ari"],
        output_dir,
        "04_max_ari_hist.pdf",
        "Histogram of Max ARI Values (Best per Dataset)",
        "Max ARI",
        color="green",
        save_data=max_ari_df,
    )

    # 5. Max ARI per Algorithm
    _create_multi_histogram(
        winners_df,
        "algorithm",
        "max_ari",
        output_dir,
        "05_max_ari_per_algorithm.pdf",
        "Max ARI Distribution when Winner is: {}",
        "Max ARI",
        color="green",
    )

    # 6. Max ARI per Strategy
    _create_multi_histogram(
        max_ari_strat,
        "strategy",
        "max_ari",
        output_dir,
        "06_max_ari_per_strategy.pdf",
        "Max ARI Distribution for Strategy: {}",
        "Max ARI",
        color="green",
    )

    # 7. Stacked Strategy
    _create_stacked_histogram(
        max_ari_strat,
        "strategy",
        "max_ari",
        output_dir,
        "07_max_ari_stacked_strategy.pdf",
        "Max ARI Distribution by Strategy (Stacked)",
        "Max ARI",
    )

    # 8. Stacked Algorithm
    _create_stacked_histogram(
        winners_df,
        "algorithm",
        "max_ari",
        output_dir,
        "08_max_ari_stacked_algorithm.pdf",
        "Max ARI Distribution by Winning Algorithm (Stacked)",
        "Max ARI",
        ylabel="Frequency (Count of Wins)",
    )

    # 9. Negative ARI Report
    _generate_negative_ari_report(df, output_dir)

    # Copy search_spaces.json
    search_spaces_src = Path(__file__).parent / "search_spaces.json"
    if search_spaces_src.exists():
        shutil.copy(search_spaces_src, output_dir / "search_spaces.json")
        logger.info(f"Copied search_spaces.json to {output_dir}")
    else:
        logger.warning(f"Could not find search_spaces.json at {search_spaces_src}")
