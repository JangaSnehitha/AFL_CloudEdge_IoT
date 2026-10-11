"""Plot round-level metrics produced by an AFL Cloud-Edge-IoT experiment.

Usage:
    python scripts/plot_results.py results/run_YYYYMMDD_HHMMSS
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path) -> list[dict[str, str]]:
    with metrics_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def save_line_plot(
    rounds: list[int],
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(rounds, values, marker="o", linewidth=2)
    plt.xlabel("Communication round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="Experiment result directory")
    args = parser.parse_args()

    run_dir = args.run_dir
    metrics_path = run_dir / "metrics.csv"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    rows = load_metrics(metrics_path)
    if not rows:
        raise ValueError(f"No metric rows found in {metrics_path}")

    rounds = [int(row["round"]) for row in rows]
    global_loss = [float(row["global_loss"]) for row in rows]
    global_accuracy = [float(row["global_accuracy"]) for row in rows]
    mean_train_loss = [float(row["mean_train_loss"]) for row in rows]

    save_line_plot(
        rounds,
        global_loss,
        "Global test loss",
        "Global test loss versus communication round",
        run_dir / "global_loss_vs_round.png",
    )

    save_line_plot(
        rounds,
        global_accuracy,
        "Global test accuracy",
        "Global test accuracy versus communication round",
        run_dir / "global_accuracy_vs_round.png",
    )

    save_line_plot(
        rounds,
        mean_train_loss,
        "Mean client training loss",
        "Mean client training loss versus communication round",
        run_dir / "mean_train_loss_vs_round.png",
    )

    print(f"Created plots in: {run_dir}")
    print(f"Rounds plotted: {len(rounds)}")
    print(f"Final global loss: {global_loss[-1]:.6f}")
    print(f"Final global accuracy: {global_accuracy[-1]:.4f}")
    print(f"Final mean train loss: {mean_train_loss[-1]:.6f}")


if __name__ == "__main__":
    main()
