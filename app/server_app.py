import csv
from datetime import datetime
from pathlib import Path

from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg

from app.task import (
    LEARNING_RATE,
    LOCAL_EPOCHS,
    NUM_CLIENTS,
    SEED,
    SimpleNet,
    set_seed,
    write_partition_summary,
)


app = ServerApp()


def create_run_directory() -> Path:
    """Create a unique output directory for one experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_config(run_dir: Path, num_rounds: int) -> None:
    """Write the exact configuration used by this experimental run."""
    config_path = run_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment: fedavg_two_client_non_iid_synthetic",
                f"seed: {SEED}",
                f"num_clients: {NUM_CLIENTS}",
                f"num_rounds: {num_rounds}",
                "fraction_train: 1.0",
                "fraction_evaluate: 1.0",
                "model: SimpleNet",
                "dataset: synthetic_2d_binary_classification",
                "evaluation_dataset: common_balanced_global_test_set",
                "client_0_distribution: 'class_0=0.80, class_1=0.20'",
                "client_1_distribution: 'class_0=0.20, class_1=0.80'",
                f"local_epochs: {LOCAL_EPOCHS}",
                f"learning_rate: {LEARNING_RATE}",
                "",
            ]
        ),
        encoding="utf-8",
    )


@app.main()
def main(grid, context: Context) -> None:
    """Run the two-client, non-IID, multi-round FedAvg baseline."""
    num_rounds = 10
    set_seed(SEED)

    run_dir = create_run_directory()
    write_config(run_dir, num_rounds)
    write_partition_summary(run_dir)

    metrics_path = run_dir / "metrics.csv"
    fieldnames = [
        "round",
        "global_loss",
        "global_accuracy",
        "mean_train_loss",
        "num_clients",
    ]
    train_loss_by_round: dict[int, float] = {}

    with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
        csv.DictWriter(metrics_file, fieldnames=fieldnames).writeheader()

    def aggregate_train_metrics(
        records: list[RecordDict],
        weighted_by_key: str,
    ) -> MetricRecord:
        """Compute weighted mean client train loss for Flower logging."""
        total_weight = 0.0
        weighted_loss = 0.0

        for record in records:
            metrics = record["metrics"]
            weight = float(metrics[weighted_by_key])
            weighted_loss += float(metrics["train_loss"]) * weight
            total_weight += weight
        mean_train_loss = weighted_loss / total_weight

        # One call occurs after every training phase; derive the round index
        # from the number of values recorded so far.
        server_round = len(train_loss_by_round) + 1
        train_loss_by_round[server_round] = mean_train_loss

        return MetricRecord({"mean_train_loss": mean_train_loss})
        


    def aggregate_evaluate_metrics(
        records: list[RecordDict],
        weighted_by_key: str,
    ) -> MetricRecord:
        """Aggregate common-test evaluation and append one CSV row."""
        total_weight = 0.0
        weighted_loss = 0.0
        weighted_accuracy = 0.0

        for record in records:
            metrics = record["metrics"]
            weight = float(metrics[weighted_by_key])
            weighted_loss += float(metrics["eval_loss"]) * weight
            weighted_accuracy += float(metrics["global_accuracy"]) * weight
            total_weight += weight

        global_loss = weighted_loss / total_weight
        global_accuracy = weighted_accuracy / total_weight

        # Flower invokes this once per evaluation phase. Row number gives round.
        existing_rows = 0
        if metrics_path.exists():
            with metrics_path.open("r", encoding="utf-8") as metrics_file:
                existing_rows = max(sum(1 for _ in metrics_file) - 1, 0)

        server_round = existing_rows + 1

        with metrics_path.open("a", newline="", encoding="utf-8") as metrics_file:
            writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
            writer.writerow(
                {
                    "round": server_round,
                    "global_loss": global_loss,
                    "global_accuracy": global_accuracy,
                    "mean_train_loss": train_loss_by_round.get(server_round, ""),
                    "num_clients": NUM_CLIENTS,
                }
            )

        print(
            f"\n[GLOBAL EVALUATION] round={server_round}/{num_rounds} "
            f"global_loss={global_loss:.6f} "
            f"global_accuracy={global_accuracy:.4f}\n"
        )

        return MetricRecord(
            {
                "global_loss": global_loss,
                "global_accuracy": global_accuracy,
            }
        )

    model = SimpleNet()
    initial_arrays = ArrayRecord(model.state_dict())

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=NUM_CLIENTS,
        min_evaluate_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
        train_metrics_aggr_fn=aggregate_train_metrics,
        evaluate_metrics_aggr_fn=aggregate_evaluate_metrics,
    )

    print(f"Results for this run will be saved in: {run_dir}")

    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )
