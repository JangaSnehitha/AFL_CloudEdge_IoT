import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


SEED = 42
NUM_CLIENTS = 2
SAMPLES_PER_CLIENT = 400
TEST_SAMPLES_PER_CLASS = 500
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.05


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleNet(nn.Module):
    """Small binary classifier for the controlled synthetic benchmark."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _make_class_samples(
    num_samples: int,
    class_label: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two separable Gaussian classes."""
    center = torch.tensor([-1.5, -1.5]) if class_label == 0 else torch.tensor([1.5, 1.5])
    features = center + 0.8 * torch.randn(num_samples, 2, generator=generator)
    labels = torch.full((num_samples,), class_label, dtype=torch.long)
    return features, labels


def _make_dataset(
    class_0_count: int,
    class_1_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    x0, y0 = _make_class_samples(class_0_count, 0, generator)
    x1, y1 = _make_class_samples(class_1_count, 1, generator)

    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([y0, y1], dim=0)

    permutation = torch.randperm(len(y), generator=generator)
    return x[permutation], y[permutation]


def get_client_data(client_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic non-IID data for one client.

    Client 0 has 80% class 0 and 20% class 1.
    Client 1 has 20% class 0 and 80% class 1.
    """
    if client_id == 0:
        class_0_count, class_1_count = 320, 80
    elif client_id == 1:
        class_0_count, class_1_count = 80, 320
    else:
        raise ValueError(f"Unsupported client_id: {client_id}")

    return _make_dataset(
        class_0_count=class_0_count,
        class_1_count=class_1_count,
        seed=SEED + client_id,
    )


def get_global_test_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a balanced test set never used by clients for training."""
    return _make_dataset(
        class_0_count=TEST_SAMPLES_PER_CLASS,
        class_1_count=TEST_SAMPLES_PER_CLASS,
        seed=SEED + 10_000,
    )


def train(
    model: nn.Module,
    client_id: int,
) -> tuple[float, int]:
    """Train a local copy of the model on one client's non-IID data."""
    set_seed(SEED + client_id)

    x, y = get_client_data(client_id)
    dataset = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED + client_id),
    )

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_examples = 0

    for _ in range(LOCAL_EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_y.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size

    return total_loss / total_examples, len(dataset)


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    """Evaluate model loss and accuracy on supplied data."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(x)
        loss = loss_fn(logits, y).item()
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == y).float().mean().item()

    return float(loss), float(accuracy)


def write_partition_summary(run_dir: Path) -> None:
    """Save a verifiable description of the two client data partitions."""
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "partition_summary.csv"

    rows = [
        {
            "client_id": 0,
            "num_samples": SAMPLES_PER_CLIENT,
            "class_0_count": 320,
            "class_1_count": 80,
            "class_0_fraction": 0.80,
            "class_1_fraction": 0.20,
        },
        {
            "client_id": 1,
            "num_samples": SAMPLES_PER_CLIENT,
            "class_0_count": 80,
            "class_1_count": 320,
            "class_0_fraction": 0.20,
            "class_1_fraction": 0.80,
        },
    ]

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
