import torch

from flwr.serverapp import ServerApp
from flwr.serverapp.strategy import FedAvg
from flwr.app import (
    ArrayRecord,
    Context,
)


app = ServerApp()


@app.main()
def main(grid, context: Context):
    """Run one round of FedAvg."""

    # Create initial global model
    from app.task import SimpleNet

    model = SimpleNet()

    # Convert model parameters to Flower ArrayRecord
    initial_arrays = ArrayRecord(model.state_dict())

    # Create FedAvg strategy
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=0.0,
        min_available_nodes=2,
    )

    # Run federated learning
    strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=1,
    )
