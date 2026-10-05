from flwr.clientapp import ClientApp
from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from app.task import SimpleNet, train


app = ClientApp()


@app.train()
def train_client(msg: Message, context: Context):
    """Train the model on a simulated client."""

    # Create model
    model = SimpleNet()

    # Load global model from server
    model.load_state_dict(
        msg.content["arrays"].to_torch_state_dict()
    )

    # Use Flower's node ID to identify the simulated client
    node_id = context.node_id

    print(f"Client with node_id={node_id} started training.")

    # Temporary mapping for our 2-client experiment
    client_id = int(node_id)

    # Train locally
    loss = train(model, client_id)

    print(
        f"Client {client_id} finished training. "
        f"Loss: {loss:.4f}"
    )

    # Updated model
    model_record = ArrayRecord(model.state_dict())

    # Training metrics
    metrics = MetricRecord(
        {
            "train_loss": loss,
            "num-examples": 4,
        }
    )

    content = RecordDict(
        {
            "arrays": model_record,
            "metrics": metrics,
        }
    )

    return Message(
        content=content,
        reply_to=msg,
    )
