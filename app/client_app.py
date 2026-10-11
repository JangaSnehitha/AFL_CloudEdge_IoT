from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from app.task import SimpleNet, evaluate, get_global_test_data, train


app = ClientApp()
def get_client_id(context: Context) -> int:
    """Return Flower's stable simulated-client partition identifier."""
    client_id = int(context.node_config["partition-id"])

    if client_id not in (0, 1):
        raise ValueError(
            f"Expected a two-client partition ID 0 or 1, got {client_id}"
        )

    return client_id

@app.train()
def train_client(msg: Message, context: Context) -> Message:
    """Train the received global model on local non-IID client data."""
    model = SimpleNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    client_id = get_client_id(context)

    print(
        f"[TRAIN] node_id={context.node_id} "
        f"mapped_client_id={client_id}"
    )

    train_loss, num_examples = train(model, client_id)

    print(
        f"[TRAIN] client_id={client_id} completed "
        f"train_loss={train_loss:.6f} num_examples={num_examples}"
    )

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(
                    {
                        "train_loss": train_loss,
                        "num-examples": num_examples,
                        "client_id": client_id,
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate_client(msg: Message, context: Context) -> Message:
    """Evaluate the received global model on the common balanced test set."""
    model = SimpleNet()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    test_x, test_y = get_global_test_data()
    global_loss, global_accuracy = evaluate(model, test_x, test_y)

    client_id = get_client_id(context)

    print(
        f"[EVALUATE] node_id={context.node_id} "
        f"partition_id={client_id} "
        f"global_loss={global_loss:.6f} "
        f"global_accuracy={global_accuracy:.4f}"
    )

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": global_loss,
                        "global_accuracy": global_accuracy,
                        "num-examples": len(test_y),
                        "client_id": client_id,
                    }
                )
            }
        ),
        reply_to=msg,
    )
