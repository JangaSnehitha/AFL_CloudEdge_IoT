import torch
from torch import nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def train(model, client_id):
    model.train()

    # Very small synthetic local dataset
    if client_id == 0:
        x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])
    else:
        x = torch.tensor([[5.0], [6.0], [7.0], [8.0]])
        y = torch.tensor([[10.0], [12.0], [14.0], [16.0]])

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for _ in range(5):
        optimizer.zero_grad()

        predictions = model(x)
        loss = loss_fn(predictions, y)

        loss.backward()
        optimizer.step()

    return float(loss.item())
