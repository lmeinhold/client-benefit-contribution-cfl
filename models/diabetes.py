from torch import nn
from torch.nn.functional import leaky_relu, sigmoid


class MLP(nn.Module):
    """
    Simple MLP model with 3 layers and dropout for the diabetes dataset

        Parameters:
            dropout: fraction of neurons to randomly exclude for dropout
    """

    def __init__(self, dropout=0.2):
        super().__init__()

        self.dense1 = nn.Linear(21, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = leaky_relu(self.dense1(x))
        x = leaky_relu(self.dense2(x))
        x = self.dropout(x)
        x = sigmoid(self.dense3(x))
        return x
