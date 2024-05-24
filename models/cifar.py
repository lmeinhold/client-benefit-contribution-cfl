"""Models for CIFAR-style image classification"""

from torch import nn


class CNN(nn.Module):
    """
    Convolutional Neural Network model with two convolutional and three linear layers. Uses MaxPooling and Dropout.

        Parameters:
            dropout: fraction of neurons to randomly exclude for dropout
    """

    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.Dropout(p=dropout),
            nn.LeakyReLU(),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        return x
