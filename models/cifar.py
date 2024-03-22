"""Models for MNIST-style 28x28 image classification"""

from torch import nn


class CNN(nn.Module):
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


class CNN_DC(nn.Module):
    def __init__(self, n_output_classes=10):
        super().__init__()

        self.n_output_classes = n_output_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=10,
                kernel_size=5
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=5
            ),
            nn.ReLU(),
            nn.Dropout()
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(11520, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, self.n_output_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        return x
