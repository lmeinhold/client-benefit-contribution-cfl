"""Models for MNIST-style 28x28 image classification"""

from torch import nn


class CNN_DC(nn.Module):
    def __init__(self, n_output_classes=10):
        super().__init__()

        self.n_output_classes = n_output_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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
            nn.Linear(8000, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, self.n_output_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        return x


class CNN(nn.Module):
    def __init__(self, dropout=0.2, n_output_classes=10):
        super().__init__()

        self.n_output_classes = n_output_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
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
            nn.Linear(16 * 4 * 4, 120),
            nn.LeakyReLU(),
            nn.Linear(120, 84),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(84, self.n_output_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        return x


class LargeCNN(nn.Module):
    def __init__(self, n_output_classes=10):
        super().__init__()

        self.n_output_classes = n_output_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0),
            nn.BatchNorm2d(6),
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
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=26,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(26),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, self.n_output_classes))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.output(x)
        return x
