import torch
from torch import nn

BATCH_NORM_DECAY = 0.01
BATCH_NORM_EPSILON = 1e-3


class ResidualBlock(nn.Module):
    """Residual Block in PyTorch"""

    def __init__(self, input_channels, channels, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(
            input_channels,
            momentum=BATCH_NORM_DECAY,
            eps=BATCH_NORM_EPSILON,
        )
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(
            channels,
            momentum=BATCH_NORM_DECAY,
            eps=BATCH_NORM_EPSILON,
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or input_channels != channels:
            self.shortcut = nn.Conv2d(
                input_channels,
                channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            self.conv_shortcut = True
        else:
            self.shortcut = nn.Identity()
            self.conv_shortcut = False

    def forward(self, inputs):
        x = self.bn1(inputs)
        x = self.relu(x)

        shortcut = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.conv_shortcut:
            return x + shortcut
        else:
            return x + inputs


class ResidualGroup(nn.Module):
    """Residual Group in PyTorch"""

    def __init__(self, input_channels, channels, num_blocks, stride):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(input_channels, channels, stride=stride),
            *[ResidualBlock(channels, channels, stride=1) for _ in range(num_blocks - 1)],
        )

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    """ResNet in PyTorch"""

    def __init__(self, num_classes, block_sizes, block_channels, block_strides):
        super().__init__()
        self.num_classes = num_classes
        self.block_sizes = block_sizes
        self.block_channels = block_channels
        self.block_strides = block_strides

        input_channels = 16
        self.conv1 = nn.Conv2d(3, input_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.blocks = []

        for i in range(len(block_sizes)):
            block = ResidualGroup(input_channels, block_channels[i], block_sizes[i], stride=block_strides[i])
            input_channels = block_channels[i]
            self.add_module(f"block{i}", block)
            self.blocks.append(block)

        self.bn = nn.BatchNorm2d(
            block_channels[-1],
            momentum=BATCH_NORM_DECAY,
            eps=BATCH_NORM_EPSILON,
        )
        self.relu = nn.ReLU()
        self.fc = nn.Linear(block_channels[-1], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)

        for block in self.blocks:
            x = block(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x).squeeze()
        x = self.fc(x)
        return x
