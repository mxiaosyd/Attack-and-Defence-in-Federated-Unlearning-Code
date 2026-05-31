"""Model definitions used by the unified benchmark."""
from __future__ import annotations

import copy
from typing import Callable

import torch
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feat_size = image_size // 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * feat_size * feat_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TinyCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * (image_size // 2) * (image_size // 2), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * image_size * image_size, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """GroupNorm avoids BatchNorm running buffers in vectorized FL recovery.

    This makes model vectorization and aggregation safer in the unified benchmark
    while preserving a ResNet-18-like architecture.
    """
    groups = min(32, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Callable[[int], nn.Module] = _group_norm,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = norm_layer(planes)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


class CIFARResNet(nn.Module):
    """CIFAR-style ResNet.

    For CIFAR-10, this uses the standard small-image ResNet stem:
    3x3 convolution, stride 1, no ImageNet-style max-pooling.

    The default layer configuration [2, 2, 2, 2] corresponds to ResNet-18.
    """

    def __init__(
        self,
        block: type[BasicBlock],
        layers: list[int],
        in_channels: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
        norm_layer: Callable[[int], nn.Module] = _group_norm,
    ):
        super().__init__()
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(
            in_channels,
            base_width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm1 = norm_layer(base_width)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, base_width, layers[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int,
        norm_layer: Callable[[int], nn.Module],
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, norm_layer=norm_layer))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(in_channels: int = 3, num_classes: int = 10, image_size: int = 32) -> nn.Module:
    """Return a CIFAR-style ResNet-18.

    ``image_size`` is accepted for compatibility with the benchmark interface.
    """
    _ = image_size
    return CIFARResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
    )


def get_model(name: str, in_channels: int, num_classes: int, image_size: int) -> nn.Module:
    name = name.lower()
    if name in {"smallcnn", "cnn"}:
        return SmallCNN(in_channels, num_classes, image_size)
    if name in {"tinycnn", "tiny"}:
        return TinyCNN(in_channels, num_classes, image_size)
    if name in {"mlp", "fc"}:
        return MLP(in_channels, num_classes, image_size)
    if name in {"resnet18", "resnet", "cifar_resnet18", "cifar-resnet18"}:
        return ResNet18(in_channels, num_classes, image_size)
    raise ValueError(f"Unknown model: {name}")


def clone_model(model: nn.Module) -> nn.Module:
    return copy.deepcopy(model)