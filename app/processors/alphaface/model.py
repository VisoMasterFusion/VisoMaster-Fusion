"""Inference-only AlphaFace swapper architecture.

Adapted from AlphaFace's official MIT-licensed implementation at
https://github.com/andrewyu90/Alphaface_Official (commit d41fbd4).
Training-only modules and dependencies are intentionally omitted.

This module exists purely so ``app/tools/export_alphaface_onnx.py`` can turn the
official checkpoint into the ONNX graph VisoMaster ships; the runtime path never
instantiates it.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class IdentityFeedingBlock(nn.Module):
    def __init__(self, output_dim: int, identity_dim: int = 512) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.fc = nn.Linear(identity_dim, output_dim)

    def forward(
        self, identity: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = self.fc(identity).unsqueeze(2).unsqueeze(3)
        # The projected identity is spatially 1x1, so its centered value is
        # exactly zero. The official AdaIN expression therefore reduces to the
        # target channel mean without changing the model's result.
        target_mean = torch.sum(target, (2, 3), keepdim=True) / (
            target.shape[2] * target.shape[3]
        )
        midpoint = int(self.output_dim / 2)
        first = projected[:, 0:midpoint, :, :]
        second = projected[:, midpoint : self.output_dim, :, :]
        first = (first + target_mean) / 2.0
        second = (second + target_mean) / 2.0
        return first, second


class OperationUnit(nn.Module):
    def __init__(self, channels: int, identity_output_dim: int, activate: bool) -> None:
        super().__init__()
        self.activate = activate
        self.Conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.activation = nn.ReLU()
        self.IFF = IdentityFeedingBlock(identity_output_dim)

    def forward(self, features: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        scale, bias = self.IFF(identity, features)
        output = F.pad(features, (1, 1, 1, 1), mode="reflect")
        output = self.Conv1(output)
        # The official code spells instance norm out as ReduceMean/Mul/Sqrt/Div.
        # F.instance_norm is numerically equivalent to within ~2e-5 and exports
        # to a single InstanceNormalization node, which TensorRT can run in FP16.
        output = F.instance_norm(output, eps=1.0e-8)
        output = torch.add(torch.mul(scale, output), bias)
        return self.activation(output) if self.activate else output


class CrossAdaptiveIdentityInjectionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.OP1 = OperationUnit(1024, 2048, activate=True)
        self.OP2 = OperationUnit(1024, 2048, activate=False)

    def forward(self, features: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        return features + self.OP2(self.OP1(features, identity), identity)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = [3, 128, 256, 512, 1024]
        kernels = [7, 3, 3, 3]
        paddings = [0, 1, 1, 1]
        strides = [1, 1, 2, 2]
        self.Encoder = nn.ModuleDict(
            {
                f"layer_{index}": nn.Sequential(
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        kernel_size=kernels[index],
                        stride=strides[index],
                        padding=paddings[index],
                    ),
                    nn.LeakyReLU(0.2),
                )
                for index in range(4)
            }
        )
        self.fusion_module = nn.ModuleDict(
            {
                f"fusion_layer_{index}": CrossAdaptiveIdentityInjectionBlock()
                for index in range(6)
            }
        )

    def forward(self, target: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
        output = F.pad(target, (3, 3, 3, 3), mode="reflect")
        for index in range(4):
            output = self.Encoder[f"layer_{index}"](output)
        for index in range(6):
            output = self.fusion_module[f"fusion_layer_{index}"](output, identity)
        return output


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Upsample = nn.Upsample(
            scale_factor=2, align_corners=False, mode="bilinear"
        )
        self.Conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.Conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.Conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.Conv4_new = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.Conv5_new = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.Conv6_new = nn.Conv2d(128, 3, kernel_size=5, padding=0)
        self.Activation_LeakyRelu = nn.LeakyReLU(0.2)
        self.Activation_Tanh = nn.Tanh()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.Upsample(features)
        output = self.Activation_LeakyRelu(self.Conv1(output))
        output = self.Upsample(output)
        output = self.Activation_LeakyRelu(self.Conv2(output))
        output = self.Activation_LeakyRelu(self.Conv3(output))
        output = self.Activation_LeakyRelu(self.Conv4_new(output))
        output = self.Activation_LeakyRelu(self.Conv5_new(output))
        output = F.pad(output, (3, 3, 3, 3), mode="reflect")
        output = self.Activation_Tanh(self.Conv6_new(output))
        return (output + 1.0) / 2.0


class AlphaFaceSwapper(nn.Module):
    """The released 256px AlphaFace swapper without training dependencies."""

    def __init__(self) -> None:
        super().__init__()
        # Attribute names match the official checkpoint exactly.
        self.E = Encoder()
        self.G = Decoder()

    def forward(
        self, target: torch.Tensor, source_embedding: torch.Tensor
    ) -> torch.Tensor:
        return self.G(self.E(target, source_embedding))
