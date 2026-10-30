"""
3D segmentation decoder for the Cardiac-SSL Swin encoder.

The encoder is shared exactly with SSL pretraining.

Deep-supervision logits are resized to nnU-Net's expected
deep-supervision target sizes.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),

            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.proj = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        x = self.proj(x)
        x = torch.cat((x, skip), dim=1)

        return self.conv(x)


class SwinSegmentationDecoder(nn.Module):

    def __init__(
        self,
        encoder_channels: Sequence[int],
        num_classes: int,
        deep_supervision: bool = True,
        deep_supervision_scales=None,
    ):
        super().__init__()

        if len(encoder_channels) != 5:
            raise ValueError(
                "Expected five Swin feature levels."
            )

        c0, c1, c2, c3, c4 = encoder_channels

        self.deep_supervision = deep_supervision
        self.deep_supervision_scales = deep_supervision_scales

        self.dec3 = DecoderStage(c4, c3, c3)
        self.dec2 = DecoderStage(c3, c2, c2)
        self.dec1 = DecoderStage(c2, c1, c1)
        self.dec0 = DecoderStage(c1, c0, c0)

        self.seg0 = nn.Conv3d(c0, num_classes, 1)
        self.seg1 = nn.Conv3d(c1, num_classes, 1)
        self.seg2 = nn.Conv3d(c2, num_classes, 1)
        self.seg3 = nn.Conv3d(c3, num_classes, 1)

    @staticmethod
    def _resize(logits, size):
        if tuple(logits.shape[2:]) == tuple(size):
            return logits

        return F.interpolate(
            logits,
            size=size,
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, features, output_size):
        x0, x1, x2, x3, x4 = features

        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)

        out0 = self._resize(
            self.seg0(d0),
            output_size,
        )

        if not self.deep_supervision:
            return out0

        raw_outputs = [
            out0,
            self.seg1(d1),
            self.seg2(d2),
            self.seg3(d3),
        ]

        if self.deep_supervision_scales is None:
            return raw_outputs

        outputs = []

        for logits, scale in zip(
            raw_outputs,
            self.deep_supervision_scales,
        ):
            target_size = tuple(
                max(1, int(round(dim * s)))
                for dim, s in zip(output_size, scale)
            )

            outputs.append(
                self._resize(logits, target_size)
            )

        return outputs
