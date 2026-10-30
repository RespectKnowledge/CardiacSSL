"""
Cardiac Swin segmentation network.

The encoder is exactly the Swin3D backbone used during SSL pretraining.
Only the decoder is segmentation-specific.

Inputs are padded internally to satisfy the Swin3D downsampling
requirements. Segmentation outputs are cropped back to the original
nnU-Net patch size.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cardiac_ssl.backbones.swin3d import (
    Swin3D,
    Swin3DConfig,
)

from downstream.nnunet.networks.swin_decoder import (
    SwinSegmentationDecoder,
)


class SwinSSLUNet(nn.Module):

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = True,
        deep_supervision_scales=None,
    ):
        super().__init__()

        config = Swin3DConfig(
            input_channels=input_channels,
            patch_size=(2, 2, 2),
            window_size=(7, 7, 7),
            embed_dim=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            patch_norm=False,
            use_checkpoint=False,
            downsample="merging",
            use_v2=False,
            normalize=True,
        )

        self.encoder = Swin3D(config)

        self.decoder = SwinSegmentationDecoder(
            encoder_channels=self.encoder.output_channels,
            num_classes=num_classes,
            deep_supervision=deep_supervision,
            deep_supervision_scales=deep_supervision_scales,
        )

    @staticmethod
    def _pad_to_factor(
        x: torch.Tensor,
        factor=(32, 32, 32),
    ):
        """
        Pad [B,C,D,H,W] on the positive side so D/H/W are
        divisible by the Swin3D downsampling factor.
        """

        d, h, w = x.shape[2:]

        pd = (factor[0] - d % factor[0]) % factor[0]
        ph = (factor[1] - h % factor[1]) % factor[1]
        pw = (factor[2] - w % factor[2]) % factor[2]

        if pd or ph or pw:
            # F.pad order for 5D tensor:
            # W-left, W-right, H-left, H-right, D-left, D-right
            x = F.pad(
                x,
                (0, pw, 0, ph, 0, pd),
                mode="constant",
                value=0,
            )

        return x

    @staticmethod
    def _crop_output(
        output,
        output_size,
    ):
        """
        Crop segmentation output(s) back to the original
        nnU-Net spatial size.

        Handles both a single tensor and deep-supervision outputs.
        """

        d, h, w = output_size

        if isinstance(output, (list, tuple)):
            cropped = []

            for i, out in enumerate(output):
                if i == 0:
                    cropped.append(out[..., :d, :h, :w])
                else:
                    cropped.append(out)

            return type(output)(cropped)

        return output[..., :d, :h, :w]

    def forward(self, x: torch.Tensor):

        original_size = tuple(x.shape[2:])

        x = self._pad_to_factor(
            x,
            factor=self.encoder.downsample_factor,
        )

        padded_size = tuple(x.shape[2:])

        features = self.encoder.forward_intermediates(x)

        output = self.decoder(
            features,
            output_size=original_size,
        )

        return self._crop_output(
            output,
            original_size,
        )
