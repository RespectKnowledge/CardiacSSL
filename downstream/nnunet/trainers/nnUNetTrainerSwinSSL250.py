"""
nnU-Net v2 trainer using the Cardiac-SSL 3D Swin encoder.

All standard nnU-Net functionality is inherited:
    - preprocessing / dataloading
    - augmentation
    - Dice + CE loss
    - deep supervision
    - optimizer / LR schedule
    - validation
    - checkpointing
    - sliding-window inference

Only the network architecture is replaced by SwinSSLUNet.

SSL encoder initialization is handled separately so the same downstream
architecture can be evaluated with Random, SimCLR, MAE, V-JEPA and
SwinDINO-3D initialization.
"""

from __future__ import annotations

import os
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import (
    PlansManager,
    ConfigurationManager,
)

from downstream.nnunet.networks.swin_nnunet import SwinSSLUNet
from downstream.nnunet.utils.load_ssl_weights import load_ssl_encoder


class nnUNetTrainerSwinSSL250(nnUNetTrainer):

    def initialize(self):
        self.num_epochs = 250
        # Swin3D requires more GPU memory than the standard nnU-Net.
        # Keep the planned patch size and all other settings unchanged.
        self.batch_size = 2

        super().initialize()

        checkpoint = os.environ.get("CARDIAC_SSL_CHECKPOINT")

        if checkpoint:
            print("=" * 72)
            print("Loading Cardiac SSL initialization")
            print("Checkpoint:", checkpoint)

            load_ssl_encoder(
                self.network,
                checkpoint,
                strict=True,
            )
        else:
            print("=" * 72)
            print("CARDIAC_SSL_CHECKPOINT not set")
            print("Using RANDOM Swin3D initialization")
            print("=" * 72)

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:

        if enable_deep_supervision:
            import numpy as np

            deep_supervision_scales = list(
                list(i)
                for i in 1 / np.cumprod(
                    np.vstack(
                        configuration_manager.pool_op_kernel_sizes
                    ),
                    axis=0,
                )
            )[:-1]
        else:
            deep_supervision_scales = None

        network = SwinSSLUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            deep_supervision_scales=deep_supervision_scales,
        )

        return network
