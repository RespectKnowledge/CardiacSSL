from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SimCLRConfig:
    """
    FM-Bench SimCLR configuration.

    Defaults preserve the historical nnSSL SimCLR objective/projector:
      - temperature = 0.5
      - 3-layer MLP projector
      - hidden/output dimension = 2048
      - InstanceNorm1d without affine parameters/running statistics
    """

    temperature: float = 0.5
    projector_hidden_dim: int = 2048
    projector_output_dim: int = 2048


class SimCLRProjectionHead(nn.Module):
    """
    Projection head used by the historical nnSSL VoCo/SimCLR models.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 2048,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.InstanceNorm1d(
                hidden_dim,
                affine=False,
                track_running_stats=False,
            ),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.InstanceNorm1d(
                hidden_dim,
                affine=False,
                track_running_stats=False,
            ),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


def gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """Gather a tensor across DDP ranks while preserving local gradients.

    In single-process training this is an identity operation.

    Under DDP, tensors from every rank are gathered and concatenated along
    the batch dimension. The tensor belonging to the current rank is
    replaced by the original local tensor so gradients flow through the
    local embeddings. DDP subsequently synchronizes parameter gradients
    across ranks.
    """
    if not dist.is_available() or not dist.is_initialized():
        return x

    world_size = dist.get_world_size()

    if world_size == 1:
        return x

    rank = dist.get_rank()

    gathered = [
        torch.empty_like(x)
        for _ in range(world_size)
    ]

    dist.all_gather(gathered, x.contiguous())

    # all_gather outputs are detached from the local autograd graph.
    # Restore this rank's original tensor to preserve its gradients.
    gathered[rank] = x

    return torch.cat(gathered, dim=0)


class NTXentLoss(nn.Module):
    """
    NT-Xent objective matching the historical nnSSL formulation.

    Unlike the old implementation, batch size is inferred dynamically.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

        self.temperature = float(temperature)

    def get_logits(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_i.ndim != 2 or z_j.ndim != 2:
            raise ValueError(
                "NTXentLoss expects two tensors with shape [N, C]."
            )

        if z_i.shape != z_j.shape:
            raise ValueError(
                f"View shapes must match, got {tuple(z_i.shape)} "
                f"and {tuple(z_j.shape)}."
            )

        batch_size = z_i.shape[0]

        if batch_size < 1:
            raise ValueError("NTXentLoss requires at least one sample.")

        # Preserve historical ordering: [z_j, z_i].
        representations = torch.cat([z_j, z_i], dim=0)

        # Inputs are normalized by SimCLR before this call, so the matrix
        # multiplication is equivalent to cosine similarity.
        similarity_matrix = representations @ representations.T

        l_pos = torch.diagonal(
            similarity_matrix,
            offset=batch_size,
        )
        r_pos = torch.diagonal(
            similarity_matrix,
            offset=-batch_size,
        )

        positives = torch.cat([l_pos, r_pos], dim=0).reshape(
            2 * batch_size,
            1,
        )

        device = similarity_matrix.device

        diag = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=device,
        )

        positive_mask = torch.zeros_like(diag)
        idx = torch.arange(batch_size, device=device)
        positive_mask[idx, idx + batch_size] = True
        positive_mask[idx + batch_size, idx] = True

        negative_mask = ~(diag | positive_mask)

        negatives = similarity_matrix[negative_mask].reshape(
            2 * batch_size,
            -1,
        )

        logits = torch.cat([positives, negatives], dim=1)
        logits = logits / self.temperature

        labels = torch.zeros(
            2 * batch_size,
            dtype=torch.long,
            device=device,
        )

        return logits, labels

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, labels = self.get_logits(z_i, z_j)

        loss = F.cross_entropy(
            logits,
            labels,
            reduction="sum",
        )
        loss = loss / logits.shape[0]

        accuracy = (logits.detach().argmax(dim=1) == 0).float().mean()

        return loss, accuracy


class SimCLR(nn.Module):
    """
    Backbone-independent SimCLR method.

    The backbone must expose:
      - ``feature_dim``
      - ``forward_pooled(x)``

    Batch format expected by ``forward``:
      {
          "all_crops": Tensor[2N, C, D, H, W],
          "reference_crop_index": N
      }

    This preserves the existing FM-Bench/nnSSL crop pipeline while keeping
    the SSL method independent of CNN, ViT, or Swin implementation details.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: SimCLRConfig | None = None,
    ):
        super().__init__()

        self.config = config or SimCLRConfig()
        self.backbone = backbone

        if not hasattr(backbone, "feature_dim"):
            raise TypeError(
                "SimCLR backbone must expose a 'feature_dim' attribute."
            )

        if not callable(getattr(backbone, "forward_pooled", None)):
            raise TypeError(
                "SimCLR backbone must implement forward_pooled(x)."
            )

        self.feature_dim = int(backbone.feature_dim)

        self.projector = SimCLRProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=self.config.projector_hidden_dim,
            output_dim=self.config.projector_output_dim,
        )

        self.loss_fn = NTXentLoss(
            temperature=self.config.temperature,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the backbone representation before the SimCLR projector.
        """
        return self.backbone.forward_pooled(x)

    def project(self, features: torch.Tensor) -> torch.Tensor:
        """
        Return unnormalized projector output.
        """
        return self.projector(features)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return L2-normalized SimCLR embeddings.
        """
        features = self.encode(x)
        projected = self.project(features)
        return F.normalize(projected, dim=1)

    def compute_loss(
        self,
        all_crops: torch.Tensor,
        reference_crop_index: int,
    ) -> dict[str, torch.Tensor]:
        if all_crops.ndim != 5:
            raise ValueError(
                "all_crops must have shape [2N, C, D, H, W]."
            )

        nref = int(reference_crop_index)

        if nref <= 0:
            raise ValueError("reference_crop_index must be positive.")

        if all_crops.shape[0] != 2 * nref:
            raise ValueError(
                "SimCLR expects two equally-sized groups of paired crops. "
                f"Got {all_crops.shape[0]} crops and "
                f"reference_crop_index={nref}."
            )

        embeddings = self.embed(all_crops)

        z_i = embeddings[:nref]
        z_j = embeddings[nref:]

        # SimCLR negatives must come from the global distributed batch.
        # Single-process training remains unchanged.
        z_i = gather_with_grad(z_i)
        z_j = gather_with_grad(z_j)

        loss, accuracy = self.loss_fn(z_i, z_j)

        return {
            "loss": loss,
            "contrastive_accuracy": accuracy,
        }

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        if not isinstance(batch, dict):
            raise TypeError("SimCLR expects a batch dictionary.")

        if "all_crops" not in batch:
            raise KeyError("SimCLR batch is missing 'all_crops'.")

        if "reference_crop_index" not in batch:
            raise KeyError(
                "SimCLR batch is missing 'reference_crop_index'."
            )

        return self.compute_loss(
            batch["all_crops"],
            batch["reference_crop_index"],
        )
