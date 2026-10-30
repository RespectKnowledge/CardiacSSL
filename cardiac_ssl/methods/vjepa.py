"""V-JEPA for volumetric 3D self-supervised learning.

FM-Bench implementation designed to compose with CNN3D, ViT3D/EVA,
and Swin3D backbones while using the common generic Trainer.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VJEPAConfig:
    """Configuration for volumetric V-JEPA."""

    predictor_dim: int = 384
    predictor_depth: int = 12
    predictor_heads: int = 12
    mlp_ratio: float = 4.0

    ema_momentum: float = 0.99925

    num_context_blocks: int = 8
    num_target_blocks: int = 2

    context_scale: tuple[float, float] = (0.15, 0.15)
    target_scale: tuple[float, float] = (0.70, 0.70)
    aspect_ratio: tuple[float, float] = (0.75, 1.5)

    mask_distance_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.predictor_dim <= 0:
            raise ValueError("predictor_dim must be positive.")

        if self.predictor_depth <= 0:
            raise ValueError("predictor_depth must be positive.")

        if self.predictor_heads <= 0:
            raise ValueError("predictor_heads must be positive.")

        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError(
                "ema_momentum must be in [0, 1]."
            )

        if self.num_context_blocks <= 0:
            raise ValueError(
                "num_context_blocks must be positive."
            )

        if self.num_target_blocks <= 0:
            raise ValueError(
                "num_target_blocks must be positive."
            )


class VJEPABackboneAdapter(nn.Module):
    """Expose a common token representation for FM-Bench backbones.

    EVA/ViT:
        native patch-token sequence [B, N, C]

    CNN/Swin:
        final dense feature grid flattened to [B, N, C]
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()

        if not hasattr(backbone, "feature_dim"):
            raise ValueError(
                "V-JEPA backbone must expose feature_dim."
            )

        self.backbone = backbone
        self.feature_dim = int(backbone.feature_dim)

        self.is_eva = (
            hasattr(backbone, "forward_tokens")
            and hasattr(backbone, "eva")
            and hasattr(backbone, "down_projection")
        )

    @staticmethod
    def _dense_to_tokens(
        features: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                "Dense backbone features must have shape "
                "[B, C, D, H, W]."
            )

        return (
            features
            .flatten(start_dim=2)
            .transpose(1, 2)
            .contiguous()
        )

    def token_grid_shape(
        self,
        x: torch.Tensor,
    ) -> tuple[int, int, int]:
        """Return the spatial grid represented by output tokens."""

        if self.is_eva:
            return tuple(
                int(v)
                for v in self.backbone._feature_shape_for_input(x)
            )

        spatial = tuple(
            int(v)
            for v in x.shape[2:]
        )

        factor = getattr(
            self.backbone,
            "downsample_factor",
            None,
        )

        if factor is None:
            raise RuntimeError(
                "Dense V-JEPA backbone must expose "
                "downsample_factor."
            )

        factor = tuple(int(v) for v in factor)

        if any(
            size % down != 0
            for size, down in zip(
                spatial,
                factor,
            )
        ):
            raise ValueError(
                f"Input shape {spatial} is not divisible "
                f"by downsample factor {factor}."
            )

        return tuple(
            size // down
            for size, down in zip(
                spatial,
                factor,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_eva:
            tokens, keep_indices = (
                self.backbone.forward_tokens(x)
            )

            if keep_indices is not None:
                raise RuntimeError(
                    "V-JEPA requires the complete EVA token "
                    "sequence. Set patch_drop_rate=0."
                )

            n_register_tokens = int(
                getattr(
                    self.backbone,
                    "num_register_tokens",
                    0,
                )
            )

            if n_register_tokens > 0:
                tokens = tokens[:, n_register_tokens:]

            return tokens

        features = self.backbone.forward_features(x)
        return self._dense_to_tokens(features)


    def _forward_eva_context(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate masked-token encoding to the ViT3D backbone."""

        forward_context = getattr(
            self.backbone,
            "forward_context_tokens",
            None,
        )

        if forward_context is None:
            raise RuntimeError(
                "EVA V-JEPA backbone must implement "
                "forward_context_tokens()."
            )

        return forward_context(
            x,
            context_indices,
        )

    def _forward_dense_context(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        full_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode CNN/Swin context from a masked input volume.

        Unlike post-hoc token gathering, hidden image regions are removed
        before the dense encoder sees the input. This is the explicit
        volumetric CNN/Swin adaptation of V-JEPA context masking.
        """
        del full_tokens

        grid_shape = self.token_grid_shape(x)
        d, h, w = grid_shape
        num_tokens = d * h * w

        if torch.any(context_indices < 0) or torch.any(
            context_indices >= num_tokens
        ):
            raise ValueError(
                "context_indices contains an index outside "
                f"the token grid {grid_shape}."
            )

        # Construct visibility on the encoder output grid using the same
        # flattened index convention as the V-JEPA mask generator.
        visibility = torch.zeros(
            x.shape[0],
            num_tokens,
            device=x.device,
            dtype=x.dtype,
        )

        visibility.scatter_(
            1,
            context_indices,
            1.0,
        )

        visibility = visibility.reshape(
            x.shape[0],
            1,
            d,
            h,
            w,
        )

        # Expand output-grid visibility back to image space. Nearest-neighbour
        # preserves binary visible/hidden regions and also supports anisotropic
        # valid input sizes.
        voxel_visibility = F.interpolate(
            visibility,
            size=tuple(int(v) for v in x.shape[2:]),
            mode="nearest",
        )

        masked_x = x * voxel_visibility

        # Dense V-JEPA requires a separate masked encoder pass for each
        # context mask. Checkpoint the expensive encoder during training so
        # Swin3D does not retain all context-encoder activation graphs.
        if self.training and torch.is_grad_enabled():
            full_tokens = torch.utils.checkpoint.checkpoint(
                self.forward,
                masked_x,
                use_reentrant=False,
            )
        else:
            full_tokens = self.forward(masked_x)

        return _gather_tokens(
            full_tokens,
            context_indices,
        )

    def forward_context(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        full_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return encoder representations at selected context locations."""

        if context_indices.ndim != 2:
            raise ValueError(
                "context_indices must have shape [B, K]."
            )

        if context_indices.shape[0] != x.shape[0]:
            raise ValueError(
                "context_indices batch size must match input batch size."
            )

        if self.is_eva:
            return self._forward_eva_context(
                x,
                context_indices,
            )

        return self._forward_dense_context(
            x,
            context_indices,
            full_tokens=full_tokens,
        )


class VJEPATargetEncoder(nn.Module):
    """Frozen exponential-moving-average copy of the context encoder."""

    def __init__(
        self,
        context_encoder: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = deepcopy(context_encoder)

        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

        self.encoder.eval()

    def train(
        self,
        mode: bool = True,
    ):
        # Target encoder always remains in evaluation mode.
        super().train(False)
        self.encoder.train(False)
        return self

    @torch.no_grad()
    def update(
        self,
        context_encoder: nn.Module,
        momentum: float,
    ) -> None:
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(
                "EMA momentum must be in [0, 1]."
            )

        context_parameters = dict(
            context_encoder.named_parameters()
        )

        target_parameters = dict(
            self.encoder.named_parameters()
        )

        if (
            context_parameters.keys()
            != target_parameters.keys()
        ):
            raise RuntimeError(
                "Context and target encoder parameter "
                "structures do not match."
            )

        for name, target_parameter in (
            target_parameters.items()
        ):
            context_parameter = context_parameters[name]

            target_parameter.mul_(momentum).add_(
                context_parameter.detach(),
                alpha=1.0 - momentum,
            )

        # Copy buffers directly. This is important for
        # normalization/statistical buffers in future backbones.
        context_buffers = dict(
            context_encoder.named_buffers()
        )
        target_buffers = dict(
            self.encoder.named_buffers()
        )

        for name, target_buffer in target_buffers.items():
            if name in context_buffers:
                target_buffer.copy_(
                    context_buffers[name]
                )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self.encoder.eval()
        return self.encoder(x)


class VJEPALoss(nn.Module):
    """Smooth-L1 loss in layer-normalized embedding space."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target must have identical "
                f"shape, got {tuple(prediction.shape)} and "
                f"{tuple(target.shape)}."
            )

        prediction = F.layer_norm(
            prediction,
            (prediction.shape[-1],),
        )

        target = F.layer_norm(
            target,
            (target.shape[-1],),
        )

        return F.smooth_l1_loss(
            prediction,
            target,
        )


class VJEPAMaskGenerator:
    """Generate contiguous block masks over a 3D token grid.

    Context and target masks are represented as flattened token indices
    [B, K]. Blocks are sampled in the actual 3D feature grid rather than
    as unrelated random token subsets.
    """

    def __init__(
        self,
        config: VJEPAConfig | None = None,
    ) -> None:
        self.config = config or VJEPAConfig()

    @staticmethod
    def _validate_grid(
        grid_shape: tuple[int, int, int],
    ) -> tuple[int, int, int]:
        if len(grid_shape) != 3:
            raise ValueError(
                "grid_shape must contain three dimensions."
            )

        grid = tuple(int(v) for v in grid_shape)

        if any(v <= 0 for v in grid):
            raise ValueError(
                f"Invalid token grid {grid}."
            )

        return grid

    @staticmethod
    def _sample_fraction(
        scale: tuple[float, float],
        device: torch.device,
    ) -> float:
        low, high = map(float, scale)

        if not 0.0 < low <= high <= 1.0:
            raise ValueError(
                "Mask scale must satisfy "
                "0 < low <= high <= 1."
            )

        if low == high:
            return low

        return float(
            torch.empty(
                (),
                device=device,
            ).uniform_(low, high).item()
        )

    def _sample_block_shape(
        self,
        grid_shape: tuple[int, int, int],
        scale: tuple[float, float],
        device: torch.device,
    ) -> tuple[int, int, int]:
        d, h, w = grid_shape

        fraction = self._sample_fraction(
            scale,
            device,
        )

        target_volume = max(
            1,
            min(
                d * h * w,
                int(round(d * h * w * fraction)),
            ),
        )

        ar_low, ar_high = map(
            float,
            self.config.aspect_ratio,
        )

        if not 0.0 < ar_low <= ar_high:
            raise ValueError(
                "aspect_ratio must satisfy "
                "0 < low <= high."
            )

        # Start from a cubic block with the requested volume.
        base = target_volume ** (1.0 / 3.0)

        # Independently perturb depth/height aspect ratios.
        ratio_d = float(
            torch.empty(
                (),
                device=device,
            ).uniform_(ar_low, ar_high).item()
        )

        ratio_h = float(
            torch.empty(
                (),
                device=device,
            ).uniform_(ar_low, ar_high).item()
        )

        bd = max(
            1,
            int(round(base * ratio_d)),
        )
        bh = max(
            1,
            int(round(base * ratio_h)),
        )

        bw = max(
            1,
            int(
                round(
                    target_volume
                    / max(1, bd * bh)
                )
            ),
        )

        bd = min(bd, d)
        bh = min(bh, h)
        bw = min(bw, w)

        return bd, bh, bw

    @staticmethod
    def _block_indices(
        grid_shape: tuple[int, int, int],
        block_shape: tuple[int, int, int],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        d, h, w = grid_shape
        bd, bh, bw = block_shape

        all_indices = []

        for _ in range(batch_size):
            sd = int(
                torch.randint(
                    0,
                    d - bd + 1,
                    (),
                    device=device,
                ).item()
            )
            sh = int(
                torch.randint(
                    0,
                    h - bh + 1,
                    (),
                    device=device,
                ).item()
            )
            sw = int(
                torch.randint(
                    0,
                    w - bw + 1,
                    (),
                    device=device,
                ).item()
            )

            zz = torch.arange(
                sd,
                sd + bd,
                device=device,
            )
            yy = torch.arange(
                sh,
                sh + bh,
                device=device,
            )
            xx = torch.arange(
                sw,
                sw + bw,
                device=device,
            )

            z, y, x = torch.meshgrid(
                zz,
                yy,
                xx,
                indexing="ij",
            )

            indices = (
                z * (h * w)
                + y * w
                + x
            ).reshape(-1)

            all_indices.append(indices)

        return torch.stack(
            all_indices,
            dim=0,
        )

    def _sample_masks(
        self,
        *,
        batch_size: int,
        grid_shape: tuple[int, int, int],
        scale: tuple[float, float],
        count: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        masks = []

        for _ in range(count):
            # One block shape per mask keeps K identical across
            # subjects, which permits efficient batched gathering.
            block_shape = self._sample_block_shape(
                grid_shape,
                scale,
                device,
            )

            masks.append(
                self._block_indices(
                    grid_shape,
                    block_shape,
                    batch_size,
                    device,
                )
            )

        return masks

    def __call__(
        self,
        batch_size: int,
        grid_shape: tuple[int, int, int],
        device: torch.device,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        grid_shape = self._validate_grid(
            grid_shape
        )

        masks_enc = self._sample_masks(
            batch_size=batch_size,
            grid_shape=grid_shape,
            scale=self.config.context_scale,
            count=self.config.num_context_blocks,
            device=device,
        )

        masks_pred = self._sample_masks(
            batch_size=batch_size,
            grid_shape=grid_shape,
            scale=self.config.target_scale,
            count=self.config.num_target_blocks,
            device=device,
        )

        return masks_enc, masks_pred


def _gather_tokens(
    tokens: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Gather token positions independently for each batch item."""

    if tokens.ndim != 3:
        raise ValueError(
            "tokens must have shape [B, N, C]."
        )

    if indices.ndim != 2:
        raise ValueError(
            "indices must have shape [B, K]."
        )

    if tokens.shape[0] != indices.shape[0]:
        raise ValueError(
            "tokens and indices batch sizes must match."
        )

    expanded = indices.unsqueeze(-1).expand(
        -1,
        -1,
        tokens.shape[-1],
    )

    return torch.gather(
        tokens,
        dim=1,
        index=expanded,
    )


def _indices_to_3d_coordinates(
    indices: torch.Tensor,
    grid_shape: tuple[int, int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert flattened token indices to normalized 3D coordinates."""

    if indices.ndim != 2:
        raise ValueError(
            "indices must have shape [B, K]."
        )

    d, h, w = (
        int(grid_shape[0]),
        int(grid_shape[1]),
        int(grid_shape[2]),
    )

    if min(d, h, w) <= 0:
        raise ValueError(
            f"Invalid grid_shape={grid_shape}."
        )

    num_tokens = d * h * w

    if torch.any(indices < 0) or torch.any(indices >= num_tokens):
        raise ValueError(
            "Token index is outside the 3D token grid."
        )

    z = torch.div(
        indices,
        h * w,
        rounding_mode="floor",
    )

    remainder = indices % (h * w)

    y = torch.div(
        remainder,
        w,
        rounding_mode="floor",
    )

    x = remainder % w

    def normalize(
        coordinate: torch.Tensor,
        size: int,
    ) -> torch.Tensor:
        coordinate = coordinate.to(dtype=dtype)

        if size == 1:
            return torch.zeros_like(coordinate)

        # Map coordinates to [-1, 1].
        return (
            2.0 * coordinate / float(size - 1)
            - 1.0
        )

    return torch.stack(
        [
            normalize(z, d),
            normalize(y, h),
            normalize(x, w),
        ],
        dim=-1,
    )


class VJEPAPredictor(nn.Module):
    """Transformer predictor for target-token embeddings."""

    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int = 384,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()

        if predictor_dim % num_heads != 0:
            raise ValueError(
                "predictor_dim must be divisible by "
                "num_heads."
            )

        self.encoder_dim = int(encoder_dim)
        self.predictor_dim = int(predictor_dim)

        self.input_projection = nn.Linear(
            self.encoder_dim,
            self.predictor_dim,
        )

        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                self.predictor_dim,
            )
        )

        # Match the source behavior: mask tokens start at zero.
        nn.init.zeros_(self.mask_token)

        # Spatial location of each target token. Coordinates are
        # normalized in the 3D token grid and projected into the
        # predictor embedding space.
        self.position_projection = nn.Linear(
            3,
            self.predictor_dim,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=self.predictor_dim,
            nhead=num_heads,
            dim_feedforward=int(
                self.predictor_dim * mlp_ratio
            ),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(
            layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(
            self.predictor_dim
        )

        self.output_projection = nn.Linear(
            self.predictor_dim,
            self.encoder_dim,
        )

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_indices: torch.Tensor,
        grid_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError(
                "context_tokens must have shape [B, N, C]."
            )

        if target_indices.ndim != 2:
            raise ValueError(
                "target_indices must have shape [B, K]."
            )

        if (
            context_tokens.shape[0]
            != target_indices.shape[0]
        ):
            raise ValueError(
                "Context and target batch sizes must match."
            )

        num_target_tokens = target_indices.shape[1]

        if num_target_tokens <= 0:
            raise ValueError(
                "At least one target token is required."
            )

        context = self.input_projection(
            context_tokens
        )

        coordinates = _indices_to_3d_coordinates(
            target_indices,
            grid_shape=grid_shape,
            dtype=context.dtype,
        )

        target_positions = self.position_projection(
            coordinates
        )

        mask_tokens = self.mask_token.expand(
            context.shape[0],
            num_target_tokens,
            -1,
        )

        # A target query now identifies both:
        #   1. an unknown target representation
        #   2. its position in the 3D token grid.
        target_queries = (
            mask_tokens
            + target_positions
        )

        sequence = torch.cat(
            [context, target_queries],
            dim=1,
        )

        sequence = self.blocks(sequence)
        sequence = self.norm(sequence)

        prediction = sequence[
            :,
            -num_target_tokens:,
        ]

        return self.output_projection(
            prediction
        )


class VJEPA(nn.Module):
    """Backbone-independent volumetric V-JEPA method."""

    def __init__(
        self,
        backbone: nn.Module,
        config: VJEPAConfig | None = None,
    ) -> None:
        super().__init__()

        self.config = config or VJEPAConfig()

        self.context_encoder = (
            VJEPABackboneAdapter(backbone)
        )

        self.target_encoder = (
            VJEPATargetEncoder(
                self.context_encoder
            )
        )

        self.predictor = VJEPAPredictor(
            encoder_dim=self.context_encoder.feature_dim,
            predictor_dim=self.config.predictor_dim,
            depth=self.config.predictor_depth,
            num_heads=self.config.predictor_heads,
            mlp_ratio=self.config.mlp_ratio,
        )

        self.mask_generator = VJEPAMaskGenerator(
            self.config
        )

        self.loss_fn = VJEPALoss()

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        # EMA target is always frozen/eval.
        self.target_encoder.train(False)

        return self

    def forward(
        self,
        batch: dict[str, object],
    ) -> dict[str, torch.Tensor]:
        x = batch.get("data")

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                "batch['data'] must be a torch.Tensor."
            )

        # -----------------------------------------------------
        # Target representation: full image, frozen EMA.
        # -----------------------------------------------------
        with torch.no_grad():
            target_tokens = self.target_encoder(x)

            target_tokens = F.layer_norm(
                target_tokens,
                (target_tokens.shape[-1],),
            )

        # -----------------------------------------------------
        # Token geometry and masks.
        # -----------------------------------------------------
        batch_size = x.shape[0]

        grid_shape = (
            self.context_encoder.token_grid_shape(x)
        )

        num_tokens = (
            grid_shape[0]
            * grid_shape[1]
            * grid_shape[2]
        )

        if target_tokens.shape[1] != num_tokens:
            raise RuntimeError(
                "V-JEPA token-grid geometry does not "
                "match target encoder token count: "
                f"grid={grid_shape}, "
                f"target_tokens={target_tokens.shape[1]}."
            )

        masks_enc, masks_pred = (
            self.mask_generator(
                batch_size=batch_size,
                grid_shape=grid_shape,
                device=x.device,
            )
        )

        # Compute every context/target mask combination.
        losses = []

        for context_indices in masks_enc:
            visible_context = (
                self.context_encoder.forward_context(
                    x,
                    context_indices,
                )
            )

            for target_indices in masks_pred:
                target = _gather_tokens(
                    target_tokens,
                    target_indices,
                )

                prediction = self.predictor(
                    visible_context,
                    target_indices=target_indices,
                    grid_shape=grid_shape,
                )

                losses.append(
                    self.loss_fn(
                        prediction,
                        target,
                    )
                )

        loss = torch.stack(losses).mean()

        return {
            "loss": loss,
            "target_tokens": target_tokens,
        }

    @torch.no_grad()
    def update_target_encoder(
        self,
        momentum: float | None = None,
    ) -> None:
        if momentum is None:
            momentum = self.config.ema_momentum

        self.target_encoder.update(
            self.context_encoder,
            momentum=momentum,
        )

    def on_after_optimizer_step(self) -> None:
        """Called by the generic FM-Bench Trainer."""
        self.update_target_encoder()
