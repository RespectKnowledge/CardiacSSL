from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy.ndimage import zoom


class DINOv2Transform3D:
    """
    3D multi-crop augmentation for DINO-style self-distillation.

    Produces:
      - 2 global crops
      - n_local_crops local crops

    Input:  [B,C,D,H,W] numpy array
    """

    def __init__(
        self,
        global_crop_size: Sequence[int],
        local_crop_size: Sequence[int] | None = None,
        n_global_crops: int = 2,
        n_local_crops: int = 4,
        augment: bool = True,
        seed: int | None = None,
    ):
        self.global_crop_size = tuple(int(x) for x in global_crop_size)

        if local_crop_size is None:
            self.local_crop_size = tuple(
                max(32, x // 2) for x in self.global_crop_size
            )
        else:
            self.local_crop_size = tuple(int(x) for x in local_crop_size)

        self.n_global_crops = int(n_global_crops)
        self.n_local_crops = int(n_local_crops)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

    def reseed_from_worker(self):
        seed = int(np.random.randint(0, 2**31 - 1))
        self.rng = np.random.default_rng(seed)

    def _random_crop(self, x, crop_size):
        _, d, h, w = x.shape
        cd, ch, cw = crop_size

        # If necessary resize before cropping.
        scale = max(
            cd / max(d, 1),
            ch / max(h, 1),
            cw / max(w, 1),
            1.0,
        )

        if scale > 1.0:
            x = zoom(
                x,
                (1.0, scale, scale, scale),
                order=1,
            )
            _, d, h, w = x.shape

        sd = int(self.rng.integers(0, d - cd + 1)) if d > cd else 0
        sh = int(self.rng.integers(0, h - ch + 1)) if h > ch else 0
        sw = int(self.rng.integers(0, w - cw + 1)) if w > cw else 0

        crop = x[
            :,
            sd:sd + cd,
            sh:sh + ch,
            sw:sw + cw,
        ].copy()

        if self.augment:
            for axis in (1, 2, 3):
                if self.rng.random() < 0.5:
                    crop = np.flip(crop, axis=axis).copy()

            # Mild intensity scaling.
            if self.rng.random() < 0.5:
                factor = float(self.rng.uniform(0.9, 1.1))
                crop *= factor

            # Mild Gaussian noise.
            if self.rng.random() < 0.2:
                std = float(self.rng.uniform(0.0, 0.05))
                crop += self.rng.normal(
                    0.0,
                    std,
                    size=crop.shape,
                ).astype(np.float32)

        return np.ascontiguousarray(crop, dtype=np.float32)

    def __call__(self, data):
        if data.ndim != 5:
            raise ValueError(
                f"Expected [B,C,D,H,W], got {data.shape}"
            )

        global_views = []
        local_views = []

        for _ in range(self.n_global_crops):
            global_views.append(
                np.stack(
                    [
                        self._random_crop(x, self.global_crop_size)
                        for x in data
                    ],
                    axis=0,
                )
            )

        for _ in range(self.n_local_crops):
            local_views.append(
                np.stack(
                    [
                        self._random_crop(x, self.local_crop_size)
                        for x in data
                    ],
                    axis=0,
                )
            )

        return {
            "global_crops": global_views,
            "local_crops": local_views,
        }
