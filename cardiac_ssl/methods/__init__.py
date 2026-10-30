"""Self-supervised learning methods."""

# Keep method modules independent so that one optional method
# does not prevent another method from being imported.

try:
    from .simclr import SimCLR, SimCLRConfig
except ImportError:
    SimCLR = None
    SimCLRConfig = None

try:
    from .mae import SwinMAE, MAEConfig
except ImportError:
    SwinMAE = None
    MAEConfig = None

try:
    from .vjepa import VJEPA, VJEPAConfig
except ImportError:
    VJEPA = None
    VJEPAConfig = None

__all__ = [
    "SimCLR",
    "SimCLRConfig",
    "SwinMAE",
    "MAEConfig",
    "VJEPA",
    "VJEPAConfig",
]

try:
    from .dinov2 import DINOv2, DINOv2Config
except ImportError:
    DINOv2 = None
    DINOv2Config = None
