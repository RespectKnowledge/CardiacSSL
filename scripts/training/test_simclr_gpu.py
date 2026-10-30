import torch

from cardiac_ssl.backbones.swin3d import Swin3D, Swin3DConfig
from cardiac_ssl.methods.simclr import SimCLR, SimCLRConfig
from cardiac_ssl.data.build import build_split_loader

device = torch.device("cuda")
print("GPU:", torch.cuda.get_device_name(0))

# -------------------------
# Swin3D backbone
# -------------------------
backbone = Swin3D(
    Swin3DConfig(
        input_channels=1,
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
)

model = SimCLR(
    backbone=backbone,
    config=SimCLRConfig(
        temperature=0.5,
        projector_hidden_dim=2048,
        projector_output_dim=2048,
    ),
).to(device)

print("Model created")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------
# Dataset702 loader
# -------------------------
loader = build_split_loader(
    split_file="data/splits/splits_final_nnssl.json",
    split="train",
    dataset_root=(
        "data/preprocessed/Dataset702_Public_MRI/"
        "nnsslPlans_onemmiso/Dataset702_Public_MRI"
    ),
    plan=(
        "data/preprocessed/Dataset702_Public_MRI/"
        "nnsslPlans.json"
    ),
    configuration="onemmiso",
    method="simclr",

    batch_size=2,
    num_workers=2,
    num_cached=2,
    iterations_per_epoch=1,

    # Swin requires D/H/W divisible by 32
    final_patch_size_override=(32, 160, 160),

    spatial_augmentation=True,
    intensity_augmentation=True,
    seed=42,
)

it = iter(loader)
batch = next(it)

print("all_crops:", tuple(batch["all_crops"].shape))

def to_device(x):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v) for v in x]
    if isinstance(x, tuple):
        return tuple(to_device(v) for v in x)
    return x

batch = to_device(batch)

torch.cuda.reset_peak_memory_stats()
model.train()

# -------------------------
# Forward
# -------------------------
with torch.autocast(
    device_type="cuda",
    dtype=torch.bfloat16,
):
    output = model(batch)
    loss = output["loss"]

print("Loss:", loss.detach().float().item())

if not torch.isfinite(loss):
    raise RuntimeError("Non-finite SimCLR loss")

# -------------------------
# Backward
# -------------------------
loss.backward()

print("Backward: PASS")
print(
    "Peak GPU memory:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB",
)

print("\n===================================")
print("SIMCLR GPU SMOKE TEST: PASS")
print("===================================")
