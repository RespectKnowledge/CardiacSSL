import torch

from cardiac_ssl.backbones.swin3d import Swin3D, Swin3DConfig
from cardiac_ssl.methods.vjepa import VJEPA, VJEPAConfig
from cardiac_ssl.data.build import build_split_loader
from cardiac_ssl.training.trainer import Trainer, TrainerConfig
from cardiac_ssl.training.schedulers import WarmupCosineScheduler

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

EPOCHS = 1000
TRAIN_ITERS = 250
VAL_ITERS = 50
BATCH_SIZE = 4
NUM_WORKERS = 8
NUM_CACHED = 8

PATCH_SIZE = (32, 160, 160)

LR = 6e-4
WEIGHT_DECAY = 0.04
WARMUP_EPOCHS = 40

OUTPUT_DIR = "outputs/pretraining/vjepa_swin3d_dataset702"

SPLIT_FILE = "data/splits/splits_final_nnssl.json"
DATASET_ROOT = (
    "data/preprocessed/Dataset702_Public_MRI/"
    "nnsslPlans_onemmiso/Dataset702_Public_MRI"
)
PLAN_FILE = (
    "data/preprocessed/Dataset702_Public_MRI/"
    "nnsslPlans.json"
)

device = torch.device("cuda:0")

print("=" * 70)
print("CARDIAC SSL - V-JEPA PRETRAINING")
print("=" * 70)
print("GPU:", torch.cuda.get_device_name(0))
print("Crop:", PATCH_SIZE)

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

model = VJEPA(
    backbone=backbone,
    config=VJEPAConfig(
        predictor_dim=384,
        predictor_depth=12,
        predictor_heads=12,
        mlp_ratio=4.0,
        ema_momentum=0.99925,
        num_context_blocks=8,
        num_target_blocks=2,
        context_scale=(0.15, 0.15),
        target_scale=(0.70, 0.70),
        aspect_ratio=(0.75, 1.50),
        mask_distance_weight=0.0,
    ),
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

train_loader = build_split_loader(
    split_file=SPLIT_FILE,
    split="train",
    dataset_root=DATASET_ROOT,
    plan=PLAN_FILE,
    configuration="onemmiso",
    method="vjepa",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    num_cached=NUM_CACHED,
    iterations_per_epoch=TRAIN_ITERS,
    final_patch_size_override=PATCH_SIZE,
    spatial_augmentation=True,
    intensity_augmentation=True,
    seed=SEED,
)

val_loader = build_split_loader(
    split_file=SPLIT_FILE,
    split="val",
    dataset_root=DATASET_ROOT,
    plan=PLAN_FILE,
    configuration="onemmiso",
    method="vjepa",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    num_cached=NUM_CACHED,
    iterations_per_epoch=VAL_ITERS,
    final_patch_size_override=PATCH_SIZE,
    spatial_augmentation=True,
    intensity_augmentation=True,
    seed=SEED + 1,
)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)

TOTAL_STEPS = EPOCHS * TRAIN_ITERS
WARMUP_STEPS = WARMUP_EPOCHS * TRAIN_ITERS

scheduler = WarmupCosineScheduler(
    optimizer=optimizer,
    warmup_steps=WARMUP_STEPS,
    total_steps=TOTAL_STEPS,
    start_lr=0.0,
    peak_lr=LR,
    final_lr=0.0,
)

config = TrainerConfig(
    epochs=EPOCHS,
    train_iterations_per_epoch=TRAIN_ITERS,
    val_iterations_per_epoch=VAL_ITERS,
    precision="bf16",
    gradient_clip_norm=None,
    output_dir=OUTPUT_DIR,
    checkpoint_every=1,
    compile_model=False,
    find_unused_parameters=False,
    stop_on_nonfinite_loss=True,
)

print("Epochs:", EPOCHS)
print("Iterations/epoch:", TRAIN_ITERS)
print("Total steps:", TOTAL_STEPS)
print("Warmup steps:", WARMUP_STEPS)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
    scheduler=scheduler,
    device=device,
)

CHECKPOINT = (
    "outputs/pretraining/vjepa_swin3d_dataset702/"
    "checkpoint_latest.pth"
)

print("Resuming from:", CHECKPOINT)
trainer.resume(CHECKPOINT)

print(
    "Resumed at epoch:",
    trainer.current_epoch,
    "global step:",
    trainer.global_step,
)

trainer.fit()
