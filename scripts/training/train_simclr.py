import os
import torch

from cardiac_ssl.backbones.swin3d import Swin3D, Swin3DConfig
from cardiac_ssl.methods.simclr import SimCLR, SimCLRConfig
from cardiac_ssl.data.build import build_split_loader
from cardiac_ssl.training.trainer import Trainer, TrainerConfig
from cardiac_ssl.training.schedulers import WarmupCosineScheduler


# ============================================================
# Reproducibility
# ============================================================
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required")

device = torch.device("cuda:0")

print("=" * 70)
print("CARDIAC SSL - SIMCLR PRETRAINING")
print("=" * 70)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)


# ============================================================
# Frozen experiment configuration
# ============================================================
EPOCHS = 1000
TRAIN_ITERS = 500
VAL_ITERS = 50

BATCH_SIZE = 2
NUM_WORKERS = 8
NUM_CACHED = 8

PATCH_SIZE = (32, 160, 160)

LR = 6e-4
WEIGHT_DECAY = 0.04
WARMUP_EPOCHS = 40

OUTPUT_DIR = "outputs/pretraining/simclr_swin3d_dataset702"

SPLIT_FILE = "data/splits/splits_final_nnssl.json"

DATASET_ROOT = (
    "data/preprocessed/Dataset702_Public_MRI/"
    "nnsslPlans_onemmiso/Dataset702_Public_MRI"
)

PLAN_FILE = (
    "data/preprocessed/Dataset702_Public_MRI/"
    "nnsslPlans.json"
)


# ============================================================
# Backbone
# ============================================================
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


# ============================================================
# SimCLR
# ============================================================
model = SimCLR(
    backbone=backbone,
    config=SimCLRConfig(
        temperature=0.5,
        projector_hidden_dim=2048,
        projector_output_dim=2048,
    ),
)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print(f"Total parameters:     {n_params:,}")
print(f"Trainable parameters: {n_trainable:,}")
print("Training crop:", PATCH_SIZE)
print("Batch size:", BATCH_SIZE)


# ============================================================
# Dataset702 loaders
# ============================================================
print("\nBuilding training loader...")

train_loader = build_split_loader(
    split_file=SPLIT_FILE,
    split="train",
    dataset_root=DATASET_ROOT,
    plan=PLAN_FILE,
    configuration="onemmiso",
    method="simclr",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    num_cached=NUM_CACHED,
    iterations_per_epoch=TRAIN_ITERS,
    final_patch_size_override=PATCH_SIZE,
    spatial_augmentation=True,
    intensity_augmentation=True,
    seed=SEED,
)

print("Building validation loader...")

val_loader = build_split_loader(
    split_file=SPLIT_FILE,
    split="val",
    dataset_root=DATASET_ROOT,
    plan=PLAN_FILE,
    configuration="onemmiso",
    method="simclr",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    num_cached=NUM_CACHED,
    iterations_per_epoch=VAL_ITERS,
    final_patch_size_override=PATCH_SIZE,
    spatial_augmentation=True,
    intensity_augmentation=True,
    seed=SEED + 1,
)


# ============================================================
# Optimizer
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.95),
)


# ============================================================
# LR scheduler
# ============================================================
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

print("\nTraining schedule")
print("Epochs:", EPOCHS)
print("Iterations/epoch:", TRAIN_ITERS)
print("Validation iterations:", VAL_ITERS)
print("Total optimization steps:", TOTAL_STEPS)
print("Warmup steps:", WARMUP_STEPS)
print("Peak LR:", LR)


# ============================================================
# Trainer
# ============================================================
trainer_config = TrainerConfig(
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

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    config=trainer_config,
    scheduler=scheduler,
    device=device,
)


print("\n" + "=" * 70)
print("STARTING SIMCLR PRETRAINING")
print("=" * 70)

# Resume interrupted training if a checkpoint exists
resume_path = os.path.join(
    OUTPUT_DIR,
    "checkpoint_latest.pth",
)

if os.path.isfile(resume_path):
    print("\nResuming from:", resume_path)
    trainer.resume(resume_path)
    print(
        "Resume state: epoch=%d, global_step=%d, best_val_loss=%s"
        % (
            trainer.current_epoch,
            trainer.global_step,
            str(trainer.best_val_loss),
        )
    )
else:
    print("\nNo checkpoint found. Starting from scratch.")

trainer.fit()

print("\n" + "=" * 70)
print("SIMCLR PRETRAINING COMPLETE")
print("=" * 70)
