import os
import torch

from cardiac_ssl.backbones.swin3d import Swin3D, Swin3DConfig
from cardiac_ssl.methods.dinov2 import DINOv2, DINOv2Config
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


# ============================================================
# Frozen experiment configuration
# ============================================================
EPOCHS = 1000
TRAIN_ITERS = 500
VAL_ITERS = 50

BATCH_SIZE = 2
NUM_WORKERS = 8
NUM_CACHED = 8

# Same global crop as MAE / SimCLR
PATCH_SIZE = (32, 160, 160)

# DINO local views.
# Must be divisible by 32 for this Swin3D implementation.
LOCAL_CROP_SIZE = (32, 96, 96)
N_LOCAL_CROPS = 4

LR = 6e-4
WEIGHT_DECAY = 0.04
WARMUP_EPOCHS = 40

OUTPUT_DIR = "outputs/pretraining/dino_swin3d_dataset702"

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
# Environment
# ============================================================
if "nnssl_preprocessed" not in os.environ:
    os.environ["nnssl_preprocessed"] = os.path.abspath(
        "data/preprocessed"
    )

print("=" * 70)
print("CARDIAC SSL - SWIN-DINO PRETRAINING")
print("=" * 70)
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Global crop:", PATCH_SIZE)
print("Local crop:", LOCAL_CROP_SIZE)
print("Number local crops:", N_LOCAL_CROPS)
print("Batch size:", BATCH_SIZE)


# ============================================================
# Swin3D backbone
# Same architecture as MAE / SimCLR
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
# DINO
# ============================================================
model = DINOv2(
    backbone=backbone,
    config=DINOv2Config(),
)

n_params = sum(p.numel() for p in model.parameters())
n_trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Total parameters:     {:,}".format(n_params))
print("Trainable parameters: {:,}".format(n_trainable))


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
    method="dinov2",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    num_cached=NUM_CACHED,
    iterations_per_epoch=TRAIN_ITERS,
    final_patch_size_override=PATCH_SIZE,
    dinov2_local_crop_size=LOCAL_CROP_SIZE,
    dinov2_n_local_crops=N_LOCAL_CROPS,
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
    method="dinov2",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    num_cached=NUM_CACHED,
    iterations_per_epoch=VAL_ITERS,
    final_patch_size_override=PATCH_SIZE,
    dinov2_local_crop_size=LOCAL_CROP_SIZE,
    dinov2_n_local_crops=N_LOCAL_CROPS,
    spatial_augmentation=True,
    intensity_augmentation=True,
    seed=SEED + 1,
)


# ============================================================
# Optimizer
# Teacher parameters have requires_grad=False, but filtering
# keeps the optimizer explicitly student-only.
# ============================================================
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
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


# ============================================================
# Resume
# ============================================================
print("\n" + "=" * 70)
print("STARTING SWIN-DINO PRETRAINING")
print("=" * 70)

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
