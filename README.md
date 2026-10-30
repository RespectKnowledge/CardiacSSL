# CardiacSwinDINO

**Hierarchical 3D Swin Transformer self-supervised learning for cardiac MRI segmentation**

This repository contains the research code for **SwinDINO-3D**, a DINO-style teacher–student self-supervised framework for volumetric cardiac MRI, and comparisons with **Swin-MAE**, **Swin-SimCLR**, **Swin-V-JEPA**, and a randomly initialized (**Scratch**) segmentation model. Downstream evaluation uses **ACDC** and **M&Ms-2**.

## 1. Repository layout

The following structure is visible in the project snapshot; subdirectories and individual script filenames must be checked against the repository before publication.

```text
Cardiac_SSL/
├── cardiac_ssl/              # Model and SSL implementation
├── configs/                  # Training and evaluation configurations
├── data/                     # Dataset configuration and split files
├── docs/                     # Documentation
├── downstream/               # nnU-Net v2 downstream integration
├── outputs/                  # Training outputs/checkpoints
├── predictions/              # Segmentation predictions
├── scripts/                  # Training / inference entry points
├── evaluate_acdc.py
├── evaluate_mnm2.py
├── evaluate_acdc_scratch.py
└── statistical_analysis_all.py
```

## 2. Environment and installation

The downstream pipeline uses **PyTorch** and **nnU-Net v2**, with a custom `SwinSSLUNet` network and `nnUNetTrainerSwinSSL` trainer. The Scratch experiment uses `nnUNetTrainerSwinSSL250`.

**Before publishing:** provide a pinned `requirements.txt` or `environment.yml`, the tested Python/PyTorch/CUDA or ROCm versions, and installation instructions for any custom nnU-Net modules.

```bash
# TODO: replace with tested installation commands from your environment
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```

## 3. Data and preprocessing

| Stage | Dataset identifier | Use |
|---|---|---|
| SSL pretraining | `Dataset702_Public_MRI` | Unlabelled cardiac MRI representation learning |
| Downstream | `Dataset703_ACDC` | Supervised fine-tuning and held-out ACDC evaluation |
| External evaluation | `Dataset704_MnM2` | M&Ms-2 evaluation of the ACDC-trained model without target-domain adaptation |

The verified SSL split file is `data/splits/splits_final_nnssl.json`, using the `onemmiso` preprocessing configuration. The verified SSL split comprises **54,916 training** and **2,964 validation** volumes (**57,880 total**). The ACDC evaluation collection contains **100 held-out volumes**; the M&Ms-2 held-out collection contains **320 volumes**.

**Data access:** The datasets are not distributed in this repository. Add the official dataset access links, licenses, required directory layout, preprocessing commands, and integrity/split checks before release. Do not publish restricted medical images.

## 4. Model architecture

The SSL methods use a common **hierarchical 3D Swin Transformer encoder**:

| Setting | Value |
|---|---|
| Input channels | 1 |
| Patch size | `(2, 2, 2)` |
| Window size | `(7, 7, 7)` |
| Embedding dimension | 48 |
| Stage depths | `(2, 2, 2, 2)` |
| Attention heads | `(3, 6, 12, 24)` |
| MLP ratio | 4.0 |
| Downsampling | Patch merging |
| Dropout / attention dropout / drop path | 0 |

SwinDINO-3D trains a student using DINO-style self-distillation and updates the teacher using an exponential moving average. Only the pretrained student encoder is transferred to the downstream segmentation model; the teacher and projection heads are not required at inference.

## 5. Self-supervised pretraining

All methods use a global crop of **`32 × 160 × 160` voxels**, AdamW with learning rate **`6e-4`**, weight decay **`0.04`**, **40 warm-up epochs**, and a cosine learning-rate schedule. Random seed: **42**. The verified training configurations are:

| Method | Batch size | Epoch target | Training iterations/epoch | Specific settings |
|---|---:|---:|---:|---|
| SwinDINO-3D | 2 | 1,000 | 500 | 2 global crops + 4 local crops (`32 × 96 × 96`) |
| Swin-MAE | 2 | 1,000 | 500 | Mask ratio 0.75; mask block size 16 |
| Swin-SimCLR | 2 | 1,000 | 500 | NT-Xent temperature 0.5 |
| Swin-V-JEPA | 4 | 1,000 | 250 | EMA momentum 0.99925; predictor dim 384, depth 12 |

The project uses spatial and MRI intensity augmentation. Check the implementation for the exact augmentation sequence and probabilities. The recorded DINO checkpoint was at epoch **789**; do not describe it as a completed 1,000-epoch run without checking the final checkpoint.

### Launch pretraining

**TODO: verify executable script names and flags** in `scripts/` or `cardiac_ssl/` and publish one working command per method. Suggested documentation format:

```bash
# SwinDINO-3D
# python <actual_dino_entrypoint>.py --config <actual_dino_config>

# Swin-MAE
# python <actual_mae_entrypoint>.py --config <actual_mae_config>

# Swin-SimCLR
# python <actual_simclr_entrypoint>.py --config <actual_simclr_config>

# Swin-V-JEPA
# python <actual_vjepa_entrypoint>.py --config <actual_vjepa_config>
```

Record checkpoint output locations, GPU type, GPU count, effective batch size, runtime and peak memory for each experiment.

## 6. Downstream segmentation

The pretrained student encoder is loaded into the **SwinSSLUNet** segmentation architecture via the custom `nnUNetTrainerSwinSSL` integration. The encoder and decoder are fine-tuned **end-to-end**. Scratch uses the same segmentation architecture initialized randomly and the `nnUNetTrainerSwinSSL250` trainer.

| Setting | SSL-initialized segmentation | Scratch |
|---|---|---|
| Batch size | 2 | 2 |
| Epochs | 1,000 | 1000 |
| Training iterations/epoch | 250 | 250 |
| Validation iterations/epoch | 50 | 50 |
| Optimizer | SGD, momentum 0.99, Nesterov | Same |
| Initial learning rate | 0.01 | 0.01 |
| Weight decay | `3e-5` | `3e-5` |
| Loss | Dice + cross-entropy, deep supervision | Same |

The verified ACDC training patch is **`20 × 256 × 224`**. The M&Ms-2 plan uses **`12 × 320 × 320`**. Runtime batch size is overridden to 2 by the custom trainer.

### Checkpoint transfer

The custom trainer reads the **`CARDIAC_SSL_CHECKPOINT`** environment variable and loads the SSL encoder with `load_ssl_encoder(..., strict=True)`.

```bash
# Example of the verified environment-variable interface;
# replace the path and nnU-Net command with your actual checkpoint and tested invocation.
export CARDIAC_SSL_CHECKPOINT=/path/to/pretrained_encoder_checkpoint

# TODO: add tested nnUNetv2_train command for Dataset703_ACDC
# TODO: add tested inference command and checkpoint selection
```

**Important evaluation distinction:** To reproduce the manuscript's cross-domain results, fine-tune on **ACDC** and evaluate the resulting checkpoint on **M&Ms-2 without additional M&Ms-2 fine-tuning or adaptation**. If the repository also contains M&Ms-2 training experiments, document them as separate experiments and do not conflate them with external evaluation.

## 7. Evaluation and statistical analysis

The project contains the following evaluation entry points:

```text
evaluate_acdc.py
evaluate_mnm2.py
evaluate_acdc_scratch.py
statistical_analysis_all.py
```

The evaluation protocol measures **Dice**, **IoU**, **HD95 (mm)** and **average surface distance (ASD, mm)** for **right ventricle (RV)**, **myocardium (MYO)** and **left ventricle (LV)**, with summary statistics. Run the scripts against held-out predictions and corresponding ground truth. Exact arguments and expected directory layout must be documented from `--help` or the source files.

## 8. Reproducibility checklist

- [ ] Pin software dependencies and specify hardware.
- [ ] Document dataset access, preprocessing, split generation and leakage checks.
- [ ] Publish verified CLI commands for all four SSL methods.
- [ ] Publish verified ACDC training and external M&Ms-2 inference commands.
- [ ] Provide checkpoint download links and expected file names.
- [ ] Document evaluation script flags, input/output paths and expected metrics.
- [ ] Include seeds and clarify checkpoint-selection rules.
- [ ] Provide measured FLOPs, GPU memory, training duration and inference latency.
- [ ] Add team affiliations, citation, license and data usage terms.

## 9. Citation and license

**Citation:** TODO — add the final manuscript citation or BibTeX entry.

**License:** TODO — choose a license compatible with the code and dependencies; data and pretrained-weight distribution may have separate restrictions.

## 11. Contact

TODO — add the project maintainer or issue-tracker instructions.
