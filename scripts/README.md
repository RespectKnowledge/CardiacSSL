# Cardiac SSL Data Pipeline

Reproducible preprocessing, quality-control and training utilities for
self-supervised learning on cardiac cine MRI.

## Pipeline

1. Raw data inventory and QC
2. Dataset provenance and duplicate checking
3. Cine-series / subject grouping
4. Downstream holdout exclusion
5. Cardiac SAX MRI preprocessing
6. Preprocessing QC
7. Subject-level train/validation split
8. Self-supervised pretraining
9. Checkpoint and experiment reporting

## Raw data

Raw datasets must never be modified in place.

`data/raw/` contains local dataset copies and is not intended for inclusion
in the public Git repository.

## Generated data

Preprocessed images, checkpoints, logs and other large generated artifacts
are excluded from Git. Scripts, configuration files, manifests/splits where
permitted, and documentation are version controlled.
