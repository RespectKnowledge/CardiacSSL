#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

ROOT = Path("predictions/ACDC")
GT_DIR = ROOT / "GT"

MODELS = {
    "Scratch": ROOT / "Scratch",
    "MAE": ROOT / "MAE",
    "SimCLR": ROOT / "SimCLR",
    "VJEPA": ROOT / "VJEPA",
    "DINO": ROOT / "DINO",
}

LABELS = {
    1: "RV",
    2: "MYO",
    3: "LV",
}


def load_image(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)

    # SimpleITK spacing is x,y,z whereas the array is z,y,x
    spacing = np.asarray(img.GetSpacing(), dtype=float)[::-1]

    return arr, spacing


def dice(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    denom = gt.sum() + pred.sum()

    if denom == 0:
        return np.nan

    return 2.0 * np.logical_and(gt, pred).sum() / denom


def iou(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return np.nan

    return np.logical_and(gt, pred).sum() / union


def get_surface(mask):
    mask = mask.astype(bool)
    eroded = binary_erosion(mask)
    return mask ^ eroded


def surface_metrics(gt, pred, spacing):

    gt_surface = get_surface(gt)
    pred_surface = get_surface(pred)

    gt_points = np.argwhere(gt_surface).astype(float)
    pred_points = np.argwhere(pred_surface).astype(float)

    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.nan, np.nan

    # Convert voxel coordinates to physical coordinates (mm)
    gt_points *= spacing
    pred_points *= spacing

    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)

    gt_to_pred = pred_tree.query(gt_points, k=1)[0]
    pred_to_gt = gt_tree.query(pred_points, k=1)[0]

    all_distances = np.concatenate([
        gt_to_pred,
        pred_to_gt
    ])

    hd95 = np.percentile(all_distances, 95)

    asd = (
        gt_to_pred.mean() +
        pred_to_gt.mean()
    ) / 2.0

    return hd95, asd


rows = []

gt_files = sorted(GT_DIR.glob("*.nii.gz"))

print("GT cases:", len(gt_files))


for model_name, pred_dir in MODELS.items():

    print("\n==============================")
    print("Evaluating:", model_name)
    print("==============================")

    pred_files = {
        p.name: p
        for p in pred_dir.glob("*.nii.gz")
    }

    matched = 0

    for gt_path in gt_files:

        pred_path = pred_files.get(gt_path.name)

        if pred_path is None:
            print("Missing prediction:", gt_path.name)
            continue

        matched += 1

        gt, spacing = load_image(gt_path)
        pred, pred_spacing = load_image(pred_path)

        if gt.shape != pred.shape:
            print(
                "SHAPE MISMATCH:",
                gt_path.name,
                gt.shape,
                pred.shape
            )
            continue

        for label, structure in LABELS.items():

            gt_mask = gt == label
            pred_mask = pred == label

            dc = dice(gt_mask, pred_mask)
            jac = iou(gt_mask, pred_mask)

            hd95, asd = surface_metrics(
                gt_mask,
                pred_mask,
                spacing
            )

            rows.append({
                "Model": model_name,
                "Case": gt_path.name,
                "Label": label,
                "Structure": structure,
                "Dice": dc,
                "IoU": jac,
                "HD95_mm": hd95,
                "ASD_mm": asd,
            })

    print("Matched cases:", matched)


df = pd.DataFrame(rows)

per_case_file = ROOT / "ACDC_metrics_per_case.csv"
df.to_csv(per_case_file, index=False)


summary = (
    df.groupby(["Model", "Structure"])
      [["Dice", "IoU", "HD95_mm", "ASD_mm"]]
      .agg(["mean", "std"])
)

summary_file = ROOT / "ACDC_metrics_summary.csv"
summary.to_csv(summary_file)


# Overall model statistics across structures/cases
overall = (
    df.groupby("Model")
      [["Dice", "IoU", "HD95_mm", "ASD_mm"]]
      .agg(["mean", "std"])
)

overall_file = ROOT / "ACDC_metrics_overall.csv"
overall.to_csv(overall_file)


print("\n========================================")
print("PER-STRUCTURE RESULTS")
print("========================================")
print(summary)

print("\n========================================")
print("OVERALL RESULTS")
print("========================================")
print(overall)

print("\nSaved:")
print(per_case_file)
print(summary_file)
print(overall_file)
