from pathlib import Path
import nibabel as nib
import pandas as pd
import numpy as np

ROOT = Path("data/raw/Dataset702_Public_MRI")
OUT = Path("data/manifests/dataset702_geometry.csv")
SUMMARY = Path("outputs/qc/dataset702_geometry_summary.txt")

files = sorted(ROOT.rglob("*.nii.gz"))

rows = []
bad = []

for i, p in enumerate(files, 1):
    try:
        img = nib.load(str(p))

        shape = tuple(int(x) for x in img.shape)
        zooms = tuple(float(x) for x in img.header.get_zooms()[:len(shape)])

        rows.append({
            "path": str(p),
            "relative_path": str(p.relative_to(ROOT)),
            "ndim": len(shape),
            "shape": "x".join(map(str, shape)),
            "sx": shape[0] if len(shape) > 0 else np.nan,
            "sy": shape[1] if len(shape) > 1 else np.nan,
            "sz": shape[2] if len(shape) > 2 else np.nan,
            "spacing_x": zooms[0] if len(zooms) > 0 else np.nan,
            "spacing_y": zooms[1] if len(zooms) > 1 else np.nan,
            "spacing_z": zooms[2] if len(zooms) > 2 else np.nan,
            "dtype": str(img.get_data_dtype()),
            "error": "",
        })

    except Exception as e:
        bad.append((str(p), repr(e)))

    if i % 5000 == 0:
        print(f"Checked {i:,}/{len(files):,}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

lines = [
    "Dataset702 NIfTI header QC",
    "=" * 70,
    f"Total files: {len(files):,}",
    f"Readable:    {len(df):,}",
    f"Failed:      {len(bad):,}",
]

if len(df):
    lines += [
        "",
        "Dimensions:",
        df["ndim"].value_counts().sort_index().to_string(),
        "",
        "Most common shapes:",
        df["shape"].value_counts().head(30).to_string(),
        "",
        "Spacing summary:",
        df[
            ["spacing_x", "spacing_y", "spacing_z"]
        ].describe().to_string(),
    ]

if bad:
    lines += ["", "FAILED FILES:"]
    for p,e in bad[:100]:
        lines.append(f"{p}\t{e}")

SUMMARY.write_text("\n".join(lines))

print("\n".join(lines))
print("\nGeometry CSV:", OUT)
print("Summary:", SUMMARY)
