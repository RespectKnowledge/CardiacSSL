from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path("data/downstream/nnUNet_raw")

datasets = {
    "Dataset703_ACDC": {
        "imagesTr": 200,
        "labelsTr": 200,
        "imagesTs": 100,
        "labelsTs": 100,
    },
    "Dataset704_MnM2": {
        "imagesTr": 320,
        "labelsTr": 320,
        "imagesVal": 80,
        "labelsVal": 80,
        "imagesTs": 320,
        "labelsTs": 320,
    },
}

failed = False

for dataset, expected in datasets.items():
    root = ROOT / dataset

    print("\n" + "=" * 70)
    print(dataset)
    print("=" * 70)

    for folder, expected_count in expected.items():
        files = sorted((root / folder).glob("*.nii.gz"))
        print(f"{folder:10s}: {len(files):4d}  expected={expected_count}")

        if len(files) != expected_count:
            failed = True

    # Verify each image has its corresponding label
    for image_folder, label_folder in [
        ("imagesTr", "labelsTr"),
        ("imagesVal", "labelsVal"),
        ("imagesTs", "labelsTs"),
    ]:
        idir = root / image_folder
        ldir = root / label_folder

        if not idir.exists():
            continue

        missing = []
        geometry_errors = []
        all_labels = set()

        for image_path in sorted(idir.glob("*_0000.nii.gz")):
            case = image_path.name.replace("_0000.nii.gz", "")
            label_path = ldir / f"{case}.nii.gz"

            if not label_path.exists():
                missing.append(case)
                continue

            img = nib.load(str(image_path))
            seg = nib.load(str(label_path))

            if img.shape != seg.shape:
                geometry_errors.append(
                    (case, img.shape, seg.shape)
                )

            if not np.allclose(img.affine, seg.affine, atol=1e-4):
                geometry_errors.append(
                    (case, "affine mismatch")
                )

            values = np.unique(np.asanyarray(seg.dataobj))
            all_labels.update(values.tolist())

        print(
            f"{image_folder}/{label_folder}: "
            f"missing={len(missing)}, "
            f"geometry_errors={len(geometry_errors)}, "
            f"labels={sorted(all_labels)}"
        )

        if missing:
            failed = True
            print("First missing:", missing[:5])

        if geometry_errors:
            failed = True
            print("First geometry errors:", geometry_errors[:5])

        if all_labels != {0, 1, 2, 3}:
            failed = True
            print("ERROR: unexpected label set")

    print("dataset.json:", (root / "dataset.json").exists())

print("\n" + "=" * 70)

if failed:
    raise SystemExit("FAIL: dataset integrity problems detected.")

print("PASS: all generated downstream datasets are internally consistent.")
