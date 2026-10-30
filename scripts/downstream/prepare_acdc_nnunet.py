from pathlib import Path
import json
import shutil

SRC = Path("data/downstream/raw/ACDC")
DST = Path("data/downstream/nnUNet_raw/Dataset703_ACDC")

imagesTr = DST / "imagesTr"
labelsTr = DST / "labelsTr"
imagesTs = DST / "imagesTs"
labelsTs = DST / "labelsTs"

for d in (imagesTr, labelsTr, imagesTs, labelsTs):
    d.mkdir(parents=True, exist_ok=True)


def convert_split(src_root, out_images, out_labels):
    n = 0

    for gt in sorted(src_root.rglob("*_frame*_gt.nii.gz")):
        image = gt.with_name(gt.name.replace("_gt.nii.gz", ".nii.gz"))

        if not image.exists():
            raise FileNotFoundError(f"Missing image for {gt}")

        case = gt.name.replace("_gt.nii.gz", "")

        shutil.copy2(
            image,
            out_images / f"{case}_0000.nii.gz"
        )
        shutil.copy2(
            gt,
            out_labels / f"{case}.nii.gz"
        )

        n += 1

    return n


n_train = convert_split(
    SRC / "training",
    imagesTr,
    labelsTr,
)

n_test = convert_split(
    SRC / "testing",
    imagesTs,
    labelsTs,
)

dataset_json = {
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "RV": 1,
        "MYO": 2,
        "LV": 3
    },
    "numTraining": n_train,
    "file_ending": ".nii.gz"
}

with open(DST / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print("ACDC nnU-Net dataset created")
print("Training volumes:", n_train)
print("Testing volumes :", n_test)
print("Output:", DST)
