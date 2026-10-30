from pathlib import Path
import json
import shutil

SRC = Path("data/downstream/raw/MnM2")
DST = Path("data/downstream/nnUNet_raw/Dataset704_MnM2")

imagesTr = DST / "imagesTr"
labelsTr = DST / "labelsTr"
imagesVal = DST / "imagesVal"
labelsVal = DST / "labelsVal"
imagesTs = DST / "imagesTs"
labelsTs = DST / "labelsTs"

# Recreate output directories cleanly.
if DST.exists():
    shutil.rmtree(DST)

for d in (
    imagesTr, labelsTr,
    imagesVal, labelsVal,
    imagesTs, labelsTs,
):
    d.mkdir(parents=True, exist_ok=True)


# Official M&Ms-2 challenge split.
# README states 160 training, 40 validation and 160 testing
# subjects included sequentially.
train_ids = {f"{i:03d}" for i in range(1, 161)}
val_ids   = {f"{i:03d}" for i in range(161, 201)}
test_ids  = {f"{i:03d}" for i in range(201, 361)}

print("Official sequential subject split:")
print("Train:", len(train_ids), "001-160")
print("Val  :", len(val_ids),   "161-200")
print("Test :", len(test_ids),  "201-360")

assert len(train_ids) == 160
assert len(val_ids) == 40
assert len(test_ids) == 160
assert not (train_ids & val_ids)
assert not (train_ids & test_ids)
assert not (val_ids & test_ids)


def locate(pid, phase):
    pattern = f"{pid}_SA_{phase}.nii.gz"
    found = list(SRC.rglob(pattern))

    if len(found) != 1:
        raise RuntimeError(
            f"{pid} {phase}: expected exactly one {pattern}, "
            f"found {len(found)}"
        )

    image = found[0]
    gt = image.with_name(
        image.name.replace(".nii.gz", "_gt.nii.gz")
    )

    if not gt.exists():
        raise FileNotFoundError(
            f"Missing GT for {image}: {gt}"
        )

    return image, gt


def convert(subject_ids, out_images, out_labels):
    n = 0

    for pid in sorted(subject_ids):
        for phase in ("ED", "ES"):
            image, gt = locate(pid, phase)

            case = f"mnm2_{pid}_{phase.lower()}"

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


n_train = convert(train_ids, imagesTr, labelsTr)
n_val   = convert(val_ids, imagesVal, labelsVal)
n_test  = convert(test_ids, imagesTs, labelsTs)

assert n_train == 320
assert n_val == 80
assert n_test == 320

dataset_json = {
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "LV": 1,
        "MYO": 2,
        "RV": 3
    },
    "numTraining": n_train,
    "file_ending": ".nii.gz"
}

with open(DST / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print("\nM&M2 nnU-Net dataset created")
print("Training subjects   : 160")
print("Training volumes    :", n_train)
print("Validation subjects : 40")
print("Validation volumes  :", n_val)
print("Testing subjects    : 160")
print("Testing volumes     :", n_test)
print("Total subjects      : 360")
print("Total ED/ES volumes :", n_train + n_val + n_test)
print("Output:", DST)
