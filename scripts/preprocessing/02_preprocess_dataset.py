from pathlib import Path
import argparse
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path("data/raw/Dataset702_Public_MRI")
OUT = Path("data/preprocessed/Dataset702_SSL")
MANIFEST = Path("data/manifests/dataset702_grouped.csv")

TARGET_DHW = (64, 128, 128)


def normalize(x):
    x = x.astype(np.float32)

    finite = np.isfinite(x)
    if not finite.any():
        raise ValueError("No finite voxels")

    x[~finite] = 0

    nz = x[x != 0]

    if nz.size > 100:
        lo, hi = np.percentile(nz, [0.5, 99.5])
        x = np.clip(x, lo, hi)

        mean = nz.mean()
        std = nz.std()
    else:
        mean = x.mean()
        std = x.std()

    if std < 1e-8:
        std = 1.0

    x = (x - mean) / std
    return x


def resize_inplane(x, target_hw=(128, 128)):
    # input H,W,D
    x = np.transpose(x, (2, 0, 1))       # D,H,W
    t = torch.from_numpy(x).float()[:, None]

    t = F.interpolate(
        t,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )

    return t[:, 0].numpy()                # D,H,W


def center_pad_crop_depth(x, target_d=64):
    d, h, w = x.shape

    if d > target_d:
        start = (d - target_d) // 2
        x = x[start:start + target_d]

    elif d < target_d:
        before = (target_d - d) // 2
        after = target_d - d - before

        x = np.pad(
            x,
            ((before, after), (0, 0), (0, 0)),
            mode="constant",
        )

    return x


def process(path):
    img = nib.load(str(path))
    x = np.asanyarray(img.dataobj)

    # Singleton 4D -> 3D
    if x.ndim == 4 and x.shape[-1] == 1:
        x = x[..., 0]

    if x.ndim != 3:
        raise ValueError(f"Unsupported shape {x.shape}")

    x = normalize(x)
    x = resize_inplane(x, (128, 128))
    x = center_pad_crop_depth(x, 64)

    if x.shape != TARGET_DHW:
        raise RuntimeError(f"Unexpected output shape {x.shape}")

    return x.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MANIFEST)

    end = len(df) if args.end is None else min(args.end, len(df))
    df = df.iloc[args.start:end]

    print(f"Processing rows {args.start}:{end}")

    ok = 0
    failed = []

    for j, (_, row) in enumerate(df.iterrows(), 1):

        src = ROOT / row["relative_path"]

        # Unique deterministic filename
        idx = args.start + j - 1
        dst = OUT / f"vol_{idx:06d}.npy"

        try:
            if not dst.exists():
                x = process(src)
                np.save(dst, x)

            ok += 1

        except Exception as e:
            failed.append((str(src), repr(e)))

        if j % 1000 == 0:
            print(
                f"{j:,}/{len(df):,} "
                f"ok={ok:,} failed={len(failed):,}",
                flush=True,
            )

    print("\nDONE")
    print("OK:", ok)
    print("FAILED:", len(failed))

    if failed:
        failfile = Path(
            f"outputs/preprocessing_logs/"
            f"failed_{args.start}_{end}.txt"
        )
        failfile.write_text(
            "\n".join(f"{p}\t{e}" for p,e in failed)
        )
        print("Failure log:", failfile)


if __name__ == "__main__":
    main()
