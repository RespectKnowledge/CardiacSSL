from pathlib import Path
from collections import Counter
import pandas as pd
import re

ROOT = Path("data/raw/Dataset702_Public_MRI")
OUT = Path("data/manifests/dataset702_inventory.csv")
SUMMARY = Path("outputs/qc/dataset702_inventory_summary.txt")

files = sorted(ROOT.rglob("*.nii.gz"))

print(f"Found {len(files):,} NIfTI files")

rows = []
patterns = Counter()

for i, p in enumerate(files, 1):

    name = p.name
    stem = name[:-7]  # remove .nii.gz

    source_pattern = "other"
    subject_id = stem
    frame_id = ""

    # patient017_frame01
    m = re.match(r"(.+)_frame(\d+)$", stem, re.I)
    if m:
        source_pattern = "frame"
        subject_id = m.group(1)
        frame_id = m.group(2)

    else:
        # C005_..._P011_time3
        m = re.match(r"(.+)_time(\d+)$", stem, re.I)
        if m:
            source_pattern = "time"
            subject_id = m.group(1)
            frame_id = m.group(2)

        else:
            # B9E0Q1_sa_18
            m = re.match(r"(.+)_sa_(\d+)$", stem, re.I)
            if m:
                source_pattern = "sa"
                subject_id = m.group(1)
                frame_id = m.group(2)

            else:
                # 1-048-005__sax_00
                m = re.match(r"(.+)_sax_(\d+)$", stem, re.I)
                if m:
                    source_pattern = "sax"
                    subject_id = m.group(1)
                    frame_id = m.group(2)

                else:
                    # Explicit ED/ES cases
                    m = re.match(r"(.+)_SA_(ED|ES)$", stem, re.I)
                    if m:
                        source_pattern = "sa_phase"
                        subject_id = m.group(1)
                        frame_id = m.group(2).upper()

                    else:
                        # Generic trailing number, e.g. patient017_12
                        m = re.match(r"(.+?)_(\d+)$", stem)
                        if m:
                            source_pattern = "trailing_number"
                            subject_id = m.group(1)
                            frame_id = m.group(2)

    patterns[source_pattern] += 1

    rows.append({
        "path": str(p),
        "relative_path": str(p.relative_to(ROOT)),
        "filename": name,
        "subject_candidate": subject_id,
        "frame_candidate": frame_id,
        "pattern": source_pattern,
        "size_bytes": p.stat().st_size,
    })

    if i % 10000 == 0:
        print(f"Processed {i:,}/{len(files):,}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

group_sizes = (
    df.groupby(["pattern", "subject_candidate"])
      .size()
      .reset_index(name="n_volumes")
)

text = []
text.append("Dataset702 inventory")
text.append("=" * 70)
text.append(f"Total NIfTI files: {len(df):,}")
text.append("")
text.append("Filename patterns:")

for k, v in patterns.most_common():
    text.append(f"  {k:20s}: {v:,}")

text.append("")
text.append(
    f"Unique candidate groups: "
    f"{df[['pattern','subject_candidate']].drop_duplicates().shape[0]:,}"
)

text.append("")
text.append("Candidate group-size distribution:")
text.append(group_sizes["n_volumes"].describe().to_string())

text.append("")
text.append("Largest candidate groups:")
text.append(
    group_sizes.sort_values("n_volumes", ascending=False)
               .head(30)
               .to_string(index=False)
)

SUMMARY.write_text("\n".join(text))

print()
print("\n".join(text))
print()
print("Inventory :", OUT)
print("Summary   :", SUMMARY)
