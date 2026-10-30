from pathlib import Path
import pandas as pd
import re
from collections import Counter

ROOT = Path("data/raw/Dataset702_Public_MRI")
GEOM = Path("data/manifests/dataset702_geometry.csv")
OUT = Path("data/manifests/dataset702_grouped.csv")
SUMMARY = Path("outputs/qc/dataset702_grouping_summary.txt")

df = pd.read_csv(GEOM)

def parse_name(filename):
    s = filename
    if s.endswith(".nii.gz"):
        s = s[:-7]

    original = s

    # nnU-Net modality suffix may occur at the end
    s = re.sub(r"_0000$", "", s, flags=re.I)

    # Examples:
    # 814_frame9_0000
    # Upgrade_18_Pre_frame2_0000
    m = re.match(r"^(.*)_frame(\d+)$", s, re.I)
    if m:
        return m.group(1), f"frame{int(m.group(2))}", "frame"

    # Example where modality marker follows frame:
    # 03183_frame10_0000_sa
    m = re.match(r"^(.*)_frame(\d+)_0000_sa$", original, re.I)
    if m:
        return m.group(1), f"frame{int(m.group(2))}", "frame_0000_sa"

    # 1-043-003__sax_12
    m = re.match(r"^(.*?)[_]+sax_(\d+)$", s, re.I)
    if m:
        return m.group(1), f"frame{int(m.group(2))}", "sax"

    # H7N4V9_sa_12
    m = re.match(r"^(.*)_sa_(\d+)$", s, re.I)
    if m:
        return m.group(1), f"frame{int(m.group(2))}", "sa"

    # C001_UIH_umr780_sax_P010_time9
    m = re.match(r"^(.*)_time(\d+)$", s, re.I)
    if m:
        return m.group(1), f"frame{int(m.group(2))}", "time"

    # 001_saed / 001_saes
    m = re.match(r"^(.*)_sa(ed|es)$", s, re.I)
    if m:
        return m.group(1), m.group(2).upper(), "sa_phase"

    # SA_ED / SA_ES variants
    m = re.match(r"^(.*)_sa[_-]?(ed|es)$", s, re.I)
    if m:
        return m.group(1), m.group(2).upper(), "sa_phase"

    # Generic trailing number, only as fallback
    m = re.match(r"^(.*)_(\d+)$", s)
    if m:
        return m.group(1), f"index{int(m.group(2))}", "trailing_number"

    return s, "", "single_or_unknown"


parsed = df["relative_path"].apply(
    lambda x: parse_name(Path(str(x)).name)
)

df["cine_id"] = [x[0] for x in parsed]
df["temporal_id"] = [x[1] for x in parsed]
df["naming_family"] = [x[2] for x in parsed]

df.to_csv(OUT, index=False)

groups = (
    df.groupby(["cine_id"], dropna=False)
      .size()
      .reset_index(name="n_volumes")
)

family_counts = Counter(df["naming_family"])

lines = []
lines.append("Dataset702 cine grouping")
lines.append("=" * 70)
lines.append(f"Volumes: {len(df):,}")
lines.append(f"Candidate cine/subject groups: {len(groups):,}")
lines.append("")
lines.append("Naming families:")

for k,v in family_counts.most_common():
    lines.append(f"  {k:25s}: {v:,}")

lines.append("")
lines.append("Volumes per candidate cine:")
lines.append(groups["n_volumes"].describe().to_string())

lines.append("")
lines.append("Group-size frequencies:")
lines.append(
    groups["n_volumes"]
    .value_counts()
    .sort_index()
    .head(50)
    .to_string()
)

lines.append("")
lines.append("Largest groups:")
lines.append(
    groups.sort_values("n_volumes", ascending=False)
          .head(40)
          .to_string(index=False)
)

lines.append("")
lines.append("Unknown/single naming examples:")

unknown = df[df["naming_family"] == "single_or_unknown"]
for x in unknown["relative_path"].head(50):
    lines.append(f"  {x}")

SUMMARY.write_text("\n".join(lines))

print("\n".join(lines))
print()
print("Manifest:", OUT)
print("Summary :", SUMMARY)
