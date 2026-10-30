import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

DATASETS = {
    "MnM2": Path("predictions/MnM2/MnM2_metrics_per_case.csv"),
    "ACDC": Path("predictions/ACDC/ACDC_metrics_per_case.csv"),
}

METRICS = ["Dice", "IoU", "HD95_mm", "ASD_mm"]
rows = []

for dataset, path in DATASETS.items():

    df = pd.read_csv(path)

    for structure in ["LV", "MYO", "RV"]:

        d = df[df["Structure"] == structure]

        for metric in METRICS:

            mae = (
                d[d["Model"] == "MAE"]
                .set_index("Case")[metric]
            )

            simclr = (
                d[d["Model"] == "SimCLR"]
                .set_index("Case")[metric]
            )

            common = mae.index.intersection(simclr.index)

            x = mae.loc[common].values
            y = simclr.loc[common].values

            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]

            diff = x - y

            if len(x) > 0 and np.any(diff != 0):
                statistic, p = wilcoxon(
                    x,
                    y,
                    zero_method="wilcox",
                    alternative="two-sided"
                )
            else:
                statistic, p = np.nan, np.nan

            rows.append({
                "Dataset": dataset,
                "Structure": structure,
                "Metric": metric,
                "N": len(x),

                "MAE_mean": np.mean(x),
                "MAE_std": np.std(x, ddof=1),

                "SimCLR_mean": np.mean(y),
                "SimCLR_std": np.std(y, ddof=1),

                "Mean_diff_MAE_minus_SimCLR":
                    np.mean(diff),

                "Wilcoxon_statistic": statistic,
                "p_raw": p
            })

res = pd.DataFrame(rows)

# Holm correction performed separately within each dataset
res["p_Holm"] = np.nan

for dataset in res["Dataset"].unique():

    idx = res["Dataset"] == dataset
    pvals = res.loc[idx, "p_raw"].values

    valid = np.isfinite(pvals)
    pv = pvals[valid]

    order = np.argsort(pv)
    m = len(pv)

    adjusted = np.empty(m)
    running_max = 0

    for rank, j in enumerate(order):
        value = (m - rank) * pv[j]
        running_max = max(running_max, value)
        adjusted[j] = min(running_max, 1.0)

    temp = np.full(len(pvals), np.nan)
    temp[valid] = adjusted

    res.loc[idx, "p_Holm"] = temp

res["Significant_Holm"] = res["p_Holm"] < 0.05

out = Path("predictions/SSL_MAE_vs_SimCLR_statistics.csv")
res.to_csv(out, index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

for dataset in DATASETS:

    print("\n" + "=" * 100)
    print(dataset)
    print("=" * 100)

    x = res[res["Dataset"] == dataset]

    print(
        x[
            [
                "Structure",
                "Metric",
                "N",
                "MAE_mean",
                "SimCLR_mean",
                "Mean_diff_MAE_minus_SimCLR",
                "p_raw",
                "p_Holm",
                "Significant_Holm"
            ]
        ].to_string(index=False)
    )

print("\nSaved:", out)
