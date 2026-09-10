"""
statistical_tests.py

Statistical analysis of region_masking_comparison.csv (300 sentences x 8
regions, paired original vs. masked BLEU/chrF/semantic-similarity).

1. Per-region paired Wilcoxon signed-rank test (masked vs original) for each
   metric, with a rank-biserial effect size and a bootstrap 95% CI on the
   mean delta. BH (Benjamini-Hochberg) FDR correction across all region x
   metric tests (8 x 3 = 24).
2. Friedman test per metric: do the 8 regions' deltas differ from each
   other at all? If significant, all-pairs post-hoc Wilcoxon with its own
   BH correction.
3. Cross-check against the independent (CLIP/SigLIP-derived) grounding
   signal already joined into the comparison CSV (`grounding_best_region`):
   for each region, Mann-Whitney U comparing delta_bleu on sentences where
   that region was independently flagged as the word's best region vs. not.

Outputs (under outputs/mmslt_random4000_v3/analysis/):
    region_stats.csv          — per region x metric: mean delta, CI, Wilcoxon p (raw+BH), effect size
    friedman_test.csv         — per metric: chi2, p (are regions different at all)
    posthoc_pairwise.csv      — per metric, all-pairs region deltas Wilcoxon (only if Friedman significant)
    grounding_crossval.csv    — per region: Mann-Whitney of delta_bleu, grounding-flagged vs not
"""
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))
from masked_dataset import MASK_REGIONS  # noqa: E402

ANALYSIS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis")
COMPARISON_CSV = os.path.join(ANALYSIS_DIR, "region_masking_comparison.csv")
METRICS = ["bleu", "chrf", "simscore"]
N_BOOT = 2000
RNG = np.random.RandomState(0)


def bh_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty(n)
    out[order] = adjusted
    return out


def bootstrap_ci(diffs, n_boot=N_BOOT, alpha=0.05):
    diffs = np.asarray(diffs)
    n = len(diffs)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.randint(0, n, n)
        boot_means[b] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def wilcoxon_with_effect_size(diffs):
    diffs = np.asarray(diffs)
    n_nonzero = int((diffs != 0).sum())
    if n_nonzero < 1:
        return dict(n_nonzero=0, statistic=np.nan, p=1.0, z=0.0, effect_size_r=0.0)
    try:
        res = stats.wilcoxon(diffs, zero_method="wilcox", mode="approx")
        # scipy's "approx" mode exposes the normal-approximation z via .zstatistic in newer
        # versions; fall back to deriving it from the p-value (two-sided) if unavailable.
        z = getattr(res, "zstatistic", None)
        if z is None:
            z = stats.norm.isf(res.pvalue / 2)
        effect_size_r = float(z) / np.sqrt(n_nonzero)
        return dict(n_nonzero=n_nonzero, statistic=float(res.statistic), p=float(res.pvalue),
                    z=float(z), effect_size_r=effect_size_r)
    except ValueError:
        # all diffs zero after dropping ties, etc.
        return dict(n_nonzero=n_nonzero, statistic=np.nan, p=1.0, z=0.0, effect_size_r=0.0)


def main():
    df = pd.read_csv(COMPARISON_CSV)
    print(f"Loaded {len(df)} rows ({df['region'].nunique()} regions)")

    # ---- 1. per-region, per-metric paired test + bootstrap CI ----
    rows = []
    for region in MASK_REGIONS:
        sub = df[df["region"] == region].sort_values("SENTENCE_ID")
        for metric in METRICS:
            diffs = (sub[f"masked_{metric}"] - sub[f"original_{metric}"]).values
            wt = wilcoxon_with_effect_size(diffs)
            ci_lo, ci_hi = bootstrap_ci(diffs)
            rows.append(dict(
                region=region, metric=metric, n=len(diffs), mean_delta=diffs.mean(),
                ci_low=ci_lo, ci_high=ci_hi, **wt,
            ))
    stats_df = pd.DataFrame(rows)
    stats_df["p_adj_bh"] = np.nan
    for metric in METRICS:
        mask = stats_df["metric"] == metric
        stats_df.loc[mask, "p_adj_bh"] = bh_adjust(stats_df.loc[mask, "p"].values)
    stats_df["significant_fdr05"] = stats_df["p_adj_bh"] < 0.05
    stats_df = stats_df.sort_values(["metric", "mean_delta"])
    stats_df.to_csv(os.path.join(ANALYSIS_DIR, "region_stats.csv"), index=False)
    print("\n=== Per-region paired Wilcoxon (BH-adjusted), delta_bleu ===")
    print(stats_df[stats_df.metric == "bleu"][
        ["region", "n", "mean_delta", "ci_low", "ci_high", "p", "p_adj_bh", "effect_size_r", "significant_fdr05"]
    ].to_string(index=False))

    # ---- 2. Friedman test across regions, per metric ----
    friedman_rows = []
    posthoc_rows = []
    for metric in METRICS:
        pivot = df.pivot(index="SENTENCE_ID", columns="region", values=f"masked_{metric}")
        orig = df.pivot(index="SENTENCE_ID", columns="region", values=f"original_{metric}")
        delta_pivot = (pivot - orig)[MASK_REGIONS]  # sentences x regions matrix of deltas
        chi2, p = stats.friedmanchisquare(*[delta_pivot[r].values for r in MASK_REGIONS])
        friedman_rows.append(dict(metric=metric, chi2=chi2, p=p, n=len(delta_pivot)))
        print(f"\nFriedman test [{metric}]: chi2={chi2:.2f}, p={p:.4g}, n={len(delta_pivot)}")

        if p < 0.05:
            pair_pvals = []
            pairs = list(combinations(MASK_REGIONS, 2))
            for r1, r2 in pairs:
                d1, d2 = delta_pivot[r1].values, delta_pivot[r2].values
                try:
                    stat, pv = stats.wilcoxon(d1, d2, zero_method="wilcox")
                except ValueError:
                    stat, pv = np.nan, 1.0
                pair_pvals.append(pv)
                posthoc_rows.append(dict(metric=metric, region_a=r1, region_b=r2,
                                          mean_delta_a=d1.mean(), mean_delta_b=d2.mean(),
                                          statistic=stat, p=pv))
            adj = bh_adjust(pair_pvals)
            for row, a in zip(posthoc_rows[-len(pairs):], adj):
                row["p_adj_bh"] = a
                row["significant_fdr05"] = a < 0.05

    pd.DataFrame(friedman_rows).to_csv(os.path.join(ANALYSIS_DIR, "friedman_test.csv"), index=False)
    if posthoc_rows:
        pd.DataFrame(posthoc_rows).to_csv(os.path.join(ANALYSIS_DIR, "posthoc_pairwise.csv"), index=False)
        print(f"wrote posthoc_pairwise.csv ({len(posthoc_rows)} pairs)")

    # ---- 3. Cross-check against independent grounding signal ----
    cross_rows = []
    base = df[df["region"] == MASK_REGIONS[0]][["SENTENCE_ID", "grounding_best_region"]].drop_duplicates()
    for region in MASK_REGIONS:
        sub = df[df["region"] == region].merge(base, on="SENTENCE_ID", suffixes=("", "_dup"))
        flagged = sub[sub["grounding_best_region"] == region]["delta_bleu"]
        not_flagged = sub[sub["grounding_best_region"] != region]["delta_bleu"]
        if len(flagged) >= 5 and len(not_flagged) >= 5:
            stat, p = stats.mannwhitneyu(flagged, not_flagged, alternative="two-sided")
        else:
            stat, p = np.nan, np.nan
        cross_rows.append(dict(
            region=region, n_flagged=len(flagged), n_not_flagged=len(not_flagged),
            median_delta_flagged=flagged.median(), median_delta_not_flagged=not_flagged.median(),
            mannwhitney_stat=stat, p=p,
        ))
    cross_df = pd.DataFrame(cross_rows)
    valid_p = cross_df["p"].dropna()
    if len(valid_p):
        adj = bh_adjust(valid_p.values)
        cross_df.loc[valid_p.index, "p_adj_bh"] = adj
        cross_df["significant_fdr05"] = cross_df["p_adj_bh"] < 0.05
    cross_df.to_csv(os.path.join(ANALYSIS_DIR, "grounding_crossval.csv"), index=False)
    print("\n=== Grounding cross-validation (delta_bleu, region flagged vs not by independent CLIP/SigLIP signal) ===")
    print(cross_df.to_string(index=False))

    print(f"\nAll stats written under {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
