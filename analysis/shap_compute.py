"""
shap_compute.py

Computes EXACT Shapley values for each of the 8 masking regions, per metric,
from the 256 per-combo predictions CSVs written by shap_run_inference.py
(analysis/shap_predictions/predictions_<combo>.csv, 60-sentence subsample —
see that script's docstring, and shap_masked_feature_extract.py in
external/SpaMo/analysis/ for the full cost-tradeoff writeup).

Definition used (masked-set convention, matching how the combo files are
named): let g(M) be a sentence's metric score when the regions in masked-set
M are masked (all other regions present/unmasked). The Shapley value for
region i is

    phi_i = (1/n) * sum over M subset of N\\{i} of
                [1 / C(n-1, |M|)] * [ g(M) - g(M union {i}) ]

i.e. the average marginal DROP in the metric from additionally masking
region i, over all 2^(n-1)=128 possible "what else is already masked"
contexts, using the standard Shapley combinatorial weights. This is
mathematically the ordinary Shapley value of "region i is present" (just
re-derived in terms of masked-sets, which is how the data is organized) --
phi_i > 0 means masking region i tends to hurt this metric; larger phi_i =
more important. Computed per sentence, then aggregated (mean + bootstrap CI)
across the subsample, matching the project's region-masking-ablation
convention (region_stats.csv).

Outputs (under analysis/ next to the other SHAP-adjacent output; pass
--out-dir to point elsewhere for SpaMo/VTaMo reuse of this same script):
    shap_values.csv   — region x metric: mean phi, 95% bootstrap CI, n
    shap_full.json    — per-sentence phi_i for every region x metric (for
                         future scatter/consistency checks against the
                         region-masking-ablation delta_bleu, analogous to
                         the TAM/GradCAM cross-checks)

Usage:
    conda activate mmslt   # (or spamo, for SpaMo/VTaMo -- pure CSV in/out, no model loading)
    python analysis/shap_compute.py
"""
import argparse
import csv
import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

MASK_REGIONS = ["eyes", "nose", "mouth", "face", "left_hand", "right_hand", "shoulder", "body"]
METRICS = ["bleu", "chrf", "simscore"]
METRIC_COLUMN = {"bleu": "bleu", "chrf": "chrf", "simscore": "semantic_similarity"}
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


def combo_name(masked_set):
    return "original" if not masked_set else "+".join(sorted(masked_set))


def all_masked_sets():
    sets = [frozenset()]
    for k in range(1, len(MASK_REGIONS) + 1):
        for combo in itertools.combinations(MASK_REGIONS, k):
            sets.append(frozenset(combo))
    return sets


def load_predictions(predictions_dir):
    """Returns {masked_set: {sentence_id: {metric: value}}}."""
    data = {}
    for masked_set in all_masked_sets():
        name = combo_name(masked_set)
        path = os.path.join(predictions_dir, f"predictions_{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing combo file: {path}")
        rows = {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows[row["SENTENCE_ID"]] = {
                    m: float(row[METRIC_COLUMN[m]]) for m in METRICS
                }
        data[masked_set] = rows
    return data


def shapley_for_region(region, sentence_id, data, metric):
    """phi_i for one region, one sentence, one metric -- exact, 128 terms."""
    others = [r for r in MASK_REGIONS if r != region]
    n = len(MASK_REGIONS)
    total = 0.0
    for k in range(len(others) + 1):
        weight = 1.0 / math.comb(n - 1, k)
        for subset in itertools.combinations(others, k):
            m = frozenset(subset)
            m_with_i = m | {region}
            g_m = data[m][sentence_id][metric]
            g_m_i = data[m_with_i][sentence_id][metric]
            total += weight * (g_m - g_m_i)
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", default=None,
                         help="dir containing predictions_<combo>.csv (default: this model's shap_predictions dir)")
    parser.add_argument("--out-dir", default=None, help="where to write shap_values.csv/shap_full.json (default: same as predictions-dir's parent 'analysis' dir, or predictions-dir itself if unclear)")
    args = parser.parse_args()

    if args.predictions_dir is None:
        mmslt_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.predictions_dir = os.path.join(mmslt_dir, "outputs", "mmslt_random4000_v3", "analysis", "shap_predictions")
    out_dir = args.out_dir or args.predictions_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading 256 combo predictions from {args.predictions_dir} ...")
    data = load_predictions(args.predictions_dir)
    sentence_ids = sorted(data[frozenset()].keys())
    print(f"{len(sentence_ids)} sentences, {len(data)} masked-set combos loaded")

    full_rows = []  # sentence_id, region, metric, phi
    for sid in sentence_ids:
        for region in MASK_REGIONS:
            for metric in METRICS:
                phi = shapley_for_region(region, sid, data, metric)
                full_rows.append(dict(sentence_id=sid, region=region, metric=metric, phi=phi))

    full_df = pd.DataFrame(full_rows)
    full_path = os.path.join(out_dir, "shap_full.csv")
    full_df.to_csv(full_path, index=False)
    print(f"wrote {full_path} ({len(full_df)} rows)")

    summary_rows = []
    for region in MASK_REGIONS:
        for metric in METRICS:
            vals = full_df[(full_df.region == region) & (full_df.metric == metric)]["phi"].values
            mean_phi = vals.mean()
            boot = np.array([vals[RNG.randint(0, len(vals), len(vals))].mean() for _ in range(N_BOOT)])
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            # two-sided bootstrap p-value: fraction of bootstrap means crossing zero, doubled
            frac_le0 = float((boot <= 0).mean())
            frac_ge0 = float((boot >= 0).mean())
            p = min(1.0, 2 * min(frac_le0, frac_ge0))
            summary_rows.append(dict(
                region=region, metric=metric, n=len(vals),
                mean_phi=mean_phi, ci_low=ci_lo, ci_high=ci_hi, p=p,
            ))
    summary_df = pd.DataFrame(summary_rows)
    summary_df["p_adj_bh"] = np.nan
    for metric in METRICS:
        mask = summary_df["metric"] == metric
        summary_df.loc[mask, "p_adj_bh"] = bh_adjust(summary_df.loc[mask, "p"].values)
    summary_df["significant_fdr05"] = summary_df["p_adj_bh"] < 0.05
    summary_df = summary_df.sort_values(["metric", "mean_phi"], ascending=[True, False])
    summary_path = os.path.join(out_dir, "shap_values.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")

    print("\n=== Shapley values (BLEU), sorted by importance (masking-harm), BH-adjusted across 8 regions ===")
    print(summary_df[summary_df.metric == "bleu"][["region", "n", "mean_phi", "ci_low", "ci_high", "p", "p_adj_bh", "significant_fdr05"]].to_string(index=False))


if __name__ == "__main__":
    main()
