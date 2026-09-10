"""
nmm_subgroup_analysis.py

Follow-up to statistical_tests.py: are the "non-manual" regions (face, eyes,
nose, mouth, shoulder) more important specifically on sentences whose GT
(English) text suggests grammar that ASL marks non-manually (negation,
conditionals/subordinate clauses, hedging, intensifiers, rhetorical wh-cleft
constructions)? This is a proxy: GT here is the English translation, not an
ASL gloss, so a text heuristic approximates (does not guarantee) where the
signer likely used a non-manual marker (raised/furrowed brows, head tilt,
head shake, mouth morpheme).

An earlier version of this script also tried "is_question" (sentence-initial
wh-word or yes/no-question starter): it turned out this test set (How2Sign
how-to narration) contains only 2 genuine questions in 300 sentences — far
too few to test — and most of what it flagged were "what/when/where + is/do"
CLEFT constructions ("What you need to do is...") that are declaratives, not
questions. Those cleft sentences are kept here under a different, correctly-
named category (is_rhetorical_cleft) since ASL commonly marks exactly this
construction non-manually (a wh-word followed immediately by its own answer),
distinct from a real interrogative.

Method: flag each of the 300 test sentences by text heuristics, then for
each candidate non-manual region x metric x flag, compare delta (masked -
original) between flagged vs not-flagged sentences with Mann-Whitney U
(unpaired), BH-FDR corrected across ALL region x metric x flag tests in one
pool (5 regions x 3 metrics x 5 flags = 75 tests).

Outputs (outputs/mmslt_random4000_v3/analysis/):
    nmm_flags.csv            — SENTENCE_ID, GT, one bool column per flag
    nmm_subgroup_stats.csv   — region x metric x flag: n, mean/median delta per group, Mann-Whitney p (raw+BH)
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

ANALYSIS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis")
COMPARISON_CSV = os.path.join(ANALYSIS_DIR, "region_masking_comparison.csv")

NMM_REGIONS = ["face", "eyes", "nose", "mouth", "shoulder"]
METRICS = ["bleu", "chrf", "simscore"]
FLAGS = ["is_negation", "is_conditional_subordinate", "is_hedge", "is_intensifier", "is_rhetorical_cleft"]

WH_WORDS = {"what", "who", "whom", "whose", "where", "when", "why", "which", "how"}

NEG_RE = re.compile(r"\b(not|never|nothing|nobody|none|no)\b|n['’]t\b", re.IGNORECASE)
COND_SUB_RE = re.compile(r"\b(if|unless|suppose|supposing|when|while|before|after)\b", re.IGNORECASE)
HEDGE_RE = re.compile(r"\b(maybe|probably|perhaps|possibly|might|i think|i guess|kind of|sort of|seems? like)\b", re.IGNORECASE)
INTENSIFIER_RE = re.compile(
    r"\b(very|really|extremely|incredibly|totally|absolutely|completely|quite|too|"
    r"best|worst|most|least|greatest|biggest|smallest|hardest|easiest|highest|lowest|largest|longest)\b",
    re.IGNORECASE,
)


def classify(gt):
    text = gt.strip()
    low = text.lower()
    words = low.split()
    first_word = re.sub(r"[^a-z]", "", words[0]) if words else ""

    is_true_question = text.endswith("?")
    is_rhetorical_cleft = (first_word in WH_WORDS) and not is_true_question
    is_negation = bool(NEG_RE.search(low))
    is_conditional_subordinate = bool(COND_SUB_RE.search(low))
    is_hedge = bool(HEDGE_RE.search(low))
    is_intensifier = bool(INTENSIFIER_RE.search(low))
    return dict(
        is_true_question=is_true_question,
        is_rhetorical_cleft=is_rhetorical_cleft,
        is_negation=is_negation,
        is_conditional_subordinate=is_conditional_subordinate,
        is_hedge=is_hedge,
        is_intensifier=is_intensifier,
    )


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


def main():
    df = pd.read_csv(COMPARISON_CSV)
    base = df[["SENTENCE_ID", "GT"]].drop_duplicates().reset_index(drop=True)
    flag_rows = base["GT"].map(classify).apply(pd.Series)
    base = pd.concat([base, flag_rows], axis=1)
    base.to_csv(os.path.join(ANALYSIS_DIR, "nmm_flags.csv"), index=False)

    print(f"Of {len(base)} test sentences:")
    print(f"  is_true_question             : {base.is_true_question.sum()}  (too few to test statistically, informational only)")
    for f in FLAGS:
        print(f"  {f:28s}: {base[f].sum()}")
    print()

    df = df.merge(base[["SENTENCE_ID"] + FLAGS], on="SENTENCE_ID")

    rows = []
    for region in NMM_REGIONS:
        sub = df[df["region"] == region]
        for metric in METRICS:
            delta_col = f"delta_{metric}"
            for flag_name in FLAGS:
                flagged = sub[sub[flag_name]][delta_col]
                not_flagged = sub[~sub[flag_name]][delta_col]
                if len(flagged) >= 5 and len(not_flagged) >= 5:
                    stat, p = stats.mannwhitneyu(flagged, not_flagged, alternative="two-sided")
                else:
                    stat, p = np.nan, np.nan
                rows.append(dict(
                    region=region, metric=metric, flag=flag_name,
                    n_flagged=len(flagged), n_not_flagged=len(not_flagged),
                    mean_delta_flagged=flagged.mean(), mean_delta_not_flagged=not_flagged.mean(),
                    median_delta_flagged=flagged.median(), median_delta_not_flagged=not_flagged.median(),
                    mannwhitney_stat=stat, p=p,
                ))
    res = pd.DataFrame(rows)
    valid = res["p"].notna()
    res.loc[valid, "p_adj_bh"] = bh_adjust(res.loc[valid, "p"].values)
    res["significant_fdr05"] = res["p_adj_bh"] < 0.05
    res = res.sort_values(["flag", "metric", "mean_delta_flagged"])
    res.to_csv(os.path.join(ANALYSIS_DIR, "nmm_subgroup_stats.csv"), index=False)

    print("=== BLEU: delta on flagged vs not-flagged sentences, per candidate non-manual region ===")
    print(res[(res.metric == "bleu")][
        ["region", "flag", "n_flagged", "n_not_flagged", "mean_delta_flagged", "mean_delta_not_flagged", "p", "p_adj_bh", "significant_fdr05"]
    ].to_string(index=False))

    print(f"\nwrote nmm_flags.csv, nmm_subgroup_stats.csv under {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
