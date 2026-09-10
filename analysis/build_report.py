"""
build_report.py

Merges predictions_original.csv with each region's predictions_<region>.csv
(all produced by run_inference.py) into a single per-sentence, per-region
comparison table, plus a per-region mean-delta summary — the deliverable
for the MMSLT region-masking ablation (grounding vs. guessing check).

Also optionally joins `stratum` / `best_region` (an independent, CLIP/SigLIP
-derived grounding signal for the same 300 sentences) from
results/sampling/how2sign_test300_unique_full.csv, as extra context.

Outputs (under outputs/mmslt_random4000_v3/analysis/):
    region_masking_comparison.csv  — 300 x 8 regions = 2400 rows
    region_delta_summary.csv       — 1 row per region, mean/median deltas
"""
import csv
import os
import sys

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(MMSLT_DIR))
sys.path.insert(0, os.path.dirname(__file__))

from masked_dataset import MASK_REGIONS  # noqa: E402

ANALYSIS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis")
SAMPLING_CSV = os.path.join(WORKSPACE_ROOT, "results", "sampling", "how2sign_test300_unique_full.csv")


def load_predictions(condition):
    path = os.path.join(ANALYSIS_DIR, f"predictions_{condition}.csv")
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["SENTENCE_ID"]] = row
    return rows


def load_grounding_context():
    """sentence_id -> {"stratum":..., "best_region":...}, best-effort."""
    if not os.path.exists(SAMPLING_CSV):
        return {}
    ctx = {}
    with open(SAMPLING_CSV, newline="") as f:
        for row in csv.DictReader(f):
            sid = row.get("SENTENCE_ID")
            if sid:
                ctx[sid] = {"stratum": row.get("stratum", ""), "best_region": row.get("best_region", "")}
    return ctx


def main():
    baseline = load_predictions("original")
    grounding_ctx = load_grounding_context()
    print(f"Baseline sentences: {len(baseline)}; grounding context rows: {len(grounding_ctx)}")

    comparison_path = os.path.join(ANALYSIS_DIR, "region_masking_comparison.csv")
    fieldnames = [
        "SENTENCE_ID", "region", "GT", "original_pred", "masked_pred",
        "original_bleu", "masked_bleu", "delta_bleu",
        "original_chrf", "masked_chrf", "delta_chrf",
        "original_simscore", "masked_simscore", "delta_simscore",
        "stratum", "grounding_best_region",
    ]

    region_deltas = {r: {"bleu": [], "chrf": [], "sim": []} for r in MASK_REGIONS}

    with open(comparison_path, "w", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for region in MASK_REGIONS:
            masked = load_predictions(region)
            missing = set(baseline) - set(masked)
            if missing:
                print(f"WARNING: {region} missing {len(missing)} sentence ids vs baseline")

            for sid, base_row in baseline.items():
                if sid not in masked:
                    continue
                m_row = masked[sid]
                ob, mb = float(base_row["bleu"]), float(m_row["bleu"])
                oc, mc = float(base_row["chrf"]), float(m_row["chrf"])
                os_, ms = float(base_row["semantic_similarity"]), float(m_row["semantic_similarity"])
                ctx = grounding_ctx.get(sid, {})

                writer.writerow({
                    "SENTENCE_ID": sid,
                    "region": region,
                    "GT": base_row["GT"],
                    "original_pred": base_row["prediction"],
                    "masked_pred": m_row["prediction"],
                    "original_bleu": f"{ob:.4f}", "masked_bleu": f"{mb:.4f}", "delta_bleu": f"{mb - ob:.4f}",
                    "original_chrf": f"{oc:.4f}", "masked_chrf": f"{mc:.4f}", "delta_chrf": f"{mc - oc:.4f}",
                    "original_simscore": f"{os_:.4f}", "masked_simscore": f"{ms:.4f}", "delta_simscore": f"{ms - os_:.4f}",
                    "stratum": ctx.get("stratum", ""),
                    "grounding_best_region": ctx.get("best_region", ""),
                })

                region_deltas[region]["bleu"].append(mb - ob)
                region_deltas[region]["chrf"].append(mc - oc)
                region_deltas[region]["sim"].append(ms - os_)

    print(f"wrote {comparison_path}")

    summary_path = os.path.join(ANALYSIS_DIR, "region_delta_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "n", "mean_delta_bleu", "mean_delta_chrf", "mean_delta_simscore"])
        rows = []
        for region, d in region_deltas.items():
            n = len(d["bleu"])
            mean_bleu = sum(d["bleu"]) / n if n else float("nan")
            mean_chrf = sum(d["chrf"]) / n if n else float("nan")
            mean_sim = sum(d["sim"]) / n if n else float("nan")
            rows.append((region, n, mean_bleu, mean_chrf, mean_sim))
        rows.sort(key=lambda r: r[2])  # most negative delta_bleu (biggest drop) first
        for region, n, mb, mc, ms in rows:
            writer.writerow([region, n, f"{mb:.4f}", f"{mc:.4f}", f"{ms:.4f}"])
            print(f"  {region:12s} n={n:3d}  mean_delta_bleu={mb:+.4f}  mean_delta_chrf={mc:+.4f}  mean_delta_sim={ms:+.4f}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
