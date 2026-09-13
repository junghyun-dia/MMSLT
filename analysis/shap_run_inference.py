"""
shap_run_inference.py

Generalization of run_inference.py for exact Shapley-value computation over
the 8 body regions: runs generation for ALL 2^8=256 region-masking
combinations (every subset of MASK_REGIONS, from "nothing masked" through
"everything masked"), each restricted to the fixed 60-sentence SHAP
subsample (results/sampling/shap_sentence_subsample_60.json — see
external/SpaMo/analysis/shap_masked_feature_extract.py's docstring for why
60, not the full 300). MMSLT masks pixels inline, so this needs no separate
feature-extraction phase (unlike SpaMo/VTaMo) — just re-running generation
per combo, which is cheap enough to redo all 256 fresh rather than special-
casing reuse of the 9 already-computed 300-sentence conditions (mixing a
300-sentence baseline with 60-sentence coalitions would break the Shapley
computation's requirement that every coalition be evaluated on the same
sentence set).

Outputs (under outputs/mmslt_random4000_v3/analysis/shap_predictions/):
    predictions_<combo>.csv   (SENTENCE_ID, GT, prediction, bleu, chrf, semantic_similarity)
(combo = "original", a single region name, or "+"-joined sorted region names)

Resumable per-combo (skips a combo if its CSV already has 60 rows).

Usage (pick GPU 4/5/6/7; always nohup for a run this long):
    conda activate mmslt
    cd external/MMSLT
    CUDA_VISIBLE_DEVICES=4 python analysis/shap_run_inference.py --shard 0 --num-shards 4
"""
import argparse
import csv
import itertools
import os
import sys
import time

import torch
import yaml
from sacrebleu.metrics import BLEU, CHRF
from transformers import MBart50TokenizerFast

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MMSLT_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from models import MMSLT  # noqa: E402
from shap_masked_dataset import ShapMaskedS2TDataset, MASK_REGIONS, combo_name  # noqa: E402

CONFIG_PATH = os.path.join(MMSLT_DIR, "configs", "config_mmslt_how2sign_random4000_v3.yaml")
CHECKPOINT_PATH = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "best_checkpoint.pth")
LANDMARKS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis", "landmarks")
OUTPUT_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis", "shap_predictions")


class Args:
    input_size = 224
    resize = 256


def all_shap_combos():
    """All 2^8 subsets of MASK_REGIONS, () first (= "original")."""
    combos = [()]
    for k in range(1, len(MASK_REGIONS) + 1):
        combos.extend(itertools.combinations(MASK_REGIONS, k))
    return combos


def build_dataset(config, tokenizer, mask_regions):
    return ShapMaskedS2TDataset(
        path=config["data"]["label_path"],
        tokenizer=tokenizer,
        config=config,
        args=Args(),
        phase="test",
        mask_regions=mask_regions,
        landmarks_dir=LANDMARKS_DIR,
    )


def load_model(config, device):
    model = MMSLT(config, Args())
    model.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')} ({CHECKPOINT_PATH})")
    return model


def get_semantic_similarity_fn():
    from sentence_transformers import SentenceTransformer, util
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    def fn(preds, refs):
        emb_p = st_model.encode(preds, convert_to_tensor=True, show_progress_bar=False)
        emb_r = st_model.encode(refs, convert_to_tensor=True, show_progress_bar=False)
        sims = util.cos_sim(emb_p, emb_r).diagonal()
        return [float(s) for s in sims]

    return fn


def already_done(csv_path, expected_rows):
    if not os.path.exists(csv_path):
        return False
    with open(csv_path) as f:
        n = sum(1 for _ in f) - 1
    return n >= expected_rows


def run_combo(regions, config, tokenizer, model, device, batch_size, sim_fn):
    combo = combo_name(regions)
    out_csv = os.path.join(OUTPUT_DIR, f"predictions_{combo}.csv")

    dataset = build_dataset(config, tokenizer, regions)
    n_expected = len(dataset)
    if already_done(out_csv, n_expected):
        print(f"[{combo}] already done ({out_csv}), skipping")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    t0 = time.time()
    rows = []
    with torch.no_grad():
        for src_input, tgt_input in dataloader:
            output = model.generate(
                src_input, max_new_tokens=150, num_beams=8,
                forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
            )
            names = src_input["name_batch"]
            ref_ids = tgt_input["input_ids"]
            for i in range(len(names)):
                pred_text = tokenizer.decode(output[i], skip_special_tokens=True)
                gt_text = tokenizer.decode(ref_ids[i], skip_special_tokens=True)
                rows.append((names[i], gt_text, pred_text))
    elapsed = time.time() - t0

    preds = [r[2] for r in rows]
    refs = [r[1] for r in rows]

    bleu_scorer = BLEU(effective_order=True)
    chrf_scorer = CHRF()
    sims = sim_fn(preds, refs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SENTENCE_ID", "GT", "prediction", "bleu", "chrf", "semantic_similarity"])
        for (sid, gt, pred), sim in zip(rows, sims):
            bleu = bleu_scorer.sentence_score(pred, [gt]).score
            chrf = chrf_scorer.sentence_score(pred, [gt]).score
            writer.writerow([sid, gt, pred, f"{bleu:.4f}", f"{chrf:.4f}", f"{sim:.4f}"])

    corpus_bleu = BLEU().corpus_score(preds, [refs]).score
    print(f"[{combo}] n={len(rows)} elapsed={elapsed:.0f}s corpus_bleu4={corpus_bleu:.2f} "
          f"mask_hit/miss={dataset.hit_count}/{dataset.miss_count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt", src_lang="de_DE", tgt_lang="de_DE",
        model_max_length=1024,
    )
    model = load_model(config, device)
    sim_fn = get_semantic_similarity_fn()

    combos = all_shap_combos()
    my_combos = [c for i, c in enumerate(combos) if i % args.num_shards == args.shard]
    print(f"Shard {args.shard}/{args.num_shards}: {len(my_combos)} of {len(combos)} total combos assigned")

    t_start = time.time()
    for n, regions in enumerate(my_combos, 1):
        run_combo(list(regions), config, tokenizer, model, device, args.batch_size, sim_fn)
        elapsed = time.time() - t_start
        rate = elapsed / n
        remaining = rate * (len(my_combos) - n)
        print(f"  [{n}/{len(my_combos)}] shard progress, elapsed={elapsed/60:.1f}min, "
              f"est. remaining={remaining/60:.1f}min", flush=True)

    print(f"Shard {args.shard} complete: {len(my_combos)} combos processed.")


if __name__ == "__main__":
    main()
