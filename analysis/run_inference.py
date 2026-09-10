"""
run_inference.py

Loads MMSLT's best_checkpoint.pth (epoch 11, the checkpoint with the best
dev BLEU-4 during the mmslt_random4000_v3 training run) and runs generation
over the How2Sign test set (300 sentences), either on the original frames
("--mask-region none") or with one body region blacked out in every frame
("--mask-region {eyes,nose,mouth,face,left_hand,right_hand,shoulder,body}").

For each sentence, records SENTENCE_ID/GT/prediction plus sentence-level
BLEU, chrF, and semantic similarity (sentence-transformers cosine sim), and
saves one CSV per condition:
    outputs/mmslt_random4000_v3/analysis/predictions_<condition>.csv
(condition = "original" for --mask-region none, else the region name).

Also prints the corpus-level BLEU-4 at the end — for --mask-region none this
should land close to the 1.58 test BLEU-4 already recorded in the training
log, as a sanity check that this script faithfully reproduces the checkpoint.

Resumable at the condition level: if the output CSV for a condition already
has all 300 rows, that condition is skipped entirely, so re-running the same
command after an interruption just picks up the remaining conditions.

Usage (single GPU, non-distributed; pick GPU 4/6/7 — never 0-3, and check
`nvidia-smi` first since other jobs may already be on 4-7 too):
    conda activate mmslt
    CUDA_VISIBLE_DEVICES=7 python run_inference.py --mask-region none
    CUDA_VISIBLE_DEVICES=7 python run_inference.py --mask-region left_hand
"""
import argparse
import csv
import gzip
import os
import pickle
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
from masked_dataset import MaskedS2TDataset, MASK_REGIONS  # noqa: E402

CONFIG_PATH = os.path.join(MMSLT_DIR, "configs", "config_mmslt_how2sign_random4000_v3.yaml")
CHECKPOINT_PATH = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "best_checkpoint.pth")
LANDMARKS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis", "landmarks")
OUTPUT_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis")

CONDITIONS = ["original"] + MASK_REGIONS  # "original" == --mask-region none


class Args:
    """Minimal stand-in for train_mmslt.py's argparse Namespace — only the
    fields S2T_Dataset/MMSLT actually read are needed for inference."""
    input_size = 224
    resize = 256


def build_dataset(config, tokenizer, mask_region):
    return MaskedS2TDataset(
        path=config["data"]["label_path"],
        tokenizer=tokenizer,
        config=config,
        args=Args(),
        phase="test",
        mask_region=mask_region,
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
        n = sum(1 for _ in f) - 1  # minus header
    return n >= expected_rows


def run_condition(mask_region_name, config, tokenizer, model, device, batch_size, limit=None):
    condition = "original" if mask_region_name in (None, "none") else mask_region_name
    mask_region = None if condition == "original" else condition
    out_csv = os.path.join(OUTPUT_DIR, f"predictions_{condition}.csv")

    dataset = build_dataset(config, tokenizer, mask_region)
    n_expected = limit or len(dataset)
    if already_done(out_csv, n_expected):
        print(f"[{condition}] already done ({out_csv}), skipping")
        return

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=dataset.collate_fn,
    )

    print(f"[{condition}] running generation over {len(dataset)} test sentences...")
    t0 = time.time()
    rows = []  # (sentence_id, gt, pred)
    n_seen = 0
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
            n_seen += len(names)
            if limit and n_seen >= limit:
                break
    elapsed = time.time() - t0
    print(f"[{condition}] generation done in {elapsed:.0f}s "
          f"(mask hit/miss: {dataset.hit_count}/{dataset.miss_count})")

    preds = [r[2] for r in rows]
    refs = [r[1] for r in rows]

    bleu_scorer = BLEU(effective_order=True)
    chrf_scorer = CHRF()
    sim_fn = get_semantic_similarity_fn()
    sims = sim_fn(preds, refs)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SENTENCE_ID", "GT", "prediction", "bleu", "chrf", "semantic_similarity"])
        for (sid, gt, pred), sim in zip(rows, sims):
            bleu = bleu_scorer.sentence_score(pred, [gt]).score
            chrf = chrf_scorer.sentence_score(pred, [gt]).score
            writer.writerow([sid, gt, pred, f"{bleu:.4f}", f"{chrf:.4f}", f"{sim:.4f}"])

    corpus_bleu = BLEU().corpus_score(preds, [refs]).score
    print(f"[{condition}] corpus BLEU-4 = {corpus_bleu:.2f} "
          f"(sanity check: --mask-region none should be close to 1.58)")
    print(f"[{condition}] wrote {out_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-region", required=True, choices=["none"] + MASK_REGIONS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only run on first N test sentences (debug)")
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

    run_condition(args.mask_region, config, tokenizer, model, device, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
