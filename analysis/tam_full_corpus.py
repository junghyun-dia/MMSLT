"""
tam_full_corpus.py

Extends cross_attention_analysis.py's Temporal Attention Map (TAM) extraction
from a curated set of 9 example sentences to the FULL 300-sentence test set,
turning it from a qualitative illustration into a quantitative grounding
signal: for each sentence, how *peaked* (concentrated on a few frames) vs.
*diffuse* (spread near-uniformly) is the decoder's attention over time?

Reuses the exact same model, checkpoint, and verified frame-index mapping as
cross_attention_analysis.py (see that file's docstring for the derivation).
Re-tokenizes each sentence's OWN prediction from predictions_original.csv
(already generated) as the teacher-forced target, instead of re-running beam
search generation, to avoid paying for expensive decoding twice.

Per-sentence stats:
    mean_entropy       — token-averaged normalized entropy of the attention
                         distribution over encoder positions (0 = fully
                         peaked on one frame, 1 = fully uniform over all
                         available frames). Low entropy = the decoder is
                         actually looking somewhere specific when generating
                         this sentence (evidence for grounding); high entropy
                         = attention barely discriminates between frames
                         (consistent with guessing from language-model priors
                         rather than the video).
    mean_peak_relpos   — token-averaged peak-frame position, normalized to
                         [0,1] by orig_len (0=start of clip, 1=end). Checks
                         for a positional bias (e.g. always attending near
                         the start) as opposed to content-dependent attention.

Cross-validates against the region-masking result already on disk: does a
sentence's attention peakiness (1 - mean_entropy) correlate with how much
region-masking hurts it overall (mean |delta_bleu| across all 8 regions from
region_masking_comparison.csv)? A positive correlation would be independent
evidence for grounding (peaked-attention sentences are also the
masking-sensitive ones); no correlation is inconclusive, not counter-evidence
(masking sensitivity and decoder-attention peakiness are different signals
about the same underlying question).

Outputs (under outputs/mmslt_random4000_v3/analysis/):
    tam_stats.csv    — per-sentence: sentence_id, n_tokens, mean_entropy, mean_peak_relpos
    tam_summary.csv  — corpus-level: mean/median entropy, Spearman r vs mean|delta_bleu|
    tam_full.json    — ALL 300 sentences' full per-token attention rows + frame_of_pos
                        (same schema as the original cross_attention_examples.json,
                        which only covered 9 curated sentences) -- for building a
                        per-example temporal-grounding viewer over the whole test set.
"""
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from transformers import MBart50TokenizerFast

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MMSLT_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from models import MMSLT  # noqa: E402
from masked_dataset import MaskedS2TDataset  # noqa: E402
from cross_attention_analysis import orig_frame_idx, Args  # noqa: E402

CONFIG_PATH = os.path.join(MMSLT_DIR, "configs", "config_mmslt_how2sign_random4000_v3.yaml")
CHECKPOINT_PATH = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "best_checkpoint.pth")
LANDMARKS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis", "landmarks")
ANALYSIS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis")
PREDICTIONS_CSV = os.path.join(ANALYSIS_DIR, "predictions_original.csv")
COMPARISON_CSV = os.path.join(ANALYSIS_DIR, "region_masking_comparison.csv")

SPECIAL_TOKENS = {"<pad>", "</s>", "<s>", "de_DE", "<unk>"}


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt", src_lang="de_DE", tgt_lang="de_DE",
        model_max_length=1024,
    )

    model = MMSLT(config, Args())
    model.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    dataset = MaskedS2TDataset(
        path=config["data"]["label_path"], tokenizer=tokenizer, config=config, args=Args(),
        phase="test", mask_region=None, landmarks_dir=LANDMARKS_DIR,
    )
    name_to_index = {dataset.raw_data[k]["name"]: i for i, k in enumerate(dataset.list)}

    preds_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Running TAM extraction over {len(preds_df)} test sentences...")

    rows = []
    full_examples = []
    with torch.no_grad():
        for n_done, rec in enumerate(preds_df.itertuples(), 1):
            sid = rec.SENTENCE_ID
            if sid not in name_to_index:
                print(f"  WARNING: {sid} not found in dataset, skipping")
                continue
            idx = name_to_index[sid]
            key = dataset.list[idx]
            orig_len = len(dataset.raw_data[key]["imgs_path"])

            src_input, _ = dataset.collate_fn([dataset[idx]])
            new_src_len = int(src_input["new_src_length_batch"][0])

            target_ids = tokenizer(rec.prediction, return_tensors="pt").input_ids.to(device)

            inputs_embeds, attention_mask = model.share_forward(src_input)
            out = model.mbart(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask.to(device),
                labels=target_ids,
                output_attentions=True,
                return_dict=True,
            )
            last_layer_cross_attn = out.cross_attentions[-1][0]  # [num_heads, tgt_len, src_len]
            attn = last_layer_cross_attn.mean(dim=0).float().cpu().numpy()  # [tgt_len, src_len]
            attn = attn[:, :new_src_len]
            attn = attn / attn.sum(axis=1, keepdims=True).clip(min=1e-8)

            tokens = tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
            frame_of_pos = [orig_frame_idx(j, orig_len) for j in range(new_src_len)]
            max_entropy = np.log(new_src_len) if new_src_len > 1 else 1.0

            entropies, peak_relpos, token_rows = [], [], []
            for t_idx, tok in enumerate(tokens):
                if tok in SPECIAL_TOKENS or t_idx >= attn.shape[0]:
                    continue
                row = attn[t_idx]
                ent = -np.sum(row * np.log(row.clip(min=1e-12)))
                entropies.append(ent / max_entropy)
                peak_j = int(np.argmax(row))
                peak_relpos.append(frame_of_pos[peak_j] / max(orig_len - 1, 1))
                token_rows.append(dict(
                    token=tok.replace("▁", " ").strip() or tok,
                    attn=[round(v, 4) for v in row.tolist()],
                    peak_frame=frame_of_pos[peak_j],
                ))

            if not entropies:
                continue
            rows.append(dict(
                sentence_id=sid, n_tokens=len(entropies),
                mean_entropy=float(np.mean(entropies)),
                mean_peak_relpos=float(np.mean(peak_relpos)),
            ))
            full_examples.append(dict(
                sentence_id=sid, gt=rec.GT, prediction=rec.prediction,
                orig_len=orig_len, new_src_len=new_src_len,
                frame_of_pos=frame_of_pos, tokens=token_rows,
            ))

            if n_done % 25 == 0 or n_done == len(preds_df):
                print(f"  [{n_done}/{len(preds_df)}] last={sid}", flush=True)

    stats_df = pd.DataFrame(rows)
    stats_path = os.path.join(ANALYSIS_DIR, "tam_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"wrote {stats_path} ({len(stats_df)} sentences)")

    full_path = os.path.join(ANALYSIS_DIR, "tam_full.json")
    with open(full_path, "w") as f:
        json.dump(full_examples, f)
    print(f"wrote {full_path} ({len(full_examples)} sentences, full per-token attention)")

    # ---- cross-validate against region-masking sensitivity ----
    comp_df = pd.read_csv(COMPARISON_CSV)
    maskability = comp_df.groupby("SENTENCE_ID")["delta_bleu"].apply(lambda s: s.abs().mean())
    merged = stats_df.merge(maskability.rename("mean_abs_delta_bleu"), left_on="sentence_id", right_index=True)

    peakiness = 1 - merged["mean_entropy"]
    r, p = stats.spearmanr(peakiness, merged["mean_abs_delta_bleu"])

    summary = dict(
        n_sentences=len(stats_df),
        mean_entropy=stats_df["mean_entropy"].mean(),
        median_entropy=stats_df["mean_entropy"].median(),
        mean_peak_relpos=stats_df["mean_peak_relpos"].mean(),
        spearman_r_peakiness_vs_maskability=r,
        spearman_p=p,
        n_matched=len(merged),
    )
    summary_path = os.path.join(ANALYSIS_DIR, "tam_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n=== TAM summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
