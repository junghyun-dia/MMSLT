"""
gradcam_full_corpus.py

Transformer-adapted Grad-CAM for MMSLT (attention x gradient, the standard
simplification of Chefer et al.'s relevance-propagation method to a single
layer): unlike TAM (tam_full_corpus.py), which just reads the raw decoder
cross-attention, this ALSO backprops from the generation loss to that same
attention tensor, so the saliency reflects "how much would strengthening
attention to this frame have increased the model's confidence in its own
output" -- not just "how much attention mass landed there".

    GradCAM(frame) = mean_heads( ReLU(attn * d(loss)/d(attn)) )

Same last-decoder-layer, teacher-forced-on-own-prediction setup and the
same verified frame-index mapping as tam_full_corpus.py / cross_attention_analysis.py
(see those files' docstrings for the derivation). Reuses orig_frame_idx from
cross_attention_analysis.py directly, so the two methods are numerically
comparable position-for-position.

Outputs (under outputs/mmslt_random4000_v3/analysis/):
    gradcam_stats.csv    — per-sentence: sentence_id, n_tokens, mean_entropy (of the GradCAM map), mean_peak_relpos
    gradcam_summary.csv  — corpus-level stats + Spearman r vs region-masking sensitivity (mean |delta_bleu|)
                            + Spearman r vs. this same sentence's own TAM peakiness (do the two methods agree per-sentence?)
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
TAM_STATS_CSV = os.path.join(ANALYSIS_DIR, "tam_stats.csv")

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
    model.eval()  # eval mode (dropout off etc.) but gradients still flow -- we backprop, not train

    dataset = MaskedS2TDataset(
        path=config["data"]["label_path"], tokenizer=tokenizer, config=config, args=Args(),
        phase="test", mask_region=None, landmarks_dir=LANDMARKS_DIR,
    )
    name_to_index = {dataset.raw_data[k]["name"]: i for i, k in enumerate(dataset.list)}

    preds_df = pd.read_csv(PREDICTIONS_CSV)
    print(f"Running GradCAM extraction over {len(preds_df)} test sentences...")

    rows = []
    full_examples = []
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

        model.zero_grad(set_to_none=True)
        inputs_embeds, attention_mask = model.share_forward(src_input)
        out = model.mbart(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(device),
            labels=target_ids,
            output_attentions=True,
            return_dict=True,
        )
        attn = out.cross_attentions[-1]  # [1, num_heads, tgt_len, src_len], requires_grad
        attn.retain_grad()
        out.loss.backward()
        grad = attn.grad
        if grad is None:
            print(f"  WARNING: {sid} got no gradient on cross-attention, skipping")
            continue

        cam = torch.relu(attn.detach() * grad)[0].mean(dim=0).float().cpu().numpy()  # [tgt_len, src_len]
        cam = cam[:, :new_src_len]
        row_sums = cam.sum(axis=1, keepdims=True)
        cam = np.divide(cam, row_sums, out=np.zeros_like(cam), where=row_sums > 1e-12)

        tokens = tokenizer.convert_ids_to_tokens(target_ids[0].tolist())
        frame_of_pos = [orig_frame_idx(j, orig_len) for j in range(new_src_len)]

        entropies, peak_relpos, token_rows = [], [], []
        for t_idx, tok in enumerate(tokens):
            if tok in SPECIAL_TOKENS or t_idx >= cam.shape[0]:
                continue
            row = cam[t_idx]
            if row.sum() <= 1e-12:
                continue  # this token's GradCAM row is all-zero (ReLU killed everything) -- uninformative, skip
            max_entropy = np.log(new_src_len) if new_src_len > 1 else 1.0
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
    stats_path = os.path.join(ANALYSIS_DIR, "gradcam_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"wrote {stats_path} ({len(stats_df)} sentences)")

    full_path = os.path.join(ANALYSIS_DIR, "gradcam_full.json")
    with open(full_path, "w") as f:
        json.dump(full_examples, f)
    print(f"wrote {full_path} ({len(full_examples)} sentences, full per-token GradCAM)")

    comp_df = pd.read_csv(COMPARISON_CSV)
    maskability = comp_df.groupby("SENTENCE_ID")["delta_bleu"].apply(lambda s: s.abs().mean())
    merged = stats_df.merge(maskability.rename("mean_abs_delta_bleu"), left_on="sentence_id", right_index=True)
    peakiness = 1 - merged["mean_entropy"]
    r_mask, p_mask = stats.spearmanr(peakiness, merged["mean_abs_delta_bleu"])

    r_tam, p_tam = float("nan"), float("nan")
    if os.path.exists(TAM_STATS_CSV):
        tam_df = pd.read_csv(TAM_STATS_CSV)[["sentence_id", "mean_entropy"]].rename(columns={"mean_entropy": "tam_entropy"})
        both = stats_df.merge(tam_df, on="sentence_id")
        if len(both) > 5:
            r_tam, p_tam = stats.spearmanr(1 - both["mean_entropy"], 1 - both["tam_entropy"])

    summary = dict(
        n_sentences=len(stats_df),
        mean_entropy=stats_df["mean_entropy"].mean(),
        median_entropy=stats_df["mean_entropy"].median(),
        mean_peak_relpos=stats_df["mean_peak_relpos"].mean(),
        spearman_r_peakiness_vs_maskability=r_mask,
        spearman_p_vs_maskability=p_mask,
        spearman_r_vs_tam_peakiness=r_tam,
        spearman_p_vs_tam=p_tam,
        n_matched=len(merged),
    )
    summary_path = os.path.join(ANALYSIS_DIR, "gradcam_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())

    print("\n=== GradCAM summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
