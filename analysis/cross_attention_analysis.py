"""
cross_attention_analysis.py

Extracts mBART decoder->encoder cross-attention for a curated set of test
sentences, using MMSLT's own best_checkpoint.pth and its own generated
prediction (teacher-forced re-forward of the exact tokens it already
produced in predictions_original.csv, so the attention reflects what
actually happened during generation, not the ground truth it never saw).

Frame-index mapping (encoder position -> original frame index): verified
empirically (see conversation) against the collate_fn padding scheme
(left_pad=8, video_length = ceil(orig_len/4)*4 + 16) and TemporalConv's
two (kernel=5 valid-conv, kernel=2 maxpool) stages. Receptive-field-center
algebra for that exact stack gives:

    orig_frame_idx(j) = clamp(round(4*j - 0.5), 0, orig_len-1)

Confirmed exactly against a real sample: orig_len=64 -> video_length=80 ->
new_src_length=17, and j=0..16 maps cleanly across frames 0..63. This holds
per-sample regardless of batch size, since collate_fn truncates each
sample's padded sequence to its own ceil(orig_len/4)*4+16 before batching
(other samples' extra padding never leaks in).

Only mBART's LAST decoder layer's cross-attention is used (averaged across
heads) — the standard convention for seq2seq attention visualization
(earlier layers mix more positional/syntactic signal, last layer is most
semantically aligned to the output word).

Outputs:
    outputs/mmslt_random4000_v3/analysis/cross_attention_examples.json
    — one entry per curated sentence: tokens, their decoded text, the
      attention row per token (list of floats over encoder positions),
      the corresponding original frame index per encoder position, and
      the argmax ("peak") frame index per token.
"""
import json
import os
import sys

import numpy as np
import torch
import yaml
from transformers import MBart50TokenizerFast

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MMSLT_DIR)
sys.path.insert(0, os.path.dirname(__file__))

from models import MMSLT  # noqa: E402
from masked_dataset import MaskedS2TDataset  # noqa: E402

CONFIG_PATH = os.path.join(MMSLT_DIR, "configs", "config_mmslt_how2sign_random4000_v3.yaml")
CHECKPOINT_PATH = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "best_checkpoint.pth")
LANDMARKS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis", "landmarks")
ANALYSIS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis")
COMPARISON_CSV = os.path.join(ANALYSIS_DIR, "region_masking_comparison.csv")

SPECIAL_TOKENS = {"<pad>", "</s>", "<s>", "de_DE", "<unk>"}


class Args:
    input_size = 224
    resize = 256


def pick_example_sentences(n_per_region=3):
    import pandas as pd
    df = pd.read_csv(COMPARISON_CSV)
    picks = []
    for region in ["left_hand", "right_hand", "body"]:
        sub = df[df.region == region].sort_values("delta_bleu").head(n_per_region)
        picks += list(sub["SENTENCE_ID"])
    return list(dict.fromkeys(picks))  # de-dup, keep order


def orig_frame_idx(j, orig_len):
    idx = round(4 * j - 0.5)
    return int(max(0, min(orig_len - 1, idx)))


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

    targets = pick_example_sentences()
    print(f"Extracting cross-attention for {len(targets)} sentences: {targets}")

    results = []
    with torch.no_grad():
        for sid in targets:
            idx = name_to_index[sid]
            key = dataset.list[idx]
            orig_len = len(dataset.raw_data[key]["imgs_path"])
            gt_text = dataset.raw_data[key]["text"]

            src_input, _ = dataset.collate_fn([dataset[idx]])
            new_src_len = int(src_input["new_src_length_batch"][0])

            output = model.generate(
                src_input, max_new_tokens=150, num_beams=8,
                forced_bos_token_id=tokenizer.lang_code_to_id["de_DE"],
            )
            pred_text = tokenizer.decode(output[0], skip_special_tokens=True)

            inputs_embeds, attention_mask = model.share_forward(src_input)
            out = model.mbart(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask.to(device),
                labels=output.to(device),
                output_attentions=True,
                return_dict=True,
            )
            last_layer_cross_attn = out.cross_attentions[-1][0]  # [num_heads, tgt_len, src_len]
            attn = last_layer_cross_attn.mean(dim=0).float().cpu().numpy()  # [tgt_len, src_len]
            attn = attn[:, :new_src_len]
            attn = attn / attn.sum(axis=1, keepdims=True).clip(min=1e-8)

            tokens = tokenizer.convert_ids_to_tokens(output[0].tolist())
            frame_of_pos = [orig_frame_idx(j, orig_len) for j in range(new_src_len)]

            token_rows = []
            for t_idx, tok in enumerate(tokens):
                if tok in SPECIAL_TOKENS:
                    continue
                if t_idx >= attn.shape[0]:
                    break
                row = attn[t_idx].tolist()
                peak_j = int(np.argmax(row))
                token_rows.append(dict(
                    token=tok.replace("▁", " ").strip() or tok,
                    attn=[round(v, 4) for v in row],
                    peak_frame=frame_of_pos[peak_j],
                ))

            results.append(dict(
                sentence_id=sid, gt=gt_text, prediction=pred_text,
                orig_len=orig_len, new_src_len=new_src_len,
                frame_of_pos=frame_of_pos, tokens=token_rows,
            ))
            print(f"  {sid}: orig_len={orig_len} new_src_len={new_src_len} n_tokens={len(token_rows)}")

    out_path = os.path.join(ANALYSIS_DIR, "cross_attention_examples.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
