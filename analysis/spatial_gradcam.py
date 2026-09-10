"""
spatial_gradcam.py

Spatial grounding for MMSLT: classic Grad-CAM (Selvaraju et al.) on the last
conv block of MMSLT's own ResNet18 frame encoder (`model.backbone.resnet.layer4`)
-- unlike SpaMo's CLIP-ViT, this is a genuine CNN inside the actual trained
model's own forward path (not a frozen offline feature cache), so no
re-engineering is needed: the same generation-loss backward pass used by
gradcam_full_corpus.py (temporal) also carries a real gradient straight into
this conv layer's spatial feature map, for every frame.

    Grad-CAM(frame) = ReLU( sum_c  GAP(d(loss)/d(activations_c)) * activations_c )

For each of the 300 test sentences, computes this for a single representative
frame -- the mode (most frequently peaked-at) frame across that sentence's
tokens' TEMPORAL GradCAM peaks (gradcam_full.json, already computed) -- so
temporal and spatial grounding are directly comparable for the same sentence
and the same underlying gradient signal, not two unrelated analyses.

Overlay is rendered on the 224x224 center-cropped, resized RGB frame (exactly
what the ResNet actually saw -- same deterministic crop as datasets.py's
data_augmentation(is_train=False): resize to 256x256, center-crop 224x224),
not the raw source frame, so pixel alignment with the CAM grid is exact.

Outputs (under outputs/mmslt_random4000_v3/analysis/):
    spatial_gradcam.json — per sentence: sentence_id, peak_frame_idx,
        cam_grid (7x7 floats, raw Grad-CAM before upsampling),
        frame_thumb_b64 (base64 JPEG, 112x112, the plain preprocessed frame
        with no heatmap baked in -- overlay is drawn client-side from cam_grid
        so the viewer can toggle it and control opacity)

Usage (pick GPU 4/5/6/7; always nohup — see project memory):
    conda activate mmslt
    cd external/MMSLT
    CUDA_VISIBLE_DEVICES=4 python analysis/spatial_gradcam.py
"""
import base64
import io
import json
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as Fn
import yaml
from PIL import Image
from torchvision import transforms
from transformers import MBart50TokenizerFast

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MMSLT_DIR)
sys.path.insert(0, os.path.dirname(__file__))

import utils as mmslt_utils  # noqa: E402
from models import MMSLT  # noqa: E402
from masked_dataset import MaskedS2TDataset  # noqa: E402
from cross_attention_analysis import Args  # noqa: E402

CONFIG_PATH = os.path.join(MMSLT_DIR, "configs", "config_mmslt_how2sign_random4000_v3.yaml")
CHECKPOINT_PATH = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "best_checkpoint.pth")
LANDMARKS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis", "landmarks")
ANALYSIS_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis")
GRADCAM_FULL_JSON = os.path.join(ANALYSIS_DIR, "gradcam_full.json")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def load_preprocessed_frame(img_path, resize, crop_rect):
    """Same pipeline as datasets.py S2T_Dataset.load_imgs, but returns the
    plain (un-normalized) RGB uint8 224x224 crop for visualization, instead
    of a normalized tensor."""
    frame_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize(resize)
    arr = np.array(img)  # H,W,3 uint8, at `resize` size
    l, t, r, b = crop_rect
    return arr[t:b, l:r]  # crop_size x crop_size x 3


def cam_to_thumb_b64(rgb_224, size=112, quality=60):
    img = Image.fromarray(rgb_224).resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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

    with open(GRADCAM_FULL_JSON) as f:
        gradcam_full = json.load(f)
    print(f"Loaded temporal GradCAM for {len(gradcam_full)} sentences to pick representative frames")

    # ---- register forward hook to capture layer4 activations + their grad ----
    captured = {}

    def hook(module, inp, out):
        out.retain_grad()
        captured["activations"] = out

    handle = model.backbone.resnet.layer4.register_forward_hook(hook)

    crop_rect, resize = mmslt_utils.data_augmentation(resize=(256, 256), crop_size=224, is_train=False)

    results = []
    for n_done, entry in enumerate(gradcam_full, 1):
        sid = entry["sentence_id"]
        if sid not in name_to_index:
            continue
        idx = name_to_index[sid]
        key = dataset.list[idx]
        imgs_path = dataset.raw_data[key]["imgs_path"]

        peak_frames = [t["peak_frame"] for t in entry["tokens"]]
        if not peak_frames:
            continue
        peak_frame_idx = Counter(peak_frames).most_common(1)[0][0]
        peak_frame_idx = max(0, min(len(imgs_path) - 1, peak_frame_idx))

        src_input, _ = dataset.collate_fn([dataset[idx]])
        target_ids = tokenizer(entry["prediction"], return_tensors="pt").input_ids.to(device)

        model.zero_grad(set_to_none=True)
        inputs_embeds, attention_mask = model.share_forward(src_input)
        out = model.mbart(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask.to(device),
            labels=target_ids, return_dict=True,
        )
        out.loss.backward()

        activations = captured.get("activations")
        if activations is None or activations.grad is None:
            print(f"  WARNING: {sid} got no layer4 gradient, skipping")
            continue

        # activations: [total_frames_in_batch, 512, 7, 7]; frame `peak_frame_idx`
        # of THIS sentence is at that same index since batch size is 1 here.
        if peak_frame_idx >= activations.shape[0]:
            continue
        act = activations[peak_frame_idx].detach()
        grad = activations.grad[peak_frame_idx].detach()
        weights = grad.mean(dim=(1, 2))  # [512], global-average-pooled gradient per channel
        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))  # [7,7]
        cam = cam / cam.max().clamp(min=1e-8)
        cam_grid = cam.cpu().numpy().tolist()

        img_path = os.path.join(dataset.img_path, imgs_path[peak_frame_idx])
        try:
            rgb_224 = load_preprocessed_frame(img_path, resize, crop_rect)
        except Exception as e:
            print(f"  WARNING: {sid} failed to load frame image {img_path}: {e}")
            continue
        thumb_b64 = cam_to_thumb_b64(rgb_224)

        results.append(dict(
            sentence_id=sid, peak_frame_idx=peak_frame_idx, n_frames=len(imgs_path),
            cam_grid=[[round(v, 4) for v in row] for row in cam_grid],
            frame_thumb_b64=thumb_b64,
        ))

        if n_done % 25 == 0 or n_done == len(gradcam_full):
            print(f"  [{n_done}/{len(gradcam_full)}] last={sid}", flush=True)

    handle.remove()

    out_path = os.path.join(ANALYSIS_DIR, "spatial_gradcam.json")
    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"wrote {out_path} ({len(results)} sentences)")


if __name__ == "__main__":
    main()
