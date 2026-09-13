"""
shap_masked_dataset.py

Generalization of masked_dataset.py for exact Shapley-value computation over
the 8 body regions: masks an arbitrary SUBSET of regions together (not just
one), and restricts the dataset to a fixed 60-sentence subsample of the
300-sentence test set (results/sampling/shap_sentence_subsample_60.json,
seed=42 — same subsample used for SpaMo/VTaMo's SHAP runs, see
external/SpaMo/analysis/shap_masked_feature_extract.py's docstring for the
cost tradeoff that subsample represents).

MMSLT masks pixels inline (no offline feature-extraction step), so unlike
SpaMo/VTaMo this doesn't need a separate multi-day extraction phase — this
dataset class is all that's needed; shap_run_inference.py drives it per
combo directly.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(MMSLT_DIR))
sys.path.insert(0, MMSLT_DIR)
sys.path.insert(0, WORKSPACE_ROOT)

import utils as mmslt_utils  # noqa: E402
from datasets import S2T_Dataset  # noqa: E402
from grounding.region_extractor import load_landmarks  # noqa: E402
from grounding.crop_utils import bbox_from_points, mask_region_in_frame  # noqa: E402

MASK_REGIONS = ["eyes", "nose", "mouth", "face", "left_hand", "right_hand", "shoulder", "body"]

DEFAULT_BBOX_KWARGS = dict(pad_ratio=0.25, square=True)
BBOX_KWARGS = {
    "shoulder": dict(pad_ratio=0.25, square=False, min_size_px=120),
}

SUBSAMPLE_PATH = os.path.join(WORKSPACE_ROOT, "results", "sampling", "shap_sentence_subsample_60.json")


def combo_name(regions):
    return "original" if not regions else "+".join(sorted(regions))


class ShapMaskedS2TDataset(S2T_Dataset):
    def __init__(self, path, tokenizer, config, args, phase, mask_regions=(), landmarks_dir=None):
        assert all(r in MASK_REGIONS for r in mask_regions), mask_regions
        super().__init__(path, tokenizer, config, args, phase)
        self.mask_regions = list(mask_regions)
        self.landmarks_dir = landmarks_dir
        self._landmark_cache = {}
        self.miss_count = 0
        self.hit_count = 0

        with open(SUBSAMPLE_PATH) as f:
            keep = set(json.load(f)["sentence_ids"])
        self.list = [k for k in self.list if self.raw_data[k]["name"] in keep]

    def __len__(self):
        # S2T_Dataset.__len__ returns len(self.raw_data) (the full, unfiltered
        # dict) rather than len(self.list) -- harmless for the base class and
        # masked_dataset.py's MaskedS2TDataset (neither ever filters self.list),
        # but must be overridden here since self.list is a strict subset.
        return len(self.list)

    def _get_landmarks(self, sentence_id):
        if sentence_id not in self._landmark_cache:
            cache_path = os.path.join(self.landmarks_dir, f"{sentence_id}.pkl")
            frame_regions = load_landmarks(cache_path)
            self._landmark_cache[sentence_id] = {fr.frame_idx: fr for fr in frame_regions}
        return self._landmark_cache[sentence_id]

    def __getitem__(self, index):
        key = self.list[index]
        sample = self.raw_data[key]

        descript_sample = self.descript_feat[key.split('/')[1]]['bert_feat']
        tgt_sample = sample['text']
        name_sample = sample['name']

        img_sample = self.load_imgs([self.img_path + x for x in sample['imgs_path']], name_sample)
        return name_sample, descript_sample, tgt_sample, img_sample

    def load_imgs(self, paths, sentence_id):
        by_idx = None
        if self.mask_regions:
            by_idx = self._get_landmarks(sentence_id)

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        if len(paths) > self.max_length:
            tmp = sorted(np.random.choice(len(paths), size=self.max_length, replace=False))
            paths = [paths[i] for i in tmp]

        imgs = torch.zeros(len(paths), 3, self.args.input_size, self.args.input_size)
        crop_rect, resize = mmslt_utils.data_augmentation(
            resize=(self.args.resize, self.args.resize),
            crop_size=self.args.input_size,
            is_train=False,
        )

        batch_image = []
        for i, img_path in enumerate(paths):
            frame_bgr = cv2.imread(img_path)
            if by_idx is not None:
                fr = by_idx.get(i)
                any_hit = False
                for region in self.mask_regions:
                    pts = fr.landmarks.get(region) if fr is not None else None
                    if pts is not None:
                        h, w = frame_bgr.shape[:2]
                        kwargs = BBOX_KWARGS.get(region, DEFAULT_BBOX_KWARGS)
                        bbox = bbox_from_points(pts, w, h, **kwargs)
                        frame_bgr = mask_region_in_frame(frame_bgr, bbox, mode="black")
                        any_hit = True
                if any_hit:
                    self.hit_count += 1
                else:
                    self.miss_count += 1
            img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_image.append(img)

        for i, img in enumerate(batch_image):
            img_resized = img.resize(resize)
            img_tensor = data_transform(img_resized).unsqueeze(0)
            imgs[i, :, :, :] = img_tensor[:, :, crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]

        return imgs
