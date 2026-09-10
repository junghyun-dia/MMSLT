"""
masked_dataset.py

Subclass of datasets.S2T_Dataset that, when constructed with `mask_region`
set to one of MASK_REGIONS, blacks out that body region in every frame
before it goes through MMSLT's normal preprocessing (resize/crop/normalize).
`mask_region=None` behaves identically to the original S2T_Dataset (used
for the baseline/"original" condition).

Only needs grounding.crop_utils (numpy/opencv, no mediapipe) plus the
per-video landmark caches built once by build_landmark_cache.py — so this
(and everything downstream) can run entirely inside the `mmslt` conda env.
"""
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
# "shoulder" is defined by only 2 (near-collinear, wide-apart) points, so the
# default square=True box balloons into a huge square that swallows the face
# and torso (verified visually) — use a thin non-square band instead.
BBOX_KWARGS = {
    "shoulder": dict(pad_ratio=0.25, square=False, min_size_px=120),
}


class MaskedS2TDataset(S2T_Dataset):
    def __init__(self, path, tokenizer, config, args, phase, mask_region=None, landmarks_dir=None):
        assert mask_region is None or mask_region in MASK_REGIONS, mask_region
        super().__init__(path, tokenizer, config, args, phase)
        self.mask_region = mask_region
        self.landmarks_dir = landmarks_dir
        self._landmark_cache = {}
        self.miss_count = 0
        self.hit_count = 0

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
        if self.mask_region is not None:
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
                pts = fr.landmarks.get(self.mask_region) if fr is not None else None
                if pts is not None:
                    h, w = frame_bgr.shape[:2]
                    kwargs = BBOX_KWARGS.get(self.mask_region, DEFAULT_BBOX_KWARGS)
                    bbox = bbox_from_points(pts, w, h, **kwargs)
                    frame_bgr = mask_region_in_frame(frame_bgr, bbox, mode="black")
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
