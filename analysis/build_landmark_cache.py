"""
build_landmark_cache.py

One-time step for the MMSLT region-masking ablation: run MediaPipe Holistic
(via grounding.region_extractor.RegionExtractor) over every frame of every
How2Sign test-set video (the same 64 jpgs MMSLT itself trains/evals on,
under /mnt/aix22303/data/how2sign/How2Sign/prepared/mmslt_random4000_v3/frames/test/),
and cache the resulting per-frame region landmarks (eyes/nose/mouth/face/
body/right_hand/left_hand pixel points) to disk.

Downstream (masked_dataset.py) only needs grounding.crop_utils
(bbox_from_points + mask_region_in_frame, pure numpy/opencv, no mediapipe)
to turn these cached landmarks into black-box masks at inference time, so
this is the only script in the pipeline that needs mediapipe.

Resumable: skips any video whose cache pickle already exists, so a killed
run can just be re-launched with the same command.

Usage:
    conda activate mmslt
    python build_landmark_cache.py [--workers 16]
"""
import argparse
import gzip
import os
import pickle
import sys
import time
from multiprocessing import Pool

import cv2

MMSLT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(MMSLT_DIR))
sys.path.insert(0, WORKSPACE_ROOT)

from grounding.region_extractor import RegionExtractor, save_landmarks  # noqa: E402

# The 8 regions we mask/analyze for MMSLT. region_extractor.REGION_NAMES is a
# separate, narrower list used by the CLIP/SigLIP grounding pipeline elsewhere
# in this repo — kept untouched — so we define our own superset here, which
# also includes "shoulder", a region_extractor addition made for this study.
MASK_REGIONS = ["eyes", "nose", "mouth", "face", "left_hand", "right_hand", "shoulder", "body"]

LABELS_TEST = "/mnt/aix22303/data/how2sign/How2Sign/prepared/mmslt_random4000_v3/labels/labels.test"
IMG_ROOT = "/mnt/aix22303/data/how2sign/How2Sign/prepared/mmslt_random4000_v3/frames/"
CACHE_DIR = os.path.join(MMSLT_DIR, "outputs", "mmslt_random4000_v3", "analysis", "landmarks")


def load_test_labels():
    with gzip.open(LABELS_TEST, "rb") as f:
        return pickle.load(f)


def process_one_video(item):
    sentence_id, imgs_path = item
    cache_path = os.path.join(CACHE_DIR, f"{sentence_id}.pkl")
    if os.path.exists(cache_path):
        return sentence_id, None  # already done, nothing to count

    extractor = RegionExtractor()
    frame_regions = []
    try:
        for idx, rel_path in enumerate(imgs_path):
            frame_bgr = cv2.imread(os.path.join(IMG_ROOT, rel_path))
            if frame_bgr is None:
                continue
            fr = extractor.process_frame(frame_bgr, idx)
            fr.crops = {}  # drop crops before caching, landmarks are enough
            frame_regions.append(fr)
    finally:
        extractor.close()

    save_landmarks(frame_regions, cache_path)

    counts = {r: 0 for r in MASK_REGIONS}
    for fr in frame_regions:
        for r in fr.landmarks:
            counts[r] += 1
    return sentence_id, {"num_frames": len(frame_regions), "counts": counts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    raw_data = load_test_labels()
    items = [(v["name"], v["imgs_path"]) for v in raw_data.values()]
    print(f"Total test videos: {len(items)}")

    already_done = sum(1 for name, _ in items if os.path.exists(os.path.join(CACHE_DIR, f"{name}.pkl")))
    print(f"Already cached: {already_done} (will be skipped)")

    total_frames = 0
    region_counts = {r: 0 for r in MASK_REGIONS}
    n_done = 0
    t0 = time.time()

    with Pool(args.workers) as pool:
        for sentence_id, stats in pool.imap_unordered(process_one_video, items):
            n_done += 1
            if stats is not None:
                total_frames += stats["num_frames"]
                for r, c in stats["counts"].items():
                    region_counts[r] += c
            if n_done % 20 == 0 or n_done == len(items):
                elapsed = time.time() - t0
                print(f"[{n_done}/{len(items)}] elapsed={elapsed:.0f}s last={sentence_id}", flush=True)

    print("\n=== Landmark detection rate (over newly-processed videos only) ===")
    if total_frames > 0:
        for r in MASK_REGIONS:
            pct = 100.0 * region_counts[r] / total_frames
            print(f"  {r:12s}: {region_counts[r]:6d}/{total_frames} frames ({pct:5.1f}%)")
    else:
        print("  (nothing new processed this run — all videos already cached)")
    print(f"\nDone. Cache dir: {CACHE_DIR}")


if __name__ == "__main__":
    main()
