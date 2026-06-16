#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KSL AIHub WORD keypoint training script - desktop tuned v5
=========================================================

Goal
----
Train a keypoint-based word classifier for realtime continuous webcam inference.
This version is tuned for a stronger desktop GPU such as RTX 5080 16GB.

Main ideas
----------
- Use all D/U/L/R/F views.
- Recommended validation split: train D/U/L/R, validate F.
- Cache RAW 65x3 keypoint sequences, not final features.
- Generate realtime-like windows every epoch with temporal crop, speed jitter,
  idle padding, hand dropout, joint dropout, time masking, coordinate noise,
  rotation jitter and z-scale jitter.
- Use a stronger TCN + attention pooling model.
- Support EMA, mixup, label smoothing, gradient accumulation, AMP, final fit on all views.

Feature format
--------------
Input feature dimension is kept as 390 for compatibility:
normalized xyz flatten 195 + temporal delta 195.

Expected folder structure
-------------------------
REAL/WORD/
  01/
    NIA_SL_WORD0001_REAL01_D/*.json
    NIA_SL_WORD0001_REAL01_F/*.json
    ...

Example command
---------------
py -3 .\ksl_keypoint_tcn_train_v5_desktop.py `
  --data_root "D:\Gradulation_Project\004.수어영상\1.Training\라벨링데이터\REAL\WORD" `
  --output_dir "D:\Gradulation_Project\outputs\ksl_keypoint_tcn_v5_desktop" `
  --coord 3d --view_filter D,U,L,R,F --split_strategy label_view_holdout --val_views F `
  --target_len 96 --epochs 140 --batch_size 96 --grad_accum_steps 1 `
  --channels 384 --blocks 6 --dropout 0.25 --lr 8e-4 `
  --label_smoothing 0.05 --mixup_alpha 0.15 --mixup_prob 0.35 `
  --ema --ema_decay 0.999 --amp --num_workers 6 --rebuild_cache --sanity_check `
  --final_fit_epochs 20 --final_fit_lr 2e-4
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
except Exception as e:
    raise RuntimeError("PyTorch is required. Install a CUDA build of torch first.") from e


FEATURE_VERSION = "ksl_pose23_xyz_delta_norm_v5_realtime_20260615"
VALID_VIEWS = ["D", "U", "L", "R", "F"]
WORD_RE = re.compile(r"(WORD\d{4})", re.IGNORECASE)
REAL_RE = re.compile(r"(REAL\d+)", re.IGNORECASE)
VIEW_RE = re.compile(r"_([DULRF])(?:$|_)", re.IGNORECASE)


@dataclass
class SampleInfo:
    sample_dir: str
    label: str
    group: str
    view: str
    num_json: int


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True


def parse_view_filter(view_filter: str) -> List[str]:
    if not view_filter:
        return []
    views = []
    for v in view_filter.split(","):
        v = v.strip().upper()
        if not v:
            continue
        if v not in VALID_VIEWS:
            raise ValueError(f"Invalid view '{v}'. Valid views: {VALID_VIEWS}")
        views.append(v)
    return sorted(set(views), key=lambda x: VALID_VIEWS.index(x))


def extract_name_info(path: Path) -> Tuple[str, str, str]:
    name = path.name
    word_m = WORD_RE.search(name) or WORD_RE.search(str(path))
    if not word_m:
        raise ValueError(f"Cannot find WORDxxxx label from path: {path}")
    label = word_m.group(1).upper()

    real_m = REAL_RE.search(name)
    real = real_m.group(1).upper() if real_m else "REALXX"

    view_m = VIEW_RE.search(name + "_")
    view = view_m.group(1).upper() if view_m else "X"

    group = f"{label}_{real}"
    return label, group, view


def discover_samples(root: Path, min_frames: int = 8, max_samples: int = 0, view_filter: str = "") -> List[SampleInfo]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    allowed_views = set(parse_view_filter(view_filter)) if view_filter else set()
    samples: List[SampleInfo] = []
    seen_dirs = set()
    scanned_dirs = 0

    print(f"[INFO] Fast scanning root: {root}")
    print(
        f"[INFO] min_frames={min_frames}, "
        f"max_samples={max_samples if max_samples > 0 else 'all'}, "
        f"view_filter={','.join(sorted(allowed_views)) if allowed_views else 'ALL'}"
    )

    for dirpath, dirnames, filenames in os.walk(root):
        scanned_dirs += 1
        if scanned_dirs % 2000 == 0:
            print(f"[SCAN] dirs={scanned_dirs:,}, samples={len(samples):,}, current={dirpath}")

        key_count = sum(1 for fn in filenames if fn.endswith("_keypoints.json"))
        if key_count < min_frames:
            continue

        d = Path(dirpath)
        if str(d) in seen_dirs:
            continue
        seen_dirs.add(str(d))

        try:
            label, group, view = extract_name_info(d)
        except ValueError:
            continue

        if allowed_views and view.upper() not in allowed_views:
            dirnames[:] = []
            continue

        samples.append(SampleInfo(str(d), label, group, view, key_count))
        if len(samples) <= 5 or len(samples) % 1000 == 0:
            print(f"[FOUND] {len(samples):,}: {d.name} frames={key_count} label={label} view={view}")

        dirnames[:] = []
        if max_samples > 0 and len(samples) >= max_samples:
            print(f"[INFO] max_samples reached: {max_samples}")
            break

    return samples


def filter_min_samples_per_class(samples: List[SampleInfo], min_samples_per_class: int) -> List[SampleInfo]:
    if min_samples_per_class <= 1:
        return samples
    counts = Counter(s.label for s in samples)
    keep = {lab for lab, c in counts.items() if c >= min_samples_per_class}
    filtered = [s for s in samples if s.label in keep]
    print(f"[INFO] min_samples_per_class={min_samples_per_class}: removed_samples={len(samples)-len(filtered)}, kept_classes={len(keep)}")
    return filtered


def limit_classes(samples: List[SampleInfo], max_classes: int, seed: int = 42) -> List[SampleInfo]:
    if max_classes <= 0:
        return samples
    labels = sorted({s.label for s in samples})
    rnd = random.Random(seed)
    rnd.shuffle(labels)
    keep = set(sorted(labels[:max_classes]))
    filtered = [s for s in samples if s.label in keep]
    print(f"[INFO] limit_classes={max_classes}: samples={len(filtered)}, classes={len(keep)}")
    return filtered


# -----------------------------------------------------------------------------
# Split strategies
# -----------------------------------------------------------------------------

def split_label_view_holdout(samples: List[SampleInfo], val_views: Sequence[str], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    val_views = {v.upper() for v in val_views if v}
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        label_to_indices[s.label].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    rnd = random.Random(seed)

    for _, idxs in sorted(label_to_indices.items()):
        idxs = sorted(idxs, key=lambda i: (samples[i].group, samples[i].view, samples[i].sample_dir))
        if len(idxs) == 1:
            train_idx.extend(idxs)
            continue

        preferred_val = [i for i in idxs if samples[i].view.upper() in val_views]
        if preferred_val:
            max_val = max(1, min(len(idxs) - 1, int(round(len(idxs) * max(val_ratio, 1.0 / len(idxs))))))
            chosen_val = preferred_val[:max_val]
        else:
            shuffled = idxs[:]
            rnd.shuffle(shuffled)
            n_val = max(1, min(len(idxs) - 1, int(round(len(idxs) * val_ratio))))
            chosen_val = sorted(shuffled[:n_val])

        val_set = set(chosen_val)
        chosen_train = [i for i in idxs if i not in val_set]
        if not chosen_train:
            chosen_train = [chosen_val.pop()]

        train_idx.extend(chosen_train)
        val_idx.extend(chosen_val)

    return sorted(train_idx), sorted(val_idx)


def split_label_stratified_random(samples: List[SampleInfo], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    label_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        label_to_indices[s.label].append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    rnd = random.Random(seed)

    for _, idxs in sorted(label_to_indices.items()):
        idxs = idxs[:]
        rnd.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs)
            continue
        n_val = max(1, min(len(idxs) - 1, int(round(len(idxs) * val_ratio))))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    return sorted(train_idx), sorted(val_idx)


def split_group_holdout(samples: List[SampleInfo], val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    groups = sorted({s.group for s in samples})
    rnd = random.Random(seed)
    rnd.shuffle(groups)
    label_to_groups: Dict[str, List[str]] = defaultdict(list)
    for g in groups:
        label_to_groups[g.split("_")[0]].append(g)

    val_groups = set()
    for _, gs in label_to_groups.items():
        if len(gs) >= 2:
            n_val = max(1, int(round(len(gs) * val_ratio)))
            val_groups.update(gs[:n_val])

    train_idx, val_idx = [], []
    for i, s in enumerate(samples):
        if s.group in val_groups:
            val_idx.append(i)
        else:
            train_idx.append(i)
    return sorted(train_idx), sorted(val_idx)


def make_split(args, samples: List[SampleInfo]) -> Tuple[List[int], List[int]]:
    if args.split_strategy == "label_view_holdout":
        val_views = parse_view_filter(args.val_views) if args.val_views else ["F"]
        return split_label_view_holdout(samples, val_views=val_views, val_ratio=args.val_ratio, seed=args.seed)
    if args.split_strategy == "label_stratified_random":
        return split_label_stratified_random(samples, val_ratio=args.val_ratio, seed=args.seed)
    if args.split_strategy == "group_holdout":
        return split_group_holdout(samples, val_ratio=args.val_ratio, seed=args.seed)
    raise ValueError(args.split_strategy)


def validate_split(samples: List[SampleInfo], train_idx: List[int], val_idx: List[int]) -> None:
    if not train_idx:
        raise RuntimeError("train set is empty")
    train_labels = {samples[i].label for i in train_idx}
    val_labels = {samples[i].label for i in val_idx}
    missing = sorted(val_labels - train_labels)
    if missing:
        raise RuntimeError(
            f"Validation has labels not in train. missing_count={len(missing)}, preview={missing[:20]}. "
            "Use label_view_holdout or label_stratified_random."
        )


def print_dataset_stats(samples: List[SampleInfo], train_idx: List[int], val_idx: List[int]) -> None:
    labels = sorted({s.label for s in samples})
    view_counts = Counter(s.view for s in samples)
    train_view_counts = Counter(samples[i].view for i in train_idx)
    val_view_counts = Counter(samples[i].view for i in val_idx)
    label_counts = Counter(s.label for s in samples)
    train_label_counts = Counter(samples[i].label for i in train_idx)
    val_label_counts = Counter(samples[i].label for i in val_idx)

    print(f"[INFO] samples: {len(samples):,}")
    print(f"[INFO] classes: {len(labels):,}")
    print(f"[INFO] all view counts: {dict(sorted(view_counts.items()))}")
    print(f"[INFO] train: {len(train_idx):,}, val: {len(val_idx):,}")
    print(f"[INFO] train view counts: {dict(sorted(train_view_counts.items()))}")
    print(f"[INFO] val view counts: {dict(sorted(val_view_counts.items()))}")

    for name, counter in [("all", label_counts), ("train", train_label_counts), ("val", val_label_counts)]:
        if counter:
            vals = list(counter.values())
            print(f"[INFO] samples/class {name}: min={min(vals)}, max={max(vals)}, mean={np.mean(vals):.2f}")


# -----------------------------------------------------------------------------
# JSON parsing / keypoint conversion
# -----------------------------------------------------------------------------

def _reshape_keypoints(arr: Sequence[float], dim: int) -> np.ndarray:
    arr_np = np.asarray(arr, dtype=np.float32)
    if arr_np.size == 0:
        return np.zeros((0, dim), dtype=np.float32)
    if arr_np.size % dim != 0:
        raise ValueError(f"Invalid keypoint length: {arr_np.size}, dim={dim}")
    return arr_np.reshape(-1, dim)


def read_people_dict(js: dict) -> dict:
    people = js.get("people", {})
    if isinstance(people, list):
        return people[0] if people else {}
    if isinstance(people, dict):
        return people
    return {}


COMMON_POSE23_NAMES = [
    "nose",
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder", "left_elbow", "left_wrist",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
    "right_eye", "left_eye", "right_ear", "left_ear",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
    "neck", "midhip",
]

AIHUB_DIRECT_POSE21_IDX = [
    0,
    2, 3, 4,
    5, 6, 7,
    9, 10, 11,
    12, 13, 14,
    15, 16, 17, 18,
    21, 24,
    19, 22,
]


def select_aihub_common_pose23(pose25_xyz: np.ndarray) -> np.ndarray:
    pose25 = np.zeros((25, 3), dtype=np.float32)
    if pose25_xyz is not None and pose25_xyz.size > 0:
        n = min(25, pose25_xyz.shape[0])
        pose25[:n] = pose25_xyz[:n, :3]

    out = np.zeros((23, 3), dtype=np.float32)
    for j, idx in enumerate(AIHUB_DIRECT_POSE21_IDX):
        out[j] = pose25[idx]

    rsho, lsho = pose25[2], pose25[5]
    rhip, lhip = pose25[9], pose25[12]

    if not (np.all(np.abs(rsho) < 1e-8) or np.all(np.abs(lsho) < 1e-8)):
        out[21] = (rsho + lsho) / 2.0
    else:
        out[21] = pose25[1]

    if not (np.all(np.abs(rhip) < 1e-8) or np.all(np.abs(lhip) < 1e-8)):
        out[22] = (rhip + lhip) / 2.0
    else:
        out[22] = pose25[8]

    return out.astype(np.float32)


def extract_frame_65x3(json_path: Path, coord: str = "3d") -> np.ndarray:
    with open(json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    p = read_people_dict(js)

    if coord.lower() == "3d":
        lh = _reshape_keypoints(p.get("hand_left_keypoints_3d", []), 4)[:21, :3]
        rh = _reshape_keypoints(p.get("hand_right_keypoints_3d", []), 4)[:21, :3]
        pose25 = _reshape_keypoints(p.get("pose_keypoints_3d", []), 4)[:25, :3]
        pose = select_aihub_common_pose23(pose25)
    elif coord.lower() == "2d":
        lh2 = _reshape_keypoints(p.get("hand_left_keypoints_2d", []), 3)[:21, :2]
        rh2 = _reshape_keypoints(p.get("hand_right_keypoints_2d", []), 3)[:21, :2]
        pose2 = _reshape_keypoints(p.get("pose_keypoints_2d", []), 3)[:25, :2]
        lh = np.concatenate([lh2, np.zeros((lh2.shape[0], 1), dtype=np.float32)], axis=1)
        rh = np.concatenate([rh2, np.zeros((rh2.shape[0], 1), dtype=np.float32)], axis=1)
        pose25 = np.concatenate([pose2, np.zeros((pose2.shape[0], 1), dtype=np.float32)], axis=1)
        pose = select_aihub_common_pose23(pose25)
    else:
        raise ValueError("--coord must be 3d or 2d")

    def pad(a: np.ndarray, n: int) -> np.ndarray:
        if a.shape[0] >= n:
            return a[:n]
        out = np.zeros((n, 3), dtype=np.float32)
        out[:a.shape[0]] = a
        return out

    return np.concatenate([pad(lh, 21), pad(rh, 21), pad(pose, 23)], axis=0).astype(np.float32)


def load_sequence_65x3(sample_dir: Path, coord: str = "3d") -> np.ndarray:
    files = sorted(sample_dir.glob("*_keypoints.json"))
    frames = []
    for fp in files:
        try:
            frames.append(extract_frame_65x3(fp, coord=coord))
        except Exception as e:
            print(f"[WARN] failed to parse {fp}: {e}", file=sys.stderr)
    if not frames:
        raise RuntimeError(f"No valid keypoint frames in {sample_dir}")
    return np.stack(frames, axis=0).astype(np.float32)


# -----------------------------------------------------------------------------
# Feature and augmentation
# -----------------------------------------------------------------------------

def valid_mask(seq: np.ndarray) -> np.ndarray:
    return ~np.all(np.abs(seq) < 1e-8, axis=-1)


def normalize_sequence(seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = seq.astype(np.float32).copy()
    T = x.shape[0]
    pose = x[:, 42:65, :]
    centers = np.zeros((T, 1, 3), dtype=np.float32)
    scales = np.ones((T, 1, 1), dtype=np.float32)
    vm = valid_mask(x)

    for t in range(T):
        pose_t = pose[t]
        valid_t = vm[t]
        neck = pose_t[21] if pose_t.shape[0] > 21 else np.zeros(3, dtype=np.float32)
        rsho = pose_t[1] if pose_t.shape[0] > 1 else np.zeros(3, dtype=np.float32)
        lsho = pose_t[4] if pose_t.shape[0] > 4 else np.zeros(3, dtype=np.float32)

        neck_ok = not np.all(np.abs(neck) < eps)
        r_ok = not np.all(np.abs(rsho) < eps)
        l_ok = not np.all(np.abs(lsho) < eps)

        if neck_ok:
            c = neck
        elif r_ok and l_ok:
            c = (rsho + lsho) / 2.0
        elif valid_t.any():
            c = x[t, valid_t].mean(axis=0)
        else:
            c = np.zeros(3, dtype=np.float32)

        if r_ok and l_ok:
            s = float(np.linalg.norm(rsho - lsho))
        elif valid_t.any():
            pts = x[t, valid_t]
            s = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        else:
            s = 1.0

        if s < eps or not np.isfinite(s):
            s = 1.0

        centers[t, 0] = c
        scales[t, 0, 0] = s

    y = np.zeros_like(x, dtype=np.float32)
    for t in range(T):
        m = vm[t]
        y[t, m] = (x[t, m] - centers[t]) / scales[t]

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -10.0, 10.0)
    return y.astype(np.float32)


def temporal_resample_flat(seq: np.ndarray, target_len: int) -> np.ndarray:
    T = seq.shape[0]
    if T == target_len:
        return seq.astype(np.float32)
    if T <= 1:
        return np.repeat(seq, target_len, axis=0).astype(np.float32)

    old_idx = np.linspace(0, 1, T, dtype=np.float32)
    new_idx = np.linspace(0, 1, target_len, dtype=np.float32)
    out = np.empty((target_len, seq.shape[1]), dtype=np.float32)
    for d in range(seq.shape[1]):
        out[:, d] = np.interp(new_idx, old_idx, seq[:, d])
    return out


def resample_keypoints(seq: np.ndarray, new_len: int) -> np.ndarray:
    T, K, C = seq.shape
    flat = seq.reshape(T, K * C)
    out = temporal_resample_flat(flat, new_len)
    return out.reshape(new_len, K, C).astype(np.float32)


def make_features(seq65: np.ndarray, target_len: int) -> np.ndarray:
    norm = normalize_sequence(seq65)
    flat = norm.reshape(norm.shape[0], -1).astype(np.float32)
    delta = np.zeros_like(flat)
    if flat.shape[0] > 1:
        delta[1:] = flat[1:] - flat[:-1]
    feat = np.concatenate([flat, delta], axis=1)
    feat = temporal_resample_flat(feat, target_len=target_len)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feat


def augment_sequence_realtime(seq: np.ndarray, args, rng: np.random.Generator) -> np.ndarray:
    """Create a synthetic realtime sliding-window sample from an isolated sign sequence."""
    out = seq.astype(np.float32).copy()
    T = out.shape[0]

    # Randomly trim beginning/end to imitate partial sliding windows.
    if args.temporal_crop_prob > 0 and rng.random() < args.temporal_crop_prob and T > 16:
        keep_ratio = float(rng.uniform(args.temporal_crop_min, 1.0))
        crop_len = max(8, int(round(T * keep_ratio)))
        crop_len = min(crop_len, T)
        start = int(rng.integers(0, T - crop_len + 1))
        out = out[start:start + crop_len]
        T = out.shape[0]

    # Speed jitter before final resampling.
    if args.speed_jitter > 0 and T > 2:
        factor = float(rng.uniform(1.0 - args.speed_jitter, 1.0 + args.speed_jitter))
        new_len = max(8, int(round(T / max(0.1, factor))))
        out = resample_keypoints(out, new_len)
        T = out.shape[0]

    # Add idle frames before/after. This is important for continuous webcam windows.
    if args.idle_pad_prob > 0 and rng.random() < args.idle_pad_prob and T > 0:
        max_pad = max(0, int(round(args.idle_pad_max_ratio * T)))
        if max_pad > 0:
            pre = int(rng.integers(0, max_pad + 1))
            post = int(rng.integers(0, max_pad + 1))
            if pre > 0:
                pre_frames = np.repeat(out[:1], pre, axis=0)
                out = np.concatenate([pre_frames, out], axis=0)
            if post > 0:
                post_frames = np.repeat(out[-1:], post, axis=0)
                out = np.concatenate([out, post_frames], axis=0)
            T = out.shape[0]

    # Small geometric jitter before normalization.
    if args.coord_noise_std > 0:
        valid = valid_mask(out)[..., None]
        noise = rng.normal(0, args.coord_noise_std, size=out.shape).astype(np.float32)
        out = out + noise * valid

    if args.z_scale_jitter > 0:
        z_scale = float(rng.uniform(1.0 - args.z_scale_jitter, 1.0 + args.z_scale_jitter))
        out[:, :, 2] *= z_scale

    if args.rotate_jitter_deg > 0:
        angle = math.radians(float(rng.uniform(-args.rotate_jitter_deg, args.rotate_jitter_deg)))
        ca, sa = math.cos(angle), math.sin(angle)
        x = out[:, :, 0].copy()
        y = out[:, :, 1].copy()
        out[:, :, 0] = ca * x - sa * y
        out[:, :, 1] = sa * x + ca * y

    # Simulate MediaPipe missing hand frames. One hand can disappear briefly.
    if args.hand_dropout_prob > 0 and rng.random() < args.hand_dropout_prob and T > 4:
        hand = 0 if rng.random() < 0.5 else 1
        sl = slice(0, 21) if hand == 0 else slice(21, 42)
        w = max(1, int(round(T * float(rng.uniform(0.04, args.hand_dropout_max_ratio)))))
        st = int(rng.integers(0, max(1, T - w + 1)))
        out[st:st + w, sl, :] = 0.0

    # Drop random joints, mostly hands.
    if args.joint_dropout_prob > 0 and rng.random() < args.joint_dropout_prob:
        n_drop = int(rng.integers(1, args.joint_dropout_max + 1))
        candidates = np.arange(0, 42)  # hands only by default
        joints = rng.choice(candidates, size=min(n_drop, len(candidates)), replace=False)
        out[:, joints, :] = 0.0

    # Time masking over all features/keypoints.
    if args.time_mask_prob > 0 and rng.random() < args.time_mask_prob and T > 4:
        w = max(1, int(round(T * float(rng.uniform(0.03, args.time_mask_max_ratio)))))
        st = int(rng.integers(0, max(1, T - w + 1)))
        out[st:st + w, :, :] = 0.0

    return out.astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset with raw sequence cache
# -----------------------------------------------------------------------------

def raw_cache_key(sample_dir: str, coord: str, cache_version: str) -> str:
    p = Path(sample_dir)
    files = sorted(p.glob("*_keypoints.json"))
    first = files[0].name if files else "NONE"
    last = files[-1].name if files else "NONE"
    s = "|".join([FEATURE_VERSION, cache_version, str(p.resolve()), coord, str(len(files)), first, last])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class KSLKeypointDataset(Dataset):
    def __init__(
        self,
        samples: List[SampleInfo],
        indices: List[int],
        label_to_idx: Dict[str, int],
        target_len: int,
        coord: str,
        cache_dir: Path,
        cache_version: str,
        is_train: bool,
        args,
        rebuild_cache: bool = False,
    ):
        self.samples = samples
        self.indices = indices
        self.label_to_idx = label_to_idx
        self.target_len = target_len
        self.coord = coord
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        self.is_train = is_train
        self.args = args
        self.rebuild_cache = rebuild_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.indices)

    def _load_raw(self, s: SampleInfo) -> np.ndarray:
        key = raw_cache_key(s.sample_dir, self.coord, self.cache_version)
        npy_path = self.cache_dir / f"{key}.npy"
        meta_path = self.cache_dir / f"{key}.json"

        if npy_path.exists() and not self.rebuild_cache:
            try:
                arr = np.load(npy_path).astype(np.float32)
                if arr.ndim == 3 and arr.shape[1:] == (65, 3) and np.isfinite(arr).all():
                    return arr
                print(f"[WARN] invalid raw cache, rebuilding: {npy_path}")
            except Exception as e:
                print(f"[WARN] cannot load raw cache, rebuilding: {npy_path} ({e})")

        arr = load_sequence_65x3(Path(s.sample_dir), coord=self.coord)
        np.save(npy_path, arr)
        try:
            meta = {
                "feature_version": FEATURE_VERSION,
                "cache_version": self.cache_version,
                "sample_dir": s.sample_dir,
                "label": s.label,
                "group": s.group,
                "view": s.view,
                "coord": self.coord,
                "raw_shape": list(arr.shape),
                "raw_mean": float(arr.mean()),
                "raw_std": float(arr.std()),
                "zero_ratio": float(np.mean(np.abs(arr) < 1e-8)),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return arr

    def __getitem__(self, idx: int):
        s = self.samples[self.indices[idx]]
        raw = self._load_raw(s)

        if self.is_train:
            worker = torch.utils.data.get_worker_info()
            base_seed = torch.initial_seed() % (2**32)
            rng = np.random.default_rng(base_seed + idx + (worker.id if worker else 0) * 1000003)
            raw = augment_sequence_realtime(raw, self.args, rng)

        feat = make_features(raw, target_len=self.target_len)
        y = self.label_to_idx[s.label]
        return torch.from_numpy(feat), torch.tensor(y, dtype=torch.long)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        y = self.conv1(x)
        y = self.bn1(y)
        y = F.silu(y, inplace=True)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.dropout(y)
        return F.silu(y + residual, inplace=True)


class KSLTinyTCN(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, channels: int = 256, blocks: int = 5, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.LayerNorm(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        dilations = [1, 2, 4, 8, 16, 32, 64][:blocks]
        self.tcn = nn.Sequential(*[TemporalBlock(channels, kernel_size=5, dilation=d, dropout=dropout) for d in dilations])
        self.head = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        y = self.tcn(x.transpose(1, 2))
        avg_pool = y.mean(dim=2)
        max_pool = y.max(dim=2).values
        return self.head(torch.cat([avg_pool, max_pool], dim=1))


class KSLTCNAttn(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, channels: int = 384, blocks: int = 6, dropout: float = 0.25):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.LayerNorm(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )
        dilations = [1, 2, 4, 8, 16, 32, 64][:blocks]
        self.tcn = nn.Sequential(*[TemporalBlock(channels, kernel_size=5, dilation=d, dropout=dropout) for d in dilations])
        self.attn = nn.Sequential(
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
        )
        self.head = nn.Sequential(
            nn.Linear(channels * 3, channels),
            nn.LayerNorm(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)       # B,T,C
        y = self.tcn(x.transpose(1, 2))  # B,C,T
        avg_pool = y.mean(dim=2)
        max_pool = y.max(dim=2).values
        attn_logits = self.attn(y).squeeze(1)  # B,T
        attn_w = torch.softmax(attn_logits, dim=1).unsqueeze(1)
        attn_pool = (y * attn_w).sum(dim=2)
        return self.head(torch.cat([avg_pool, max_pool, attn_pool], dim=1))


def build_model(model_type: str, input_dim: int, num_classes: int, channels: int, blocks: int, dropout: float) -> nn.Module:
    if model_type == "tiny_tcn":
        return KSLTinyTCN(input_dim=input_dim, num_classes=num_classes, channels=channels, blocks=blocks, dropout=dropout)
    if model_type == "tcn_attn":
        return KSLTCNAttn(input_dim=input_dim, num_classes=num_classes, channels=channels, blocks=blocks, dropout=dropout)
    raise ValueError(f"Unknown model_type: {model_type}")


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def make_smoothed_onehot(y: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    with torch.no_grad():
        off = smoothing / max(1, num_classes - 1)
        out = torch.full((y.size(0), num_classes), off, device=y.device, dtype=torch.float32)
        out.scatter_(1, y[:, None], 1.0 - smoothing)
    return out


def compute_loss(logits: torch.Tensor, y: torch.Tensor, num_classes: int, label_smoothing: float) -> torch.Tensor:
    if label_smoothing > 0:
        target_probs = make_smoothed_onehot(y, num_classes, label_smoothing)
        return soft_cross_entropy(logits, target_probs)
    return F.cross_entropy(logits, y)


def apply_mixup(feat: torch.Tensor, y: torch.Tensor, num_classes: int, alpha: float, label_smoothing: float):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    perm = torch.randperm(feat.size(0), device=feat.device)
    mixed_feat = lam * feat + (1.0 - lam) * feat[perm]
    y1 = make_smoothed_onehot(y, num_classes, label_smoothing)
    y2 = make_smoothed_onehot(y[perm], num_classes, label_smoothing)
    mixed_y = lam * y1 + (1.0 - lam) * y2
    return mixed_feat, mixed_y


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        if self.backup is None:
            return
        sd = model.state_dict()
        for k, v in self.backup.items():
            sd[k].copy_(v)
        self.backup = None


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss, total, correct1, correct5 = 0.0, 0, 0, 0
    for feat, y in loader:
        feat = feat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(feat)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        pred1 = logits.argmax(dim=1)
        correct1 += int((pred1 == y).sum().item())
        k = min(5, logits.shape[1])
        top5 = logits.topk(k, dim=1).indices
        correct5 += int((top5 == y[:, None]).any(dim=1).sum().item())
    return {"loss": total_loss / max(1, total), "top1": correct1 / max(1, total), "top5": correct5 / max(1, total), "total": total}


def train_one_epoch(model, loader, optimizer, device, args, scaler=None, ema: Optional[ModelEMA] = None):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    num_classes = args.num_classes_runtime

    for step, (feat, y) in enumerate(loader, start=1):
        feat = feat.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        use_mixup = args.mixup_alpha > 0 and args.mixup_prob > 0 and random.random() < args.mixup_prob
        if use_mixup:
            feat, soft_y = apply_mixup(feat, y, num_classes, args.mixup_alpha, args.label_smoothing)
        else:
            soft_y = None

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits = model(feat)
            if soft_y is not None:
                loss = soft_cross_entropy(logits, soft_y)
            else:
                loss = compute_loss(logits, y, num_classes, args.label_smoothing)
            loss_to_backward = loss / max(1, args.grad_accum_steps)

        if scaler is not None:
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        if step % args.grad_accum_steps == 0 or step == len(loader):
            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)

        total_loss += float(loss.item()) * y.size(0)
        total += y.size(0)
        correct += int((logits.argmax(dim=1) == y).sum().item())

    return {"loss": total_loss / max(1, total), "top1": correct / max(1, total), "total": total}


@torch.no_grad()
def sanity_check_batch(model, loader, device, labels: List[str]):
    try:
        feat, y = next(iter(loader))
    except StopIteration:
        print("[WARN] empty train loader; cannot run sanity check")
        return
    feat = feat.to(device)
    y = y.to(device)
    logits = model(feat)
    loss = F.cross_entropy(logits, y)
    feat_np = feat.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    unique_y = sorted(set(int(v) for v in y_np.tolist()))[:10]
    preview = [labels[i] for i in unique_y if i < len(labels)]
    print(f"[SANITY] feat_shape={tuple(feat.shape)}, mean={feat_np.mean():.5f}, std={feat_np.std():.5f}, absmax={np.abs(feat_np).max():.5f}, loss={float(loss.item()):.4f}")
    print(f"[SANITY] labels_preview={preview}")


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float, meta: dict, ema: Optional[ModelEMA] = None):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "meta": meta,
    }
    if ema is not None:
        ckpt["ema_state"] = ema.shadow
    torch.save(ckpt, path)


def write_csv_log(path: Path, row: dict):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def make_train_sampler(samples: List[SampleInfo], train_idx: List[int]) -> WeightedRandomSampler:
    label_counts = Counter(samples[i].label for i in train_idx)
    weights = [1.0 / max(1, label_counts[samples[i].label]) for i in train_idx]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="outputs/ksl_keypoint_tcn_v5_desktop")
    ap.add_argument("--coord", type=str, default="3d", choices=["3d", "2d"])
    ap.add_argument("--target_len", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--channels", type=int, default=384)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--model_type", type=str, default="tcn_attn", choices=["tiny_tcn", "tcn_attn"])
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_frames", type=int, default=8)
    ap.add_argument("--cache_dir", type=str, default="")
    ap.add_argument("--cache_version", type=str, default=FEATURE_VERSION)
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--compile", action="store_true", help="Optional torch.compile. On Windows, use only if it works in your environment.")
    ap.add_argument("--limit_samples", type=int, default=0)
    ap.add_argument("--limit_classes", type=int, default=0)
    ap.add_argument("--min_samples_per_class", type=int, default=2)
    ap.add_argument("--view_filter", type=str, default="D,U,L,R,F")
    ap.add_argument("--split_strategy", type=str, default="label_view_holdout", choices=["label_view_holdout", "label_stratified_random", "group_holdout"])
    ap.add_argument("--val_views", type=str, default="F")
    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--sanity_check", action="store_true")

    # realtime-style augmentation
    ap.add_argument("--temporal_crop_prob", type=float, default=0.80)
    ap.add_argument("--temporal_crop_min", type=float, default=0.72)
    ap.add_argument("--speed_jitter", type=float, default=0.25)
    ap.add_argument("--idle_pad_prob", type=float, default=0.65)
    ap.add_argument("--idle_pad_max_ratio", type=float, default=0.35)
    ap.add_argument("--coord_noise_std", type=float, default=0.010)
    ap.add_argument("--z_scale_jitter", type=float, default=0.15)
    ap.add_argument("--rotate_jitter_deg", type=float, default=5.0)
    ap.add_argument("--hand_dropout_prob", type=float, default=0.35)
    ap.add_argument("--hand_dropout_max_ratio", type=float, default=0.18)
    ap.add_argument("--joint_dropout_prob", type=float, default=0.25)
    ap.add_argument("--joint_dropout_max", type=int, default=4)
    ap.add_argument("--time_mask_prob", type=float, default=0.25)
    ap.add_argument("--time_mask_max_ratio", type=float, default=0.12)

    # optimization
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--mixup_alpha", type=float, default=0.15)
    ap.add_argument("--mixup_prob", type=float, default=0.35)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--eval_ema", action="store_true", default=True)
    ap.add_argument("--final_fit_epochs", type=int, default=20)
    ap.add_argument("--final_fit_lr", type=float, default=2e-4)
    ap.add_argument("--final_fit_disable_mixup", action="store_true", default=True)
    args = ap.parse_args()

    seed_everything(args.seed)

    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_cache_version = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.cache_version)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / f"raw_cache_{safe_cache_version}_{args.coord}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Discovering samples...")
    samples = discover_samples(data_root, min_frames=args.min_frames, max_samples=args.limit_samples, view_filter=args.view_filter)
    if not samples:
        raise RuntimeError(f"No sample directories found under {data_root}")

    samples = filter_min_samples_per_class(samples, args.min_samples_per_class)
    samples = limit_classes(samples, args.limit_classes, seed=args.seed)
    if not samples:
        raise RuntimeError("No samples left after filtering")

    labels = sorted({s.label for s in samples})
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}
    args.num_classes_runtime = len(labels)

    train_idx, val_idx = make_split(args, samples)
    validate_split(samples, train_idx, val_idx)
    print_dataset_stats(samples, train_idx, val_idx)

    print(f"[INFO] split_strategy: {args.split_strategy}")
    if args.split_strategy == "label_view_holdout":
        print(f"[INFO] val_views: {args.val_views}")
    print(f"[INFO] cache_dir: {cache_dir}")
    print(f"[INFO] feature_version: {FEATURE_VERSION}")
    print(f"[INFO] cache_version: {args.cache_version}")
    if args.rebuild_cache:
        print("[INFO] rebuild_cache=True: existing raw cache files will be ignored and overwritten")

    meta = {
        "feature_version": FEATURE_VERSION,
        "cache_version": args.cache_version,
        "coord": args.coord,
        "view_filter": args.view_filter,
        "target_len": args.target_len,
        "input_dim": 390,
        "num_classes": len(labels),
        "labels": labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "split_strategy": args.split_strategy,
        "val_views": args.val_views,
        "landmark_order": "left_hand_21 + right_hand_21 + common_pose_23",
        "common_pose23_names": COMMON_POSE23_NAMES,
        "normalization": "center=neck/shoulder fallback, scale=shoulder distance/bbox fallback, missing landmarks stay zero",
        "feature": "normalized xyz flatten 195 + temporal delta 195 = 390",
        "model": {
            "name": "KSLTCNAttn" if args.model_type == "tcn_attn" else "KSLTinyTCN",
            "model_type": args.model_type,
            "channels": args.channels,
            "blocks": args.blocks,
            "dropout": args.dropout,
        },
        "train_args": vars(args),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(out_dir / "samples_manifest.json", "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in samples], f, ensure_ascii=False, indent=2)
    with open(out_dir / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"train": [asdict(samples[i]) for i in train_idx], "val": [asdict(samples[i]) for i in val_idx]}, f, ensure_ascii=False, indent=2)

    train_ds = KSLKeypointDataset(samples, train_idx, label_to_idx, args.target_len, args.coord, cache_dir, args.cache_version, True, args, args.rebuild_cache)
    val_ds = KSLKeypointDataset(samples, val_idx, label_to_idx, args.target_len, args.coord, cache_dir, args.cache_version, False, args, args.rebuild_cache)

    train_sampler = make_train_sampler(samples, train_idx) if args.balanced_sampler else None
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    # DataLoader does not accept prefetch_factor when num_workers=0.
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] total_vram_gb: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")

    model = build_model(args.model_type, 390, len(labels), args.channels, args.blocks, args.dropout).to(device)
    if args.compile:
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile failed; continuing without compile: {e}")

    if args.sanity_check:
        sanity_check_batch(model, train_loader, device, labels)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    best_top1 = -1.0
    best_path = out_dir / "ksl_keypoint_tcn_best.pt"
    last_path = out_dir / "ksl_keypoint_tcn_last.pt"
    log_path = out_dir / "history.csv"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args, scaler=scaler, ema=ema)

        if ema is not None and args.eval_ema:
            ema.apply_to(model)
            val_metrics = evaluate(model, val_loader, device) if len(val_ds) > 0 else {"loss": math.nan, "top1": 0.0, "top5": 0.0, "total": 0}
            ema.restore(model)
        else:
            val_metrics = evaluate(model, val_loader, device) if len(val_ds) > 0 else {"loss": math.nan, "top1": 0.0, "top5": 0.0, "total": 0}

        scheduler.step()
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "train_total": train_metrics["total"],
            "val_total": val_metrics["total"],
            "elapsed_sec": elapsed,
        }
        write_csv_log(log_path, row)
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] lr={lr_now:.6g} "
            f"train_loss={train_metrics['loss']:.4f} train_top1={train_metrics['top1']*100:.2f}% "
            f"val_loss={val_metrics['loss']:.4f} val_top1={val_metrics['top1']*100:.2f}% val_top5={val_metrics['top5']*100:.2f}% "
            f"time={elapsed:.1f}s"
        )

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_top1, meta, ema=ema)
        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            if ema is not None and args.eval_ema:
                ema.apply_to(model)
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_top1, meta, ema=ema)
                ema.restore(model)
            else:
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_top1, meta, ema=ema)
            print(f"[BEST] saved {best_path} val_top1={best_top1*100:.2f}%")

    print("[INFO] Main training done.")

    final_path = out_dir / "ksl_keypoint_tcn_final.pt"
    if args.final_fit_epochs > 0:
        print(f"[INFO] Starting final fit on all views for {args.final_fit_epochs} epochs")
        # Load best model before final fitting.
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        if ema is not None:
            ema = ModelEMA(model, decay=args.ema_decay)

        all_idx = list(range(len(samples)))
        final_ds = KSLKeypointDataset(samples, all_idx, label_to_idx, args.target_len, args.coord, cache_dir, args.cache_version, True, args, False)
        final_loader = DataLoader(final_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **loader_kwargs)

        old_mixup_prob = args.mixup_prob
        if args.final_fit_disable_mixup:
            args.mixup_prob = 0.0
        final_optimizer = torch.optim.AdamW(model.parameters(), lr=args.final_fit_lr, weight_decay=args.weight_decay)
        final_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=max(1, args.final_fit_epochs), eta_min=args.min_lr)

        for ep in range(1, args.final_fit_epochs + 1):
            t0 = time.time()
            m = train_one_epoch(model, final_loader, final_optimizer, device, args, scaler=scaler, ema=ema)
            final_scheduler.step()
            print(f"[FinalFit {ep:03d}/{args.final_fit_epochs}] lr={final_optimizer.param_groups[0]['lr']:.6g} loss={m['loss']:.4f} top1={m['top1']*100:.2f}% time={time.time()-t0:.1f}s")
        args.mixup_prob = old_mixup_prob

        if ema is not None:
            ema.apply_to(model)
            save_checkpoint(final_path, model, None, None, args.epochs + args.final_fit_epochs, best_top1, meta, ema=ema)
            ema.restore(model)
        else:
            save_checkpoint(final_path, model, None, None, args.epochs + args.final_fit_epochs, best_top1, meta, ema=None)
        print(f"[FINAL] saved {final_path}")
    else:
        shutil.copy2(best_path, final_path)
        print(f"[FINAL] copied best to {final_path}")

    print("[DONE]")
    print(f"Best model:  {best_path}")
    print(f"Final model: {final_path}")
    print(f"Meta:        {out_dir / 'meta.json'}")
    print(f"History:     {log_path}")


if __name__ == "__main__":
    main()
