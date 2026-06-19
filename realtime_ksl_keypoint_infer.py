#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KSL realtime webcam inference - v5
==================================

Compatible with ksl_keypoint_tcn_train_v5_desktop.py.
Supports both tiny_tcn and tcn_attn checkpoints through checkpoint meta.

This script continuously runs a sliding-window classifier on webcam keypoints.
It can average predictions from multiple recent window sizes such as 48,72,96
frames, which is useful because users perform signs at different speeds.
"""

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = ImageDraw = ImageFont = None

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "mediapipe is required. Example: pip install mediapipe\n"
        "If you see `module mediapipe has no attribute solutions`, use Python 3.10 and mediapipe 0.10.x."
    ) from e


# -----------------------------------------------------------------------------
# Model definitions. Must match training script.
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
        x = self.input_proj(x)
        y = self.tcn(x.transpose(1, 2))
        avg_pool = y.mean(dim=2)
        max_pool = y.max(dim=2).values
        attn_logits = self.attn(y).squeeze(1)
        attn_w = torch.softmax(attn_logits, dim=1).unsqueeze(1)
        attn_pool = (y * attn_w).sum(dim=2)
        return self.head(torch.cat([avg_pool, max_pool, attn_pool], dim=1))


# -----------------------------------------------------------------------------
# Feature functions. Must match training script.
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


def make_features(seq65: np.ndarray, target_len: int) -> np.ndarray:
    norm = normalize_sequence(seq65)
    flat = norm.reshape(norm.shape[0], -1).astype(np.float32)
    delta = np.zeros_like(flat)
    if flat.shape[0] > 1:
        delta[1:] = flat[1:] - flat[:-1]
    feat = np.concatenate([flat, delta], axis=1)
    feat = temporal_resample_flat(feat, target_len=target_len)
    return np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# -----------------------------------------------------------------------------
# MediaPipe extraction
# -----------------------------------------------------------------------------

def lm_to_np(lm) -> np.ndarray:
    if lm is None:
        return np.zeros(3, dtype=np.float32)
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def extract_hand_21(hand_landmarks) -> np.ndarray:
    out = np.zeros((21, 3), dtype=np.float32)
    if hand_landmarks is None:
        return out
    for i, lm in enumerate(hand_landmarks.landmark[:21]):
        out[i] = lm_to_np(lm)
    return out


def extract_pose23_from_mediapipe(pose_landmarks) -> np.ndarray:
    out = np.zeros((23, 3), dtype=np.float32)
    if pose_landmarks is None:
        return out
    lms = pose_landmarks.landmark

    def get(i: int) -> np.ndarray:
        if i < 0 or i >= len(lms):
            return np.zeros(3, dtype=np.float32)
        return lm_to_np(lms[i])

    nose = get(0)
    rsho, relb, rwri = get(12), get(14), get(16)
    lsho, lelb, lwri = get(11), get(13), get(15)
    rhip, rknee, rankle = get(24), get(26), get(28)
    lhip, lknee, lankle = get(23), get(25), get(27)
    reye, leye, rear, lear = get(5), get(2), get(8), get(7)
    lheel, rheel, lfoot, rfoot = get(29), get(30), get(31), get(32)

    vals = [
        nose,
        rsho, relb, rwri,
        lsho, lelb, lwri,
        rhip, rknee, rankle,
        lhip, lknee, lankle,
        reye, leye, rear, lear,
        lheel, rheel,
        lfoot, rfoot,
        np.zeros(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    ]
    for i, v in enumerate(vals):
        out[i] = v

    if not (np.all(np.abs(rsho) < 1e-8) or np.all(np.abs(lsho) < 1e-8)):
        out[21] = (rsho + lsho) / 2.0
    if not (np.all(np.abs(rhip) < 1e-8) or np.all(np.abs(lhip) < 1e-8)):
        out[22] = (rhip + lhip) / 2.0
    return out.astype(np.float32)


def extract_keypoints_65(results) -> np.ndarray:
    lh = extract_hand_21(results.left_hand_landmarks)
    rh = extract_hand_21(results.right_hand_landmarks)
    pose23 = extract_pose23_from_mediapipe(results.pose_landmarks)
    return np.concatenate([lh, rh, pose23], axis=0).astype(np.float32)


def fill_missing_hands_with_previous(kp: np.ndarray, state: dict, max_hold_frames: int = 6) -> np.ndarray:
    out = kp.copy()
    lh, rh = out[:21], out[21:42]
    lh_missing = np.all(np.abs(lh) < 1e-8)
    rh_missing = np.all(np.abs(rh) < 1e-8)

    if lh_missing:
        if state.get("last_lh") is not None and state.get("lh_hold", 0) < max_hold_frames:
            out[:21] = state["last_lh"]
            state["lh_hold"] = state.get("lh_hold", 0) + 1
    else:
        state["last_lh"] = lh.copy()
        state["lh_hold"] = 0

    if rh_missing:
        if state.get("last_rh") is not None and state.get("rh_hold", 0) < max_hold_frames:
            out[21:42] = state["last_rh"]
            state["rh_hold"] = state.get("rh_hold", 0) + 1
    else:
        state["last_rh"] = rh.copy()
        state["rh_hold"] = 0

    return out.astype(np.float32)


def hand_missing_ratios(seq65: np.ndarray) -> Tuple[float, float]:
    lh = seq65[:, :21, :]
    rh = seq65[:, 21:42, :]
    return float(np.all(np.abs(lh) < 1e-8, axis=(1, 2)).mean()), float(np.all(np.abs(rh) < 1e-8, axis=(1, 2)).mean())


def motion_energy(seq65: np.ndarray) -> float:
    if seq65.shape[0] < 2:
        return 0.0
    hands = seq65[:, :42, :]
    vm = valid_mask(hands)
    diff = np.diff(hands, axis=0)
    valid = vm[1:] & vm[:-1]
    if not valid.any():
        return 0.0
    mag = np.linalg.norm(diff, axis=-1)
    return float(mag[valid].mean())


# -----------------------------------------------------------------------------
# Model loading and prediction
# -----------------------------------------------------------------------------

def build_model_from_meta(meta: dict, num_classes: int):
    input_dim = int(meta.get("input_dim", 390))
    model_cfg = meta.get("model", {})
    model_type = model_cfg.get("model_type", "tiny_tcn")
    channels = int(model_cfg.get("channels", 256))
    blocks = int(model_cfg.get("blocks", 5))
    dropout = float(model_cfg.get("dropout", 0.25))

    if model_type == "tcn_attn" or model_cfg.get("name") == "KSLTCNAttn":
        return KSLTCNAttn(input_dim=input_dim, num_classes=num_classes, channels=channels, blocks=blocks, dropout=dropout)
    return KSLTinyTCN(input_dim=input_dim, num_classes=num_classes, channels=channels, blocks=blocks, dropout=dropout)


def load_model(model_path: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)
    meta = ckpt.get("meta", {})
    labels = meta.get("labels")
    if labels is None:
        idx_to_label = meta.get("idx_to_label", {})
        labels = [idx_to_label[str(i)] if str(i) in idx_to_label else idx_to_label[i] for i in range(len(idx_to_label))]

    model = build_model_from_meta(meta, num_classes=len(labels)).to(device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    target_len = int(meta.get("target_len", 96))
    return model, labels, meta, target_len


@torch.no_grad()
def predict_probs_for_windows(model, seq_buffer: deque, target_len: int, window_sizes: List[int], device: torch.device) -> torch.Tensor:
    seq_all = np.stack(list(seq_buffer), axis=0).astype(np.float32)
    feats = []
    valid_windows = []
    for w in window_sizes:
        if seq_all.shape[0] >= w:
            seq = seq_all[-w:]
            feat = make_features(seq, target_len=target_len)
            feats.append(feat)
            valid_windows.append(w)
    if not feats:
        seq = seq_all
        feats.append(make_features(seq, target_len=target_len))
    x = torch.from_numpy(np.stack(feats, axis=0)).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return probs.mean(dim=0)


def parse_window_sizes(s: str, max_window: int) -> List[int]:
    if not s:
        return [max_window]
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v > 0:
            out.append(v)
    out = sorted(set(out))
    if max_window not in out:
        out.append(max_window)
    return sorted(out)


def should_commit(pred_history: deque, conf_threshold: float, majority_ratio: float, min_history: int):
    if len(pred_history) < min_history:
        return None
    recent = list(pred_history)[-min_history:]
    idxs = [x[0] for x in recent]
    probs = [x[1] for x in recent]
    counts = Counter(idxs)
    best_idx, count = counts.most_common(1)[0]
    ratio = count / len(recent)
    avg_prob = float(np.mean([p for i, p in recent if i == best_idx]))
    if ratio >= majority_ratio and avg_prob >= conf_threshold:
        return best_idx, avg_prob, ratio
    return None


def stable_candidates_from_topk_history(
    topk_history: deque,
    stable_topk: int = 5,
    rank_decay: float = 0.65,
) -> List[Tuple[int, float, float, float]]:
    """Accumulate evidence from recent Top-K predictions.

    Returns:
        [(class_idx, score, count_ratio, last_seen_prob), ...]

    Why this is better than only storing Top-1:
      - If the correct word repeatedly appears as rank 2~5, it still gets score.
      - Recent predictions get slightly larger age weights.
      - Lower ranks get lower score via rank_decay.
    """
    if not topk_history:
        return []

    hist = list(topk_history)
    n = len(hist)
    scores: Dict[int, float] = {}
    counts: Counter = Counter()
    last_seen_prob: Dict[int, float] = {}
    total_age_weight = 0.0

    stable_topk = max(1, int(stable_topk))
    rank_decay = float(rank_decay)

    for age, preds in enumerate(hist):
        # Older predictions still count, but recent predictions count more.
        age_weight = 0.60 + 0.40 * ((age + 1) / max(1, n))
        total_age_weight += age_weight

        for rank, (idx, prob) in enumerate(preds[:stable_topk]):
            rank_weight = rank_decay ** rank
            contribution = float(prob) * rank_weight * age_weight
            scores[idx] = scores.get(idx, 0.0) + contribution
            counts[idx] += 1
            last_seen_prob[idx] = float(prob)

    denom = max(1e-8, total_age_weight)
    out = []
    for idx, score in scores.items():
        out.append((idx, score / denom, counts[idx] / max(1, n), last_seen_prob.get(idx, 0.0)))

    out.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    return out


def choose_stable_commit(
    candidates: List[Tuple[int, float, float, float]],
    min_score: float,
    min_count_ratio: float,
    min_margin: float,
    min_last_prob: float = 0.0,
):
    """Choose a commit candidate from accumulated Top-K evidence.

    candidates item format: (class_idx, stable_score, count_ratio, last_seen_prob).

    The first three thresholds check accumulated evidence across recent windows.
    min_last_prob additionally prevents committing a candidate whose latest
    smoothed probability is still too weak, even if it was stable for a while.
    """
    if not candidates:
        return None
    best = candidates[0]
    second_score = candidates[1][1] if len(candidates) > 1 else 0.0
    margin = best[1] - second_score
    last_prob = best[3]
    if (
        best[1] >= min_score
        and best[2] >= min_count_ratio
        and margin >= min_margin
        and last_prob >= min_last_prob
    ):
        return best[0], best[1], best[2], margin, last_prob
    return None


def load_label_map(path: str) -> Dict[str, str]:
    """Load WORDxxxx -> Korean/gloss mapping from a JSON file.

    Supported formats:
      1) {"WORD1501": "고민", ...}
      2) {"WORD1501": {"name": "고민"}, ...}
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] label map not found: {p}")
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        key = str(k).strip()
        if isinstance(v, dict):
            name = v.get("name") or v.get("gloss") or v.get("display") or v.get("word") or v.get("label")
        else:
            name = v
        if name is not None and str(name).strip():
            out[key] = str(name).strip()
    return out


def format_label(label: str, label_map: Dict[str, str], show_word_id: bool = True) -> str:
    word = label_map.get(label)
    if not word:
        return label
    return f"{label} | {word}" if show_word_id else word


_FONT_CACHE = {}

def _has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in str(text))


def _get_korean_font(size: int):
    if ImageFont is None:
        return None
    key = int(size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/malgunbd.ttf",
        "C:/Windows/Fonts/gulim.ttc",
        "C:/Windows/Fonts/batang.ttc",
    ]
    font = None
    for fp in candidates:
        try:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, key)
                break
        except Exception:
            pass
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    _FONT_CACHE[key] = font
    return font


def put_text(img, text, org, scale=0.8, color=(0, 255, 0), thickness=2):
    """Draw text on a BGR OpenCV image. Uses PIL for Korean text.

    cv2.putText cannot render Korean properly, so non-ASCII text is drawn with
    the Windows Malgun Gothic font when available.
    """
    text = str(text)
    if not _has_non_ascii(text) or Image is None or ImageDraw is None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return

    font_size = max(12, int(scale * 30))
    font = _get_korean_font(font_size)
    if font is None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return

    # OpenCV uses BGR, PIL uses RGB.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    x, y = org
    # cv2 org is baseline-ish; PIL org is top-left. Adjust a little upward.
    y = max(0, y - font_size)
    rgb_color = (int(color[2]), int(color[1]), int(color[0]))

    # Simple thickness effect by drawing around the target point.
    if thickness > 1:
        for dx in range(-thickness + 1, thickness):
            for dy in range(-thickness + 1, thickness):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=rgb_color)
    img[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--label_map_path", type=str, default="", help="Optional JSON mapping WORDxxxx -> Korean/gloss")
    ap.add_argument("--hide_word_id", action="store_true", help="Show only mapped Korean/gloss text, not WORD id")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--window_frames", type=int, default=96)
    ap.add_argument("--multi_windows", type=str, default="48,72,96")
    ap.add_argument("--min_window_frames", type=int, default=32)
    ap.add_argument("--predict_every", type=int, default=3)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--conf_threshold", type=float, default=0.60)
    ap.add_argument("--majority_ratio", type=float, default=0.60)
    ap.add_argument("--smooth_history", type=int, default=8)
    ap.add_argument("--min_commit_history", type=int, default=5)
    ap.add_argument("--cooldown_sec", type=float, default=0.9)
    ap.add_argument("--repeat_cooldown_sec", type=float, default=1.8)
    ap.add_argument("--max_missing_ratio", type=float, default=0.55)
    ap.add_argument("--hold_missing_hands", type=int, default=6)
    ap.add_argument("--no_reset_after_commit", action="store_true",
                    help="Do not clear sequence/probability buffers after a word is committed. Default clears them to prevent stale-window commits.")
    ap.add_argument("--reset_hand_state_after_commit", action="store_true",
                    help="Also clear cached previous hand landmarks after commit. Useful when stale hand-hold causes repeated predictions.")
    ap.add_argument("--min_motion_energy", type=float, default=0.0, help="0 disables idle filtering")
    ap.add_argument("--prob_ema_alpha", type=float, default=0.55, help="EMA smoothing for full probability vector. 0 disables.")
    ap.add_argument("--stable_history", type=int, default=24, help="Number of recent prediction steps used for Top-K evidence accumulation.")
    ap.add_argument("--stable_topk", type=int, default=5, help="How many ranks from each prediction are accumulated.")
    ap.add_argument("--rank_decay", type=float, default=0.65, help="Weight decay for lower ranks in Top-K accumulation.")
    ap.add_argument("--stable_min_score", type=float, default=0.025, help="Minimum accumulated score required for commit.")
    ap.add_argument("--stable_min_count_ratio", type=float, default=0.30, help="Minimum ratio of recent predictions where candidate appeared in Top-K.")
    ap.add_argument("--stable_margin", type=float, default=0.005, help="Minimum score margin over second candidate for commit.")
    ap.add_argument("--commit_min_last_prob", type=float, default=0.0,
                    help="Minimum latest smoothed probability of the stable candidate required for commit. Useful to block low-confidence Top-1 commits.")
    ap.add_argument("--commit_require_current_top1", action="store_true",
                    help="Only commit when the stable candidate is also the current smoothed Top-1.")
    ap.add_argument("--commit_min_current_top1_prob", type=float, default=0.0,
                    help="Minimum probability of the current smoothed Top-1 required for commit.")
    ap.add_argument("--show_raw_topk", action="store_true", help="Also show raw/EMA Top-K below stable candidates.")
    ap.add_argument("--no_stable_commit", action="store_true", help="Use old Top-1 majority commit instead of stable Top-K evidence commit.")
    ap.add_argument("--no_mirror_display", action="store_true")
    ap.add_argument("--draw_landmarks", action="store_true")
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, labels, meta, target_len = load_model(Path(args.model_path), device)
    label_map = load_label_map(args.label_map_path)
    show_word_id = not args.hide_word_id
    window_sizes = parse_window_sizes(args.multi_windows, args.window_frames)

    print("[INFO] model loaded:", args.model_path)
    print("[INFO] device:", device)
    print("[INFO] classes:", len(labels))
    print("[INFO] label_map entries:", len(label_map))
    print("[INFO] target_len:", target_len)
    print("[INFO] model:", meta.get("model", {}))
    print("[INFO] multi_windows:", window_sizes)
    print("[INFO] stable_topk:", {
        "prob_ema_alpha": args.prob_ema_alpha,
        "stable_history": args.stable_history,
        "stable_topk": args.stable_topk,
        "rank_decay": args.rank_decay,
        "stable_min_score": args.stable_min_score,
        "stable_min_count_ratio": args.stable_min_count_ratio,
        "stable_margin": args.stable_margin,
        "commit_min_last_prob": args.commit_min_last_prob,
        "commit_require_current_top1": args.commit_require_current_top1,
        "commit_min_current_top1_prob": args.commit_min_current_top1_prob,
        "stable_commit": not args.no_stable_commit,
    })

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera: {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    seq_buffer = deque(maxlen=args.window_frames)
    pred_history = deque(maxlen=args.smooth_history)  # old Top-1 history, kept as fallback
    topk_history = deque(maxlen=args.stable_history)  # new Top-K evidence history
    committed_words = []
    last_commit_label = None
    last_commit_time = 0.0
    frame_count = 0
    last_topk = []          # raw/EMA Top-K
    last_stable_topk = []   # accumulated stable Top-K: (idx, score, count_ratio, last_prob)
    ema_probs = None
    last_status = "warming up"
    hand_state = {"last_lh": None, "last_rh": None, "lh_hold": 0, "rh_hold": 0}

    prev_time = time.time()
    fps = 0.0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] camera frame read failed")
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True

            kp = extract_keypoints_65(results)
            kp = fill_missing_hands_with_previous(kp, hand_state, args.hold_missing_hands)
            seq_buffer.append(kp)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)

            display = frame.copy()
            if not args.no_mirror_display:
                display = cv2.flip(display, 1)

            if args.draw_landmarks:
                mp_drawing.draw_landmarks(display, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(display, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(display, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            if len(seq_buffer) >= args.min_window_frames and frame_count % args.predict_every == 0:
                seq_np = np.stack(list(seq_buffer), axis=0).astype(np.float32)
                lh_miss, rh_miss = hand_missing_ratios(seq_np)
                energy = motion_energy(seq_np[-min(len(seq_np), args.window_frames):])

                if lh_miss > args.max_missing_ratio or rh_miss > args.max_missing_ratio:
                    last_status = f"hand unstable LH {lh_miss:.2f} RH {rh_miss:.2f}"
                elif args.min_motion_energy > 0 and energy < args.min_motion_energy:
                    last_status = f"idle/motion low {energy:.5f}"
                else:
                    try:
                        probs = predict_probs_for_windows(model, seq_buffer, target_len, window_sizes, device)

                        # 1) Smooth the full probability vector, not just Top-1.
                        if args.prob_ema_alpha > 0:
                            a = float(args.prob_ema_alpha)
                            if ema_probs is None:
                                ema_probs = probs
                            else:
                                ema_probs = a * probs + (1.0 - a) * ema_probs
                            probs_for_rank = ema_probs
                        else:
                            probs_for_rank = probs

                        # 2) Get Top-K from the smoothed probability vector.
                        k_for_history = min(max(args.topk, args.stable_topk), probs_for_rank.numel())
                        vals, inds = torch.topk(probs_for_rank, k=k_for_history)
                        last_topk = [(int(i.item()), float(v.item())) for i, v in zip(inds, vals)]

                        best_idx, best_prob = last_topk[0]
                        pred_history.append((best_idx, best_prob))
                        topk_history.append(last_topk)

                        # 3) Accumulate evidence from recent Top-K lists.
                        last_stable_topk = stable_candidates_from_topk_history(
                            topk_history,
                            stable_topk=args.stable_topk,
                            rank_decay=args.rank_decay,
                        )

                        if last_stable_topk:
                            s_idx, s_score, s_ratio, s_last_prob = last_stable_topk[0]
                            s_label = labels[s_idx] if 0 <= s_idx < len(labels) else str(s_idx)
                            s_label = format_label(s_label, label_map, show_word_id)
                            last_status = (
                                f"stable {s_label} score={s_score:.3f} "
                                f"seen={s_ratio:.2f} lastp={s_last_prob:.3f} energy={energy:.5f}"
                            )
                        else:
                            last_status = f"predicting energy={energy:.5f}"

                        # 4) Commit. Default is stable Top-K evidence commit.
                        if args.no_stable_commit:
                            old_commit = should_commit(
                                pred_history,
                                args.conf_threshold,
                                args.majority_ratio,
                                args.min_commit_history,
                            )
                            commit = None
                            if old_commit is not None:
                                c_idx, c_prob, c_ratio = old_commit
                                commit = (c_idx, c_prob, c_ratio, 0.0)
                        else:
                            commit = choose_stable_commit(
                                last_stable_topk,
                                min_score=args.stable_min_score,
                                min_count_ratio=args.stable_min_count_ratio,
                                min_margin=args.stable_margin,
                                min_last_prob=args.commit_min_last_prob,
                            )

                        if commit is not None and len(topk_history) >= args.min_commit_history:
                            # Stable commit returns 5 values. Old Top-1 fallback returns 4 values.
                            if len(commit) == 5:
                                c_idx, c_score, c_ratio, c_margin, c_last_prob = commit
                            else:
                                c_idx, c_score, c_ratio, c_margin = commit
                                c_last_prob = c_score

                            current_top1_idx = last_topk[0][0] if last_topk else None
                            current_top1_prob = last_topk[0][1] if last_topk else 0.0

                            commit_block_reason = None
                            if args.commit_require_current_top1 and c_idx != current_top1_idx:
                                commit_block_reason = "stable candidate is not current top1"
                            elif current_top1_prob < args.commit_min_current_top1_prob:
                                commit_block_reason = f"current top1 prob low {current_top1_prob:.3f}"

                            if commit_block_reason is not None:
                                raw_label = labels[c_idx] if 0 <= c_idx < len(labels) else str(c_idx)
                                display_label = format_label(raw_label, label_map, show_word_id)
                                last_status = (
                                    f"commit blocked {display_label}: {commit_block_reason} "
                                    f"score={c_score:.3f} seen={c_ratio:.2f} lastp={c_last_prob:.3f}"
                                )
                            else:
                                label = labels[c_idx]
                                display_label = format_label(label, label_map, show_word_id)
                                elapsed = now - last_commit_time
                                is_repeat = (last_commit_label == label)
                                required_cooldown = args.repeat_cooldown_sec if is_repeat else args.cooldown_sec
                                if elapsed >= required_cooldown:
                                    committed_words.append(display_label)
                                    last_commit_label = label
                                    last_commit_time = now

                                    # IMPORTANT:
                                    # After a commit, the sliding window still contains the just-finished sign.
                                    # If we keep seq_buffer / stable scores alive, the same high-score evidence
                                    # can remain during the following idle frames and cause another false commit.
                                    # So by default, clear all inference evidence but keep committed_words.
                                    pred_history.clear()
                                    topk_history.clear()
                                    ema_probs = None
                                    last_topk = []
                                    last_stable_topk = []
                                    if not args.no_reset_after_commit:
                                        seq_buffer.clear()
                                        if args.reset_hand_state_after_commit:
                                            hand_state = {"last_lh": None, "last_rh": None, "lh_hold": 0, "rh_hold": 0}

                                    last_status = (
                                        f"COMMIT {display_label} "
                                        f"score={c_score:.3f} seen={c_ratio:.2f} margin={c_margin:.3f} "
                                        f"lastp={c_last_prob:.3f} top1p={current_top1_prob:.3f} "
                                        f"| reset={'off' if args.no_reset_after_commit else 'on'}"
                                    )
                    except Exception as e:
                        last_status = f"predict error: {e}"

            put_text(display, f"FPS {fps:.1f}  buffer {len(seq_buffer)}/{args.window_frames}", (20, 30), 0.7, (255, 255, 255), 2)
            put_text(display, f"Status: {last_status}", (20, 62), 0.7, (0, 255, 255), 2)

            y0 = 100
            put_text(display, "Stable candidates:", (20, y0), 0.72, (0, 255, 255), 2)
            y0 += 32
            for rank, item in enumerate(last_stable_topk[:args.topk], start=1):
                idx, score, count_ratio, last_prob = item
                raw_label = labels[idx] if 0 <= idx < len(labels) else str(idx)
                label = format_label(raw_label, label_map, show_word_id)
                put_text(
                    display,
                    f"{rank}. {label}: score={score:.3f} seen={count_ratio:.2f} p={last_prob:.3f}",
                    (20, y0 + 34 * (rank - 1)),
                    0.68,
                    (0, 255, 0),
                    2,
                )

            if args.show_raw_topk:
                y_raw = y0 + 34 * (args.topk + 1)
                put_text(display, "Raw/EMA Top-K:", (20, y_raw), 0.68, (255, 255, 255), 2)
                y_raw += 30
                for rank, (idx, prob) in enumerate(last_topk[:args.topk], start=1):
                    raw_label = labels[idx] if 0 <= idx < len(labels) else str(idx)
                    label = format_label(raw_label, label_map, show_word_id)
                    put_text(display, f"{rank}. {label}: {prob:.3f}", (20, y_raw + 28 * (rank - 1)), 0.58, (255, 255, 255), 1)

            sent = " ".join(committed_words[-8:])
            put_text(display, f"Committed: {sent}", (20, display.shape[0] - 35), 0.75, (255, 255, 0), 2)
            put_text(display, "Q/ESC quit | R reset buffer | C clear words", (20, display.shape[0] - 10), 0.6, (255, 255, 255), 1)

            cv2.imshow("KSL realtime keypoint inference v5", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                seq_buffer.clear()
                pred_history.clear()
                topk_history.clear()
                ema_probs = None
                hand_state = {"last_lh": None, "last_rh": None, "lh_hold": 0, "rh_hold": 0}
                last_status = "buffer reset"
            if key in (ord('c'), ord('C')):
                committed_words.clear()
                last_commit_label = None
                pred_history.clear()
                topk_history.clear()
                ema_probs = None
                last_status = "committed words cleared"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
