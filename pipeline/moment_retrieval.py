"""
Moment retrieval (Step 4).

We compare two zero-shot strategies:

  1) **CG-DETR via Lighthouse**   (https://github.com/line/lighthouse)
     Pre-trained on QVHighlights with CLIP features. Localises a text query
     to a temporal window in a video.

  2) **CLIP zero-shot baseline**   (no extra repo dependency)
     For each query, encode the text with CLIP and encode evenly-sampled
     frames with CLIP, then find the contiguous window whose mean cosine
     similarity to the query is highest. Uses non-maximum suppression to
     keep the top-K windows.

Both strategies expose the same `predict(query, video_path) -> [(start, end, score), ...]`
interface so the rest of the pipeline can swap them transparently.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class Window:
    start: float
    end: float
    score: float

    def as_tuple(self):
        return (self.start, self.end, self.score)


# Helper: paraphrase a query into a few variations
def paraphrase_query(q: str) -> list[str]:
    """Return 3 lightweight variants of the original query for query-fusion."""
    base = q.strip().rstrip(".")
    return [
        base,
        f"a video clip showing {base.lower()}",
        f"footage of {base.lower()}",
    ]


# 1) Lighthouse CG-DETR wrapper
class LighthouseCGDETR:
    def __init__(self,
                 ckpt: str = "results/cg_detr/qvhighlight/clip/best.ckpt",
                 device: str = "cpu",
                 feature_name: str = "clip"):
        from lighthouse.models import CGDETRPredictor 
        self.model = CGDETRPredictor(ckpt, device=device, feature_name=feature_name)
        self._cache = {}

    def _encode(self, video_path: str):
        if video_path not in self._cache:
            self._cache[video_path] = self.model.encode_video(video_path)
        return self._cache[video_path]

    def predict(self, query: str, video_path: str, top_k: int = 1,
                fuse_paraphrases: bool = False) -> list[Window]:
        v = self._encode(video_path)
        if fuse_paraphrases:
            wins: list[Window] = []
            for q in paraphrase_query(query):
                pred = self.model.predict(q, v)
                for w in pred["pred_relevant_windows"][:top_k]:
                    wins.append(Window(float(w[0]), float(w[1]), float(w[2])))
            wins = nms(wins, iou_thresh=0.5)
            return wins[:top_k]
        pred = self.model.predict(query, v)
        return [Window(float(w[0]), float(w[1]), float(w[2]))
                for w in pred["pred_relevant_windows"][:top_k]]


# 2) CLIP-based moment retrieval baseline
class CLIPMomentRetrieval:
    """
    Lightweight zero-shot moment retrieval:
      - Sample frames at `fps_sample` Hz.
      - Encode them with CLIP image tower.
      - Encode text query with CLIP text tower.
      - Compute cosine similarity per frame.
      - Slide a window of length `win_seconds` and rank by mean similarity.
      - Return top-K windows after temporal NMS.
    """

    def __init__(self,
                 model_name: str = "openai/clip-vit-base-patch32",
                 device: str | None = None,
                 fps_sample: float = 1.0):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.fps_sample = fps_sample
        self._cache = {}

    # --------------------------------------------------------------- #
    def _frame_features(self, video_path: str):
        if video_path in self._cache:
            return self._cache[video_path]
        import cv2, torch
        from PIL import Image
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(round(fps / self.fps_sample)))
        frames, timestamps = [], []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
                timestamps.append(idx / fps)
            idx += 1
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames sampled from {video_path}")

        feats = []
        bs = 32
        for i in range(0, len(frames), bs):
            batch = frames[i:i + bs]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                f = self.model.get_image_features(**inputs)
            # transformers >=5.x returns BaseModelOutputWithPooling; older returns tensor
            if not isinstance(f, torch.Tensor):
                f = f.pooler_output
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
        feats = np.concatenate(feats, axis=0)
        timestamps = np.array(timestamps)
        self._cache[video_path] = (feats, timestamps, total / fps)
        return self._cache[video_path]

    # --------------------------------------------------------------- #
    def _text_feature(self, query: str) -> np.ndarray:
        import torch
        inputs = self.processor(text=[query], return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            t = self.model.get_text_features(**inputs)
        if not isinstance(t, torch.Tensor):
            t = t.pooler_output
        t = t / t.norm(dim=-1, keepdim=True)
        return t.cpu().numpy()[0]

    # --------------------------------------------------------------- #
    def predict(self, query: str, video_path: str, top_k: int = 1,
                win_seconds: float | list[float] | None = None,
                fuse_paraphrases: bool = False) -> list[Window]:
        """
        If `win_seconds` is None, multiple window lengths are tried (4, 8,
        16, 32 s) and the highest-mean-similarity window across all sizes is
        returned. This makes the baseline competitive with CG-DETR which
        also predicts variable-length windows.
        """
        feats, ts, duration = self._frame_features(video_path)

        if fuse_paraphrases:
            queries = paraphrase_query(query)
            txt = np.stack([self._text_feature(q) for q in queries], axis=0).mean(0)
            txt = txt / (np.linalg.norm(txt) + 1e-12)
        else:
            txt = self._text_feature(query)

        sims = feats @ txt  # (N,)

        if win_seconds is None:
            win_seconds = [4, 8, 16, 32]
        elif isinstance(win_seconds, (int, float)):
            win_seconds = [win_seconds]

        all_wins: list[Window] = []
        for ws in win_seconds:
            win = max(1, int(round(ws * self.fps_sample)))
            if win >= len(sims):
                continue
            cs = np.concatenate([[0.0], np.cumsum(sims)])
            means = (cs[win:] - cs[:-win]) / win
            order = np.argsort(-means)
            # Take a generous shortlist; NMS later.
            for i in order[:max(5, top_k * 3)]:
                s = float(ts[i])
                e = float(ts[min(len(ts) - 1, i + win - 1)])
                all_wins.append(Window(s, e, float(means[i])))
        all_wins = nms(all_wins, iou_thresh=0.3)
        return all_wins[:top_k]


# Temporal non-maximum suppression
def _iou(a: Window, b: Window) -> float:
    s = max(a.start, b.start)
    e = min(a.end, b.end)
    inter = max(0.0, e - s)
    union = (a.end - a.start) + (b.end - b.start) - inter
    return inter / union if union > 0 else 0.0


def nms(windows: list[Window], iou_thresh: float = 0.5) -> list[Window]:
    """Keep highest-scoring windows, suppress ones with IoU > threshold."""
    sorted_w = sorted(windows, key=lambda w: -w.score)
    kept: list[Window] = []
    for w in sorted_w:
        if all(_iou(w, k) < iou_thresh for k in kept):
            kept.append(w)
    return kept


# Post-processing helpers
def merge_overlapping(windows: list[Window],
                      max_gap: float = 1.0) -> list[Window]:
    """Merge near-adjacent windows. Keeps max score among merged."""
    if not windows:
        return []
    ws = sorted(windows, key=lambda w: w.start)
    merged = [ws[0]]
    for w in ws[1:]:
        last = merged[-1]
        if w.start - last.end <= max_gap:
            merged[-1] = Window(
                start=last.start,
                end=max(last.end, w.end),
                score=max(last.score, w.score),
            )
        else:
            merged.append(w)
    return merged


def filter_short(windows: list[Window], min_seconds: float = 1.0) -> list[Window]:
    return [w for w in windows if (w.end - w.start) >= min_seconds]
