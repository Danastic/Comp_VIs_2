from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

_TIME_RE = re.compile(r"(\d{1,2})[:.](\d{2})")


def time_to_seconds(s: str) -> float:
    """Parse 'MM:SS' (the spec format) tolerating 'M:SS' / 'MM.SS' / extra spaces."""
    s = s.strip().replace(" ", "")
    m = _TIME_RE.fullmatch(s)
    if not m:
        # explicit fallback for floats
        try:
            return float(s)
        except ValueError as e:
            raise ValueError(f"Cannot parse time string {s!r}") from e
    return int(m.group(1)) * 60 + int(m.group(2))


def seconds_to_time(t: float) -> str:
    t = max(0, int(round(t)))
    return f"{t // 60:02d}:{t % 60:02d}"


# Annotation parsing
@dataclass
class Event:
    description: str
    start: float 
    end: float 
    saliency: int  #  from -2 to 2 

    def to_dict(self):
        return asdict(self)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


# matches "MM:SS - MM:SS" (with several allowed dashes/spaces).
_RANGE_RE = re.compile(
    r"(\d{1,2}[:.]\d{2})\s*[\-–—]\s*(\d{1,2}[:.]\d{2})"
)


def parse_annotation_file(path: str | os.PathLike) -> list[Event]:
    """
    Parse annotation files, each non-empty line looks like:
        <leading idx>, MM:SS - MM:SS, <rating>
    """
    events: list[Event] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            # strip leading "<num>\t" or "<num>." idx if exists
            line = re.sub(r"^\s*\d+\s*[\t.\)]\s*", "", line)
            m = _RANGE_RE.search(line)
            if not m:
                continue  
            start = time_to_seconds(m.group(1))
            end = time_to_seconds(m.group(2))
            # desc = everything before the timestamp
            desc = line[: m.start()].rstrip(" ,").strip()
            # saliency = anything after the timestamp
            tail = line[m.end():].strip(" ,\t")
            try:
                saliency = int(tail.split(",")[0].strip()) if tail else 0
            except ValueError:
                saliency = 0
            if end < start:
                start, end = end, start
            events.append(Event(desc, start, end, saliency))
    return events


def load_group_annotations(annotations_dir: str | os.PathLike,
                           group_id: int = 9) -> dict[str, dict[str, list[Event]]]:
    """
    Returns: { video_name -> { annotator_id -> [Event, ...] } }
    Filenames : [GroupID]-[Dataset]-[Video_name]-[StudentID].txt
    """
    annotations_dir = Path(annotations_dir)
    out: dict[str, dict[str, list[Event]]] = {}
    pattern = re.compile(rf"^{group_id}-([^-]+)-(video_\d+)-([^.]+)\.txt$")
    for f in sorted(annotations_dir.glob("*.txt")):
        m = pattern.match(f.name)
        if not m:
            continue
        _, video, annotator = m.groups()
        out.setdefault(video, {})[annotator] = parse_annotation_file(f)
    return out

# IoU and matching

def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """IoU between two [start, end] intervals."""
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0.0, e - s)
    union = max(a[1], a[0]) - min(a[0], a[1]) + max(b[1], b[0]) - min(b[0], b[1]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def best_match(query: tuple[float, float],
               candidates: list[tuple[float, float]]) -> tuple[int, float]:
    """return (idx, iou) of best-overlapping candidate. (-1, 0) if none."""
    best_idx, best_iou = -1, 0.0
    for i, c in enumerate(candidates):
        iou = temporal_iou(query, c)
        if iou > best_iou:
            best_iou, best_idx = iou, i
    return best_idx, best_iou

# Inter-annotator agreement

def cohens_kappa(a: Iterable[int], b: Iterable[int]) -> float:
    """
    Cohen's Kappa for two label sequences of equal length
    Returns 0 if no variance
    """
    a = list(a); b = list(b)
    assert len(a) == len(b)
    if not a:
        return 0.0
    labels = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    if 1 - pe == 0:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def agreement_report(events_a: list[Event], events_b: list[Event],
                     iou_threshold: float = 0.3) -> dict:
    """
    Compute three forms of agreement between two annotators on the same video:

    1) Event-presence kappa: project both annotators onto a 1-second timeline
       (event/no-event) and compute kappa.
    2) Saliency-rating kappa: among matched events (best IoU >= threshold),
       compare assigned saliency ratings.
    3) Mean IoU of matched event pairs (greedy, by max IoU first).
    """
    if not events_a or not events_b:
        return dict(timeline_kappa=0.0, saliency_kappa=0.0,
                    mean_iou=0.0, n_matched=0)

    # 1) timeline kappa
    end_t = int(np.ceil(max(e.end for e in events_a + events_b)))
    timeline_a = np.zeros(end_t + 1, dtype=int)
    timeline_b = np.zeros(end_t + 1, dtype=int)
    for e in events_a:
        timeline_a[int(e.start): int(e.end) + 1] = 1
    for e in events_b:
        timeline_b[int(e.start): int(e.end) + 1] = 1
    timeline_kappa = cohens_kappa(timeline_a.tolist(), timeline_b.tolist())

    # 2) greedy match by IoU, then kappa over saliency ratings of matched pairs
    iou_matrix = np.zeros((len(events_a), len(events_b)))
    for i, ea in enumerate(events_a):
        for j, eb in enumerate(events_b):
            iou_matrix[i, j] = temporal_iou((ea.start, ea.end), (eb.start, eb.end))
    matched_a, matched_b, ious = [], [], []
    used_a, used_b = set(), set()
    flat = sorted(((iou_matrix[i, j], i, j)
                   for i in range(len(events_a))
                   for j in range(len(events_b))),
                  reverse=True)
    for iou, i, j in flat:
        if iou < iou_threshold:
            break
        if i in used_a or j in used_b:
            continue
        matched_a.append(events_a[i].saliency)
        matched_b.append(events_b[j].saliency)
        ious.append(iou)
        used_a.add(i); used_b.add(j)

    saliency_kappa = cohens_kappa(matched_a, matched_b) if matched_a else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    return dict(
        timeline_kappa=float(timeline_kappa),
        saliency_kappa=float(saliency_kappa),
        mean_iou=mean_iou,
        n_matched=len(matched_a),
        n_a=len(events_a),
        n_b=len(events_b),
    )


# Sentence-embedding similarity
def cosine_sim_matrix(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-12)
    b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def event_coverage(predicted: list[str],
                   annotated: list[str],
                   embed_fn,
                   threshold: float = 0.5) -> dict:
    """
    For each annotated event, find the most similar predicted event by cosine
    similarity over sentence embeddings. Coverage = fraction of annotated
    events whose best match exceeds `threshold`.

    `embed_fn` takes list[str] -> np.ndarray (N, D).
    """
    if not predicted or not annotated:
        return dict(coverage=0.0, sims=[], best_idx=[])
    pred_emb = embed_fn(predicted)
    ann_emb = embed_fn(annotated)
    sim = cosine_sim_matrix(ann_emb, pred_emb)
    best_idx = sim.argmax(axis=1).tolist()
    best_sim = sim.max(axis=1).tolist()
    covered = sum(1 for s in best_sim if s >= threshold)
    return dict(
        coverage=covered / len(annotated),
        sims=best_sim,
        best_idx=best_idx,
        sim_matrix=sim,
    )
