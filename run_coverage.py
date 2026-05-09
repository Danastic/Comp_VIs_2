"""Compute VLM event-coverage vs ground truth (sentence-BERT cosine similarity)."""
import json, sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from pipeline.utils import load_group_annotations, temporal_iou, Event, event_coverage

def merge_annotators(by_ann):
    keys = sorted(by_ann.keys())
    if len(keys) == 1: return list(by_ann[keys[0]])
    a, b = by_ann[keys[0]], by_ann[keys[1]]
    used_b = set(); out = []
    for ea in a:
        best_j, best_iou = -1, 0.0
        for j, eb in enumerate(b):
            if j in used_b: continue
            iou = temporal_iou((ea.start, ea.end), (eb.start, eb.end))
            if iou > best_iou: best_iou, best_j = iou, j
        if best_iou >= 0.3:
            eb = b[best_j]
            out.append(Event(f"{ea.description} || {eb.description}",
                             max(ea.start, eb.start), min(ea.end, eb.end),
                             max(ea.saliency, eb.saliency)))
            used_b.add(best_j)
        else:
            out.append(ea)
    for j, eb in enumerate(b):
        if j not in used_b: out.append(eb)
    out.sort(key=lambda e: e.start)
    return out

print("Loading sentence-BERT...")
import torch
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu",
)
embed = lambda texts: sbert.encode(list(texts), convert_to_numpy=True)

ann = load_group_annotations("Annotations", group_id=9)
gt = {v: merge_annotators(by) for v, by in ann.items()}

vlm_path = Path("outputs/vlm/_combined.json")
if not vlm_path.exists():
    print("ERROR: outputs/vlm/_combined.json not found. Run run_vlm.py first.")
    sys.exit(1)
vlm_outputs = json.load(open(vlm_path))

import pandas as pd
rows = []
out_eval = Path("outputs/eval"); out_eval.mkdir(parents=True, exist_ok=True)
for v, evs in gt.items():
    if v not in vlm_outputs: continue
    ann_texts = [e.description.split(" || ")[0] for e in evs]
    for strat, payload in vlm_outputs[v].items():
        pred_texts = [e["description"] for e in payload["events"]]
        if not pred_texts: continue
        cov = event_coverage(pred_texts, ann_texts, embed_fn=embed, threshold=0.5)
        rows.append({
            "video": v, "strategy": strat,
            "n_ann": len(ann_texts), "n_pred": len(pred_texts),
            "coverage_at_0.5": float(cov["coverage"]),
            "mean_best_sim": float(np.mean(cov["sims"])),
            "max_best_sim": float(np.max(cov["sims"])),
        })
        with open(out_eval / f"coverage_{v}_{strat}.json", "w") as fh:
            json.dump({"sims": cov["sims"], "best_idx": cov["best_idx"],
                       "ann": ann_texts, "pred": pred_texts}, fh, indent=2)

df = pd.DataFrame(rows).sort_values(["video", "strategy"])
df.to_csv(out_eval / "coverage.csv", index=False)
print("\nCoverage table:")
print(df.to_string(index=False))
print(f"\nWrote {out_eval/'coverage.csv'}")
