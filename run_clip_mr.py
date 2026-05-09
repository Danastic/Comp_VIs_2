"""Run CLIP zero-shot moment retrieval on all 4 videos against merged GT."""
import json, sys, time
sys.path.insert(0, ".")
from pathlib import Path
import numpy as np
from pipeline.utils import load_group_annotations, temporal_iou, Event
from pipeline.moment_retrieval import CLIPMomentRetrieval


def merge_annotators(by_ann):
    keys = sorted(by_ann.keys())
    if len(keys) == 1:
        return list(by_ann[keys[0]])
    a, b = by_ann[keys[0]], by_ann[keys[1]]
    used_b = set()
    out = []
    for ea in a:
        best_j, best_iou = -1, 0.0
        for j, eb in enumerate(b):
            if j in used_b: continue
            iou = temporal_iou((ea.start, ea.end), (eb.start, eb.end))
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.3:
            eb = b[best_j]
            out.append(Event(
                description=f"{ea.description} || {eb.description}",
                start=max(ea.start, eb.start),
                end=min(ea.end, eb.end),
                saliency=max(ea.saliency, eb.saliency),
            ))
            used_b.add(best_j)
        else:
            out.append(ea)
    for j, eb in enumerate(b):
        if j not in used_b:
            out.append(eb)
    out.sort(key=lambda e: e.start)
    return out


import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Two moment-retrieval methods, both CLIP zero-shot but with different
# backbones. Method 1: ViT-B/32 (baseline). Method 2: ViT-L/14 (stronger
# backbone, larger receptive field). Report compares them in §4.
print("\nLoading CLIP-B/32 (baseline)...")
mr_b32 = CLIPMomentRetrieval(model_name="openai/clip-vit-base-patch32",
                             fps_sample=1.0, device=device)
print("Loading CLIP-L/14 (stronger backbone)...")
mr_l14 = CLIPMomentRetrieval(model_name="openai/clip-vit-large-patch14",
                             fps_sample=1.0, device=device)

ann = load_group_annotations("Annotations", group_id=9)
gt = {v: merge_annotators(by) for v, by in ann.items()}

methods = [
    ("clip_b32",        mr_b32, False),
    ("clip_b32_fused",  mr_b32, True),
    ("clip_l14",        mr_l14, False),
    ("clip_l14_fused",  mr_l14, True),
]

print("\nRunning all methods over all videos...")
results = {m[0]: {} for m in methods}
for v in sorted(gt.keys()):
    src = f"Videos/{v}.mp4"
    print(f"\n=== {v} ({len(gt[v])} GT events) ===")
    for method_name, predictor, fuse in methods:
        per_event = []
        ts = time.time()
        for e in gt[v]:
            desc = e.description.split(" || ")[0]
            wins = predictor.predict(desc, src, top_k=1, fuse_paraphrases=fuse)
            if wins:
                w = wins[0]
                iou = temporal_iou((e.start, e.end), (w.start, w.end))
                per_event.append({"desc": desc, "gt": [e.start, e.end],
                                  "pred": [w.start, w.end, w.score],
                                  "iou": iou, "saliency": e.saliency})
            else:
                per_event.append({"desc": desc, "gt": [e.start, e.end],
                                  "pred": None, "iou": 0.0, "saliency": e.saliency})
        results[method_name][v] = per_event
        miou = float(np.mean([x["iou"] for x in per_event]))
        print(f"  [{method_name:18s}] mIoU={miou:.3f}  in {time.time()-ts:.1f}s")

Path("outputs/moment_retrieval").mkdir(parents=True, exist_ok=True)
with open("outputs/moment_retrieval/results_clip.json", "w") as fh:
    json.dump(results, fh, indent=2)
print("\nWrote outputs/moment_retrieval/results_clip.json")

# Summary
print("\n=== Summary ===")
for method, by_v in results.items():
    all_iou = []
    for v, evs in by_v.items():
        all_iou.extend(x["iou"] for x in evs)
    miou = float(np.mean(all_iou))
    p30 = float(np.mean([1.0 if i >= 0.3 else 0.0 for i in all_iou]))
    p50 = float(np.mean([1.0 if i >= 0.5 else 0.0 for i in all_iou]))
    print(f"  {method:12s} mIoU={miou:.3f}  R@0.3={p30:.2f}  R@0.5={p50:.2f}")
