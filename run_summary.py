"""Generate summary videos using the best-IoU moment-retrieval method."""
import json, sys
sys.path.insert(0, ".")
import numpy as np
from pathlib import Path
from pipeline.summary_video import make_summary
from pipeline.moment_retrieval import Window, merge_overlapping, filter_short

mr_path = Path("outputs/moment_retrieval/results_clip.json")
results = json.load(open(mr_path))

# Pick best method by overall mean IoU
def overall_miou(method_results):
    all_iou = [x["iou"] for v in method_results.values() for x in v]
    return float(np.mean(all_iou)) if all_iou else 0.0

best_method = max(results.keys(), key=lambda m: overall_miou(results[m]))
print(f"Best method: {best_method}  (mIoU = {overall_miou(results[best_method]):.3f})")

out_dir = Path("outputs/summaries"); out_dir.mkdir(parents=True, exist_ok=True)
for v, evs in results[best_method].items():
    src = f"Videos/{v}.mp4"
    segments = []
    for x in evs:
        if x["pred"] is None: continue
        s, e, _ = x["pred"]
        segments.append((s, e))
    if not segments:
        print(f"{v}: no segments")
        continue

    # Post-process: filter <1.5s, merge near-adjacent.
    wins = [Window(s, e, 1.0) for s, e in segments if (e - s) >= 1.5]
    wins = merge_overlapping(wins, max_gap=1.0)
    segs = sorted([(w.start, w.end) for w in wins], key=lambda x: x[0])

    out = out_dir / f"{v}_summary.mp4"
    try:
        make_summary(src, segs, str(out))
        total = sum(e - s for s, e in segs)
        print(f"{v}: {len(segs)} clips ({total:.1f}s) -> {out}")
    except Exception as exc:
        print(f"{v}: FAILED - {exc}")
