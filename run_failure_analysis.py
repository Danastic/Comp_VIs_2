"""Failure analysis: IoU broken down by event category and saliency rating."""
import json, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from pathlib import Path

results = json.load(open("outputs/moment_retrieval/results_clip.json"))

def categorize(desc: str) -> str:
    d = desc.lower()
    if any(k in d for k in ["dance", "dancing", "band", "performance",
                             "performs", "music", "play", "violin", "flute",
                             "orchestra", "cello", "instrument"]):
        return "performance"
    if any(k in d for k in ["speech", "speaker", "says", "tells", "explains",
                             "introduces", "describes", "thanks", "anecdote"]):
        return "speech/dialogue"
    if any(k in d for k in ["crowd", "cheer", "applaud", "watch", "spectator",
                             "smile", "laugh"]):
        return "crowd/reaction"
    if any(k in d for k in ["bee", "hive", "honey", "frame", "jar", "queen",
                             "comb", "syrup"]):
        return "beekeeping"
    return "other"

rows = []
for method, by_v in results.items():
    for v, evs in by_v.items():
        for x in evs:
            rows.append({
                "method": method, "video": v,
                "iou": x["iou"], "saliency": x["saliency"],
                "category": categorize(x["desc"]),
                "desc": x["desc"][:80],
            })
df = pd.DataFrame(rows)

by_method = (df.groupby("method")
             .agg(mean_IoU=("iou", "mean"),
                  R_at_0_3=("iou", lambda s: float((s >= 0.3).mean())),
                  R_at_0_5=("iou", lambda s: float((s >= 0.5).mean())),
                  n=("iou", "count"))
             .round(3))
by_method.to_csv("outputs/eval/moment_retrieval_summary.csv")
print("=== Method comparison ===")
print(by_method)

by_method_cat = (df.groupby(["method", "category"])
                 .agg(mean_IoU=("iou", "mean"), n=("iou", "count"))
                 .round(3))
by_method_cat.to_csv("outputs/eval/failure_analysis_by_category.csv")
print("\n=== By category ===")
print(by_method_cat)

by_method_sal = (df.groupby(["method", "saliency"])
                 .agg(mean_IoU=("iou", "mean"), n=("iou", "count"))
                 .round(3))
by_method_sal.to_csv("outputs/eval/failure_analysis_by_saliency.csv")
print("\n=== By saliency ===")
print(by_method_sal)

# Worst predictions per method
print("\n=== Worst predictions (CLIP-L/14) ===")
worst = df[df.method == "clip_l14"].nsmallest(8, "iou")
for _, r in worst.iterrows():
    print(f"  IoU={r.iou:.2f}  cat={r.category:15s}  {r.desc}")
