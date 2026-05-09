"""Run SmolVLM2 event detection on all 4 videos.

Uses CPU because the laptop has Blackwell (sm_120) and stable PyTorch's
prebuilt wheels do not yet ship sm_120 kernels. CPU is slow (~5-10 minutes
per video) but produces valid results.
"""
import os, json, sys, time
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, ".")

from pathlib import Path
from pipeline.vlm_events import SmolVLM2

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SmolVLM2-2.2B-Instruct on {device}...")
t = time.time()
vlm = SmolVLM2(device=device)
print(f"Loaded in {time.time()-t:.1f}s")

VIDEOS = sorted(Path("Videos").glob("*.mp4"))
out_dir = Path("outputs/vlm")
out_dir.mkdir(parents=True, exist_ok=True)

all_out = {}
for video in VIDEOS:
    name = video.stem
    print(f"\n=== {name} ===")
    out = {}

    # SmolVLM2-2.2B has an 8192-token context. ~1085 tokens per frame at
    # default resolution => 6 frames is the safe ceiling.
    print("  [direct/events]")
    t = time.time()
    raw1, ev1 = vlm.detect_events_video(str(video), num_frames=6, timed=False)
    out["direct_events"] = {"raw": raw1, "events": [e.__dict__ for e in ev1]}
    print(f"    {len(ev1)} events parsed in {time.time()-t:.1f}s")

    print("  [direct/timed]")
    t = time.time()
    raw2, ev2 = vlm.detect_events_video(str(video), num_frames=6, timed=True)
    out["direct_timed"] = {"raw": raw2, "events": [e.__dict__ for e in ev2]}
    print(f"    {len(ev2)} events parsed in {time.time()-t:.1f}s")

    print("  [caption-aggregate]")
    t = time.time()
    raw3, ev3, caps = vlm.detect_events_caption_aggregate(str(video), num_frames=8)
    out["caption_aggregate"] = {"raw": raw3, "events": [e.__dict__ for e in ev3], "captions": caps}
    print(f"    {len(ev3)} events parsed in {time.time()-t:.1f}s")

    all_out[name] = out
    with open(out_dir / f"{name}.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

with open(out_dir / "_combined.json", "w", encoding="utf-8") as fh:
    json.dump(all_out, fh, indent=2)
print("\nDone. Results in outputs/vlm/")
