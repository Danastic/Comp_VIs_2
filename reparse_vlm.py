"""Re-parse already-generated VLM raw outputs with the updated parser."""
import json, sys
sys.path.insert(0, ".")
from pathlib import Path
from pipeline.vlm_events import parse_events

vlm_dir = Path("outputs/vlm")
combined = {}
for fp in sorted(vlm_dir.glob("video_*.json")):
    name = fp.stem
    data = json.load(open(fp))
    for strat, payload in data.items():
        evs = parse_events(payload["raw"])
        payload["events"] = [e.__dict__ for e in evs]
    combined[name] = data
    with open(fp, "w") as fh:
        json.dump(data, fh, indent=2)

with open(vlm_dir / "_combined.json", "w") as fh:
    json.dump(combined, fh, indent=2)

# Print summary
print(f"{'video':<10} {'strategy':<20} {'#events':>8}")
print("-" * 45)
for v, data in sorted(combined.items()):
    for strat, payload in data.items():
        print(f"{v:<10} {strat:<20} {len(payload['events']):>8}")
