# Group 9 — Second Assignment Submission

Pipeline: annotation → IAA → VLM event detection → moment retrieval →
IoU evaluation → summary video generation.

## Files

```
.
├── Group9_Second_Assignment_CV2026.ipynb   ← MAIN deliverable, run top-to-bottom
├── Report.md                               ← 1500–2000 word report
├── Presentation_outline.md                 ← 5–7 minute presentation script
├── pipeline/
│   ├── utils.py             # annotation parsing, IoU, Cohen's Kappa, embedding similarity
│   ├── vlm_events.py        # SmolVLM2 wrapper + 3 prompting strategies
│   ├── moment_retrieval.py  # Lighthouse CG-DETR + CLIP zero-shot baseline
│   └── summary_video.py     # ffmpeg/OpenCV summary video creation
├── Annotations/             # group-9 annotations (provided)
├── Videos/                  # video_33..36.mp4
├── outputs/                 # populated by running the notebook
│   ├── eval/                # iaa.json, coverage.csv, IoU summary, failure analysis
│   ├── vlm/                 # raw VLM outputs per video / strategy
│   ├── moment_retrieval/    # results.json
│   └── summaries/           # video_<id>_summary.mp4
├── build_notebook.py        # script that generated the .ipynb
└── demo_code/               # original demo notebooks (reference)
```

## How to run

```bash
# Most GPUs (CUDA 12.4):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Blackwell GPUs (RTX 50xx, sm_120) — the stable wheels do not include
# sm_120 kernels; use nightly cu128 instead:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

pip install --user transformers sentence-transformers num2words av opencv-python pillow matplotlib pandas

# Run end-to-end:
jupyter notebook Group9_Second_Assignment_CV2026.ipynb
# or run individual scripts:
python run_clip_mr.py          # CLIP-B/32 + CLIP-L/14 moment retrieval
python run_vlm.py              # SmolVLM2 event detection (3 prompt strategies)
python run_coverage.py         # VLM coverage vs annotations
python run_summary.py          # Generate summary videos
python run_failure_analysis.py # IoU breakdown by category & saliency
```

The pipeline downloads ~5 GB of model weights on first run (SmolVLM2-2.2B,
CLIP-B/32, CLIP-L/14, MiniLM-L6). Approximate runtime on an RTX 5070
laptop with PyTorch nightly cu128: ~3 minutes for moment retrieval on
all 4 videos, ~5 minutes for SmolVLM2 on all 4 videos.

## Final results summary

| Metric | Value |
|---|---|
| Inter-annotator timeline κ (mean across 4 videos) | 0.316 |
| VLM coverage @ 0.5 sim (best strategy: caption_aggregate on video 36) | 0.42 |
| Moment retrieval mean IoU (CLIP-B/32) | 0.089 |
| Moment retrieval mean IoU (CLIP-L/14) | **0.113** |
| Moment retrieval R@IoU=0.3 (CLIP-L/14) | 0.211 |
| Summary videos generated | 4 (28–40 s each) |
