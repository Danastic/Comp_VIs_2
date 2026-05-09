# Group 9 — Presentation outline (5–7 minutes)

Speaker notes are written for two presenters (A, B). Each slide title is
followed by the time budget, the visual asset, and the talking points.

---

## Slide 1 — Title (15 s, A)
*Visual:* group name, video IDs, course logo.
- Group 9 — TVSUM 33–36
- Today: pipeline from annotation to video summary.

## Slide 2 — Our four videos (30 s, A)
*Visual:* one keyframe per video with one-line caption.
- 33: graduation flash mob
- 34: ICC street flash mob
- 35: classical-music train ad
- 36: beekeeper interview
- Why this matters: very different visual styles → stress-tests the model.

## Slide 3 — Pipeline overview (45 s, A)
*Visual:* the 6-step block diagram from the report.
- Annotate → IAA → VLM events → Moment retrieval → IoU → Summary
- Built as a single notebook + a `pipeline/` package.

## Slide 4 — Annotations & inter-annotator agreement (60 s, B)
*Visual:* the IAA table from §2 of the report.
- 3 metrics: timeline κ, saliency κ, mean IoU of matched events.
- Headline: **boundaries agree (mIoU 0.55–0.72), counts and saliency
  don't** (timeline κ 0.15–0.48, saliency κ 0.23 mean).
- Most subjective video: 36 (beekeeper) — saliency κ = −0.235 (worse
  than chance) — no central climax.

## Slide 5 — VLM event detection (60 s, B)
*Visual:* side-by-side of the three prompts' output on video 35.
- SmolVLM2-2.2B; chose it over Qwen2.5 because the processor accepts
  video/frames natively and fits in 8 GB.
- 3 prompts: direct/events, direct/timed, caption-aggregate.
- Coverage: direct/events most reliable (4–5 events parsed/video);
  direct/timed mostly fails (often produces a single window or no
  timestamps); caption-aggregate works on video 36 (cov@0.5 = 0.42).
- Two failure modes: dialogue events disappear (no ASR) and the model
  loops/repeats (fixed with `repetition_penalty=1.3, no_repeat_ngram=4`).

## Slide 6 — Moment retrieval (60 s, A)
*Visual:* per-video bar chart of mean IoU for CLIP-B/32 vs CLIP-L/14
with paraphrase fusion overlay.
- We attempted Lighthouse CG-DETR but the official checkpoint is no
  longer downloadable; fell back to two CLIP backbones.
- **CLIP-L/14 wins**: mean IoU **0.113** vs CLIP-B/32's 0.089;
  R@IoU=0.3 of 0.211 vs 0.140.
- **Paraphrase fusion did not help** in our setting (small drop in
  mean IoU on both backbones).
- Per-category: performance 0.174, beekeeping 0.070,
  **speech/dialogue 0.000** — CLIP is text-blind.

## Slide 7 — One complete example (90 s, B)
*Visual:* video 35 (classical-music train).
- 13 ground-truth events in the merged annotation set.
- Show the 4 VLM-detected events (SmolVLM2 direct/events strategy).
- Show CLIP-L/14 predicted windows for those 4 queries (mIoU=0.333,
  R@0.3=0.75 — best of our four videos).
- Play the generated summary clip (28 s, 8 clips).
- Note what is missing: conductor and advertisement-text-reveal events
  (mIoU=0.000) because they require reading text, not recognising images.

## Slide 8 — Findings & critical reflections (45 s, A)
*Visual:* bullet list.
- **Subjectivity shapes the summary.** Our saliency ratings diverged most
  on video 36 (beekeeper, κ = −0.235): one annotator prioritised narrative
  events, the other visual demonstrations — and the generated summary
  reflects neither coherently because CLIP picks whichever is most
  visually distinctive, not most narratively important.
- Pretrained zero-shot moment retrieval works well on visually distinctive
  events (performance mIoU 0.174) but collapses on speech/dialogue (0.000).
- Using VLM events as queries (spec §4) gave mIoU 0.206 vs 0.113 on
  annotated events — but only because VLM queries are coarser and easier.
- Coherence in interview-style videos is bottlenecked by ASR, not vision.

## Slide 9 — Future work (30 s, B)
- **Add Whisper-tiny ASR** — biggest expected gain; dialogue events score
  0.000 mean IoU under any of our methods.
- Re-introduce saliency via LLM-judge re-ranking.
- Restore the CG-DETR comparison once a working public checkpoint is
  available (lighthouse repo is currently broken).

## Slide 10 — Q&A (rest of time)

---

### Notes for live demo
- Have `outputs/summaries/video_35_summary.mp4` open in a browser tab.
- Be ready to read out loud the 9 GT events (slide 7) so the audience
  can match them to clip transitions in the summary.
