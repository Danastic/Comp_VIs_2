# Group 9 — Second Assignment Report
**Event Detection and Video Summarization (CV 2026)**

## 1. Introduction

This report accompanies our pipeline for the second assignment. We were
assigned TVSUM videos 33–36: a graduation-day flash-mob speech (33), a
street flash mob for the ICC World T20 (34), an in-train classical-orchestra
advertisement (35), and a beekeeping interview (36). Each video was
annotated independently by both group members. We then built a 6-step
pipeline that (1) measures inter-annotator agreement, (2) detects events
with a Vision-Language Model, (3) measures coverage of the annotated
events in the VLM output, (4) localises each event with two
moment-retrieval methods, (5) evaluates IoU vs. our annotations, and
(6) renders a concatenated summary video. All code lives under
`pipeline/` and is invoked from
[Group9_Second_Assignment_CV2026.ipynb](Group9_Second_Assignment_CV2026.ipynb).

## 2. Annotations and inter-annotator agreement

We followed the format `<description>, MM:SS - MM:SS, <subjectivity>` and
produced between 8 and 15 events per video. We watched each video
independently and only compared annotations once both files were written,
so agreement is genuine (not negotiated).

**Subjectivity choices.** We used the full −2…+2 scale. Ratings of +2 went
to structurally indispensable moments — e.g. the first musician revealing
themselves on the train (video 35, 00:39–00:45) or the beekeeper opening
a hive for the first time (video 36, 00:59–01:06): a viewer who missed
these would not understand the video's core premise. Ratings of +1 covered
events that add important context but are not strictly necessary (e.g. the
conductor appearing in video 35). Zero went to events whose relevance depends
on prior knowledge (a baby watching musicians is charming, not narrative).
Negative ratings marked events we included because they enrich the summary
but a viewer can follow the story without them — e.g. the crowd laughing at
an anecdote or a close-up of a honeycomb frame. This subjectivity directly
influenced the summaries: the dialogue-heavy graduation speech (video 33) saw
the widest disagreement on which speech segments merited +2 vs −1, because
the emotional weight of words is invisible to a purely visual pipeline.

We computed three complementary agreement metrics:

| Metric | What it measures |
|---|---|
| **Timeline κ** | Cohen's Kappa over a 1-second binary event-vs-silence timeline |
| **Mean IoU** of matched events (greedy 1-to-1, IoU ≥ 0.3) | Boundary precision when both of us flagged the same event |
| **Saliency κ** | Cohen's Kappa over the −2…+2 rating among matched events |

Results (from `outputs/eval/iaa.json`):

| Video | #A | #B | timeline κ | saliency κ | mean IoU |
|---|---:|---:|---:|---:|---:|
| 33 | 10 | 9 | **0.148** | 0.600 | 0.655 |
| 34 | 8 | 8 | **0.478** | 0.357 | 0.703 |
| 35 | 9 | 10 | **0.384** | 0.226 | 0.554 |
| 36 | 15 | 11 | **0.254** | −0.235 | 0.717 |

Three observations the rubric specifically asks us to discuss:

1. **Boundary agreement is high (mIoU 0.55–0.72) but timeline-coverage
   agreement is moderate (κ 0.15–0.48).** This means we usually drew
   similar windows around events we *both* noticed, but we disagreed
   about *how many* events to mark. Video 33 is the extreme case: one of
   us marked 10 events including several short crowd reactions, the
   other absorbed those into longer dance segments.
2. **Saliency agreement collapses on the beekeeping video (κ = −0.235,
   worse than chance).** The video has no central climax, and one of us
   prioritised *narrative* events (introduction, role explanation) while
   the other prioritised *visual demonstrations* (opening hives, showing
   combs).
3. **Saliency is the most subjective dimension.** Mean saliency-κ across
   the four videos is 0.24, consistent with the assignment's claim that
   subjectivity ratings are the loosest layer of annotation. We
   consequently chose **not** to use saliency for clip selection in §5 —
   we keep all events that pass the IoU threshold rather than weighting
   by importance.

For the rest of the pipeline we built a **merged ground-truth set** per
video by greedily matching events with IoU ≥ 0.3, intersecting their
windows (the stricter timestamp), and taking max-saliency. Unmatched
events from either annotator are kept. This produces 10–19 GT events per
video (15, 10, 13, 19 respectively).

## 3. VLM event detection

We used **HuggingFaceTB/SmolVLM2-2.2B-Instruct**. We chose it over
Qwen2.5-VL-3B because SmolVLM2's processor accepts video paths or
sampled frames natively, runs comfortably in our 8 GB GPU, and matches
the demo notebook so the grader can reproduce results without a
Qwen license.

We compared three prompting strategies:

* **direct/events** — pass 6 uniformly sampled frames; ask for
  `Salient event N: …` numbered list.
* **direct/timed** — same, but ask for `[MM:SS-MM:SS]` per event.
* **caption-aggregate** — caption each frame individually, then prompt
  SmolVLM2 again to merge the captions into a numbered list. This
  follows the demo notebook's hinted strategy.

Coverage is measured by encoding both annotated and predicted event
descriptions with `sentence-transformers/all-MiniLM-L6-v2` and computing
cosine similarity. An annotated event is "covered" if its best-matching
predicted event has similarity ≥ 0.5.

Coverage table (from `outputs/eval/coverage.csv`, `mean_best_sim` = mean
similarity of each annotated event's best match in the prediction set):

| video | strategy | n_ann | n_pred | cov@0.5 | mean best sim |
|---|---|---:|---:|---:|---:|
| 33 | direct_events | 15 | 4 | 0.00 | 0.235 |
| 33 | direct_timed | 15 | 1 | 0.00 | 0.167 |
| 33 | caption_aggregate | 15 | 2 | 0.13 | 0.392 |
| 34 | direct_events | 10 | 4 | **0.30** | **0.428** |
| 34 | direct_timed | 10 | 5 | 0.10 | 0.361 |
| 34 | caption_aggregate | 10 | 2 | 0.10 | 0.344 |
| 35 | direct_events | 13 | 4 | 0.08 | 0.325 |
| 35 | direct_timed | 13 | 1 | 0.00 | 0.177 |
| 35 | caption_aggregate | 13 | 1 | 0.00 | 0.114 |
| 36 | direct_events | 19 | 5 | 0.21 | **0.430** |
| 36 | direct_timed | 19 | 1 | 0.00 | 0.018 |
| 36 | caption_aggregate | 19 | 1 | **0.42** | **0.491** |

**Findings.**

* **direct/events** is the most reliable strategy in terms of *number*
  of events extracted (4–5 per video) and is the best by `mean_best_sim`
  for video 34 (0.428).
* **direct/timed** is the worst strategy: SmolVLM2 either refuses to
  produce timestamps or hallucinates one all-encompassing window
  (e.g. `00:00–00:05`). The prompt history shows that variants like
  "use MM:SS-MM:SS format" or "include explicit timestamps" did not
  improve this. SmolVLM2 simply has no time-grounded supervision.
* **caption-aggregate** runs the model 7+ times per video. It produces
  fewer, longer events (often a single sentence describing the whole
  video) — high coverage on video 36 (0.42) but low elsewhere.

Across the board, **mean similarity is below the 0.5 coverage threshold**.
This is the expected zero-shot ceiling for a 2.2 B VLM looking at 6 stills.

**Failure modes observed.**

* *Dialogue-driven events* are missed. SmolVLM2 does not transcribe
  speech, so events such as "the host makes a joke about onions" or
  "the speaker concludes by urging the students to let go" surface only
  as "person speaking on stage".
* *Repetitive choreography* (videos 33 and 34) collapses into one event,
  while our annotations break the same footage into 3–5 formation
  changes.
* *Beekeeping vocabulary* (video 36) is partially dropped. SmolVLM2
  mentions "person in white suit holds a frame" but not "honey super"
  or "queen-bee cage".
* *Repetition collapse*. With greedy decoding and many similar frames,
  SmolVLM2-2.2B repeats the same event description over and over. We
  added `repetition_penalty=1.3` and `no_repeat_ngram_size=4`, which
  removed obvious loops but caused short outputs in some cases.

**Prompt history.** Earlier prompts (`"describe the video"`) produced
free-form paragraphs that our regex parser could not split. The
templated `Salient event N:` instruction, plus a "*Return only the
numbered list*" suffix, increased successfully parsed events from ~40 %
to ~95 % of model outputs. Adding the bracketed timestamp slot did not
improve timestamp accuracy. Removing angle brackets (`<…>`) from the
prompt was necessary because they caused the chat-template to emit zero
tokens (an interaction with the SmolVLM2 special-token vocabulary).

## 4. Moment retrieval

The assignment asks us to compare two moment-retrieval models. We
attempted to use Lighthouse CG-DETR (the demo's recommended model), but
the pre-trained checkpoint is not on HuggingFace and the official
download script is broken on the current `main` branch. Rather than
ship a half-working CG-DETR integration, we report **two zero-shot CLIP
variants** that share the same retrieval code path but differ in
backbone capacity:

* **CLIP-B/32 zero-shot** — `openai/clip-vit-base-patch32`. 88 M params.
* **CLIP-L/14 zero-shot** — `openai/clip-vit-large-patch14`. 304 M params.

For each query we sample frames at 1 Hz, extract image features, encode
the query with the text tower, compute cosine similarity per frame,
average over candidate windows of length **{4, 8, 16, 32} s**, and pick
the highest-mean window after temporal NMS (IoU ≥ 0.3). Both methods
optionally support **paraphrase fusion**: three deterministic rephrasings
(the original, "a video clip showing X", "footage of X"), retrieved
separately and merged with NMS. Implementation is in
`pipeline/moment_retrieval.py`; frames are sampled at 1 Hz, which is
unconstrained by video length since CLIP operates per-frame.

**Spec-aligned evaluation: VLM-detected events as queries.** Following
the assignment pipeline (§4), we first ran CLIP-L/14 using the
VLM-detected `direct_events` descriptions as queries (17 events across
4 videos), measuring IoU against the nearest ground-truth annotation:

| video | VLM queries | mIoU | R@0.3 |
|---|---:|---:|---:|
| 33 | 4 | 0.040 | 0.000 |
| 34 | 4 | 0.186 | 0.250 |
| 35 | 4 | 0.333 | 0.750 |
| 36 | 5 | 0.255 | 0.400 |
| **overall** | **17** | **0.206** | **0.353** |

The per-query mIoU (0.206) is higher than the per-annotated-event mIoU
below (0.113), but this is a smaller and easier set: 4–5 coarse VLM
events vs 10–19 fine-grained annotated events per video. The VLM's
generic descriptions ("crowd dancing", "people in train") happen to
align well with CLIP's image-text space, which shares the same
coarse-grained vocabulary.

**Fine-grained evaluation: annotated events as queries.** To stress-test
the retrieval against precise event descriptions, we also ran all four
method variants on our 57 merged ground-truth events:

| method | mean IoU | R@IoU=0.3 | R@IoU=0.5 |
|---|---:|---:|---:|
| CLIP-B/32          | 0.089 | 0.140 | 0.070 |
| CLIP-B/32 + fusion | 0.082 | 0.105 | 0.053 |
| **CLIP-L/14**      | **0.113** | **0.211** | 0.088 |
| CLIP-L/14 + fusion | 0.107 | 0.193 | **0.105** |

**Take-aways.**

* **The larger backbone helps.** CLIP-L/14 improves mean IoU by ~27 %
  and R@0.3 by ~50 % over CLIP-B/32, at ~3× inference cost (still
  <20 s/video on our GPU).
* **Paraphrase fusion does not help** in our setting. Both backbones see
  a small drop in mean IoU and R@0.3. R@0.5 increases slightly for
  CLIP-L/14 + fusion (0.105 vs 0.088), suggesting fusion narrows windows
  but hurts coarse alignment — the opposite of what CG-DETR-style
  cross-attention over multiple queries would achieve.
* **Per-category breakdown** (CLIP-L/14, annotated queries):
  * **performance** events (29 of 57 GT events): mean IoU **0.174** —
    best category; CLIP recognises instruments and dancing well.
  * **crowd/reaction** (3 events): 0.133.
  * **beekeeping** (14 events): 0.070 — CLIP knows "bees" but cannot
    distinguish "shows a frame" from "opens a hive".
  * **speech/dialogue** (9 events): **0.000** — CLIP is text-blind.
* **Saliency vs IoU is non-monotonic.** Saliency +2 events (21): IoU
  0.151. Saliency +1 (14): 0.091. Saliency 0 (11): 0.027. Saliency −1
  (8): 0.127. Saliency −2 (3): 0.222. The dip at 0/+1 corresponds to
  transitional moments with little distinctive visual content; the rise
  at −2 is a small-sample artefact.

## 5. Video summary generation

For each video we take the predicted windows from the **best-IoU method
(CLIP-L/14)**, drop windows shorter than 1.5 s, merge windows separated
by ≤ 1 s, sort chronologically, and concatenate with the OpenCV writer
(falling back from ffmpeg, which isn't on the laptop's `PATH`). The
resulting summary videos live in `outputs/summaries/`:

| Video | #clips after merge | Total summary length | Source length |
|---|---:|---:|---:|
| 33 | 9 | 31 s | ~7:30 |
| 34 | 9 | 28 s | ~2:25 |
| 35 | 8 | 28 s | ~2:25 |
| 36 | 11 | 40 s | ~4:00 |

Reflecting on the four characteristics the assignment lists:

* **Relevance.** Performance-heavy clips dominate: most CLIP picks for
  videos 33–35 are dancing or instrument-playing segments. Beekeeping
  picks are weaker — generic shots of hands and frames rather than the
  named demonstrations.
* **Coverage.** The summaries cover the visually distinctive moments
  well (saliency-2 events: 0.151 mean IoU) but mis-cover dialogue-only
  events (0.000 mean IoU). Counting only events with IoU ≥ 0.3, the
  summaries cover 21 % of the GT events — a low but honest reflection
  of zero-shot CLIP's ceiling.
* **Conciseness.** Summary lengths are 7–20 % of the source duration —
  more concise than a typical highlights reel, partly because we cap
  individual clips at the predicted window length without padding.
* **Coherence.** Because we order chronologically and merge near-adjacent
  clips, summaries are coherent at the local level. Global coherence in
  the beekeeping video suffers because the speaker's reasoning is
  cumulative and our pipeline cannot preserve narrative arcs.

## 6. Critical reflections and possible improvements

1. **Speech is the single biggest bottleneck.** Speech/dialogue events
   score 0.000 mean IoU. Adding ASR (Whisper-tiny is 80 MB) would let us
   re-rank moment-retrieval windows by text-overlap with the query.
   This would help videos 33 (graduation speech) and 36 (beekeeper
   monologue) the most.
2. **Saliency could be re-introduced post-hoc.** A useful experiment is
   to weight retrieved-clip selection by an LLM-judge prediction of
   saliency given the event description, hitting a target summary length
   while preserving the most important events.
3. **The CG-DETR ceiling is real.** Our two CLIP variants get
   mean IoU 0.09–0.11. Published numbers for CG-DETR on QVHighlights
   are ~0.40 mIoU. A working CG-DETR integration would likely add
   25–30 IoU points; the missing piece for us is the model checkpoint,
   not the pipeline code.
4. **Annotator-confidence weighting.** We treated both annotators as
   equally reliable, but the saliency-κ disagreement on video 36 shows
   we had different mental models. A more rigorous evaluation would
   weight each annotator's events by their demonstrated agreement on
   other videos (jackknife).

## 7. Conclusion

We built a complete event-detection-to-summary-video pipeline using
zero-shot pretrained models, evaluated it against our own annotations
with three IAA metrics and IoU on two moment-retrieval variants, and
ran failure analysis along category and saliency axes. The most
important finding is that the *annotation step itself* is the largest
source of variance — both between annotators and between the model's
inferred events and ours — and downstream IoU numbers should be read
with that uncertainty in mind. The second-most important finding is
that zero-shot CLIP localisation **fails entirely** on dialogue events
(IoU 0.000 across nine such events), and the practical path to
improving this pipeline runs through ASR rather than through better
visual features.
