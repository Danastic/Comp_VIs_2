# Group 9 — Presentation script (≈ 6 minutes)

Two presenters: **A** and **B**. Read aloud at a normal pace
(~150 words/minute). Lines in *italics* are stage directions, not
spoken. Time budget per slide is shown in **[brackets]**.

The full delivery comes to **about 6 minutes**, leaving a 1-minute
buffer for a slow opening or extra Q&A.

---

## Slide 1 — Title  **[15 s · A]**

> Hi, we're Group 9. Today we'll show how we built a system that
> watches a video, picks out the moments that matter, and stitches
> them into a short summary clip — the way a movie trailer
> summarises a film.

*(A nods to B.)*

---

## Slide 2 — Four very different videos  **[30 s · A]**

> We worked with four short videos from the TVSUM dataset.
> A graduation flash mob with a long speech.
> A street flash mob in India during a cricket world cup.
> A flash-mob orchestra on a metro train.
> And a beekeeper interview with a long voice-over.
>
> We picked these on purpose — they look very different from each
> other. That way we can see where our system holds up and where it
> falls apart.

---

## Slide 3 — Six steps, one notebook  **[35 s · A]**

> Our pipeline has six steps.
>
> *(point at each box as you go)*
>
> One: **annotate**. We mark the important moments by hand.
> Two: **agreement**. We measure how much the two of us agreed.
> Three: a **vision-language model** lists what's happening in
> the video.
> Four: a **search model** uses those descriptions to find the
> exact start and end times.
> Five: we **score** how close the predictions are to our
> annotations.
> And six: we **stitch the predicted clips** into one summary video.
>
> All of this lives in a single notebook plus a small Python
> package. Over to you, **B**.

---

## Slide 4 — Two annotators rarely fully agree  **[50 s · B]**

> Before trusting any model, we wanted to know how much *we*
> agreed with each other.
>
> We measured three things.
> First, **did we mark the same moments as eventful?** Numbers
> ranged from 0.15 to 0.48 — moderate. We picked broadly the same
> parts of each video, but we disagreed on how many separate events
> to draw.
>
> Second, **when we did agree on an event, did we draw the same
> start and end times?** Here we did much better — between 0.55 and
> 0.72 overlap. So we tend to draw similar windows, we just don't
> always agree on how to slice them up.
>
> Third, **did we agree on which events were essential?** This is
> where it falls apart. The beekeeper video is below chance —
> meaning we *disagreed more than random.* One of us prioritised
> the spoken explanations, the other prioritised the visual
> demonstrations. Both are valid, and that's the whole point of
> subjectivity.

---

## Slide 5 — Three ways of asking the same question  **[50 s · B]**

> For Step 3 we used a small vision-language model called
> **SmolVLM2** — basically an AI that reads video frames and
> writes about them.
>
> We tried three different ways of asking it the same question.
>
> *(point to first card)*  **"Just give me the events"** — works
> best. Four to five short events per video, parsed cleanly.
>
> *(point to second card)*  **"Events with timestamps"** — almost
> always fails. The model isn't trained to know *when* things
> happen, so it either refuses or returns one window covering
> everything.
>
> *(point to third card)*  **"Caption every frame, then merge"** —
> hit-or-miss. Works on the beekeeper video, where it covers 42 %
> of our annotated events; doesn't work elsewhere.
>
> The big failure across all three: **anything that depends on
> speech is invisible to a vision model.** A spoken joke turns
> into "a man stands at a microphone".

---

## Slide 6 — A bigger model is the cheapest improvement  **[55 s · A]**

> Now Step 4 — finding *when* each event happens. We tried two
> versions of the same approach, using a model called CLIP that
> matches text to images.
>
> *(point to chart)*
>
> The smaller, faster CLIP got an overlap score of 0.09. The
> larger, slower CLIP got 0.11 — a 27 % improvement just from a
> bigger backbone. The bigger one also finds the right rough
> region 21 % of the time, vs 14 % for the smaller one.
>
> We also tried something the assignment suggests: rephrasing
> each query three different ways and combining the results.
> Surprisingly, **it didn't help** — and sometimes made things
> slightly worse. We think averaging three rephrased queries
> blurs the strongest signal in the original.
>
> Breaking it down by what kind of event: performance scenes get
> 0.17 overlap, the beekeeper gets 0.07, and speech is **zero**.

---

## Slide 7 — Walk-through: the train orchestra  **[75 s · B]**

> Let's walk through one full example. This is video 35 — the
> classical-music flash mob on a metro train.
>
> *(point to left column)*
>
> We annotated 13 key moments — from the announcement on the
> screen, through each musician revealing themselves, to the
> climax with the full orchestra.
>
> *(point to right column)*
>
> Our AI listed four salient events: a flutist, a group of
> violins, passengers reacting, and the train arriving.
>
> Our search model — CLIP-Large — then located those four events.
> On this video it scored its best result of all four:
> **0.33 mean overlap, finding the right region 75 % of the time.**
>
> Let's watch the summary it produced.
>
> *(play `outputs/summaries/video_35_summary.mp4`,
>   28 seconds, 8 clips)*
>
> What it gets right: the orchestra reveal, the violins joining
> in, the climax. What it misses: the conductor — that's a single
> person standing up briefly — and the closing text overlay
> that reveals it's an advertisement, because the model can't
> read text.

---

## Slide 8 — Three things we learned  **[40 s · A]**

> Three takeaways.
>
> One. **Subjectivity matters more than the choice of model.**
> Two humans annotating the same video already disagree on what
> matters. An AI working from one of those views will reflect
> *that* bias.
>
> Two. **Off-the-shelf vision models work well on what they can
> see, and fail on what they can only hear.** Performances and
> dance: good. Speeches and jokes: zero overlap. There's no
> shortcut around this — we need speech recognition.
>
> Three. **A bigger model is the cheapest single improvement.**
> Just swapping the small CLIP for a larger one gave us a 27 %
> gain with no other changes.

---

## Slide 9 — What we'd do with another week  **[25 s · B]**

> If we had more time, we'd do three things.
>
> One: add **speech recognition**, so dialogue events stop being
> invisible. This is the single biggest win available.
>
> Two: re-rank clips by **importance**, not just relevance, so
> the summary respects our saliency ratings.
>
> Three: bring back **CG-DETR**, the model the assignment
> originally suggested. Its checkpoint is currently offline, but
> if it returns, that's our second-method comparison done
> properly.

---

## Slide 10 — Thank you  **[5 s · A]**

> Thank you. Happy to take questions.

---

## Word count and timing

| Slide | Words spoken | Target time |
|---|---:|---:|
| 1 | 35 | 0:15 |
| 2 | 70 | 0:30 |
| 3 | 95 | 0:35 |
| 4 | 145 | 0:50 |
| 5 | 145 | 0:50 |
| 6 | 145 | 0:55 |
| 7 | 175 | 1:15 (incl. clip) |
| 8 | 105 | 0:40 |
| 9 | 65 | 0:25 |
| 10 | 7 | 0:05 |
| **Total** | **~990** | **~6:00** |

If you want to land at 7 minutes, slow down the delivery on slides
4, 5, 6 and pause for 2–3 seconds between bullets — these are
where the audience needs the most processing time. If you want to
land at 5 minutes, cut the second example in slide 5
("doesn't work elsewhere") and trim slide 9 to one bullet.

---

## Q&A cheat-sheet

**Why CLIP and not CG-DETR (the demo's recommended model)?**
> "We tried — its official checkpoint isn't downloadable anymore.
> The lighthouse repo's download script returns 404 and the
> author's HuggingFace account doesn't host it. We used two CLIP
> backbones to still have a model-size comparison."

**Why is your overlap so low (~0.11)?**
> "Zero-shot CLIP has no temporal training. Published CG-DETR
> trained on QVHighlights gets ~0.40 — about 4× better — because
> it learns the localisation task end-to-end."

**Why does paraphrase fusion *hurt*?**
> "Our hypothesis: averaging text embeddings of three rephrasings
> blurs the most discriminative direction in embedding space.
> CG-DETR-style models that handle multiple queries internally
> probably profit from the same input."

**How did you choose the 1.5 s minimum clip length?**
> "Empirical — clips shorter than that never contained enough of
> the event to be useful, and they made the summaries feel
> jittery. Anything between 1 s and 2 s worked similarly; we
> settled on 1.5 s."

**How did you handle the long graduation video (video 33, 7 min)?**
> "CLIP is per-frame, so video length doesn't matter for our
> retrieval method. We sample at 1 Hz regardless of duration."

---

## Glossary (for if the audience asks)

- **Overlap / IoU**: how well a predicted time-window matches a
  true one. 1.0 is identical, 0.0 means no overlap.
- **Cohen's Kappa (κ)**: agreement between two raters, corrected
  for chance. 1.0 = perfect, 0.0 = random, negative = worse than
  random.
- **CLIP**: a model that learns to match text descriptions to
  images.
- **VLM**: a vision-language model — takes images and text,
  produces text.
- **Zero-shot**: using a pretrained model on a new task without
  retraining it.
