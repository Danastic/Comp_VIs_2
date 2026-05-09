# Group 9 — Presentation script (5–7 minutes)

Two presenters: **A** and **B**. Each slide shows: time, who's
speaking, what's on screen, and what to say. Plain English, no jargon
unless we explain it on the spot.

---

## Slide 1 — Title (15 s, A)
*On screen:* Group name, the four video thumbnails.

> "Hi, we're Group 9. Today we'll show how we built a system that
> watches a video, picks out the most important moments, and stitches
> them into a short summary clip — the way a movie trailer summarises a
> film."

---

## Slide 2 — The four videos we worked with (30 s, A)
*On screen:* one keyframe per video with a one-line caption.

> "We were given four short videos from the TVSUM dataset:
> a graduation flash mob, a street flash mob in India, a flash-mob
> orchestra on a metro train, and an interview with a beekeeper.
> We picked these on purpose — they look very different from each
> other, so we can see where our system holds up and where it breaks."

---

## Slide 3 — How the pipeline works (45 s, A)
*On screen:* simple block diagram with six boxes connected by arrows.

> "Our system has six steps. First, we **annotate** the videos by hand
> — we mark which moments matter and how much. Then we **measure how
> much we agreed with each other** — because two people will pick
> different moments. Step three: a **vision-language model watches the
> video and lists what it thinks is happening**. Step four: a
> **search model uses those descriptions to find the exact start and
> end times** in the video. Step five: we **score how close those
> predictions are to our annotations**. And finally we **cut and
> stitch the predicted clips into one summary video**."

---

## Slide 4 — How much did we agree on annotations? (60 s, B)
*On screen:* a small table with three numbers per video.

> "Before trusting any model, we wanted to know how much *we* agreed
> with each other. We measured three things:
>
> 1. **Did we mark the same moments as eventful?** — using a standard
>    agreement score called Cohen's Kappa. Numbers ranged from 0.15 to
>    0.48 — moderate. We picked broadly the same parts of each video,
>    but disagreed on how many separate events to mark.
> 2. **When we did agree on an event, did we draw the same start–end
>    window?** — measured by overlap, average **0.55 to 0.72**. Pretty
>    high. So when we picked the same moment, we usually drew similar
>    boundaries.
> 3. **Did we agree on which events were essential vs nice-to-have?** —
>    very low, sometimes negative. The beekeeper video is the worst:
>    one of us thought the *spoken explanations* mattered most, the
>    other thought the *visual demonstrations* mattered most. Both are
>    valid — that's the whole point of subjectivity."

---

## Slide 5 — Step 3: asking the AI to describe the video (60 s, B)
*On screen:* same video, three columns showing the model's outputs
under three different prompts.

> "We used a small vision-language model called **SmolVLM2** —
> basically an AI that can look at video frames and write about them.
> We tried three different ways of asking it the same question:
>
> - **Just give me the events**: works best, gives us four to five
>   short events per video.
> - **Give me events with timestamps**: it almost always refuses, or
>   gives a single window covering the whole video. The model just
>   isn't trained to know *when* things happen.
> - **Describe each frame, then summarise**: works on the beekeeper
>   video — covers 42 % of the events we annotated — but is hit-or-miss
>   on the others.
>
> The biggest failure: **anything that depends on dialogue is invisible
> to the model.** It can't hear the speech, so events like 'the speaker
> tells a joke' get reduced to 'a man stands at a microphone'."

---

## Slide 6 — Step 4: finding when each event happens (60 s, A)
*On screen:* bar chart comparing the two methods.

> "Once we have a description of an event, we need to find *where* in
> the video it happens. We tried two versions of the same approach,
> using a model called **CLIP** that matches text to images.
>
> The smaller, faster version (**CLIP-B/32**) got an overlap score of
> 0.089. The larger, slower version (**CLIP-L/14**) got **0.113** — a
> 27 % improvement just from using a bigger model. The bigger one also
> finds the right rough region 21 % of the time, vs 14 % for the small
> one.
>
> We also tried something the assignment suggests: rephrasing each
> query three different ways and combining the results. Surprisingly,
> **it didn't help** — and sometimes made things slightly worse. We
> think this is because averaging three rephrased text embeddings
> blurs the strongest signal in the original query."

---

## Slide 7 — A complete example: the train orchestra (90 s, B)
*On screen:* video 35 plus annotations and predictions side-by-side.
Play the 28-second summary clip.

> "Let's walk through one full example. This is video 35 — a flash-mob
> orchestra on a metro train. We annotated 13 key events.
>
> Our AI listed four salient events from this video. We then asked our
> search model to find each one. On video 35 it scored its best result
> of all four videos — overlap of 0.333, finding the right region
> three quarters of the time.
>
> Here's the summary it produced — 28 seconds, 8 clips, in
> chronological order. *(play clip)*
>
> What it gets right: the orchestra reveal, the violins joining,
> the climax. What it misses: the conductor — because that's a
> single person standing up briefly — and the closing text overlay
> revealing it's an advertisement, because the model can't read text."

---

## Slide 8 — What we learned (45 s, A)
*On screen:* three bullet points, big text.

> "Three takeaways.
>
> First, **subjectivity shapes the summary more than the model does.**
> Two humans annotating the same video already disagree on what
> matters; an AI working from one of those views will reflect *that*
> bias.
>
> Second, **off-the-shelf vision models work well on things they can
> see and fail on things they can only hear.** Performances and dance:
> good. Speeches and jokes: zero overlap. There's no shortcut around
> this — you need to add speech recognition.
>
> Third, **a bigger model is the cheapest improvement.** Just swapping
> CLIP-B for CLIP-L gave us a 27 % gain with no other changes."

---

## Slide 9 — What we'd do next (30 s, B)
*On screen:* three bullets.

> "If we had another week, we'd:
>
> 1. **Add speech recognition** (Whisper) so dialogue events stop
>    being invisible. This is the biggest single win available.
> 2. **Re-rank clips by importance**, not just relevance, so the
>    summary respects our saliency ratings.
> 3. **Bring back CG-DETR**, the model the assignment originally
>    recommended — its checkpoint is currently unavailable online,
>    but if it returns, that's our second-method comparison done
>    properly."

---

## Slide 10 — Thank you / Questions (rest of time)
*On screen:* the full pipeline diagram with our headline numbers
overlaid.

> "Thank you. Happy to take questions about any of the steps."

---

## Live-demo notes (for the presenters)

- Have `outputs/summaries/video_35_summary.mp4` ready to play in a
  full-screen window before the talk starts.
- During slide 7, before playing the clip, **read aloud the four VLM
  event descriptions** so the audience can match them to the cuts in
  the summary.
- If asked "why CLIP and not CG-DETR?" — answer briefly: "the official
  checkpoint isn't downloadable anymore; we used two CLIP backbones to
  still have a model-size comparison."
- If asked "why is overlap so low (~0.11)?" — answer: "zero-shot CLIP
  has no temporal training; the published CG-DETR result is 0.40
  because it was trained for this task."

## Glossary (in case the audience asks)

- **IoU / overlap**: how well a predicted time-window matches a true
  one — 1.0 is identical, 0.0 means no overlap.
- **Cohen's Kappa**: how much two people agree, *correcting for chance*.
  0.0 means random, 1.0 means perfect, negative means worse than random.
- **CLIP**: a model that learns to match text descriptions to images.
- **VLM** (vision-language model): an AI that takes images plus text
  and produces text.
- **Zero-shot**: using a pretrained model on a new task without
  retraining it.
