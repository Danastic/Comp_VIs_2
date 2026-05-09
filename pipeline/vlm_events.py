"""
VLM-based event detection .

Uses HuggingFaceTB/SmolVLM2-2.2B-Instruct (default) to read a video and
return an ordered list of salient events.

Two output modes are supported :

  - mode="events":   text list of "Event 1: ...\nEvent 2: ..." (no times).
  - mode="timed":    list of "Event 1 [MM:SS-MM:SS]: ..." style strings.

We also ship a tiny "frame caption + LLM aggregation" fallback strategy
implemented purely with the VLM (no separate LLM): we sample N frames, ask
SmolVLM2 to describe each frame, then ask SmolVLM2 to summarise the
descriptions into a numbered event list. This was suggested by the assignment
demo notebook and lets us compare prompting strategies in the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# Output dataclasses
@dataclass
class VLMEvent:
    description: str
    start: float | None = None
    end: float | None = None


# Prompts
PROMPT_EVENTS = (
    "These frames are sampled in chronological order from a short video. "
    "Retrieve all of the salient events shown. Reply with a numbered list, "
    "one event per line, starting with 'Salient event 1:'. "
    "Each line must be one short sentence. Do not add commentary."
)

PROMPT_EVENTS_TIMED = (
    "These frames are sampled in chronological order from a short video. "
    "Retrieve all of the salient events shown, with approximate timestamps "
    "in MM:SS-MM:SS format. Reply with a numbered list. Each line must be "
    "exactly: 'Salient event N [MM:SS-MM:SS]: short description'. "
    "Do not add commentary."
)

PROMPT_FRAME_CAPTION = (
    "Describe what is happening in this frame in one short sentence. "
    "Focus on actions and key visual elements."
)

PROMPT_AGGREGATE = (
    "Below are short captions describing frames sampled from a video, in "
    "chronological order. Combine them into a list of salient events that "
    "summarise the video. Merge captions describing the same event. Use the "
    "format 'Salient event N: <one-sentence description>'. Be concise."
)


# Parsers
_EVENT_LINE_RE = re.compile(
    r"(?im)^\s*(?:salient\s+event\s*\d+|event\s*\d+|\d+|[-*•])"
    r"(?:\s*\[(\d{1,2}[:.]\d{2})\s*[-–]\s*(\d{1,2}[:.]\d{2})\])?"
    r"\s*[:\.\)\-]?\s*(.+?)\s*$"
)


def _t(s: str) -> float:
    s = s.strip().replace(".", ":")
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def parse_events(text: str) -> list[VLMEvent]:
    """Extract events from the VLM's free-form output.

    Four parsing levels, in order:
      1. Line-by-line numbered/bulleted list (covers well-formatted output).
      2. Inline 'Salient event N/digits' markers anywhere in the text.
      3. Sentence split on '.' / ';' followed by an uppercase word.
      4. Semicolon split for single-line prose dumps (e.g. "Salient Event N: a; b; c").
    """
    events: list[VLMEvent] = []
    for line in text.splitlines():
        m = _EVENT_LINE_RE.match(line)
        if m:
            t0, t1, desc = m.group(1), m.group(2), m.group(3).strip(" .")
            if not desc:
                continue
            ev = VLMEvent(description=desc)
            if t0 and t1:
                ev.start, ev.end = _t(t0), _t(t1)
            events.append(ev)
    if events:
        return events

    # 2: split on any "Salient event N" marker 
    chunks = re.split(r"(?i)salient\s+event\s*[\w\d]+\s*[:\-\.]?", text)
    if len(chunks) > 1:
        for c in chunks[1:]:
            c = c.strip().split("\n")[0].strip(" .,-")
            if c:
                events.append(VLMEvent(description=c))
        if events:
            return events

    #3: sentence split on '. ' or '; ' before an uppercase letter.
    cleaned = re.sub(r"\s+", " ", text).strip()
    for sent in re.split(r"(?<=[.;])\s+(?=[A-Z])", cleaned):
        sent = sent.strip(" .,-;\"'")
        if 8 <= len(sent) <= 250:
            events.append(VLMEvent(description=sent))
    if events:
        return events

    #4: semicolon split for single long-line prose dumps.
    body = re.sub(r"(?i)^salient\s+event\s*[\w\d]*\s*[:\-\.]?\s*", "", cleaned)
    for chunk in body.split(";"):
        chunk = chunk.strip(" .,-\"'")
        if 8 <= len(chunk) <= 250:
            events.append(VLMEvent(description=chunk))
    return events


# Frame sampling (uniform, deterministic)
def sample_frames_uniform(video_path: str, num_frames: int = 12):
    """
    Uniformly sample `num_frames` frames from a video, returning PIL Images
    plus the corresponding timestamps in seconds.
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0:
        cap.release()
        raise RuntimeError(f"No frames in {video_path}")
    indices = [int(round(i * (total - 1) / max(1, num_frames - 1)))
               for i in range(num_frames)]
    frames, timestamps = [], []
    target = set(indices)
    cur = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in target:
            import cv2 as _cv2
            frame = _cv2.cvtColor(frame, _cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            timestamps.append(cur / fps)
        cur += 1
    cap.release()
    return frames, timestamps


# SmolVLM2 wrapper
class SmolVLM2:
    """Thin wrapper around HuggingFaceTB/SmolVLM2-2.2B-Instruct."""

    def __init__(self,
                 model_path: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                 device: str | None = None):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).to(device)
        self.model.eval()

    # ------------------------------------------------------------------- #
    def _generate(self, messages, max_new_tokens: int = 384) -> str:
        import torch
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype if self.device == "cuda" else None)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
            )
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        # SmolVLM2 echoes the prompt; keep only what comes after the last
        # "Assistant:" marker (used in its chat template).
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1]
        return text.strip()

    # ------------------------------------------------------------------- #
    def detect_events_video(self, video_path: str,
                            num_frames: int = 16,
                            timed: bool = False) -> tuple[str, list[VLMEvent]]:
        """Direct video -> events.

        We sample frames with OpenCV and send them as images. This bypasses
        SmolVLM2's torchvision_io/PyAV video loader, which intermittently
        fails on Windows with `swscaler: Failed initializing scaling graph`.
        Functionally equivalent to the native video path on the demo
        notebook's videos.
        """
        prompt = PROMPT_EVENTS_TIMED if timed else PROMPT_EVENTS
        frames, timestamps = sample_frames_uniform(video_path, num_frames)
        if timed:
            preface = (
                "These frames were sampled in chronological order from a "
                "single video at the following timestamps (seconds): "
                + ", ".join(f"{t:.1f}" for t in timestamps)
                + ".\n"
            )
            text_prompt = preface + prompt
        else:
            text_prompt = prompt

        content = [{"type": "image", "url": fr} for fr in frames]
        content.append({"type": "text", "text": text_prompt})
        messages = [{"role": "user", "content": content}]
        raw = self._generate(messages, max_new_tokens=512)
        return raw, parse_events(raw)

    # ------------------------------------------------------------------- #
    def detect_events_caption_aggregate(self, video_path: str,
                                        num_frames: int = 12
                                        ) -> tuple[str, list[VLMEvent], list[str]]:
        """Frame caption -> aggregate strategy."""
        frames, timestamps = sample_frames_uniform(video_path, num_frames)
        captions = []
        for fr in frames:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "url": fr},
                    {"type": "text", "text": PROMPT_FRAME_CAPTION},
                ],
            }]
            captions.append(self._generate(messages, max_new_tokens=64))

        joined = "\n".join(
            f"[{int(t)//60:02d}:{int(t)%60:02d}] {c}"
            for t, c in zip(timestamps, captions)
        )
        agg_prompt = PROMPT_AGGREGATE + "\n\nCaptions:\n" + joined
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": agg_prompt}],
        }]
        raw = self._generate(messages, max_new_tokens=512)
        return raw, parse_events(raw), captions
