"""
Summary video generator

Given a list of (start, end, label) segments and the source video, produce a
single concatenated mp4 ordered chronologically.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _ffmpeg_extract(video: str, out: str, start: float, end: float) -> None:
    duration = max(0.1, end - start)
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(video),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast",
        "-c:a", "aac",
        "-loglevel", "error",
        out,
    ]
    subprocess.run(cmd, check=True)


def _ffmpeg_concat(parts: list[str], out: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        list_path = fh.name
        for p in parts:
            fh.write(f"file '{Path(p).resolve().as_posix()}'\n")
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", "-loglevel", "error", out,
        ]
        subprocess.run(cmd, check=True)
    finally:
        os.unlink(list_path)


def make_summary_ffmpeg(video_path: str, segments: Sequence[tuple[float, float]],
                        out_path: str) -> None:
    segs = sorted(segments, key=lambda s: s[0])
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (s, e) in enumerate(segs):
            p = os.path.join(tmp, f"part_{i:03d}.mp4")
            _ffmpeg_extract(video_path, p, s, e)
            parts.append(p)
        if not parts:
            raise RuntimeError("No segments to concatenate.")
        _ffmpeg_concat(parts, out_path)


# OpenCV fallback (no audio).
def make_summary_opencv(video_path: str,
                        segments: Sequence[tuple[float, float]],
                        out_path: str) -> None:
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for s, e in sorted(segments, key=lambda x: x[0]):
        cap.set(cv2.CAP_PROP_POS_MSEC, s * 1000.0)
        end_ms = e * 1000.0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if cap.get(cv2.CAP_PROP_POS_MSEC) > end_ms:
                break
            writer.write(frame)
    cap.release()
    writer.release()


def make_summary(video_path: str,
                 segments: Sequence[tuple[float, float]],
                 out_path: str) -> str:
    """Concatenate `segments` of `video_path` into `out_path`. Returns out_path."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if _have_ffmpeg():
        make_summary_ffmpeg(video_path, segments, out_path)
    else:
        make_summary_opencv(video_path, segments, out_path)
    return out_path
