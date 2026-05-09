"""
Build a clean PDF presentation deck from the speaker script.

Output: Group9_Presentation.pdf  (16:9 landscape, ~10 slides).
Uses matplotlib's PdfPages so we don't need extra dependencies.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path

# --------------------------------------------------------------------------- #
# Style — high-contrast, readable from the back of a classroom.
# --------------------------------------------------------------------------- #
SLIDE_W, SLIDE_H = 16, 9          # inches at 100 dpi
BG     = "#FFFFFF"
INK    = "#1B2A41"                # near-navy main text
ACCENT = "#E63946"                # red accent for highlights
MUTED  = "#7A8DA1"                # light grey for footers
SOFT   = "#F1F4F8"                # band/box fill

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def new_slide(pdf, title: str, kicker: str | None = None,
              page_no: int | None = None, total: int | None = None):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SLIDE_W); ax.set_ylim(0, SLIDE_H)
    ax.axis("off")

    # Top accent band
    ax.add_patch(Rectangle((0, SLIDE_H - 0.18), SLIDE_W, 0.18,
                           facecolor=ACCENT, edgecolor="none"))
    # Footer
    if page_no is not None:
        ax.text(SLIDE_W - 0.35, 0.25,
                f"{page_no}/{total}",
                ha="right", va="center", fontsize=10, color=MUTED)
    ax.text(0.35, 0.25, "Group 9 — Event Detection & Video Summarization",
            ha="left", va="center", fontsize=10, color=MUTED)

    # Kicker (small label above the title)
    if kicker:
        ax.text(0.5, SLIDE_H - 0.65, kicker.upper(),
                ha="left", va="top", fontsize=12,
                color=ACCENT, weight="bold")
    # Title
    ax.text(0.5, SLIDE_H - 1.05, title,
            ha="left", va="top", fontsize=28, color=INK, weight="bold")
    return fig, ax


def bullet(ax, lines, x=0.6, y_top=6.4, line_h=0.65, fontsize=18):
    for i, t in enumerate(lines):
        ax.text(x, y_top - i * line_h, "•", ha="left", va="top",
                fontsize=fontsize, color=ACCENT, weight="bold")
        ax.text(x + 0.4, y_top - i * line_h, t, ha="left", va="top",
                fontsize=fontsize, color=INK, wrap=True)


def big_quote(ax, text, x=0.6, y=4.5, fontsize=22):
    ax.text(x, y, text, ha="left", va="center", fontsize=fontsize,
            color=INK, weight="medium", wrap=True)


# --------------------------------------------------------------------------- #
# Build slides
# --------------------------------------------------------------------------- #
out = Path("Group9_Presentation.pdf")
TOTAL = 10
with PdfPages(out) as pdf:

    # --- 1: Title --------------------------------------------------------- #
    fig, ax = new_slide(pdf, "From annotation to summary video",
                        kicker="Computer Vision 2026 · Group 9",
                        page_no=1, total=TOTAL)
    ax.text(0.5, 4.8, "Group 9", fontsize=42, color=INK, weight="bold")
    ax.text(0.5, 3.9, "TVSUM videos 33 — 36",
            fontsize=22, color=MUTED)
    ax.text(0.5, 1.3, "We built a system that watches a video, picks the\n"
                       "moments that matter, and stitches them into a short clip.",
            fontsize=16, color=INK)
    pdf.savefig(fig); plt.close(fig)

    # --- 2: Four videos --------------------------------------------------- #
    fig, ax = new_slide(pdf, "Four very different videos",
                        kicker="What we worked with",
                        page_no=2, total=TOTAL)
    cards = [
        ("Video 33", "Graduation flash mob",
         "speech + dance, 7 min, lots of dialogue"),
        ("Video 34", "ICC street flash mob",
         "fast group dance, 2.5 min, no speech"),
        ("Video 35", "Train classical-music flash mob",
         "musicians revealing themselves, 2.5 min"),
        ("Video 36", "Beekeeper interview",
         "instructional voice-over, 4 min, no climax"),
    ]
    box_w, box_h, gap = 6.8, 2.6, 0.5
    for i, (h, t, sub) in enumerate(cards):
        col, row = i % 2, i // 2
        x = 0.6 + col * (box_w + gap)
        y = 4.7 - row * (box_h + gap)
        ax.add_patch(FancyBboxPatch((x, y), box_w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=SOFT, edgecolor="none"))
        ax.text(x + 0.3, y + box_h - 0.5, h, fontsize=14, color=ACCENT,
                weight="bold")
        ax.text(x + 0.3, y + box_h - 1.1, t, fontsize=18, color=INK,
                weight="bold")
        ax.text(x + 0.3, y + 0.7, sub, fontsize=14, color=INK, wrap=True)
    pdf.savefig(fig); plt.close(fig)

    # --- 3: Pipeline ------------------------------------------------------ #
    fig, ax = new_slide(pdf, "Six steps, one notebook",
                        kicker="How the pipeline works",
                        page_no=3, total=TOTAL)
    steps = [
        ("Annotate", "we mark the\nimportant moments"),
        ("Agreement", "did annotator A\nagree with B?"),
        ("VLM events", "AI lists what\nis happening"),
        ("Find times", "search model finds\nstart/end times"),
        ("Score IoU", "how close are\npredictions to truth?"),
        ("Summary", "stitch clips into\none short video"),
    ]
    box_w = 2.4; box_h = 2.6
    total_w = len(steps) * box_w + (len(steps) - 1) * 0.15
    x0 = (SLIDE_W - total_w) / 2
    y0 = 3.2
    for i, (h, sub) in enumerate(steps):
        x = x0 + i * (box_w + 0.15)
        ax.add_patch(FancyBboxPatch((x, y0), box_w, box_h,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    facecolor=SOFT, edgecolor="none"))
        ax.text(x + box_w / 2, y0 + box_h - 0.55, str(i + 1),
                ha="center", fontsize=24, color=ACCENT, weight="bold")
        ax.text(x + box_w / 2, y0 + box_h - 1.25, h,
                ha="center", fontsize=15, color=INK, weight="bold")
        ax.text(x + box_w / 2, y0 + 0.7, sub,
                ha="center", fontsize=11, color=INK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_w + 0.13, y0 + box_h / 2),
                        xytext=(x + box_w + 0.02, y0 + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax.text(SLIDE_W / 2, 1.6, "Built as a single notebook + a small Python package",
            ha="center", fontsize=14, color=MUTED, style="italic")
    pdf.savefig(fig); plt.close(fig)

    # --- 4: IAA ----------------------------------------------------------- #
    fig, ax = new_slide(pdf, "Two annotators rarely fully agree",
                        kicker="Step 1–2 · Annotation agreement",
                        page_no=4, total=TOTAL)
    headers = ["Video", "Same moments?\n(timeline κ)",
               "Same boundaries?\n(mean IoU)",
               "Same importance?\n(saliency κ)"]
    rows = [
        ["33", "0.15", "0.65", "0.60"],
        ["34", "0.48", "0.70", "0.36"],
        ["35", "0.38", "0.55", "0.23"],
        ["36", "0.25", "0.72", "−0.24"],
    ]
    table_x = 1.0; table_y = 5.2; col_w = [2.0, 3.5, 3.5, 3.5]
    # header
    cx = table_x
    for j, h in enumerate(headers):
        ax.add_patch(Rectangle((cx, table_y), col_w[j], 1.0,
                               facecolor=INK, edgecolor="none"))
        ax.text(cx + col_w[j] / 2, table_y + 0.5, h, ha="center", va="center",
                fontsize=12, color="white", weight="bold")
        cx += col_w[j]
    # rows
    for ri, row in enumerate(rows):
        cx = table_x
        y = table_y - (ri + 1) * 0.6
        fill = SOFT if ri % 2 == 0 else "#FFFFFF"
        for j, cell in enumerate(row):
            ax.add_patch(Rectangle((cx, y), col_w[j], 0.6,
                                   facecolor=fill, edgecolor="#E0E5EC"))
            color = ACCENT if (j == 3 and cell.startswith("−")) else INK
            ax.text(cx + col_w[j] / 2, y + 0.3, cell, ha="center", va="center",
                    fontsize=14, color=color)
            cx += col_w[j]

    bullet(ax, [
        "Boundaries agree well (0.55–0.72) — when we both pick a moment, we draw similar windows.",
        "Counts of events disagree more — one of us is a 'splitter', the other a 'joiner'.",
        "Importance is the most subjective — beekeeper video is below chance.",
    ], x=0.7, y_top=2.4, line_h=0.55, fontsize=15)
    pdf.savefig(fig); plt.close(fig)

    # --- 5: VLM event detection ------------------------------------------ #
    fig, ax = new_slide(pdf, "Three ways of asking the same question",
                        kicker="Step 3 · Event detection with SmolVLM2",
                        page_no=5, total=TOTAL)
    cols = [
        ("Just give me events",
         "Most reliable.\n4–5 events parsed\nper video.",
         "✓ best overall"),
        ("Events with timestamps",
         "Almost always fails.\nThe model has no\nidea when things happen.",
         "✗ avoid"),
        ("Caption frames, then merge",
         "Hit-or-miss.\nWorks on the beekeeper\nvideo (covers 42 %).",
         "≈ situational"),
    ]
    cw = 4.7; ch = 4.2; cy = 2.5
    cx0 = (SLIDE_W - 3 * cw - 2 * 0.4) / 2
    for i, (h, body, verdict) in enumerate(cols):
        x = cx0 + i * (cw + 0.4)
        ax.add_patch(FancyBboxPatch((x, cy), cw, ch,
                                    boxstyle="round,pad=0.02,rounding_size=0.2",
                                    facecolor=SOFT, edgecolor="none"))
        ax.text(x + cw / 2, cy + ch - 0.6, h, ha="center", fontsize=17,
                color=ACCENT, weight="bold", wrap=True)
        ax.text(x + cw / 2, cy + ch - 2.2, body, ha="center", fontsize=14,
                color=INK)
        ax.text(x + cw / 2, cy + 0.4, verdict, ha="center", fontsize=14,
                color=INK, weight="bold")
    ax.text(SLIDE_W / 2, 1.4,
            "Big failure: anything that depends on speech is invisible to the model.",
            ha="center", fontsize=15, color=ACCENT, weight="bold")
    pdf.savefig(fig); plt.close(fig)

    # --- 6: Moment retrieval (with chart) ---------------------------------#
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SLIDE_W); ax.set_ylim(0, SLIDE_H); ax.axis("off")
    ax.add_patch(Rectangle((0, SLIDE_H - 0.18), SLIDE_W, 0.18,
                           facecolor=ACCENT, edgecolor="none"))
    ax.text(0.35, 0.25, "Group 9 — Event Detection & Video Summarization",
            ha="left", va="center", fontsize=10, color=MUTED)
    ax.text(SLIDE_W - 0.35, 0.25, f"6/{TOTAL}",
            ha="right", va="center", fontsize=10, color=MUTED)
    ax.text(0.5, SLIDE_H - 0.65, "STEP 4 · FINDING WHEN EACH EVENT HAPPENS",
            ha="left", va="top", fontsize=12, color=ACCENT, weight="bold")
    ax.text(0.5, SLIDE_H - 1.05, "A bigger model is the cheapest improvement",
            ha="left", va="top", fontsize=28, color=INK, weight="bold")

    # Inset bar chart
    ax2 = fig.add_axes([0.075, 0.18, 0.45, 0.55])
    methods = ["CLIP-B/32", "CLIP-B/32\n+ rephrase", "CLIP-L/14", "CLIP-L/14\n+ rephrase"]
    miou = [0.089, 0.082, 0.113, 0.107]
    r03  = [0.140, 0.105, 0.211, 0.193]
    x_pos = range(len(methods))
    bw = 0.35
    bars1 = ax2.bar([p - bw/2 for p in x_pos], miou, bw, label="Mean overlap",
                    color=ACCENT, edgecolor="none")
    bars2 = ax2.bar([p + bw/2 for p in x_pos], r03, bw, label="Hit rate (overlap ≥ 0.3)",
                    color=INK, edgecolor="none")
    for b, v in zip(bars1, miou):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                 ha="center", fontsize=10, color=INK)
    for b, v in zip(bars2, r03):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}",
                 ha="center", fontsize=10, color=INK)
    ax2.set_xticks(list(x_pos)); ax2.set_xticklabels(methods, fontsize=11)
    ax2.set_ylim(0, 0.28)
    ax2.set_yticks([0, 0.1, 0.2])
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)
    ax2.legend(loc="upper left", frameon=False, fontsize=11)
    ax2.set_title("Higher is better", color=MUTED, fontsize=11, loc="left")

    bullet(ax, [
        "Bigger CLIP wins by 27 % overlap with no other change.",
        "Rephrasing the query 3× actually hurt — averaging blurs the\nstrongest signal.",
        "By event type:  performance 0.17,  beekeeping 0.07,  speech 0.00",
    ], x=8.6, y_top=6.4, line_h=0.85, fontsize=15)
    pdf.savefig(fig); plt.close(fig)

    # --- 7: Complete example --------------------------------------------- #
    fig, ax = new_slide(pdf, "Walk-through: the train orchestra",
                        kicker="A complete example · Video 35",
                        page_no=7, total=TOTAL)
    # left: GT events
    ax.text(0.5, 6.8, "What we annotated  (13 events)",
            fontsize=15, color=INK, weight="bold")
    gt = [
        "00:17 — Sign reads 'Classical special train'",
        "00:39 — A flutist begins playing",
        "00:48 — A man on the oboe joins",
        "01:00 — A man with a violin stands up",
        "01:24 — Full orchestra plays, crowd reacts",
        "01:41 — Conductor stands and conducts",
        "02:05 — Train pulls in, performance ends",
    ]
    for i, t in enumerate(gt):
        ax.text(0.7, 6.2 - i * 0.45, "•", color=ACCENT, fontsize=14, weight="bold")
        ax.text(0.95, 6.2 - i * 0.45, t, fontsize=13, color=INK)

    # right: VLM + retrieval
    ax.text(8.5, 6.8, "What our system did", fontsize=15, color=INK, weight="bold")
    ax.text(8.7, 6.2, "VLM listed 4 salient events:", fontsize=13, color=INK)
    vlm = [
        "1.  A person plays a flute on a train",
        "2.  A group plays violins together",
        "3.  Passengers watch and record",
        "4.  Train arrives at the station",
    ]
    for i, t in enumerate(vlm):
        ax.text(8.9, 5.7 - i * 0.4, t, fontsize=12, color=INK)
    ax.text(8.7, 3.7,
            "CLIP-L/14 located these 4 events:",
            fontsize=13, color=INK)
    ax.text(8.9, 3.2, "Mean overlap: 0.333  (best of all 4 videos)",
            fontsize=14, color=ACCENT, weight="bold")
    ax.text(8.9, 2.8, "Hit rate at overlap ≥ 0.3: 75 %",
            fontsize=14, color=ACCENT, weight="bold")
    ax.text(8.9, 2.3, "Generated summary: 28 s, 8 clips, in order",
            fontsize=13, color=INK)
    ax.text(8.9, 1.85, "Misses: conductor, closing text overlay",
            fontsize=12, color=MUTED, style="italic")
    pdf.savefig(fig); plt.close(fig)

    # --- 8: Findings ----------------------------------------------------- #
    fig, ax = new_slide(pdf, "Three things we learned",
                        kicker="Findings",
                        page_no=8, total=TOTAL)
    findings = [
        ("Subjectivity matters more than model choice.",
         "Two humans annotating the same video already disagree about\nwhat is essential. Any AI will reflect *one* of those views."),
        ("Vision-only models are speech-blind.",
         "Performances and dance: works.\nSpeeches, jokes, narration: zero overlap. Need ASR to fix."),
        ("Bigger model = cheapest win.",
         "Swapping CLIP-B for CLIP-L gave +27 % overlap with no other\nchange. Compute is cheaper than engineering effort."),
    ]
    y_top = 6.5
    for i, (h, body) in enumerate(findings):
        y = y_top - i * 1.7
        ax.text(0.6, y, f"{i+1}.", fontsize=22, color=ACCENT, weight="bold")
        ax.text(1.3, y, h, fontsize=20, color=INK, weight="bold")
        ax.text(1.3, y - 0.7, body, fontsize=14, color=INK)
    pdf.savefig(fig); plt.close(fig)

    # --- 9: Future work -------------------------------------------------- #
    fig, ax = new_slide(pdf, "What we'd do with another week",
                        kicker="Next steps",
                        page_no=9, total=TOTAL)
    todos = [
        ("Add speech recognition (Whisper)",
         "Biggest single win: dialogue events currently\nscore zero overlap; ASR makes them retrievable."),
        ("Re-rank clips by importance",
         "Use saliency ratings as weights so the summary\nfavours essential moments over visually flashy ones."),
        ("Bring back CG-DETR",
         "The model the assignment originally suggested.\nIts pretrained checkpoint is currently offline,\nbut it would beat zero-shot CLIP by ~25 IoU points."),
    ]
    for i, (h, body) in enumerate(todos):
        y = 6.6 - i * 1.7
        ax.add_patch(Rectangle((0.5, y - 0.85), 0.15, 1.3, facecolor=ACCENT,
                               edgecolor="none"))
        ax.text(0.85, y, h, fontsize=18, color=INK, weight="bold")
        ax.text(0.85, y - 0.7, body, fontsize=14, color=INK)
    pdf.savefig(fig); plt.close(fig)

    # --- 10: Thank you --------------------------------------------------- #
    fig, ax = new_slide(pdf, "Thank you",
                        kicker="Questions?",
                        page_no=10, total=TOTAL)
    ax.text(SLIDE_W / 2, 5.0, "Group 9 · TVSUM 33–36",
            ha="center", fontsize=26, color=INK, weight="bold")
    ax.text(SLIDE_W / 2, 4.1,
            "Annotation → VLM → Retrieval → Summary",
            ha="center", fontsize=18, color=MUTED)
    ax.text(SLIDE_W / 2, 2.6,
            "Headline: subjectivity is the dominant signal,\nspeech is the bottleneck, bigger models help.",
            ha="center", fontsize=15, color=INK, style="italic")
    pdf.savefig(fig); plt.close(fig)

print(f"Wrote {out.resolve()}")
