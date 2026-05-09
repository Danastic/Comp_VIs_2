"""
Render Presentation_script.md as a polished printable PDF.

Output: Group9_Presentation_script.pdf  (A4 portrait, ~6 pages).

Uses ReportLab Platypus (flowable layout) — handles auto-pagination,
headers, blockquotes, lists, tables and inline formatting.
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)


# --------------------------------------------------------------------------- #
# Colour palette — match the slide deck.
# --------------------------------------------------------------------------- #
INK    = colors.HexColor("#1B2A41")
ACCENT = colors.HexColor("#E63946")
MUTED  = colors.HexColor("#7A8DA1")
SOFT   = colors.HexColor("#F1F4F8")


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                    fontName="Helvetica-Bold", fontSize=20, leading=24,
                    textColor=INK, spaceBefore=4, spaceAfter=10)
H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                    fontName="Helvetica-Bold", fontSize=14, leading=18,
                    textColor=ACCENT, spaceBefore=18, spaceAfter=6,
                    keepWithNext=True)
H3 = ParagraphStyle("H3", parent=styles["Heading3"],
                    fontName="Helvetica-Bold", fontSize=11, leading=14,
                    textColor=INK, spaceBefore=10, spaceAfter=4,
                    keepWithNext=True)
BODY = ParagraphStyle("Body", parent=styles["Normal"],
                      fontName="Helvetica", fontSize=10.5, leading=14,
                      textColor=INK, spaceAfter=6)
QUOTE = ParagraphStyle("Quote", parent=BODY,
                       fontSize=11.5, leading=15.5,
                       textColor=INK, leftIndent=18, rightIndent=8,
                       spaceBefore=2, spaceAfter=8,
                       borderPadding=2,
                       borderWidth=0,
                       fontName="Helvetica")
STAGE = ParagraphStyle("Stage", parent=BODY,
                       fontName="Helvetica-Oblique", fontSize=9.5,
                       textColor=MUTED, leftIndent=18, spaceBefore=0,
                       spaceAfter=4)
BULLET = ParagraphStyle("Bullet", parent=BODY,
                        leftIndent=14, bulletIndent=2)
CODE = ParagraphStyle("Code", parent=BODY,
                      fontName="Courier", fontSize=9.5)


# --------------------------------------------------------------------------- #
# Header / footer drawing
# --------------------------------------------------------------------------- #
def _on_page(canvas, doc):
    canvas.saveState()
    page_w, page_h = A4
    # Top accent stripe
    canvas.setFillColor(ACCENT)
    canvas.rect(0, page_h - 0.55*cm, page_w, 0.55*cm, stroke=0, fill=1)
    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2*cm, 1.2*cm,
                      "Group 9 — Event Detection & Video Summarization")
    canvas.drawRightString(page_w - 2*cm, 1.2*cm, f"Page {doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Inline formatting: convert minimal markdown to ReportLab's tag subset.
# --------------------------------------------------------------------------- #
def _inline(text: str) -> str:
    # Escape the few HTML-ish characters ReportLab cares about, BUT not the
    # tags we will emit ourselves. We do this by escaping first, then
    # transforming markdown.
    s = (text.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))
    # **bold**, then *italic* (allowing _italic_ too).
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", s)
    s = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", s)
    # `code`
    s = re.sub(r"`([^`]+?)`", r'<font face="Courier" size="9.5">\1</font>', s)
    return s


# --------------------------------------------------------------------------- #
# Markdown → flowables. Hand-rolled because pandoc isn't installed and we only
# need to render the small subset our script actually uses.
# --------------------------------------------------------------------------- #
def md_to_flowables(md: str):
    flow = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.rstrip()

        # Blank line
        if not stripped.strip():
            i += 1
            continue

        # Horizontal rule
        if stripped.strip() in ("---", "***"):
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", thickness=0.5,
                                   color=MUTED, spaceBefore=0, spaceAfter=8))
            i += 1
            continue

        # H1 / H2 / H3
        if stripped.startswith("# "):
            flow.append(Paragraph(_inline(stripped[2:].strip()), H1))
            i += 1; continue
        if stripped.startswith("## "):
            flow.append(Paragraph(_inline(stripped[3:].strip()), H2))
            i += 1; continue
        if stripped.startswith("### "):
            flow.append(Paragraph(_inline(stripped[4:].strip()), H3))
            i += 1; continue

        # Blockquote — collect contiguous '> ' lines into one paragraph
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                buf.append(lines[i].lstrip()[1:].strip())
                i += 1
            text = " ".join(b for b in buf if b)
            flow.append(Paragraph(_inline(text), QUOTE))
            continue

        # Stage directions in italics on their own line: *(...)*
        m = re.match(r"^\*\((.+)\)\*\s*$", stripped)
        if m:
            flow.append(Paragraph(f"<i>({m.group(1)})</i>", STAGE))
            i += 1; continue

        # Table — leading "|" followed by another "|" line later; collect.
        if stripped.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            # Parse header / separator / rows
            rows = [[c.strip() for c in row.strip().strip("|").split("|")]
                    for row in tbl_lines]
            # Drop the separator row (---|---|---)
            cleaned = [r for r in rows
                       if not all(re.fullmatch(r":?-+:?", c.strip())
                                  for c in r if c.strip())]
            # Header (row 0) uses a white-text style so it shows on the
            # dark navy header background; body rows use the regular ink style.
            HEADER_CELL = ParagraphStyle(
                "HeaderCell", parent=BODY,
                fontName="Helvetica-Bold",
                textColor=colors.white,
            )
            data = []
            for r_idx, row in enumerate(cleaned):
                style_for_row = HEADER_CELL if r_idx == 0 else BODY
                data.append([Paragraph(_inline(cell), style_for_row)
                             for cell in row])
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SOFT, colors.white]),
                ("BOX", (0, 0), (-1, -1), 0.5, MUTED),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, MUTED),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            # Force header bold by overlaying a font style.
            for col_i, _ in enumerate(data[0]):
                pass  # header cells already use BODY; bold is via inline if needed.
            flow.append(tbl)
            flow.append(Spacer(1, 8))
            continue

        # Bullet list — fold 2-space-indented continuation lines into the
        # previous bullet so multi-line markdown items render as one paragraph.
        if stripped.lstrip().startswith(("- ", "* ")):
            while i < len(lines):
                cur = lines[i]
                if cur.lstrip().startswith(("- ", "* ")):
                    txt = cur.lstrip()[2:].rstrip()
                    i += 1
                    while i < len(lines):
                        nxt = lines[i]
                        if (nxt.startswith((" ", "\t"))
                                and nxt.strip()
                                and not nxt.lstrip().startswith(("- ", "* "))):
                            txt += " " + nxt.strip()
                            i += 1
                        else:
                            break
                    flow.append(Paragraph(
                        f"•&nbsp;&nbsp;{_inline(txt)}", BULLET))
                else:
                    break
            flow.append(Spacer(1, 4))
            continue

        # Default: a normal paragraph (collect until blank line).
        buf = []
        while i < len(lines) and lines[i].strip() and not _line_is_special(lines[i]):
            buf.append(lines[i].rstrip())
            i += 1
        flow.append(Paragraph(_inline(" ".join(buf)), BODY))
    return flow


def _line_is_special(line: str) -> bool:
    s = line.lstrip()
    return (s.startswith(("# ", "## ", "### ", "> ", ">", "|"))
            or s.startswith(("- ", "* "))
            or s.strip() in ("---", "***"))


# --------------------------------------------------------------------------- #
# Build the PDF
# --------------------------------------------------------------------------- #
def build(in_md: Path, out_pdf: Path) -> None:
    md = in_md.read_text(encoding="utf-8")
    flow = md_to_flowables(md)
    doc = SimpleDocTemplate(
        str(out_pdf), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Group 9 — Presentation script",
        author="Group 9",
    )
    doc.build(flow, onFirstPage=_on_page, onLaterPages=_on_page)


if __name__ == "__main__":
    build(Path("Presentation_script.md"),
          Path("Group9_Presentation_script.pdf"))
    print("Wrote Group9_Presentation_script.pdf")
