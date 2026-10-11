"""Biomedical-aware chunker for BC5CDR.

BC5CDR abstracts are short; most yield 1 chunk. The chunker still needs
to handle multi-chunk cases correctly because Phase 3 windows and the
Phase 3 space discriminators, which CQ1 reads, depend on it.

Non-breaking abbreviations include et al., Fig., i.v., i.p., p.o., vs.,
approx., e.g., i.e., and single-uppercase-letter+period initials.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: int
    text: str
    doc_offset: int              # char offset of chunk start in full_text
    sentences: list[str] = field(default_factory=list)


_ABBREV = [
    "et al.", "Fig.", "Figs.", "fig.", "figs.",
    "i.v.", "i.p.", "p.o.", "s.c.", "i.m.",
    "vs.", "cf.", "approx.", "e.g.", "i.e.",
    "St.", "Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.",
    "min.", "max.", "avg.", "no.", "No.",
    "ca.", "wt.", "conc.", "temp.", "vol.",
    "mg.", "ml.", "kg.",
]

# Placeholder that will not appear in real biomedical text.
_PROT_TOKEN = "\x00A\x00"


def _protect(text: str) -> str:
    out = text
    for ab in _ABBREV:
        out = out.replace(ab, ab.replace(".", _PROT_TOKEN))
    # Single-uppercase-letter followed by period (initials): "J. Smith"
    out = re.sub(r"\b([A-Z])\.\s", lambda m: m.group(1) + _PROT_TOKEN + " ", out)
    return out


def _unprotect(text: str) -> str:
    return text.replace(_PROT_TOKEN, ".")


_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Regex-based biomedical sentence splitter."""
    if not text.strip():
        return []
    prot = _protect(text)
    parts = _SPLIT.split(prot)
    return [_unprotect(p).strip() for p in parts if p.strip()]


def _rough_tokens(s: str) -> int:
    return int(len(s.split()) * 1.3) + 1


def chunk_document(full_text: str, max_chunk_tokens: int = 512,
                   sentence_overlap: int = 1) -> list[Chunk]:
    """Greedy sentence accumulation with N-sentence overlap between chunks."""
    sents = split_sentences(full_text)
    if not sents:
        return []

    # Compute char offset for each sentence in full_text (best-effort search).
    offsets: list[int] = []
    cursor = 0
    for s in sents:
        i = full_text.find(s, cursor)
        if i < 0:
            i = full_text.find(s)     # fall back to global search
        if i < 0:
            i = cursor
        offsets.append(i)
        cursor = i + len(s)

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_tokens = 0
    buf_start_idx = 0                 # index into sents where chunk begins

    def flush(end_idx: int) -> None:
        nonlocal buf, buf_tokens, buf_start_idx
        if not buf:
            return
        text = " ".join(buf)
        doc_offset = offsets[buf_start_idx] if buf_start_idx < len(offsets) else 0
        chunks.append(Chunk(
            chunk_id=len(chunks), text=text, doc_offset=doc_offset,
            sentences=list(buf)))
        # Prepare overlap for the next chunk.
        overlap = buf[-sentence_overlap:] if sentence_overlap > 0 else []
        buf = list(overlap)
        buf_tokens = sum(_rough_tokens(s) for s in buf)
        buf_start_idx = end_idx - len(overlap) + 1 if overlap else end_idx + 1

    for i, s in enumerate(sents):
        s_tok = _rough_tokens(s)
        if buf and buf_tokens + s_tok > max_chunk_tokens:
            flush(i - 1)
        if not buf:
            buf_start_idx = i
        buf.append(s)
        buf_tokens += s_tok

    if buf:
        text = " ".join(buf)
        doc_offset = offsets[buf_start_idx] if buf_start_idx < len(offsets) else 0
        chunks.append(Chunk(
            chunk_id=len(chunks), text=text, doc_offset=doc_offset,
            sentences=list(buf)))

    return chunks


# ---------------------------------------------------------------------
#  Abbreviation detection: "LONG_FORM (SHORT)" where SHORT is 2-10 upper.
# ---------------------------------------------------------------------

_ABBR_PATTERN = re.compile(
    r"([A-Za-z][\w\s\-]{2,50}?)\s*\(([A-Z][A-Z0-9\-]{1,9})\)")


def extract_abbreviations(text: str) -> dict[str, str]:
    """Return {short_form: long_form} for LONG (SHORT) style hits."""
    out: dict[str, str] = {}
    for m in _ABBR_PATTERN.finditer(text):
        long_form = m.group(1).strip()
        short = m.group(2).strip()
        # Only keep when short's letters plausibly abbreviate long.
        long_words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", long_form)
        if not long_words:
            continue
        initials = "".join(w[0] for w in long_words).upper()
        # Loose match: at least one letter of short appears in the initials.
        if not any(c in initials for c in short):
            continue
        if short not in out:
            out[short] = long_form
    return out
