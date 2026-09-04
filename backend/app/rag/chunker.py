"""
VeritasAI V1 — Article Chunker
================================
Splits raw article text into overlapping, paragraph/sentence-aware chunks
suitable for BM25 + MiniLM hybrid retrieval.

Design
------
* **Deterministic** — same input always produces the same output.
* **No new dependencies** — uses only the Python standard library (``re``).
* **Paragraph-first** — splits on blank lines first, then on sentence
  boundaries when a paragraph is still too long, then hard-splits by word
  count as a last resort.
* **Overlap** — the last ``CHUNK_OVERLAP_WORDS`` words of chunk N are
  prepended to chunk N+1, preserving cross-boundary context.

Constants
---------
CHUNK_TARGET_WORDS : int
    Soft target word count per chunk (≈ 200 words).
CHUNK_OVERLAP_WORDS : int
    Number of words from the tail of one chunk to prepend to the next (≈ 25).
CHUNK_MIN_WORDS : int
    Chunks shorter than this are merged into the previous chunk instead of
    being returned as standalone items.  Prevents single-sentence orphans.
"""

from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Soft target number of words per chunk.
CHUNK_TARGET_WORDS: int = 200

#: Words from the end of chunk N prepended to chunk N+1.
CHUNK_OVERLAP_WORDS: int = 25

#: Chunks with fewer words than this are merged into the preceding chunk.
CHUNK_MIN_WORDS: int = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Return the approximate word count of *text* (whitespace-split)."""
    return len(text.split())


def _split_into_paragraphs(text: str) -> List[str]:
    """
    Split *text* on blank lines (one or more ``\\n`` with optional spaces).

    Returns a list of non-empty, stripped paragraph strings.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_into_sentences(text: str) -> List[str]:
    """
    Split *text* into sentences using punctuation boundaries.

    Uses a simple heuristic: split after ``.``, ``!``, or ``?`` followed by
    whitespace and an uppercase letter (or end-of-string).  This avoids
    splitting on abbreviations like ``Mr.`` or ``U.S.`` in most cases.
    """
    # Split after sentence-ending punctuation.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'])", text)
    return [p.strip() for p in parts if p.strip()]


def _combine_sentences_into_chunks(
    sentences: List[str],
    target: int,
    overlap: int,
    min_words: int,
) -> List[str]:
    """
    Pack *sentences* into chunks of approximately *target* words, with
    *overlap* words of trailing context carried into the next chunk.

    Parameters
    ----------
    sentences : list[str]
        Pre-split sentence strings.
    target : int
        Target word count per chunk.
    overlap : int
        Overlap word count between consecutive chunks.
    min_words : int
        Minimum words for a standalone chunk; smaller items are merged.

    Returns
    -------
    list[str]
        List of chunk strings.
    """
    chunks: List[str] = []
    current_words: List[str] = []

    for sentence in sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue

        # If adding this sentence would push us over the target AND we already
        # have something, flush the current chunk first.
        if current_words and (_word_count(" ".join(current_words)) + len(sentence_words)) > target:
            chunk_text = " ".join(current_words).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Start next chunk with the overlap tail of the previous chunk.
            current_words = current_words[-overlap:] if overlap > 0 else []

        current_words.extend(sentence_words)

    # Flush the final chunk.
    if current_words:
        chunk_text = " ".join(current_words).strip()
        if chunk_text:
            chunks.append(chunk_text)

    # Merge orphan chunks (too short) into the previous chunk.
    merged: List[str] = []
    for chunk in chunks:
        if merged and _word_count(chunk) < min_words:
            merged[-1] = merged[-1] + " " + chunk
        else:
            merged.append(chunk)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
    min_words: int = CHUNK_MIN_WORDS,
) -> List[str]:
    """
    Split *text* into overlapping, passage-sized chunks.

    Strategy
    --------
    1. Split on blank lines → paragraphs.
    2. For each paragraph that fits within *target_words*, treat it as a
       single chunk candidate.
    3. For paragraphs that are too long, split by sentence boundary and
       then pack sentences up to *target_words*.
    4. Apply overlap: the last *overlap_words* words of a chunk are
       prepended to the next.
    5. Merge chunks shorter than *min_words* into their predecessor.

    Parameters
    ----------
    text : str
        Raw article text (may contain newlines, mixed whitespace, etc.).
    target_words : int
        Soft upper bound on words per chunk.
    overlap_words : int
        Number of trailing words to carry into the next chunk.
    min_words : int
        Minimum words for a standalone chunk.

    Returns
    -------
    list[str]
        Ordered list of chunk strings.  Returns ``[text]`` when *text* is
        short enough to be a single chunk.  Returns ``[]`` for empty input.

    Examples
    --------
    >>> chunks = chunk_text("First paragraph.\\n\\nSecond paragraph.")
    >>> len(chunks) >= 1
    True
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Fast path: article is short enough to be a single chunk.
    if _word_count(text) <= target_words:
        return [text]

    # Step 1 — split on blank lines.
    paragraphs = _split_into_paragraphs(text)

    # Step 2 — collect sentence lists per paragraph, respecting target size.
    all_sentences: List[str] = []
    for para in paragraphs:
        if _word_count(para) <= target_words:
            # Paragraph fits as-is: treat it as one "sentence" for packing.
            all_sentences.append(para)
        else:
            # Paragraph is oversized: break it into sentences.
            all_sentences.extend(_split_into_sentences(para))

    if not all_sentences:
        return [text]

    # Step 3 — pack sentences into chunks with overlap.
    chunks = _combine_sentences_into_chunks(
        all_sentences,
        target=target_words,
        overlap=overlap_words,
        min_words=min_words,
    )

    return chunks if chunks else [text]
