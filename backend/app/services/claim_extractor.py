"""
VeritasAI V1 — Claim Extractor Service

Responsibilities:
  - Accept a raw free-text claim.
  - Normalise, clean, and decompose it into one or more atomic claims.
  - Extract named entities (people, places, organisations, dates, events).
  - Output a structured representation suitable for downstream pipeline stages.

TODO: Implement this module.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Entity:
    """A named entity extracted from a claim."""

    text: str
    label: str           # e.g. PERSON, LOCATION, ORG, DATE, EVENT
    start: int           # character offset in original claim
    end: int
    confidence: float = 0.0


@dataclass
class ExtractedClaim:
    """Structured representation of a single atomic claim."""

    original: str
    normalised: str
    atomic_claims: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    language: str = "en"


class ClaimExtractorService:
    """
    Decomposes raw input text into structured, verifiable atomic claims.

    Usage (once implemented):
        extractor = ClaimExtractorService()
        result = await extractor.extract("Yesterday two cars had an accident near the post office in Damoh.")
    """

    async def extract(self, raw_claim: str, language: str = "en") -> ExtractedClaim:
        """
        Parse and structure a raw claim.

        Args:
            raw_claim: Free-text claim entered by the user.
            language: ISO 639-1 language code.

        Returns:
            ExtractedClaim with normalised text, atomic sub-claims, and entities.

        TODO:
          - Integrate spaCy or a similar NLP library for NER and sentence splitting.
          - Handle multi-sentence inputs by splitting into individual atomic claims.
          - Normalise casing, punctuation, and pronouns.
          - Extract temporal expressions (dates, times) using dateparser or duckling.
          - Detect claim language if not explicitly provided.
        """
        raise NotImplementedError("ClaimExtractorService.extract is not yet implemented.")

    def _normalise(self, text: str) -> str:
        """
        Basic text normalisation (strip, lower, whitespace collapse).

        TODO: Extend with more robust cleaning and abbreviation expansion.
        """
        raise NotImplementedError

    def _extract_entities(self, text: str) -> List[Entity]:
        """
        Run NER on the claim text.

        TODO: Integrate spaCy en_core_web_sm or a HuggingFace NER pipeline.
        """
        raise NotImplementedError

    def _decompose(self, text: str) -> List[str]:
        """
        Split a compound claim into atomic, individually verifiable statements.

        TODO: Use dependency parsing to identify conjunctions and split accordingly.
        """
        raise NotImplementedError
