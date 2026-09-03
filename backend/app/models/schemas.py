"""
VeritasAI V1 — Pydantic Schemas
Request / Response contracts for all API endpoints.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────────────

class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"


# ── Sub-models ────────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """
    A single piece of evidence surfaced during claim verification.

    Includes semantic relevance from the Embedding Service and NLI
    stance scores from the Transformer Service.  Content is intentionally
    excluded from the response to keep payloads small; a snippet is
    provided for display purposes.
    """

    title: str = Field("", description="Page or article title.")
    url: str = Field("", description="Source URL.")
    source_name: str = Field("", description="Domain / publication name.")
    relevance_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Cosine-similarity relevance to the claim (0–1).",
    )
    nli_label: str = Field(
        "NEUTRAL",
        description="NLI stance: ENTAILMENT, CONTRADICTION, NEUTRAL, or SKIPPED.",
    )
    entailment_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Softmax probability for ENTAILMENT.",
    )
    contradiction_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Softmax probability for CONTRADICTION.",
    )
    neutral_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Softmax probability for NEUTRAL.",
    )
    published_at: Optional[str] = Field(
        None, description="Publication date if discoverable."
    )
    content_snippet: Optional[str] = Field(
        None, description="First ~200 characters of the evidence text."
    )


class SourceReference(BaseModel):
    """Metadata about a source used during verification."""

    url: str
    title: Optional[str] = None
    domain: Optional[str] = None
    published_at: Optional[str] = None


class TransformerResult(BaseModel):
    """
    Output from the NLP transformer (NLI) analysis stage.

    ``label`` is the dominant class; the three ``*_score`` fields hold the
    full softmax probability distribution so downstream stages (Evidence Engine,
    Credibility Engine) can apply their own thresholds.
    """

    label: str = Field(
        ...,
        description="Dominant NLI label: ENTAILMENT, CONTRADICTION, or NEUTRAL.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the dominant label (0–1).",
    )
    entailment_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Softmax probability for ENTAILMENT.",
    )
    contradiction_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Softmax probability for CONTRADICTION.",
    )
    neutral_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Softmax probability for NEUTRAL.",
    )


class WebEvaluationResult(BaseModel):
    """Output from the web evidence evaluation stage."""

    query_used: str
    results_retrieved: int
    top_evidence: List[EvidenceItem] = []


class RAGResult(BaseModel):
    """Output from the RAG retrieval stage."""

    chunks_retrieved: int
    evidence: List[EvidenceItem] = []


class GraphResult(BaseModel):
    """Output from the knowledge-graph consistency check."""

    entities_matched: int = 0
    relations_found: int = 0
    consistency_score: float = 0.0


class CredibilityResult(BaseModel):
    """Aggregated credibility scoring output."""

    overall_score: float = Field(0.0, ge=0.0, le=1.0)
    component_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component score breakdown.",
    )


# ── Request / Response ────────────────────────────────────────────────────────

class VerifyRequest(BaseModel):
    """Payload for POST /api/v1/verify."""

    claim: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="The news claim or statement to verify.",
        examples=["Yesterday two cars had an accident near the post office in Damoh."],
    )
    language: str = Field(
        "en",
        description="ISO 639-1 language code of the claim.",
    )
    options: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional per-request overrides (e.g. disable specific pipeline stages).",
    )


class VerifyResponse(BaseModel):
    """Response from POST /api/v1/verify."""

    claim: str
    verdict: VerificationStatus = Field(
        ..., description="Final deterministic verdict: VERIFIED, UNVERIFIED, or CONTRADICTED."
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence (0–1).")
    explanation: str = Field(..., description="Human-readable explanation of the verdict.")

    # Key scoring signals surfaced for transparency
    support_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Credibility-weighted support signal (0–1).",
    )
    contradiction_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Credibility-weighted contradiction signal (0–1).",
    )

    # Flattened evidence list (enriched with NLI fields)
    evidence: List[EvidenceItem] = []
    sources: List[SourceReference] = []

    # Optional pipeline debug fields (populated when stages complete)
    credibility: Optional[CredibilityResult] = None
