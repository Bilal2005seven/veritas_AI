"""
VeritasAI — Gemini Verification Schemas
Contracts for POST /api/v1/gemini-verify
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class GeminiVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"
    MISLEADING = "MISLEADING"


class GeminiClaimDetail(BaseModel):
    claim: str = Field(..., description="Factual sub-claim identified.")
    verdict: GeminiVerdict = Field(..., description="Verdict for this claim: SUPPORTED, CONTRADICTED, UNCERTAIN, or MISLEADING.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimated confidence based on available evidence (0.0 to 1.0).")
    explanation: str = Field(..., description="Reasoning and evidence alignment for this claim.")


class GeminiEvidenceItem(BaseModel):
    title: str = Field(..., description="Headline or title of the evidence source.")
    snippet: str = Field(..., description="Relevant quotation or factual finding.")
    source: str = Field(..., description="Name of the publisher or domain.")
    url: Optional[str] = Field(None, description="Direct URL to source if available.")


class GeminiSourceItem(BaseModel):
    title: str = Field(..., description="Title of the source.")
    url: str = Field(..., description="Verified URL from search grounding or user input.")
    source: str = Field(..., description="Domain name or publisher.")


class GeminiVerifyResponse(BaseModel):
    verdict: GeminiVerdict = Field(..., description="Primary verdict: SUPPORTED, CONTRADICTED, UNCERTAIN, or MISLEADING.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Gemini's estimated confidence based on the available evidence.")
    claim: str = Field(..., description="The primary claim investigated.")
    summary: str = Field(..., description="Evidence-based summary (e.g. 'Based on the available evidence...').")
    claims: List[GeminiClaimDetail] = Field(default_factory=list, description="Sub-claims evaluated.")
    supporting_evidence: List[GeminiEvidenceItem] = Field(default_factory=list, description="Evidence supporting the claim.")
    contradicting_evidence: List[GeminiEvidenceItem] = Field(default_factory=list, description="Evidence conflicting with the claim.")
    sources: List[GeminiSourceItem] = Field(default_factory=list, description="Verified external sources cited.")
    red_flags: List[str] = Field(default_factory=list, description="Temporal shifts, missing context, or distortions.")
    search_queries: List[str] = Field(default_factory=list, description="Search queries executed during web grounding.")
