from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class VerdictEnum(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNCERTAIN = "UNCERTAIN"
    MISLEADING = "MISLEADING"

class ClaimDetail(BaseModel):
    claim: str = Field(..., description="Specific factual claim identified")
    verdict: VerdictEnum = Field(..., description="Verdict for this specific claim")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this claim's verdict (0.0 to 1.0)")
    explanation: str = Field(..., description="Reasoning for the verdict on this claim")

class EvidenceItem(BaseModel):
    title: str = Field(..., description="Title of the article or evidence source")
    snippet: str = Field(..., description="Relevant quotation or factual finding")
    source: str = Field(..., description="Name of the publisher or domain")
    url: Optional[str] = Field(None, description="Direct URL to source if available")

class SourceItem(BaseModel):
    title: str = Field(..., description="Title of cited source")
    url: str = Field(..., description="Verified URL of cited source")
    source: str = Field(..., description="Domain name or publisher")

class VerificationResponse(BaseModel):
    verdict: VerdictEnum = Field(..., description="Primary verdict: SUPPORTED, CONTRADICTED, UNCERTAIN, or MISLEADING")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Estimated confidence based on available evidence")
    claim: str = Field(..., description="The central claim investigated")
    summary: str = Field(..., description="Evidence-based summary starting with 'Based on the available evidence...'")
    claims: List[ClaimDetail] = Field(default_factory=list, description="List of individual sub-claims verified")
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list, description="Evidence supporting the claim")
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list, description="Evidence contradicting the claim")
    sources: List[SourceItem] = Field(default_factory=list, description="All external sources referenced or grounded")
    red_flags: List[str] = Field(default_factory=list, description="Warning flags, temporal shifts, missing context, or distortions")
    search_queries: List[str] = Field(default_factory=list, description="Web search queries executed during grounding")

class HealthResponse(BaseModel):
    status: str
    model: str
    service: str
    version: str = "1.0.0"
