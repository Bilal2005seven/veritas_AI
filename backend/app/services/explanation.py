"""
VeritasAI V1 — Explanation Service

Responsibilities:
  - Receive the full pipeline outputs and credibility result.
  - Generate a concise, human-readable explanation for the final verdict.
  - Optionally produce a structured breakdown (evidence bullets, confidence bars).

TODO: Implement this module.
"""

from typing import List, Optional

from app.models.schemas import (
    CredibilityResult,
    EvidenceItem,
    GraphResult,
    RAGResult,
    TransformerResult,
    VerificationStatus,
    WebEvaluationResult,
)


class ExplanationService:
    """
    Produces human-readable explanations for claim verification verdicts.

    Usage (once implemented):
        svc = ExplanationService()
        text = await svc.generate(
            claim="...",
            status=VerificationStatus.VERIFIED,
            credibility=credibility_result,
            evidence=[...],
        )
    """

    async def generate(
        self,
        claim: str,
        status: VerificationStatus,
        credibility: CredibilityResult,
        evidence: List[EvidenceItem],
        transformer: Optional[TransformerResult] = None,
        web: Optional[WebEvaluationResult] = None,
        rag: Optional[RAGResult] = None,
        graph: Optional[GraphResult] = None,
    ) -> str:
        """
        Generate a plain-English explanation for the verification result.

        Args:
            claim:       The original claim text.
            status:      Final VerificationStatus verdict.
            credibility: Aggregated credibility scores.
            evidence:    Top evidence items used in the decision.
            transformer: NLP result (optional, for richer explanation).
            web:         Web evaluation result (optional).
            rag:         RAG retrieval result (optional).
            graph:       Graph result (optional).

        Returns:
            A human-readable explanation string.

        TODO:
          - Template-based: craft sentences from structured pipeline outputs.
          - LLM-based (optional): call an LLM with a summarisation prompt
            that includes the claim, verdict, and top evidence snippets.
          - Keep explanations concise (≤3 sentences by default).
          - Cite sources inline when available.
        """
        raise NotImplementedError("ExplanationService.generate is not yet implemented.")

    def _verdict_sentence(self, status: VerificationStatus, score: float) -> str:
        """
        Build the opening verdict sentence.

        TODO: Use status and score to fill a template like:
          "This claim has been {status} with {score:.0%} confidence."
        """
        raise NotImplementedError

    def _evidence_summary(self, evidence: List[EvidenceItem]) -> str:
        """
        Summarise the top evidence items into a readable string.

        TODO:
          - Pick top-3 evidence items by relevance_score.
          - Format as: "According to [source], ..."
        """
        raise NotImplementedError
