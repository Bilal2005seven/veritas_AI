"""
VeritasAI V1 — Gemini Explanation Service
==========================================
Generates a concise, human-readable explanation for the deterministic
verification verdict produced by the Evidence + Credibility pipeline.

Responsibilities
----------------
* Accept the claim, verdict, credibility assessment, and top evidence.
* Build a structured prompt that explicitly tells Gemini NOT to override
  the deterministic verdict — only to explain it.
* Call the Gemini REST API (generativelanguage.googleapis.com) via httpx.
* Return the explanation string, or ``None`` on any failure.

Design decisions
----------------
* **No SDK dependency** — uses ``httpx.AsyncClient`` (already a project dep).
* **Fail-open** — returns ``None`` on timeout, auth error, missing key, or
  any unexpected error.  The orchestration layer generates a local fallback.
* **Key never in responses** — the API key is only used in HTTP headers and
  never forwarded to callers.
* **Single call** — no streaming, no retries at this layer.  V1 simplicity.
"""

from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from app.services.credibility_engine import CredibilityAssessment
    from app.services.evidence_engine import EvidenceEngineResult

logger = logging.getLogger(__name__)

# Gemini REST endpoint template.  The model is inserted at call time.
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Maximum tokens we request from Gemini.
_MAX_OUTPUT_TOKENS: int = 512

# Maximum evidence items to include in the prompt (keeps prompt short).
_PROMPT_MAX_EVIDENCE: int = 5


class GeminiService:
    """
    Wraps the Gemini REST API for explanation generation.

    Parameters
    ----------
    api_key : str
        Google Gemini API key.  When empty, :meth:`explain` immediately
        returns ``None`` without making a network call.
    model : str
        Gemini model identifier (e.g. ``"gemini-2.0-flash"``).
    timeout : float
        HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.7-flash",
        timeout: float = 15.0,
    ) -> None:
        self._api_key = api_key.strip()
        self._model = model.strip()
        self._timeout = timeout

    async def explain(
        self,
        *,
        claim: str,
        verdict: str,
        confidence: float,
        assessment: "CredibilityAssessment",
        evidence_result: "EvidenceEngineResult",
    ) -> Optional[str]:
        """
        Generate a human-readable explanation for the given verdict.

        The prompt explicitly instructs Gemini not to override the
        deterministic verdict — only to explain it using the supplied evidence.

        Parameters
        ----------
        claim : str
            Original user claim.
        verdict : str
            The deterministic verdict string (``VERIFIED``, ``UNVERIFIED``,
            or ``CONTRADICTED``).
        confidence : float
            Confidence score (0–1) from the credibility engine.
        assessment : CredibilityAssessment
            Full credibility assessment from CredibilityEngine.
        evidence_result : EvidenceEngineResult
            Full output from EvidenceEngine.

        Returns
        -------
        Optional[str]
            Explanation text, or ``None`` on any failure.
        """
        if not self._api_key:
            logger.info("GeminiService: no API key configured — skipping Gemini call.")
            return None

        prompt = self._build_prompt(
            claim=claim,
            verdict=verdict,
            confidence=confidence,
            assessment=assessment,
            evidence_result=evidence_result,
        )

        url = _GEMINI_ENDPOINT.format(model=self._model)
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": _MAX_OUTPUT_TOKENS,
                "temperature": 0.3,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "x-goog-api-key": self._api_key,
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                data = response.json()
        except httpx.TimeoutException:
            logger.warning("GeminiService: request timed out after %.1fs.", self._timeout)
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "GeminiService: HTTP %s — %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return None
        except httpx.RequestError as exc:
            logger.warning("GeminiService: request error — %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("GeminiService: unexpected error — %s", exc)
            return None

        # Parse the response.
        try:
            text = (
                data["candidates"][0]["content"]["parts"][0]["text"]
            ).strip()
            if not text:
                logger.warning("GeminiService: empty response text.")
                return None
            return text
        except (KeyError, IndexError, TypeError) as exc:
            logger.warning("GeminiService: could not parse response — %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(
        *,
        claim: str,
        verdict: str,
        confidence: float,
        assessment: "CredibilityAssessment",
        evidence_result: "EvidenceEngineResult",
    ) -> str:
        """
        Construct the structured prompt sent to Gemini.

        The prompt includes:
        - The original claim.
        - The deterministic verdict and confidence.
        - Supporting evidence summaries.
        - Contradicting evidence summaries.
        - Explicit instruction not to override the verdict.
        """
        lines: list[str] = [
            "You are a fact-checking assistant. Your task is to explain a "
            "pre-computed verification verdict to a reader in plain, neutral language.",
            "",
            "IMPORTANT: Do not override the deterministic verdict. "
            "Explain the provided verdict using the supplied evidence.",
            "",
            f"CLAIM: {claim}",
            f"VERDICT: {verdict}",
            f"CONFIDENCE: {confidence:.0%}",
            f"CREDIBILITY ASSESSMENT: {assessment.reasoning}",
            "",
        ]

        # Add supporting evidence
        supporting = evidence_result.supporting_evidence[:_PROMPT_MAX_EVIDENCE]
        if supporting:
            lines.append("SUPPORTING EVIDENCE:")
            for i, ev in enumerate(supporting, 1):
                lines.append(
                    f"  {i}. [{ev.source_name}] {ev.title} "
                    f"(relevance: {ev.relevance_score:.2f}, "
                    f"entailment: {ev.entailment_score:.2f})"
                )
                if ev.content:
                    snippet = ev.content[:200].replace("\n", " ")
                    lines.append(f"     Snippet: \"{snippet}\"")
            lines.append("")

        # Add contradicting evidence
        contradicting = evidence_result.contradicting_evidence[:_PROMPT_MAX_EVIDENCE]
        if contradicting:
            lines.append("CONTRADICTING EVIDENCE:")
            for i, ev in enumerate(contradicting, 1):
                lines.append(
                    f"  {i}. [{ev.source_name}] {ev.title} "
                    f"(relevance: {ev.relevance_score:.2f}, "
                    f"contradiction: {ev.contradiction_score:.2f})"
                )
                if ev.content:
                    snippet = ev.content[:200].replace("\n", " ")
                    lines.append(f"     Snippet: \"{snippet}\"")
            lines.append("")

        lines += [
            "INSTRUCTIONS:",
            "- Write 2–4 sentences explaining why this verdict was reached.",
            "- Reference the specific evidence sources above.",
            "- Do not introduce new facts or make up sources.",
            "- Use plain language suitable for a general audience.",
            "- Do NOT say the verdict is wrong or should be different.",
        ]

        return "\n".join(lines)
