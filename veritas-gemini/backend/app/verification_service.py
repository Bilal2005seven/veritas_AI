import logging
import re
from typing import Optional
import httpx
from fastapi import HTTPException, UploadFile, status

from app.gemini_service import gemini_service
from app.prompt_service import build_verification_prompt
from app.schemas import (
    ClaimDetail,
    EvidenceItem,
    SourceItem,
    VerdictEnum,
    VerificationResponse,
)

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

class VerificationService:
    async def fetch_url_context(self, url: str) -> Optional[str]:
        """Fetch title and visible text from public URL for context."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    text = resp.text
                    # Simple extraction of title and body text without heavy parser
                    title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
                    title = title_match.group(1).strip() if title_match else ""
                    # Strip scripts and styles
                    cleaned_html = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.IGNORECASE | re.DOTALL)
                    # Strip tags
                    clean_text = re.sub(r"<[^>]+>", " ", cleaned_html)
                    clean_text = re.sub(r"\s+", " ", clean_text).strip()
                    context = f"Title: {title}\nArticle Snippet: {clean_text[:2500]}"
                    return context
        except Exception as e:
            logger.warning("Could not fetch URL context for %s: %s", url, e)
        return None

    async def verify(
        self,
        claim: Optional[str] = None,
        url: Optional[str] = None,
        image: Optional[UploadFile] = None,
    ) -> VerificationResponse:
        """Coordinate multimodal verification pipeline."""
        # 1. Validation: At least one input required
        claim_clean = claim.strip() if claim and claim.strip() else None
        url_clean = url.strip() if url and url.strip() else None

        if not claim_clean and not url_clean and not image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one input must be provided: a text claim, a news URL, or a screenshot image.",
            )

        # 2. URL validation
        url_content = None
        if url_clean:
            if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid URL format. URL must start with http:// or https://",
                )
            url_content = await self.fetch_url_context(url_clean)

        # 3. Image validation and reading
        image_bytes: Optional[bytes] = None
        image_mime: Optional[str] = None
        if image:
            if image.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported image format: {image.content_type}. Please upload JPG, PNG, or WEBP.",
                )
            image_bytes = await image.read()
            if len(image_bytes) > MAX_IMAGE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image size exceeds limit of 10MB (received {len(image_bytes)/(1024*1024):.1f}MB).",
                )
            image_mime = image.content_type

        # 4. Build prompt
        prompt = build_verification_prompt(
            claim=claim_clean,
            url=url_clean,
            url_content=url_content,
            has_image=bool(image_bytes),
        )

        # 5. Call Gemini Service
        try:
            raw_data = await gemini_service.verify_claim(
                prompt_text=prompt,
                image_bytes=image_bytes,
                image_mime_type=image_mime,
            )
        except Exception as e:
            logger.exception("Error executing verification: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Verification failed: {str(e)}",
            )

        # 6. Normalize and validate response into Pydantic schema
        verdict_str = str(raw_data.get("verdict", "UNCERTAIN")).upper().strip()
        if verdict_str not in {v.value for v in VerdictEnum}:
            verdict_str = "UNCERTAIN"

        confidence_val = raw_data.get("confidence", 0.5)
        try:
            confidence_float = max(0.0, min(1.0, float(confidence_val)))
        except (ValueError, TypeError):
            confidence_float = 0.5

        claims_list = []
        for c in raw_data.get("claims", []):
            try:
                c_verdict = str(c.get("verdict", "UNCERTAIN")).upper().strip()
                if c_verdict not in {v.value for v in VerdictEnum}:
                    c_verdict = "UNCERTAIN"
                c_conf = max(0.0, min(1.0, float(c.get("confidence", confidence_float))))
                claims_list.append(
                    ClaimDetail(
                        claim=c.get("claim", ""),
                        verdict=VerdictEnum(c_verdict),
                        confidence=c_conf,
                        explanation=c.get("explanation", ""),
                    )
                )
            except Exception:
                continue

        supporting = [
            EvidenceItem(
                title=e.get("title", "Supporting Evidence"),
                snippet=e.get("snippet", ""),
                source=e.get("source", "Web Source"),
                url=e.get("url"),
            )
            for e in raw_data.get("supporting_evidence", [])
            if isinstance(e, dict)
        ]

        contradicting = [
            EvidenceItem(
                title=e.get("title", "Contradicting Evidence"),
                snippet=e.get("snippet", ""),
                source=e.get("source", "Web Source"),
                url=e.get("url"),
            )
            for e in raw_data.get("contradicting_evidence", [])
            if isinstance(e, dict)
        ]

        sources = []
        for s in raw_data.get("sources", []):
            if isinstance(s, dict) and s.get("url"):
                sources.append(
                    SourceItem(
                        title=s.get("title", "Cited Source"),
                        url=s.get("url", ""),
                        source=s.get("source", "Web Source"),
                    )
                )

        response = VerificationResponse(
            verdict=VerdictEnum(verdict_str),
            confidence=confidence_float,
            claim=raw_data.get("claim") or claim_clean or "Submitted Claim",
            summary=raw_data.get("summary") or "Based on the available evidence, insufficient details were found.",
            claims=claims_list,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            sources=sources,
            red_flags=raw_data.get("red_flags", []),
            search_queries=raw_data.get("search_queries", []),
        )

        return response

verification_service = VerificationService()
