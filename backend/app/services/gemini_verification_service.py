"""
VeritasAI — Gemini Verification Service
Independent multimodal verification service using Gemini 3.7 Flash and Google Search grounding.
Completely isolated from existing RAG / NLI / Credibility / Verdict pipelines.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional
import httpx
from fastapi import HTTPException, UploadFile, status
from google import genai
from google.genai import types
from google.genai.errors import ClientError

from app.config import settings
from app.models.gemini_schemas import (
    GeminiClaimDetail,
    GeminiEvidenceItem,
    GeminiSourceItem,
    GeminiVerdict,
    GeminiVerifyResponse,
)
from app.services.gemini_verification_prompt import (
    GEMINI_VERIFICATION_SYSTEM_PROMPT,
    build_gemini_verify_user_prompt,
)

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


class GeminiVerificationService:
    """Independent verification engine powered strictly by Gemini 3.7 Flash."""

    def __init__(self):
        self._api_key = settings.GEMINI_API_KEY
        self._model = settings.GEMINI_MODEL or "gemini-3.7-flash"
        self._client: Optional[genai.Client] = None
        if self._api_key:
            self._client = genai.Client(api_key=self._api_key)
        else:
            logger.warning("GeminiVerificationService: GEMINI_API_KEY is not configured.")

    async def _fetch_url_snippet(self, url: str) -> Optional[str]:
        """Fetch title and visible snippet from public URL for context."""
        try:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code == 200:
                    text = resp.text
                    title_match = re.search(r"<title>(.*?)</title>", text, re.IGNORECASE | re.DOTALL)
                    title = title_match.group(1).strip() if title_match else ""
                    cleaned = re.sub(r"<(script|style).*?>.*?</\1>", "", text, flags=re.IGNORECASE | re.DOTALL)
                    clean_text = re.sub(r"<[^>]+>", " ", cleaned)
                    clean_text = re.sub(r"\s+", " ", clean_text).strip()
                    return f"Title: {title}\nArticle Excerpt: {clean_text[:2500]}"
        except Exception as exc:
            logger.warning("Could not fetch public URL context for %s: %s", url, exc)
        return None

    def _extract_grounding_metadata(self, candidate) -> tuple[List[Dict[str, str]], List[str]]:
        """Extract citations and web queries from candidate grounding metadata."""
        sources: List[Dict[str, str]] = []
        queries: List[str] = []
        if not candidate:
            return sources, queries

        metadata = getattr(candidate, "grounding_metadata", None)
        if not metadata:
            return sources, queries

        raw_queries = getattr(metadata, "web_search_queries", None)
        if raw_queries:
            queries.extend([str(q) for q in raw_queries if q])

        chunks = getattr(metadata, "grounding_chunks", None)
        if chunks:
            seen_urls = set()
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", "")
                    title = getattr(web, "title", "") or "Web Source"
                    if uri and uri not in seen_urls:
                        seen_urls.add(uri)
                        domain = title
                        if "://" in uri:
                            try:
                                domain = uri.split("/")[2]
                            except Exception:
                                domain = title
                        sources.append({
                            "title": title,
                            "url": uri,
                            "source": domain,
                        })

        return sources, queries

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse valid JSON from Gemini output."""
        if not text:
            raise ValueError("Empty response received from model.")

        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start : end + 1])

        return json.loads(cleaned)

    async def verify(
        self,
        claim: Optional[str] = None,
        url: Optional[str] = None,
        image: Optional[UploadFile] = None,
    ) -> GeminiVerifyResponse:
        """Execute evidence-based verification using Gemini 3.7 Flash."""
        if not self._client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Gemini API is not configured. Please set GEMINI_API_KEY in .env",
            )

        claim_clean = claim.strip() if claim and claim.strip() else None
        url_clean = url.strip() if url and url.strip() else None

        if not claim_clean and not url_clean and not image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one input must be provided: a claim, article URL, or image upload.",
            )

        # Validate URL
        url_content = None
        if url_clean:
            if not (url_clean.startswith("http://") or url_clean.startswith("https://")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid URL. Must start with http:// or https://",
                )
            url_content = await self._fetch_url_snippet(url_clean)

        # Validate and read image
        image_bytes: Optional[bytes] = None
        image_mime: Optional[str] = None
        if image:
            if image.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported image type: {image.content_type}. Allowed: JPG, JPEG, PNG, WEBP.",
                )
            image_bytes = await image.read()
            if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image size exceeds 10MB limit ({len(image_bytes)/(1024*1024):.1f}MB).",
                )
            image_mime = image.content_type

        # Build prompt
        prompt_text = build_gemini_verify_user_prompt(
            claim=claim_clean,
            url=url_clean,
            url_content=url_content,
            has_image=bool(image_bytes),
        )

        contents: List[Any] = []
        if image_bytes and image_mime:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
        contents.append(prompt_text)

        search_config = types.GenerateContentConfig(
            system_instruction=GEMINI_VERIFICATION_SYSTEM_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
        )

        response = None
        grounding_sources: List[Dict[str, str]] = []
        grounding_queries: List[str] = []

        try:
            logger.info("GeminiVerificationService: Requesting %s with Google Search grounding...", self._model)
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=search_config,
            )
        except ClientError as exc:
            err_msg = str(exc)
            logger.warning("GeminiVerificationService ClientError on %s: %s", self._model, err_msg)
            # Handle rate-limit / resource quota on search grounding on 3.7-flash
            if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "not supported" in err_msg:
                logger.info("Using search grounding fallback via gemini-2.5-flash...")
                try:
                    fallback_config = types.GenerateContentConfig(
                        system_instruction=GEMINI_VERIFICATION_SYSTEM_PROMPT,
                        tools=[types.Tool(google_search=types.GoogleSearch())],
                        temperature=0.2,
                    )
                    search_fallback = self._client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=fallback_config,
                    )
                    cand = search_fallback.candidates[0] if search_fallback.candidates else None
                    grounding_sources, grounding_queries = self._extract_grounding_metadata(cand)
                    grounded_findings = search_fallback.text or ""

                    reasoning_prompt = (
                        f"{prompt_text}\n\n"
                        f"REAL-TIME SEARCH GROUNDING FINDINGS:\n\"\"\"\n{grounded_findings}\n\"\"\"\n"
                        "Analyze this evidence and output the complete structured JSON verification response."
                    )
                    reasoning_contents = []
                    if image_bytes and image_mime:
                        reasoning_contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
                    reasoning_contents.append(reasoning_prompt)

                    response = self._client.models.generate_content(
                        model=self._model,
                        contents=reasoning_contents,
                        config=types.GenerateContentConfig(
                            system_instruction=GEMINI_VERIFICATION_SYSTEM_PROMPT,
                            temperature=0.2,
                            thinking_config=types.ThinkingConfig(thinking_budget=1024),
                        ),
                    )
                except Exception as inner_exc:
                    logger.warning("Search fallback failed: %s. Falling back to direct %s reasoning.", inner_exc, self._model)
                    response = self._client.models.generate_content(
                        model=self._model,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=GEMINI_VERIFICATION_SYSTEM_PROMPT,
                            temperature=0.2,
                            thinking_config=types.ThinkingConfig(thinking_budget=1024),
                        ),
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini API error: {err_msg}",
                )
        except Exception as exc:
            logger.exception("GeminiVerificationService unexpected failure: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Verification execution error: {str(exc)}",
            )

        # Harvest citations and queries from grounding metadata
        if response and response.candidates:
            cand = response.candidates[0]
            resp_sources, resp_queries = self._extract_grounding_metadata(cand)
            if resp_sources:
                grounding_sources.extend(resp_sources)
            if resp_queries:
                grounding_queries.extend(resp_queries)

        response_text = response.text if response else ""
        try:
            data = self._parse_json(response_text)
        except Exception as parse_err:
            logger.error("Failed to parse Gemini response as JSON: %s. Raw: %s", parse_err, response_text[:300])
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Model returned an unparseable response. Please retry.",
            )

        # Normalize verdict
        verdict_raw = str(data.get("verdict", "UNCERTAIN")).upper().strip()
        if verdict_raw not in {v.value for v in GeminiVerdict}:
            verdict_raw = "UNCERTAIN"

        # Normalize confidence
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
        except (ValueError, TypeError):
            confidence = 0.5

        # Format sub-claims
        claims_list: List[GeminiClaimDetail] = []
        for c in data.get("claims", []):
            if isinstance(c, dict):
                c_verdict = str(c.get("verdict", "UNCERTAIN")).upper().strip()
                if c_verdict not in {v.value for v in GeminiVerdict}:
                    c_verdict = "UNCERTAIN"
                try:
                    c_conf = max(0.0, min(1.0, float(c.get("confidence", confidence))))
                except (ValueError, TypeError):
                    c_conf = confidence
                claims_list.append(
                    GeminiClaimDetail(
                        claim=c.get("claim", ""),
                        verdict=GeminiVerdict(c_verdict),
                        confidence=c_conf,
                        explanation=c.get("explanation", ""),
                    )
                )

        # Supporting evidence
        supporting = [
            GeminiEvidenceItem(
                title=e.get("title", "Supporting Evidence"),
                snippet=e.get("snippet", ""),
                source=e.get("source", "Web Source"),
                url=e.get("url"),
            )
            for e in data.get("supporting_evidence", [])
            if isinstance(e, dict)
        ]

        # Contradicting evidence
        contradicting = [
            GeminiEvidenceItem(
                title=e.get("title", "Contradicting Evidence"),
                snippet=e.get("snippet", ""),
                source=e.get("source", "Web Source"),
                url=e.get("url"),
            )
            for e in data.get("contradicting_evidence", [])
            if isinstance(e, dict)
        ]

        # Merge sources
        sources: List[GeminiSourceItem] = []
        seen_urls = set()
        for s in data.get("sources", []):
            if isinstance(s, dict) and s.get("url"):
                url_str = s.get("url", "")
                if url_str not in seen_urls:
                    seen_urls.add(url_str)
                    sources.append(
                        GeminiSourceItem(
                            title=s.get("title", "Cited Source"),
                            url=url_str,
                            source=s.get("source", "Web Source"),
                        )
                    )

        # Add any grounding sources not yet in the list
        for gs in grounding_sources:
            if gs["url"] not in seen_urls:
                seen_urls.add(gs["url"])
                sources.append(
                    GeminiSourceItem(
                        title=gs["title"],
                        url=gs["url"],
                        source=gs["source"],
                    )
                )

        # If user supplied a URL, ensure it is included in sources
        if url_clean and url_clean not in seen_urls:
            sources.insert(
                0,
                GeminiSourceItem(
                    title="User Supplied Article URL",
                    url=url_clean,
                    source=url_clean.split("/")[2] if "://" in url_clean else "Submitted URL",
                ),
            )

        # Merge queries
        all_queries = list(data.get("search_queries", []))
        for q in grounding_queries:
            if q not in all_queries:
                all_queries.append(q)

        return GeminiVerifyResponse(
            verdict=GeminiVerdict(verdict_raw),
            confidence=confidence,
            claim=data.get("claim") or claim_clean or "Submitted Claim",
            summary=data.get("summary") or "Based on the available evidence, insufficient data was retrieved.",
            claims=claims_list,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            sources=sources,
            red_flags=data.get("red_flags", []),
            search_queries=all_queries,
        )


gemini_verification_service = GeminiVerificationService()
