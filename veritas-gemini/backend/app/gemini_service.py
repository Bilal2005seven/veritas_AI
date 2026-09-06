import json
import logging
import re
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError

from app.config import settings
from app.prompt_service import SYSTEM_VERIFICATION_PROMPT

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY is not configured in backend/.env!")
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY) if settings.GEMINI_API_KEY else None
        self.model_name = settings.GEMINI_MODEL

    def _extract_grounding_info(self, candidate) -> tuple[List[Dict[str, str]], List[str]]:
        """Extract sources and search queries from candidate grounding metadata."""
        sources = []
        queries = []
        if not candidate:
            return sources, queries

        metadata = getattr(candidate, "grounding_metadata", None)
        if not metadata:
            return sources, queries

        # Extract search queries
        raw_queries = getattr(metadata, "web_search_queries", None)
        if raw_queries:
            queries.extend([str(q) for q in raw_queries if q])

        # Extract grounding chunks (sources)
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

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Clean and parse JSON from model output."""
        if not text:
            raise ValueError("Empty response text from model")

        # Strip markdown fences
        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^```\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Find first '{' and last '}'
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx : end_idx + 1]
            return json.loads(json_str)

        return json.loads(cleaned)

    async def verify_claim(
        self,
        prompt_text: str,
        image_bytes: Optional[bytes] = None,
        image_mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform multimodal verification using Gemini 3.7 Flash and Google Search grounding."""
        if not self.client:
            raise RuntimeError("Gemini API key is not configured. Please set GEMINI_API_KEY in backend/.env")

        contents: List[Any] = []

        # Add image if provided
        if image_bytes and image_mime_type:
            part = types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type)
            contents.append(part)

        # Add text prompt
        contents.append(prompt_text)

        # Configure Google Search grounding tool
        search_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_VERIFICATION_PROMPT,
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.2,
        )

        response = None
        used_model = self.model_name
        grounding_sources: List[Dict[str, str]] = []
        grounding_queries: List[str] = []

        try:
            logger.info("Calling %s with Google Search grounding...", used_model)
            response = self.client.models.generate_content(
                model=used_model,
                contents=contents,
                config=search_config,
            )
        except ClientError as e:
            err_str = str(e)
            logger.warning("Gemini API ClientError with %s: %s", used_model, err_str)
            # If rate limited (429) or tool not supported on specific flash tier, fallback to search on gemini-2.5-flash
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "not supported" in err_str:
                logger.info("Attempting web search grounding via gemini-2.5-flash fallback...")
                try:
                    search_fallback_resp = self.client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=contents,
                        config=search_config,
                    )
                    cand = search_fallback_resp.candidates[0] if search_fallback_resp.candidates else None
                    grounding_sources, grounding_queries = self._extract_grounding_info(cand)
                    grounded_context = search_fallback_resp.text or ""

                    # Now pass search context to Gemini 3.7 Flash for deep reasoning and verdict
                    logger.info("Passing grounded web evidence to %s for final reasoning...", self.model_name)
                    reasoning_prompt = (
                        f"{prompt_text}\n\n"
                        f"CURRENT WEB SEARCH GROUNDING FINDINGS:\n\"\"\"\n{grounded_context}\n\"\"\"\n"
                        "Synthesize this evidence and return the complete structured verification JSON."
                    )
                    reasoning_config = types.GenerateContentConfig(
                        system_instruction=SYSTEM_VERIFICATION_PROMPT,
                        temperature=0.2,
                    )
                    reasoning_contents = []
                    if image_bytes and image_mime_type:
                        reasoning_contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type))
                    reasoning_contents.append(reasoning_prompt)

                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=reasoning_contents,
                        config=reasoning_config,
                    )
                except Exception as inner_e:
                    logger.warning("Search fallback failed: %s. Invoking %s directly.", inner_e, self.model_name)
                    direct_config = types.GenerateContentConfig(
                        system_instruction=SYSTEM_VERIFICATION_PROMPT,
                        temperature=0.2,
                    )
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=contents,
                        config=direct_config,
                    )
            else:
                raise e
        except Exception as e:
            logger.error("Unexpected error during Gemini verification: %s", e)
            raise e

        # Extract grounding metadata if available from the final response
        if response and response.candidates:
            cand = response.candidates[0]
            resp_sources, resp_queries = self._extract_grounding_info(cand)
            if resp_sources:
                grounding_sources.extend(resp_sources)
            if resp_queries:
                grounding_queries.extend(resp_queries)

        response_text = response.text if response else ""
        data = self._parse_json_response(response_text)

        # Merge grounded search queries if missing in model json
        if grounding_queries:
            existing_queries = set(data.get("search_queries", []))
            for q in grounding_queries:
                if q not in existing_queries:
                    data.setdefault("search_queries", []).append(q)

        # Merge grounded sources if missing in model json
        if grounding_sources:
            existing_urls = {s.get("url") for s in data.get("sources", []) if s.get("url")}
            for s in grounding_sources:
                if s["url"] not in existing_urls:
                    data.setdefault("sources", []).append(s)
                    existing_urls.add(s["url"])

        return data

gemini_service = GeminiService()
