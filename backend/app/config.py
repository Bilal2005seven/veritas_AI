"""
VeritasAI V1 — Application Configuration
All settings are driven by environment variables (see .env.example).
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str = "change-me-in-production"

    # ── CORS ─────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # ── NLP / Transformer ────────────────────────────────────────────────────
    TRANSFORMER_MODEL_NAME: str = "cross-encoder/nli-MiniLM2-L6-H768"

    # ── Embedding / RAG ──────────────────────────────────────────────────────
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_STORE_PATH: str = "./data/vector_store"

    # ── Web Search ───────────────────────────────────────────────────────────
    SERPER_API_KEY: str = ""
    WEB_SEARCH_MAX_RESULTS: int = 10

    # ── Knowledge Graph ──────────────────────────────────────────────────────
    # TODO: Add Neo4j / graph-db credentials when graph_engine is implemented.
    GRAPH_DB_URI: str = ""
    GRAPH_DB_USER: str = ""
    GRAPH_DB_PASSWORD: str = ""

    # ── Credibility ──────────────────────────────────────────────────────────
    CREDIBILITY_THRESHOLD_VERIFIED: float = 0.75
    CREDIBILITY_THRESHOLD_CONTRADICTED: float = 0.35

    # ── Gemini ───────────────────────────────────────────────────────────────
    # API key for Google Gemini (generativelanguage.googleapis.com).
    # Used only for explanation generation; never exposed in API responses.
    GEMINI_API_KEY: str = ""
    # Model identifier — expected value: gemini-3.7-flash
    GEMINI_MODEL: str = "gemini-3.7-flash"
    # Timeout in seconds for the Gemini API call.
    GEMINI_TIMEOUT: float = 15.0

    # ── Verdict Decision Thresholds ───────────────────────────────────────────
    # Minimum credibility-weighted support_score to consider a claim VERIFIED.
    # Rationale: 0.35 requires meaningful support from ≥1 credible source with
    # good relevance, preventing a single low-quality result from triggering
    # VERIFIED.
    VERDICT_SUPPORT_THRESHOLD: float = 0.35

    # Minimum credibility-weighted contradiction_score to consider CONTRADICTED.
    # Rationale: Set slightly lower than support (0.30) because a single credible
    # source actively contradicting the claim is a meaningful negative signal
    # worth surfacing.
    VERDICT_CONTRADICTION_THRESHOLD: float = 0.30

    # Minimum confidence for any non-UNVERIFIED verdict.
    # Rationale: With the exponential formula 1 - exp(-n/5), confidence=0.25
    # corresponds to ~1.5 unique domains — effectively requiring ≥2 independent
    # sources before we commit to VERIFIED or CONTRADICTED.
    VERDICT_CONFIDENCE_MIN: float = 0.25

    # Minimum number of usable (non-empty, non-skipped) evidence items required
    # to reach any verdict other than UNVERIFIED.
    VERDICT_MIN_USABLE_EVIDENCE: int = 1

    # Dominance multiplier: if one side's score exceeds the other's by this
    # factor, the dominant side wins the verdict (even without meeting both
    # absolute thresholds cleanly).
    VERDICT_DOMINANCE_FACTOR: float = 2.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
