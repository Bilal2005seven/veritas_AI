"""
VeritasAI — Tests for POST /api/v1/gemini-verify
"""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models.gemini_schemas import GeminiVerdict, GeminiVerifyResponse

client = TestClient(app)


def test_gemini_verify_empty_inputs_rejected():
    """Verify that submitting no claim, url, or image returns HTTP 400."""
    response = client.post("/api/v1/gemini-verify", data={})
    assert response.status_code == 400
    assert "At least one input must be provided" in response.json()["detail"]


def test_gemini_verify_invalid_url_rejected():
    """Verify that malformed URLs return HTTP 400."""
    response = client.post(
        "/api/v1/gemini-verify",
        data={"url": "ftp://invalid-protocol.com/article"},
    )
    assert response.status_code == 400
    assert "Invalid URL" in response.json()["detail"]


def test_gemini_verify_invalid_image_type_rejected():
    """Verify that non-image uploads (e.g. text/plain or application/pdf) return HTTP 400."""
    fake_file = io.BytesIO(b"%PDF-1.4 fake pdf content")
    response = client.post(
        "/api/v1/gemini-verify",
        files={"image": ("test.pdf", fake_file, "application/pdf")},
    )
    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["detail"]


def test_gemini_verify_mocked_success():
    """Verify that a successful Gemini verification maps properly to GeminiVerifyResponse schema."""
    mock_response = GeminiVerifyResponse(
        verdict=GeminiVerdict.SUPPORTED,
        confidence=0.92,
        claim="Test Claim Verified",
        summary="Based on the available evidence, this claim is supported.",
        claims=[],
        supporting_evidence=[],
        contradicting_evidence=[],
        sources=[],
        red_flags=[],
        search_queries=["test claim search"],
    )

    with patch(
        "app.services.gemini_verification_service.gemini_verification_service.verify",
        new=AsyncMock(return_value=mock_response),
    ):
        response = client.post(
            "/api/v1/gemini-verify",
            data={"claim": "Test Claim Verified"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"] == "SUPPORTED"
        assert data["confidence"] == 0.92
        assert data["claim"] == "Test Claim Verified"
        assert "search_queries" in data
