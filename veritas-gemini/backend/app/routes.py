from typing import Optional
from fastapi import APIRouter, File, Form, UploadFile

from app.config import settings
from app.schemas import HealthResponse, VerificationResponse
from app.verification_service import verification_service

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint confirming API status and model."""
    return HealthResponse(
        status="healthy",
        model=settings.GEMINI_MODEL,
        service="Veritas Gemini News Verification",
        version="1.0.0",
    )

@router.post("/api/v1/verify", response_model=VerificationResponse, tags=["Verification"])
async def verify_news_claim(
    claim: Optional[str] = Form(None, description="Text claim to verify"),
    url: Optional[str] = Form(None, description="Public news/article URL"),
    image: Optional[UploadFile] = File(None, description="Screenshot/image containing claim"),
):
    """
    Verify news claims using Gemini 3.7 Flash with Google Search grounding and multimodal analysis.
    Accepts text claim, article URL, image file, or any combination.
    """
    return await verification_service.verify(claim=claim, url=url, image=image)
