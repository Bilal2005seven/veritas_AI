import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env from backend root is loaded
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
    PORT: int = int(os.getenv("PORT", "8002"))
    HOST: str = os.getenv("HOST", "0.0.0.0")
    CORS_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5174,http://127.0.0.1:5174,http://localhost:3000,http://localhost:5173",
        ).split(",")
        if origin.strip()
    ]

settings = Settings()
