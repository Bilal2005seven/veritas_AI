# Veritas Gemini Backend

FastAPI backend for Veritas Gemini, powered by Gemini 3.7 Flash and Google Search grounding.

## Setup & Running

```bash
cd backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY

uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

## Endpoints

- `GET /health` - Service health status
- `POST /api/v1/verify` - Multi-modal verification endpoint (accepts `claim`, `url`, and/or `image`)
