from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy.orm import Session

from src.services.groq.factory import make_groq_client
from src.services.ollama.factory import make_ollama_client

from src.dependencies import get_db_session
from src.repositories.request_log import LogsRepository
from src.schemas.ai_model import LogsCreate

from src.schemas.ai_model import Request, Response

from fastapi import Cookie, Response as FastAPIResponse
from uuid import uuid4



router = APIRouter()

VALID_LABELS = {"simple", "medium", "complex"}
DEFAULT_DIFFICULTY = "medium"  # fallback when the classifier returns garbage


def _parse_difficulty(raw: str) -> str:
    """Normalise the classifier output to one of the three known labels.

    Returns the DEFAULT_DIFFICULTY when the label is unrecognised, which
    keeps the system running instead of crashing on an unexpected value.
    """
    label = raw.strip().lower()
    return label if label in VALID_LABELS else DEFAULT_DIFFICULTY


def _build_routing_table(groq_client, ollama_client) -> dict:
    """Return the routing map after the clients have been initialised."""
    return {
        "simple": {
            "handler": ollama_client.easy_task,
            "model_name": "Ollama Model (Simple)",
        },
        "medium": {
            "handler": groq_client.medium_task,
            "model_name": "Groq Model (Medium)",
        },
        "complex": {
            "handler": groq_client.medium_task,  # upgrade to a dedicated method when available
            "model_name": "Groq Model (Complex)",
        },
    }



# Clients are stateless (or internally pooled), so we instantiate them once
# at module load time — exactly as main.py does.
_groq_client = make_groq_client()
_ollama_client = make_ollama_client()
_ROUTE_TABLE = _build_routing_table(_groq_client, _ollama_client)

@router.post(
    "/route",
    status_code=status.HTTP_200_OK,
    response_model=Response,
    summary="Classify a prompt, route it to the best model, and return the answer.",
)
def route(
    body: Request,
    response: FastAPIResponse,
    db: Session = Depends(get_db_session),
    session_id: str = Cookie(default=None),  # ← Changed from Header to Cookie
):
    # If no session ID in cookie, create one and set it
    if not session_id:
        session_id = str(uuid4())
        response.set_cookie(
            key="session_id",           # Cookie name
            value=session_id,
            httponly=True,              # Security: prevent JavaScript access
            max_age=86400,              # 24 hours (adjust as needed)
            samesite="lax",             # CSRF protection
            path="/"                    # Available for all routes
        )
    
    prompt = body.prompt

    # 1) Classify ──────────────────────────────────────────────────────
    raw_label = _ollama_client.classify(prompt)
    difficulty = _parse_difficulty(raw_label)

    # 2) Route ─────────────────────────────────────────────────────────
    route = _ROUTE_TABLE.get(difficulty)
    if route is None:  # should never happen, but defensive
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No route configured for difficulty '{difficulty}'.",
        )

    handler = route["handler"]
    model_name = route["model_name"]
    llm_response: str = handler(prompt)

    # 3) Log with session ID ───────────────────────────────────────────
    repo = LogsRepository(db, session_id = session_id)  # ← Pass session_id to repository
    log_entry = LogsCreate(
        model_name=model_name,
        difficulty=difficulty,
        prompt=prompt,
        llm_response=llm_response,
    )
    repo.create(log_entry)

    # 4) Respond ───────────────────────────────────────────────────────
    return Response(
        llm_response=llm_response,
    )