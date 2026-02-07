from uuid import uuid4
import logging
from sqlalchemy.orm import Session

from fastapi import APIRouter, Depends, HTTPException, Header, status
from fastapi import Cookie, Response as FastAPIResponse

from src.services.groq.factory import make_groq_client
from src.services.ollama.factory import make_ollama_client
from src.dependencies import get_db_session
from src.repositories.request_log import LogsRepository
from src.schemas.ai_model import LogsCreate
from src.schemas.ai_model import Request, Response


router = APIRouter()

VALID_LABELS = ["simple", "medium", "complex"]
DEFAULT_DIFFICULTY = "medium"  

def parse_difficulty(raw: str) -> str:
    """Normalise the classifier output to one of the three known labels."""

    label = raw.strip().lower()
    if label in VALID_LABELS:
        return label
    else:
        return DEFAULT_DIFFICULTY


def build_routing_table(groq_client, ollama_client) -> dict:
    """Return the routing map after the clients have been initialised."""

    return {
        "simple": {
            "handler": ollama_client.easy_task,
            "model_name": "phi3:mini",
        },
        "medium": {
            "handler": groq_client.medium_task,
            "model_name": "llama-3.3-70b-versatile",
        },
        "complex": {
            "handler": groq_client.complex_task,  
            "model_name": "openai/gpt-oss-120b",
        },
    }

# Instantiate the clients once at module loading time.
groq_client = make_groq_client()
ollama_client = make_ollama_client()
ROUTE_TABLE = build_routing_table(groq_client, ollama_client)

# POST method to get the user prompt.
@router.post(
    "/route",
    status_code=status.HTTP_200_OK,
    response_model=Response,
    summary='Prompt Router',
    description="Classify a prompt, route it to the best model, and return the answer.",
    response_description="LLM response against user prompt",
    tags=["Main Logic"]
)
def route(
    body: Request,
    response: FastAPIResponse,
    db: Session = Depends(get_db_session),
    session_id: str = Cookie(default=None),  
) -> Response:
    # If no session ID in cookie, create one and set it
    if not session_id:
        session_id = str(uuid4())
        response.set_cookie(
            key="session_id",           
            value=session_id,
            httponly=True,              
            max_age=86400,              
            samesite="lax",             
            path="/"                    
        )
    
    prompt = body.prompt

    
    # Add the logs to the Postgresql DB
    repo = LogsRepository(db, session_id = session_id) 
    recent_logs = repo.get_recent_logs(limit=5)
    
    if recent_logs:
        logging.info(f"Found {len(recent_logs)} previous messages for context")
    else:
        logging.info("No previous conversation history found")
    
    # Classify the prompt into valid labels.
    raw_label = ollama_client.classify(prompt)
    difficulty = parse_difficulty(raw_label)

    # Route the prompt to the dedicated model on basis of label.
    route = ROUTE_TABLE.get(difficulty)
    if route is None:  
        logging.error("Classifier model is unable to do prompt classification.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"No route configured for difficulty '{difficulty}'.",
        )

    handler = route["handler"]
    model_name = route["model_name"]
    llm_response: str = handler(prompt, recent_logs=recent_logs)
    


    log_entry = LogsCreate(
        model_name=model_name,
        difficulty=difficulty,
        prompt=prompt,
        llm_response=llm_response,
    )
    repo.create(log_entry)

    # Return the llm_response according to the defined Response Schema 
    return Response(
        llm_response=llm_response,
    )

@router.post(
    "/heartbeat",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Keep Session Alive",
    description="Update session activity timestamp to prevent timeout",
    tags=["Session Management"]
)
def heartbeat(
    db: Session = Depends(get_db_session),
    session_id: str = Cookie(default=None),
):
    """
    Frontend calls this periodically to keep the session active.
    Updates last_activity timestamp.
    """
    if session_id:
        repo = LogsRepository(db, session_id=session_id)
        repo._update_session_activity()
        logging.debug(f"Heartbeat received for session {session_id}")
    return None


@router.post(
    "/session/end",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="End Session Immediately",
    description="Delete all logs for the current session when user closes tab",
    tags=["Session Management"]
)
def end_session(
    db: Session = Depends(get_db_session),
    session_id: str = Cookie(default=None),
):
    """
    Called when user closes the browser tab.
    Immediately deletes all session data.
    """
    if session_id:
        repo = LogsRepository(db, session_id=session_id)
        deleted = repo.delete_all()
        logging.info(f"Session ended by user - deleted {deleted} logs for session {session_id}")
    return None


