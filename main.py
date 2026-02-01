"""Simple interactive script that classifies each user message,
routes it to a model, and logs the conversation to the database."""

from src.services.hf.factory import make_huggingface_client
from src.services.groq.factory import make_groq_client
from src.services.ollama.factory import make_ollama_client

from src.database import get_db_session, get_database
from src.repositories.request_log import LogsRepository
from src.schemas.ai_model import LogsCreate


def main():
    """Initialize the database, then run one demo request and log it."""

    # Explicitly initialize the database (creates engine, tables, etc.)
    database = get_database()
    # At this point PostgreSQLDatabase.startup() has been called via the factory

    # Initialize model clients
    #hf_client = make_huggingface_client()
    groq_client = make_groq_client()
    ollama_client = make_ollama_client()
    print("Chat started (single demo run).")

    user_query = "If I have 3 apples and buy 5 more, then give away 2, how many do I have?"

    # 1) Classify
    cls_text = ollama_client.classify(user_query)
    cls_text = cls_text.lower().strip()
    print(cls_text)

    if cls_text == "simple":
        difficulty = "simple"
    elif cls_text =="medium":
        difficulty = "medium"
    elif cls_text == "complex" :
        difficulty = "complex"
    else:
         difficulty = "medium"  # safe default

    print("..............")
    print(difficulty)
    # 2) Route to model
    if difficulty == "simple":
        response = ollama_client.easy_task(user_query)
        model_name = "HF Model (Simple)"
    elif difficulty == "medium":
        response = groq_client.medium_task(user_query)
        model_name = "Groq Model (Medium)"
    else:
        response = groq_client.medium_task(user_query)
        model_name = "Groq Model (Complex)"

    # 3) Log to DB using the initialized database
    with get_db_session() as session:
        repo = LogsRepository(session)
        log = LogsCreate(
            model_name=model_name,
            difficulty=difficulty,
            prompt=user_query,
            llm_response=response,
        )
        repo.create(log)

    # 4) Show reply
    print(f"\nModel ({model_name}): {response}\n")


if __name__ == "__main__":
    main()
