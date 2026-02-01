from functools import lru_cache
from typing import TYPE_CHECKING, Annotated, Generator, Optional

if TYPE_CHECKING:
    from fastapi import Depends, Request
    from sqlalchemy.orm import Session
else:
    try:
        from fastapi import Depends, Request
        from sqlalchemy.orm import Session
    except ImportError:
        pass

from src.config import Settings
from src.db.interfaces.base import BaseDatabase
from src.services.ollama.client import Ollama_Client


@lru_cache
def get_settings() -> Settings:
    """Get application settings."""
    return Settings()


def get_request_settings(request: Request) -> Settings:
    """Get settings from the request state."""
    return request.app.state.settings


def get_database(request: Request) -> BaseDatabase:
    """Get database from the request state."""
    return request.app.state.database


def get_db_session(database: Annotated[BaseDatabase, Depends(get_database)]) -> Generator[Session, None, None]:
    """Get database session dependency."""
    with database.get_session() as session:
        yield session


# Dependency annotations
SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[BaseDatabase, Depends(get_database)]
SessionDep = Annotated[Session, Depends(get_db_session)]
