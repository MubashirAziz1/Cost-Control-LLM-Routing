import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, String, Text, Integer
from src.db.interfaces.postgresql import Base


class Info_Logs(Base):
    __tablename__ = "info"

    # Core Data Columns
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    sequence = Column(Integer, nullable=False)
    prompt = Column(String, nullable=False)
    llm_response = Column(String)
    difficulty = Column(String)
    model_name = Column(String)

