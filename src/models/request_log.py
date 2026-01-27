import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, String, Text, Integer
from src.db.interfaces.postgresql import Base


class Info_Logs(Base):
    __tablename__ = "info"

    # Core Data Columns
    sequence = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(String, nullable=False)
    llm_response = Column(String)
    difficulty = Column(String)
    model_name = Column(String)

