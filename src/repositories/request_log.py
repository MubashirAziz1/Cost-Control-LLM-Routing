from datetime import datetime
from typing import List, Optional
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.models.request_log import Info_Logs
from src.schemas.ai_model import LogsCreate


class LogsRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, logs: LogsCreate) -> Info_Logs:
        db_logs = Info_Logs(**logs.model_dump())
        self.session.add(db_logs)
        self.session.commit()
        self.session.refresh(db_logs)
        return db_logs

    def get_recent_logs(self, limit: int = 5) -> list[Info_Logs]:
        """ Retrieve the most recent logs from the current session. """

        logs = self.session.query(Info_Logs).order_by(Info_Logs.sequence.asc()).all()
        return logs[-limit:] if len(logs) > limit else logs
    

    def get_count(self) -> int:
        stmt = select(func.count(Info_Logs.sequence))
        return self.session.scalar(stmt) or 0




