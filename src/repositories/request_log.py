from typing import List, Optional
import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from src.models.request_log import Info_Logs
from src.schemas.ai_model import LogsCreate
from sqlalchemy import func


  
class LogsRepository:
    def __init__(self, session: Session, session_id: str = None):
        self.session = session
        self.session_id = session_id or str(uuid.uuid4())

    def create(self, logs: LogsCreate) -> Info_Logs:
        
        self._update_session_activity()
        max_sequence = (
                self.session.query(func.max(Info_Logs.sequence))
                .filter(Info_Logs.session_id == self.session_id)
                .scalar()
            ) or 0
    
        next_sequence = max_sequence + 1

        db_logs = Info_Logs(
            # Don't set id - let it auto-increment
            session_id=self.session_id,  # ← Use session_id here
            sequence=next_sequence,
            **logs.model_dump()
        )
        self.session.add(db_logs)
        self.session.commit()
        self.session.refresh(db_logs)
        return db_logs
    
    def _update_session_activity(self):
        """Update last_activity timestamp for current session."""
        self.session.query(Info_Logs).filter(
            Info_Logs.session_id == self.session_id
        ).update({"last_activity": datetime.now(timezone.utc)})
        self.session.commit()


    def get_recent_logs(self, limit: int = 5) -> list[Info_Logs]:
        """ Retrieve the most recent logs from the current session. """

        logs = self.session.query(Info_Logs).order_by(Info_Logs.sequence.asc()).all()
        return logs[-limit:] if len(logs) > limit else logs
    

    def get_count(self) -> int:
        stmt = select(func.count(Info_Logs.sequence))
        return self.session.scalar(stmt) or 0
    
    def delete_all(self) -> int:
        """Delete all logs for the current session."""
        deleted_count = (
            self.session.query(Info_Logs)
            .filter(Info_Logs.session_id == self.session_id)  # ← Filter by session_id
            .delete()
        )
        self.session.commit()
        return deleted_count
    
    @staticmethod
    def cleanup_inactive_sessions(session: Session, inactive_minutes: int = 30) -> int:
        """
        Delete sessions inactive for more than inactive_minutes.
        Returns the number of logs deleted.
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=inactive_minutes)
        
        # Delete all logs from inactive sessions
        deleted_count = (
            session.query(Info_Logs)
            .filter(Info_Logs.last_activity < cutoff_time)
            .delete(synchronize_session=False)
        )
        session.commit()
        
        return deleted_count
    





