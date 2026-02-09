from typing import List, Optional
import uuid

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
        
        max_sequence = (
                self.session.query(func.max(Info_Logs.sequence))
                .filter(Info_Logs.session_id == self.session_id)
                .scalar()
            ) or 0
    
        next_sequence = max_sequence + 1
        db_logs = Info_Logs(
            session_id=self.session_id,  
            sequence=next_sequence,
            **logs.model_dump()
        )
        self.session.add(db_logs)
        self.session.commit()
        self.session.refresh(db_logs)
        return db_logs


    def get_recent_logs(self, limit: int = 5) -> list[Info_Logs]:
        """ Retrieve the most recent logs from the current session in chronological order. """
        logs = (
            self.session.query(Info_Logs)
            .filter(Info_Logs.session_id == self.session_id)
            .order_by(Info_Logs.sequence.desc())  
            .limit(limit)  
            .all()
        )
        
        # Reverse to get chronological order 
        return list(reversed(logs))



    





