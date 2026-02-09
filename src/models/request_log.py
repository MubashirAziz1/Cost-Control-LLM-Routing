from sqlalchemy import  Column, String, Integer
from src.db.interfaces.postgresql import Base


class Info_Logs(Base):
    __tablename__ = "info"

    # Core Data Columns
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    sequence = Column(Integer, nullable=False)
    prompt = Column(String, nullable=False)
    llm_response = Column(String, nullable=False)
    difficulty = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    
 

