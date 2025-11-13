from database import Base
from sqlalchemy import Column,Integer,String,LargeBinary,DateTime,Float, ARRAY
from datetime import datetime
class FileDB(Base):
    __tablename__= "prediction"
    id= Column(Integer,primary_key=True)
    image_path= Column(String)
    emotion=Column(ARRAY(String))
    confidence=Column(ARRAY(Float))
    created_at=Column(DateTime, default=datetime.utcnow)
