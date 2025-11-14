from fastapi import Depends, FastAPI, UploadFile,File
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import Base, engine, session_local
from models import FileDB
from sqlalchemy_utils import database_exists, create_database
from pipeline.detect_and_predict import detect_face
import cv2
import numpy as np
from models import FileDB
app= FastAPI(title="emotional detector")
if not database_exists(engine.url):
    create_database(engine.url)

Base.metadata.create_all(bind=engine)
def get_db():
   db=session_local()
   try:
        yield db
   finally:
        db.close
   
@app.get("/")
def loading():
   return 'Hello Am an Ai Emotional Detector'
@app.post('/predict_emotion')
async def predict_emotion(file: UploadFile =File(...), db: Session=Depends(get_db)):
   data_file= await file.read()
   # the type of data_file is bytes so i need to transform it to numpy array for opencv
   byte_to_numpy= np.frombuffer(data_file, np.uint8)
   #decode the array into an image
   img= cv2.imdecode(byte_to_numpy, cv2.IMREAD_COLOR)

   print(type(img))
   img, score,emotion=detect_face(img)
   print(type(score))
   cv2.imwrite("images/" + file.filename, img)
   print(score, emotion)
   if isinstance(score, np.ndarray):
    score = score.tolist()
   if isinstance(emotion, np.ndarray):
    emotion = emotion.tolist()

# convert everything to basic Python types
   score = [float(s) for s in score]
   emotion = [str(e) for e in emotion]
   import os
   path = os.path.join('images', file.filename)   
   db_file = FileDB(
        image_path= path,
        emotion=emotion,
        confidence= score
    )
   db.add(db_file)
   db.commit()
   db.refresh(db_file)
   return {"confidence": emotion, "score": score}
@app.get('/image_prediction')
def pred(db:Session= Depends(get_db)):
   items= db.query(FileDB).all()
   return items
