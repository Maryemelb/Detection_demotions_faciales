import pytest
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.detect_and_predict import training
from main import predict_emotion
import cv2
from fastapi.testclient import TestClient
from main import app
from database import session_local
@pytest.fixture
def img():
    current_path= os.getcwd()
    img_path= os.path.join(current_path, 'images','men.jpg')
    with open(img_path, "rb") as f:
        yield {"file": ("men.jpg", f, "image/jpeg")}

def test_model():
    current_path= os.getcwd()
    model_path= os.path.join(current_path, 'best_model.keras')
    assert os.path.exists(model_path)
def test_prediction(img):
    testClient= TestClient(app)
    response= testClient.post('/predict_emotion',files=img )
    #response is a fastapi object so i need to transform it to json
    data= response.json()
    print('test',data['confidence'][0])
    assert isinstance(data['confidence'][0] ,str)
    assert response.status_code ==200
    