import uvicorn
import csv
import numpy as np
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.models import load_model
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
import io
import cv2
import os
import warnings
from PIL import Image
import io
warnings.filterwarnings('ignore')
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import pandas as pd
import logging



logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
IMG_WIDTH = 150
IMG_HEIGHT = 150
BATCH_SIZE = 32

# App creation and model loading
app = FastAPI()
model = load_model("/mnt/c/project/document_recognition/model (1).h5")


# Set up templates directory
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory="templates")

# Путь к папке со статическими файлами
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/items", response_class=HTMLResponse)
async def read_items(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/prediction_result", response_class=HTMLResponse)
async def show_prediction(request: Request, predictions: str = None):
    return templates.TemplateResponse("prediction.html", {"request": request, "predictions": predictions})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load the image
    # Load the image
    contents = await file.read()

    # Чтение изображения с помощью PIL
    img = Image.open(io.BytesIO(contents))

    # Изменение размера изображения используя 'image_utils'
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))

    # Преобразование изображения в numpy array
    x = image_utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Нормализация изображения
    x /= 255.

    # Предсказание класса изображения
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction)

    return {"predictions": int(predicted_class)} # Assuming your model predicts a single class, not probabilities.



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
