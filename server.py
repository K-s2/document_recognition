import uvicorn
import csv
import numpy as np
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

# App creation and model loading
app = FastAPI()
model = load_model("/mnt/d/project/tractor_forecasting_website/my_model.h5")


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
    image = Image.open(io.BytesIO(contents))
    image = image.resize((150, 150))

    # Convert the image to numpy array and reshape
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Predict the class of the image
    predictions = model.predict(image)

    # Get the class with the highest probability
    highest_probability_index = np.argmax(predictions)

    return {"predictions": int(highest_probability_index)} # Assuming your model predicts classes, not probabilities.



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
