# app.py
import mlflow
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from schemas import PredictIn, PredictOut

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
from PIL import Image

def get_model():
    model = mlflow.pytorch.load_model(model_uri="./sk_model")
    return model


MODEL = get_model()

# Create a FastAPI instance
app = FastAPI()

file_path = './monday'


@app.post("/predict", response_model=PredictOut)
def predict(data: PredictIn) -> PredictOut:
    df = pd.DataFrame([data.dict()])

    def load_and_preprocess_image(image_path):
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        image_path = os.path.join(file_path, image_path)
        image = Image.open(image_path).convert("RGB")

        normalized_tensor = transform(image)
        normalized_tensor = torch.unsqueeze(normalized_tensor, dim=0)

        return normalized_tensor



    pred = MODEL(load_and_preprocess_image(df["image_path"][0])).max(1, keepdim = True)[1].item()
    return PredictOut(target=pred)
