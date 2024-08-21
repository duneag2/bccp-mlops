# schemas.py
from pydantic import BaseModel


class PredictIn(BaseModel):
    image_path: str


class PredictOut(BaseModel):
    target: int
