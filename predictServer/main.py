from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import List, Any
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import os

app = FastAPI()

N_THREADS = 4
RANDOM_STATE = 42
MISSING_VALUES = [1]
MODEL_PATH = "catboost_dota_winner_model.cbm"

model = None

def load_model():
    global model
    if model is None:
        model = CatBoostClassifier()
        model.load_model(MODEL_PATH)

@app.on_event("startup")
def startup_event():
    np.random.seed(RANDOM_STATE)
    load_model()

@app.post("/predict")
async def predict_endpoint(request: Request):
    data: List[Any] = await request.json()
    print("DEBUG: received data:", data)
    if model is not None:
        if model.feature_names_ is not None:
            feature_names = list(model.feature_names_)
        else:
            raise RuntimeError("Model feature_names_ is not set")
        print("DEBUG: Model expects features:", feature_names)
        print("DEBUG: Your data length:", len(data))
        if len(data) != len(feature_names):
            raise ValueError(f"Model expects {len(feature_names)} features, but got {len(data)}")
        row = [str(x) for x in data]
        df = pd.DataFrame([row], columns=feature_names)
        print("DEBUG: DataFrame columns:", df.columns)
        print("DEBUG: DataFrame values:", df.values)
        print("DEBUG: Model cat_features:", model.get_cat_feature_indices())
        print("DEBUG: Model predict_proba:", model.predict_proba(df))
        proba = model.predict_proba(df)[0][1]
        prediction = bool(proba > 0.5)
        return JSONResponse(content={"prediction": prediction, "probability": proba})
    else:
        raise RuntimeError("Model is not loaded") 


