# -*- coding: utf-8 -*-
"""
Created on Sat May 24 22:07:13 2025

@author: kurow
"""

from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# --- APIキー認証用設定 ---
API_KEY = "your_secret_api_key_here"  # 任意の秘密キーに置き換えてください
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API KEY"
        )

# --- モデルロード ---
model = joblib.load('usdjpy_model.pkl')

# --- 入力・出力スキーマ ---
class PredictRequest(BaseModel):
    features: list[float]  # 25個の特徴量を想定

class PredictResponse(BaseModel):
    prediction: int

@app.get("/warm")
def warm():
    return {"status": "ok"}


# --- 予測エンドポイント（APIキー認証付き） ---
@app.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest, okasuke587694: str = Security(get_api_key)):
    df = pd.DataFrame([data.features])
    pred = model.predict(df)[0]
    return {"prediction": pred}

@app.get("/features")
def get_features():
    import json
    with open("features.json", "r") as f:
        data = json.load(f)
    return data
