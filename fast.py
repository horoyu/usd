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
API_KEY = "okasuke587694"  # 任意の秘密キーに置き換えてください
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
@app.get("/predict", response_model=PredictResponse)
def predict(okasuke587694: str = Security(get_api_key)):
    import yfinance as yf
    import pandas as pd
    from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
    from ta.volatility import AverageTrueRange, BollingerBands
    from datetime import datetime
    # 1. データ取得（直近60営業日程度を取得）
    df = yf.download("JPY=X", period="3mo")  # 必要最低限の期間

    # 2. テクニカル指標計算
    close = df[['Close']].squeeze()
    high = df[['High']].squeeze()
    low = df[['Low']].squeeze()
    open_ = df[['Open']].squeeze()

    df['ema_10'] = EMAIndicator(close=close, window=10).ema_indicator()
    df['ema_50'] = EMAIndicator(close=close, window=50).ema_indicator()
    df['sma_20'] = SMAIndicator(close=close, window=20).sma_indicator()
    macd = MACD(close=close)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['adx'] = ADXIndicator(high=high, low=low, close=close, window=14).adx()
    df['rsi'] = RSIIndicator(close=close, window=14).rsi()
    df['roc'] = ROCIndicator(close=close, window=12).roc()
    stoch = StochasticOscillator(high=high, low=low, close=close)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['atr'] = AverageTrueRange(high=high, low=low, close=close).average_true_range()
    bb = BollingerBands(close=close)
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    df['bb_width'] = df['bb_bbh'] - df['bb_bbl']

    # 3. ローソク足統計
    df['body'] = close - open_
    df['upper_shadow'] = high - df[['Close', 'Open']].max(axis=1)
    df['lower_shadow'] = df[['Close', 'Open']].min(axis=1) - low
    df['price_change_pct'] = close.pct_change() * 100

    # 4. 欠損除去
    df.dropna(inplace=True)

    # 5. 最新の1件だけをfeaturesに
    latest_features = df.iloc[[-1]].copy()# DataFrame形式で1行保持

    
    #df = pd.DataFrame([data.features])
    
    pred = model.predict(latest_features)[0]
    return {"prediction": pred}





