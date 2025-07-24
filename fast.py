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
    from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
    from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
    from ta.volatility import AverageTrueRange, BollingerBands
    from datetime import datetime

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from datetime import date
    from pytrends.request import TrendReq
    
    
    start = "2024-01-01"
    end = date.today()
    
    # 1. データ取得
    df = yf.download("JPY=X", start=start, end=end)
    
    # 2. テクニカル指標追加
    close = df[['Close']].squeeze()
    high = df[['High']].squeeze()
    low = df[['Low']].squeeze()
    open_ = df[['Close']].squeeze().shift(1)
    
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
    
    from fredapi import Fred
    fred = Fred(api_key="79e81d3176dbac9d2cb49523d1441ca0")
    
    df["cpi_us"] = fred.get_series("CPIAUCSL", start, end).resample("D").ffill()
    df["gdp_us"] = fred.get_series("GDPC1", start, end).resample("D").interpolate()
    df["dgs10"] = fred.get_series('DGS10', observation_start=start, observation_end=end).resample("D").ffill()
    
    nikkei = yf.download("^N225", start=start, end=end)[['Close']]
    sp500 = yf.download("^GSPC", start=start, end=end)[['Close']]
    gold = yf.download("GC=F", start=start, end=end)[['Close']]
    oil = yf.download("CL=F", start=start, end=end)[['Close']]
    
    df = df.join(nikkei.rename(columns={'Close': 'nikkei_close'}))
    df = df.join(sp500.rename(columns={'Close': 'sp500_close'}))
    df = df.join(gold.rename(columns={'Close': 'gold_close'}))
    df = df.join(oil.rename(columns={'Close': 'oil_close'}))
    
    pytrends = TrendReq()
    pytrends.build_payload(["USDJPY"], timeframe=f"{start} {end}")
    df["trend"] = pytrends.interest_over_time()["USDJPY"].resample("D").ffill()
    
    
    # 4. ターゲット（翌日上昇なら1）
    df['target'] = (close.shift(-1) > close).astype(int)
    
    # 5. 欠損除去
    df.dropna(inplace=True)
    
    # 6. 特徴量・ターゲット分離
    features = df.drop(columns=['target'])
    X = features.select_dtypes(include='number')
    y = df['target']
    
    # 7. 訓練・テスト分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 8. モデル訓練
    model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=14, max_features='sqrt', random_state=42)
    model.fit(X_train, y_train)
    
    # 9. 予測・評価
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # 5. 最新の1件だけをfeaturesに
    latest_features = df.iloc[[-1]].copy()# DataFrame形式で1行保持
    
    #df = pd.DataFrame([data.features])
    
    pred = model.predict(latest_features)[0]
    return {"prediction": pred}

