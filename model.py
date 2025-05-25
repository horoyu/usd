# -*- coding: utf-8 -*-
"""
Created on Sat May 24 19:15:35 2025

@author: kurow

"""

#!pip install fastapi uvicorn yfinance ta pandas scikit-learn joblib
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, ROCIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datetime import datetime

# 1. データ取得
df = yf.download("JPY=X", start="2005-01-01", end=datetime.today().strftime('%Y-%m-%d'))

# 2. テクニカル指標追加
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


# 保存
import joblib
joblib.dump(model, 'usdjpy_model.pkl')



# features
# 10. 最新データのfeaturesを1行抽出してJSON形式で保存
latest_features = X.iloc[[-1]]  # 最後の1行（最新日）
features_list = latest_features.values.flatten().tolist()

# JSONファイルとして保存
import json
with open("features.json", "w") as f:
    json.dump({"features": features_list}, f, indent=2)

print("最新の特徴量をfeatures.jsonに保存しました。")




"""
# 可視化
from sklearn.tree import plot_tree

estimator = model.estimators_[0]
plot_tree(estimator, feature_names=df.columns, filled=True)



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np


# 評価
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')

print("TimeSeries CV Accuracy Scores:", scores)
print("Mean Accuracy: {:.4f}".format(scores.mean()))
"""



import requests

headers = {
    "access_token": "okasuke587694"
}

res = requests.get("https://usd-2hzx.onrender.com/predict", headers=headers)
print(res.json())