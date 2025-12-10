#!/usr/bin/env python3
"""
AI Market Scanner for crypto -> Telegram alerts

How it works (simple, practical):
- Uses ccxt to fetch recent OHLCV from Binance
- Builds a small feature set per symbol (returns, SMA slope, RSI, ATR, vol change)
- Labels history by forward return (next N periods)
- Trains a RandomForestClassifier on recent history (fast)
- Predicts next-period direction and probability
- Sends Telegram alert when prob_up or prob_down exceed thresholds
- Avoids duplicate alerts for the same symbol within cooldown

NOT financial advice. Use demo / small allocation. Tune thresholds, features and model as needed.
"""

import os
import time
import math
import json
from typing import List, Dict, Any
import threading

import pandas as pd
import numpy as np
import requests

# ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# exchange
try:
    import ccxt
except Exception:
    ccxt = None

# ----------------------------
# CONFIGURE
# ----------------------------
# Telegram (YOU PROVIDED THESE)
TELEGRAM_BOT_TOKEN = "8118160409:AAHyJulC-sth62TY4GqgJ2Uq2i5MINzgNEA"
TELEGRAM_CHAT_ID = "5783515858"

# Exchange and symbols
EXCHANGE_ID = "binance"
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "ADA/USDT"]
TIMEFRAME = "15m"
HIST_PERIOD_BARS = 500

# Prediction settings
FUTURE_PERIOD = 1
LABEL_THRESHOLD = 0.0025
PROB_UP_THRESHOLD = 0.72
PROB_DOWN_THRESHOLD = 0.72

# Loop & cooldown
LOOP_INTERVAL_SECONDS = 60 * 5
ALERT_COOLDOWN_SECONDS = 60 * 60

# Model hyperparams
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 6
RANDOM_STATE = 42

USE_CCXT = True if ccxt is not None else False

_last_alert_time: Dict[str, float] = {}

# ----------------------------
# UTILITIES
# ----------------------------
def send_telegram_message(text: str) -> Dict[str, Any]:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def now_ts() -> float:
    return time.time()

def in_cooldown(symbol: str) -> bool:
    ts = _last_alert_time.get(symbol)
    if ts is None:
        return False
    return (now_ts() - ts) < ALERT_COOLDOWN_SECONDS

def set_cooldown(symbol: str):
    _last_alert_time[symbol] = now_ts()

# ----------------------------
# INDICATORS / FEATURES
# ----------------------------
def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df['high']; low = df['low']; close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def compute_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(length, min_periods=1).mean()
    ma_down = down.rolling(length, min_periods=1).mean()
    rs = ma_up / (ma_down + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_sma_slope(df: pd.DataFrame, length: int = 20) -> float:
    sma = df['close'].rolling(length).mean()
    if len(sma) < 3:
        return 0.0
    return (sma.iloc[-1] - sma.iloc[-3]) / (sma.iloc[-3] + 1e-9)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_3'] = df['close'].pct_change(3)
    df['ret_5'] = df['close'].pct_change(5)
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_diff'] = (df['sma_10'] - df['sma_20']) / (df['sma_20'] + 1e-9)
    df['rsi_14'] = compute_rsi(df)
    df['atr_14'] = compute_atr(df, 14)
    df['vol_change'] = df['volume'].pct_change(1)

    df = df.dropna().copy()
    df['atr_norm'] = df['atr_14'] / (df['close'] + 1e-9)

    feature_cols = ['ret_1','ret_3','ret_5','sma_diff','rsi_14','atr_norm','vol_change']
    return df, feature_cols

# ----------------------------
# FETCH OHLCV
# ----------------------------
def fetch_ohlcv_ccxt(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    exchange_class = getattr(ccxt, EXCHANGE_ID)
    ex = exchange_class({'enableRateLimit': True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('time')
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    return df

# ----------------------------
# LABELING & MODEL TRAINING
# ----------------------------
def label_future_returns(df: pd.DataFrame, future_period: int = FUTURE_PERIOD, threshold: float = LABEL_THRESHOLD) -> pd.Series:
    future_ret = df['close'].pct_change(periods=future_period).shift(-future_period)
    labels = pd.Series(0, index=df.index)
    labels[future_ret >= threshold] = 1
    labels[future_ret <= -threshold] = -1
    return labels

def train_model_for_symbol(df: pd.DataFrame, feature_cols: List[str]) -> Any:
    X = df[feature_cols].values
    y = df['label'].values

    mask = (y != 0)
    if mask.sum() < 20:
        return None

    X_train, X_test, y_train, y_test =
        train_test_split(X[mask], y[mask], test_size=0.2, random_state=RANDOM_STATE, stratify=y[mask])

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=1
    )
    model.fit(X_train_s, y_train)

    try:
        acc = model.score(X_test_s, y_test)
    except:
        acc = None

    return {"model": model, "scaler": scaler, "acc": acc}

# ----------------------------
# MAIN SYMBOL PROCESSING
# ----------------------------
def analyze_and_alert_symbol(symbol: str):
    try:
        df = fetch_ohlcv_ccxt(symbol, TIMEFRAME, limit=HIST_PERIOD_BARS)
    except Exception as e:
        print(f"[{symbol}] fetch error:", e)
        return

    df, feature_cols = build_features(df)
    if df.shape[0] < 80:
        print(f"[{symbol}] not enough bars ({df.shape[0]})")
        return

    df['label'] = label_future_returns(df)

    trained = train_model_for_symbol(df, feature_cols)
    if trained is None:
        print(f"[{symbol}] insufficient labeled samples")
        return

    model = trained['model']
    scaler = trained['scaler']

    X_last = df[feature_cols].iloc[-1:].values
    X_last_s = scaler.transform(X_last)
    probs = model.predict_proba(X_last_s)[0]
    classes = model.classes_
    prob_map = {cls: probs[i] for i, cls in enumerate(classes)}

    prob_up = float(prob_map.get(1, 0.0))
    prob_down = float(prob_map.get(-1, 0.0))

    last_close = df['close'].iloc[-1]
    change_1 = df['ret_1'].iloc[-1]
    sma_slope = compute_sma_slope(df)
    rsi = df['rsi_14'].iloc[-1]

    print(f"[{symbol}] close={last_close} up={prob_up:.2f} down={prob_down:.2f}")

    if prob_up >= PROB_UP_THRESHOLD and not in_cooldown(symbol):
        msg = (
            f"*Bullish signal* for *{symbol}*\n"
            f"Price: {last_close}\n"
            f"Up Probability: {prob_up:.2f}\n"
            f"RSI: {rsi:.1f}\n"
        )
        send_telegram_message(msg)
        set_cooldown(symbol)

    if prob_down >= PROB_DOWN_THRESHOLD and not in_cooldown(symbol):
        msg = (
            f"*Bearish signal* for *{symbol}*\n"
            f"Price: {last_close}\n"
            f"Down Probability: {prob_down:.2f}\n"
            f"RSI: {rsi:.1f}\n"
        )
        send_telegram_message(msg)
        set_cooldown(symbol)

# ----------------------------
# SCANNER LOOP
# ----------------------------
def scanner_loop(symbols: List[str] = SYMBOLS, interval_seconds: int = LOOP_INTERVAL_SECONDS):
    print("🔥 AI Crypto Market Scanner Started")
    print("Monitoring:", symbols)

    while True:
        t0 = now_ts()
        threads = []

        for s in symbols:
            th = threading.Thread(target=analyze_and_alert_symbol, args=(s,))
            th.start()
            threads.append(th)
            time.sleep(0.2)

        for th in threads:
            th.join()

        elapsed = now_ts() - t0
        sleep_for = max(5, interval_seconds - elapsed)
        print(f"Cycle complete. Sleeping {sleep_for}s")
        time.sleep(sleep_for)

# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    try:
        scanner_loop()
    except KeyboardInterrupt:
        print("Stopped.")
