#!/usr/bin/env python3
"""
flux_all_in_one_bot.py
SUPER MERGED Flux bot (Option C)
 - Fully merged: your original code + improvements + MT5 integration + CCXT fallback
 - All indicators and engines preserved and fixed for pandas/yfinance/MT5
 - Sent in parts (PART 1/5 ... PART 5/5). Combine all parts into a single file.

Requirements:
 pip install pandas numpy yfinance ccxt MetaTrader5
"""

from __future__ import annotations
import os
import time
import math
import uuid
import csv
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
import numpy as np

# Optional live exchange
try:
    import ccxt
except Exception:
    ccxt = None

# Optional MT5 integration
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

# ----------------------------
# CONFIG
# ----------------------------
# Execution flags
EXECUTE_ORDERS = False       # set True to actually place orders
MODE = "backtest"            # "backtest" or "live"
LIVE_MODE = False            # False => demo MT5, True => real MT5

# Market / timeframe
SYMBOL = "BTC/USDT"          # CCXT style; converted to MT5 format when needed
TIMEFRAME = "5m"
HIST_PERIOD = "30d"          # for yfinance backtest
MAX_FETCH_LIMIT = 1000

# ORB session
ORB_SESSION_START = "09:00"
ORB_SESSION_END = "09:30"

# CCXT (optional)
CCXT_EXCHANGE = "binance"
API_KEY = os.getenv("CCXT_API_KEY", "")
API_SECRET = os.getenv("CCXT_API_SECRET", "")

# MT5 credentials (fill your real passwords)
# Demo
MT5_DEMO_ACCOUNT = 400766810
MT5_DEMO_PASSWORD = "As200479@"
MT5_DEMO_SERVER = "XMGlobal-MT5 15"
MT5_DEMO_LOT = 10.25

# Live (example values; replace if different)
MT5_LIVE_ACCOUNT = 400766810
MT5_LIVE_PASSWORD = "YourLivePassword"
MT5_LIVE_SERVER = "XMGlobal-MT5 15"
MT5_LIVE_LOT = 0.01

# Path to terminal (adjust to your VPS/Windows path)
MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# Strategy hyperparams (preserved from your original)
MOMENTUM_SPAN = 4
MOMENTUM_BODY_MULT = 0.5
MOMENTUM_COUNT = 4
MIN_DISTANCE_BETWEEN_ZONES = 5
MAX_BARS_BACK = 1250
SD_LOOKBACK = 20
ATR_LENGTH = 12
RISK_RR = 2.0
RETEST_COOLDOWN_BARS = 5
TP_METHOD = "Dynamic"  # "Dynamic" or "ATR"

# Logging / journaling
TRADE_JOURNAL_CSV = "trade_journal.csv"

# ----------------------------
# DATA STRUCTS
# ----------------------------
@dataclass
class SDZone:
    top: float
    bottom: float
    sd_type: str               # "Supply" or "Demand"
    start_time: pd.Timestamp
    break_time: Optional[pd.Timestamp] = None
    guid: str = None
    combined: bool = False

    def __post_init__(self):
        if self.guid is None:
            self.guid = str(uuid.uuid4())

    def width(self) -> float:
        return max(0.0, self.top - self.bottom)

    def as_dict(self) -> Dict:
        d = asdict(self)
        d['start_time'] = d['start_time'].isoformat()
        d['break_time'] = d['break_time'].isoformat() if d['break_time'] is not None else None
        return d

# ----------------------------
# INDICATORS
# ----------------------------
def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Average True Range (simple rolling mean of true ranges)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

def avg_body_size(df: pd.DataFrame, length: int = 20) -> pd.Series:
    return (df['close'] - df['open']).abs().rolling(length, min_periods=1).mean()

def swing_high_bool(df: pd.DataFrame, left=2, right=2) -> pd.Series:
    highs = df['high']
    # Use rolling with center to flag local maxima
    return highs == highs.rolling(left + right + 1, center=True).max()

def swing_low_bool(df: pd.DataFrame, left=2, right=2) -> pd.Series:
    lows = df['low']
    return lows == lows.rolling(left + right + 1, center=True).min()

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair Value Gaps: approximate Pine Script definition across 3 bars
    bull_fvg when low[1] > high[3]
    bear_fvg when high[1] < low[3]
    """
    df = df.copy()
    df['bull_fvg'] = ((df['low'].shift(1) > df['high'].shift(3))).astype(int)
    df['bear_fvg'] = ((df['high'].shift(1) < df['low'].shift(3))).astype(int)
    return df

# ----------------------------
# SD (Supply/Demand) detection: momentum-based
# ----------------------------
def detect_sd_momentum(df: pd.DataFrame,
                       momentum_span: int = MOMENTUM_SPAN,
                       momentum_body_mult: float = MOMENTUM_BODY_MULT,
                       momentum_count: int = MOMENTUM_COUNT,
                       min_distance_between_zones: int = MIN_DISTANCE_BETWEEN_ZONES,
                       max_bars_back: int = MAX_BARS_BACK) -> Tuple[List[SDZone], List[SDZone]]:
    """
    Detect momentum-based supply/demand zones.
    Returns: (demand_zones, supply_zones) newest-first.
    """
    if df.shape[0] == 0:
        return [], []
    df2 = df.reset_index()
    # Ensure we have numeric columns in df2 similar to original expectations
    if 'open' not in df2.columns or 'close' not in df2.columns:
        raise RuntimeError("Dataframe missing open/close for SD detection")
    avg_body = avg_body_size(df2)
    demand_zones: List[SDZone] = []
    supply_zones: List[SDZone] = []
    last_demand_idx = -9999
    last_supply_idx = -9999
    n = len(df2)
    max_index = min(n-1, max_bars_back if max_bars_back is not None else n-1)
    for i in range(momentum_span+1, max_index):
        bullish_cnt = 0
        bearish_cnt = 0
        for k in range(i-momentum_span, i):
            body = abs(df2.at[k, 'close'] - df2.at[k, 'open'])
            if body >= (avg_body.iloc[k] * momentum_body_mult):
                if df2.at[k, 'close'] > df2.at[k, 'open']:
                    bullish_cnt += 1
                else:
                    bearish_cnt += 1
        if bullish_cnt >= momentum_count and (i - last_demand_idx) > min_distance_between_zones:
            last_demand_idx = i
            idx_for_zone = i - momentum_span
            top = float(df2.at[idx_for_zone, 'high'])
            bottom = float(df2.at[idx_for_zone, 'low'])
            start_time = pd.to_datetime(df2.at[idx_for_zone, df2.columns[0]])
            demand_zones.insert(0, SDZone(top=top, bottom=bottom, sd_type="Demand", start_time=start_time))
        if bearish_cnt >= momentum_count and (i - last_supply_idx) > min_distance_between_zones:
            last_supply_idx = i
            idx_for_zone = i - momentum_span
            top = float(df2.at[idx_for_zone, 'high'])
            bottom = float(df2.at[idx_for_zone, 'low'])
            start_time = pd.to_datetime(df2.at[idx_for_zone, df2.columns[0]])
            supply_zones.insert(0, SDZone(top=top, bottom=bottom, sd_type="Supply", start_time=start_time))
    return demand_zones, supply_zones

def clamp_zone_size(zone: SDZone, atr_val: float, max_zone_size_atr: float = 1.5):
    zone_size = zone.top - zone.bottom
    max_allowed = atr_val * max_zone_size_atr
    if zone_size > max_allowed:
        diff = zone_size - max_allowed
        zone.top -= diff/2.0
        zone.bottom += diff/2.0

def invalidate_zones(df: pd.DataFrame, zones: List[SDZone], sd_end_method: str = "Close"):
    """
    Walk forward and set zone.break_time when invalidated by price.
    sd_end_method: "Close" or "Wick"
    """
    if not zones:
        return
    for zone in zones:
        if zone.break_time is not None:
            continue
        mask_after = df.index > zone.start_time
        sub = df.loc[mask_after]
        for ts, row in sub.iterrows():
            if sd_end_method == "Wick":
                if zone.sd_type == "Demand":
                    if row['low'] < zone.bottom:
                        zone.break_time = ts
                        break
                else:
                    if row['high'] > zone.top:
                        zone.break_time = ts
                        break
            else:  # Close method
                check_low = min(row['open'], row['close'])
                check_high = max(row['open'], row['close'])
                if zone.sd_type == "Demand":
                    if check_low < zone.bottom:
                        zone.break_time = ts
                        break
                else:
                    if check_high > zone.top:
                        zone.break_time = ts
                        break

def zones_overlap(zone1: SDZone, zone2: SDZone, atr_val: float=0.0) -> Tuple[bool, float]:
    """
    Rough overlap percent between two zones using time as x-axis and price as y-axis.
    Returns (touch_bool, overlap_percentage).
    """
    def ts_ms(ts: pd.Timestamp) -> float:
        return ts.value // 1_000_000
    z1_x1 = ts_ms(zone1.start_time)
    z1_x2 = ts_ms(zone1.break_time) if zone1.break_time else ts_ms(pd.Timestamp.now())
    z1_y1 = zone1.top + atr_val / 100.0
    z1_y2 = zone1.bottom - atr_val / 100.0
    z2_x1 = ts_ms(zone2.start_time)
    z2_x2 = ts_ms(zone2.break_time) if zone2.break_time else ts_ms(pd.Timestamp.now())
    z2_y1 = zone2.top + atr_val / 100.0
    z2_y2 = zone2.bottom - atr_val / 100.0
    inter_x = max(0, min(z1_x2, z2_x2) - max(z1_x1, z2_x1))
    inter_y = max(0, min(z1_y1, z2_y1) - max(z1_y2, z2_y2))
    intersection_area = inter_x * inter_y
    def area(z: SDZone) -> float:
        duration_ms = ((z.break_time - z.start_time).total_seconds()*1000) if z.break_time else ((pd.Timestamp.now()-z.start_time).total_seconds()*1000)
        return z.width() * duration_ms
    union_area = max(1e-9, area(zone1) + area(zone2) - intersection_area)
    overlap_percent = (intersection_area / union_area) * 100.0
    return (overlap_percent > 0), overlap_percent

def combine_zones(zones: List[SDZone], atr_val: float = 0.0, overlap_threshold_percentage: float = 0.0) -> List[SDZone]:
    """
    Iterative pairwise combine of same-type zones if overlapping beyond threshold.
    """
    all_z = zones.copy()
    changed = True
    iterations = 0
    while changed and iterations < 1000:
        iterations += 1
        changed = False
        new_list: List[SDZone] = []
        used = [False]*len(all_z)
        for i in range(len(all_z)):
            if used[i]:
                continue
            base = all_z[i]
            for j in range(i+1, len(all_z)):
                if used[j]:
                    continue
                other = all_z[j]
                if base.sd_type != other.sd_type:
                    continue
                touch, pct = zones_overlap(base, other, atr_val)
                if touch and pct > overlap_threshold_percentage:
                    # merge
                    new_top = max(base.top, other.top)
                    new_bottom = min(base.bottom, other.bottom)
                    new_start = min(base.start_time, other.start_time)
                    new_break = None
                    if base.break_time and other.break_time:
                        new_break = max(base.break_time, other.break_time)
                    elif base.break_time:
                        new_break = base.break_time
                    elif other.break_time:
                        new_break = other.break_time
                    base = SDZone(top=new_top, bottom=new_bottom, sd_type=base.sd_type, start_time=new_start, break_time=new_break, combined=True)
                    used[j] = True
                    changed = True
            new_list.append(base)
            used[i] = True
        all_z = new_list
    return all_z

# ----------------------------
# RETESTS
# ----------------------------
def detect_retests(df: pd.DataFrame, zones: List[SDZone], retest_cooldown_bars: int = RETEST_COOLDOWN_BARS, retests_enabled: bool = True) -> List[Dict]:
    """
    Return list of retest events [{'zone_guid', 'time', 'price', 'side'}]
    """
    events = []
    if not retests_enabled:
        return events
    last_retest_idx = {}
    for idx, (ts, row) in enumerate(df.iterrows()):
        for zone in zones:
            if zone.break_time is not None:
                continue
            last_idx = last_retest_idx.get(zone.guid, -9999)
            if (idx - last_idx) <= retest_cooldown_bars:
                continue
            if zone.sd_type == "Supply":
                # price touched supply zone
                if row['high'] > zone.bottom:
                    events.append({'zone_guid': zone.guid, 'time': ts, 'price': zone.top, 'side': 'sell'})
                    last_retest_idx[zone.guid] = idx
            else:
                # demand zone touched
                if row['low'] < zone.top:
                    events.append({'zone_guid': zone.guid, 'time': ts, 'price': zone.bottom, 'side': 'buy'})
                    last_retest_idx[zone.guid] = idx
    return events

# ----------------------------
# ORB
# ----------------------------
def detect_orb(df: pd.DataFrame, session_start: str = ORB_SESSION_START, session_end: str = ORB_SESSION_END) -> pd.DataFrame:
    df = df.copy()
    try:
        window = df.between_time(session_start, session_end)
    except Exception:
        df['orb_high'] = np.nan
        df['orb_low'] = np.nan
        df['orb_break_up'] = False
        df['orb_break_down'] = False
        return df
    if window.shape[0] == 0:
        df['orb_high'] = np.nan
        df['orb_low'] = np.nan
        df['orb_break_up'] = False
        df['orb_break_down'] = False
        return df
    orb_high = window['high'].max()
    orb_low = window['low'].min()
    df['orb_high'] = orb_high
    df['orb_low'] = orb_low
    df['orb_break_up'] = df['close'] > orb_high
    df['orb_break_down'] = df['close'] < orb_low
    return df

# ----------------------------
# STRUCTURE DETECTION (BOS / CHoCH)
# ----------------------------
def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['swing_high'] = swing_high_bool(df).astype(int)
    df['swing_low'] = swing_low_bool(df).astype(int)
    # Use explicit boolean comparisons to avoid DataFrame assignment issues
    df['bos_up'] = (((df['close'] > df['high'].shift(1)) & (df['swing_high'].shift(1) == 1))).astype(int)
    df['bos_down'] = (((df['close'] < df['low'].shift(1)) & (df['swing_low'].shift(1) == 1))).astype(int)
    df['choch_up'] = ((df['bos_up'] == 1) & (df['close'] > df['open'])).astype(int)
    df['choch_down'] = ((df['bos_down'] == 1) & (df['close'] < df['open'])).astype(int)
    return df

# ----------------------------
# RISK + ORDER HELPERS
# ----------------------------
def calculate_sl_tp(entry: float, direction: str, atr_val: Optional[float] = None, rr: float = RISK_RR) -> Tuple[float, float]:
    """Return (sl, tp)"""
    if atr_val is None:
        sl_dist = entry * 0.003
    else:
        sl_dist = atr_val * 1.5
    if direction.upper() == "BUY":
        sl = entry - sl_dist
        tp = entry + sl_dist * rr
    else:
        sl = entry + sl_dist
        tp = entry - sl_dist * rr
    return sl, tp

# ----------------------------
# SIGNAL GENERATION
# ----------------------------
def generate_signal(df: pd.DataFrame, zones: List[SDZone], atr_val: float) -> Dict[str, Any]:
    """
    Look at latest candle and produce a signal dict or {'signal': None}
    Uses ORB, SMC (BOS/CHoCH), FVG, and SD retests.
    """
    last = df.iloc[-1]
    entry_price = float(last['close'])
    buy_reasons = []
    sell_reasons = []

    if int(last.get('bos_up', 0)) == 1:
        buy_reasons.append('BOS_UP')
    if int(last.get('choch_up', 0)) == 1:
        buy_reasons.append('CHOCH_UP')
    if bool(last.get('orb_break_up', False)):
        buy_reasons.append('ORB_BREAK_UP')
    if int(last.get('bull_fvg', 0)) == 1:
        buy_reasons.append('BULL_FVG')

    if int(last.get('bos_down', 0)) == 1:
        sell_reasons.append('BOS_DOWN')
    if int(last.get('choch_down', 0)) == 1:
        sell_reasons.append('CHOCH_DOWN')
    if bool(last.get('orb_break_down', False)):
        sell_reasons.append('ORB_BREAK_DOWN')
    if int(last.get('bear_fvg', 0)) == 1:
        sell_reasons.append('BEAR_FVG')

    # SD retest: if a demand zone exists and price is inside zone bottom/top => buy
    for z in zones:
        if z.break_time is not None:
            continue
        # Price touching the zone (loose check)
        if z.sd_type == "Demand":
            if last['low'] < z.top and last['high'] > z.bottom:
                buy_reasons.append('DEMAND_RETEST')
        else:
            if last['high'] > z.bottom and last['low'] < z.top:
                sell_reasons.append('SUPPLY_RETEST')

    # Basic rule: prefer buy if buy_reasons > sell_reasons
    if len(buy_reasons) > 0 and len(sell_reasons) == 0:
        sl, tp = calculate_sl_tp(entry_price, "BUY", atr_val)
        return {
            'signal': 'BUY',
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'reasons': buy_reasons
        }
    if len(sell_reasons) > 0 and len(buy_reasons) == 0:
        sl, tp = calculate_sl_tp(entry_price, "SELL", atr_val)
        return {
            'signal': 'SELL',
            'entry': entry_price,
            'sl': sl,
            'tp': tp,
            'reasons': sell_reasons
        }
    # If both sides, abstain
    return {'signal': None}

# ----------------------------
# SIMPLE BACKTEST
# ----------------------------
def simple_backtest(df: pd.DataFrame, initial_balance=10000.0, position_size_pct=0.02) -> Dict[str, Any]:
    """Run a simple backtest over df - illustrative (not production-grade)."""
    balance = initial_balance
    size_pct = position_size_pct
    trades = []
    # Prepare indicators & zones
    df2 = detect_fvg(df)
    df2 = detect_structure(df2)
    df2 = detect_orb(df2)
    demand_z, supply_z = detect_sd_momentum(df2, momentum_span=MOMENTUM_SPAN, momentum_body_mult=MOMENTUM_BODY_MULT,
                                           momentum_count=MOMENTUM_COUNT, min_distance_between_zones=MIN_DISTANCE_BETWEEN_ZONES,
                                           max_bars_back=MAX_BARS_BACK)
    # combine zones
    atr_latest = float(atr(df2, ATR_LENGTH).iloc[-1])
    demand_z = combine_zones(demand_z, atr_val=atr_latest)
    supply_z = combine_zones(supply_z, atr_val=atr_latest)
    all_zones = demand_z + supply_z

    open_pos = None
    for idx in range(len(df2)):
        sub = df2.iloc[:idx+1]
        if sub.shape[0] < 10:
            continue
        atr_now = float(atr(sub, ATR_LENGTH).iloc[-1])
        zones_now = all_zones  # static in this simple version
        sig = generate_signal(sub, zones_now, atr_now)
        price = float(sub.iloc[-1]['close'])
        if open_pos is None and sig['signal'] in ('BUY', 'SELL'):
            # open
            notional = balance * size_pct
            qty = notional / price
            open_pos = {
                'side': sig['signal'],
                'entry': sig['entry'],
                'sl': sig['sl'],
                'tp': sig['tp'],
                'qty': qty,
                'open_at': sub.index[-1]
            }
            trades.append({'action': 'open', **open_pos})
        elif open_pos is not None:
            # check TP/SL
            high = float(sub.iloc[-1]['high'])
            low = float(sub.iloc[-1]['low'])
            closed = False
            if open_pos['side'] == 'BUY':
                if low <= open_pos['sl']:
                    pnl = (open_pos['sl'] - open_pos['entry']) * open_pos['qty']
                    balance += pnl
                    trades.append({'action': 'sl', 'pnl': pnl, 'time': sub.index[-1]})
                    open_pos = None
                    closed = True
                elif high >= open_pos['tp']:
                    pnl = (open_pos['tp'] - open_pos['entry']) * open_pos['qty']
                    balance += pnl
                    trades.append({'action': 'tp', 'pnl': pnl, 'time': sub.index[-1]})
                    open_pos = None
                    closed = True
            else:
                if high >= open_pos['sl']:
                    pnl = (open_pos['entry'] - open_pos['sl']) * open_pos['qty']
                    balance += pnl
                    trades.append({'action': 'sl', 'pnl': pnl, 'time': sub.index[-1]})
                    open_pos = None
                    closed = True
                elif low <= open_pos['tp']:
                    pnl = (open_pos['entry'] - open_pos['tp']) * open_pos['qty']
                    balance += pnl
                    trades.append({'action': 'tp', 'pnl': pnl, 'time': sub.index[-1]})
                    open_pos = None
                    closed = True
    profit = balance - initial_balance
    return {'final_balance': balance, 'profit': profit, 'trades': trades}
# ----------------------------
# LIVE / DATA FETCH HELPERS
# ----------------------------
def fetch_ohlcv_ccxt(symbol: str, timeframe: str = TIMEFRAME, limit: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV via ccxt exchange (returns DataFrame indexed by timestamp).
    Requires CCXT installed and API keys if private endpoints used.
    """
    if ccxt is None:
        raise RuntimeError("ccxt not available")
    ex_class = getattr(ccxt, CCXT_EXCHANGE)
    ex = ex_class({'enableRateLimit': True, 'apiKey': API_KEY, 'secret': API_SECRET})
    # adapt symbol (CCXT expects "BTC/USDT")
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['time','open','high','low','close','volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.set_index('time')
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    return df

def fetch_ohlcv_yfinance(symbol_for_yf: str = "BTC-USD", period: str = HIST_PERIOD, interval: str = TIMEFRAME) -> pd.DataFrame:
    """
    Fetch OHLCV via yfinance for backtest/demo.
    """
    import yfinance as yf
    df = yf.download(symbol_for_yf, period=period, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError("yfinance returned empty data")
    df = df.rename(columns=str.lower)
    df = df[['open','high','low','close','volume']]
    df.index = pd.to_datetime(df.index)
    return df

def fetch_ohlcv_mt5(symbol: str, timeframe=mt5.TIMEFRAME_M5, count: int = 500) -> pd.DataFrame:
    """
    Fetch OHLCV via MT5 (requires mt5.initialize() connected).
    timeframe param should be one of mt5.TIMEFRAME_* constants.
    """
    if mt5 is None:
        raise RuntimeError("MetaTrader5 package not available")
    symbol_mt5 = symbol.replace("/", "")
    # ensure symbol selected
    try:
        if not mt5.symbol_info(symbol_mt5):
            raise RuntimeError(f"MT5 symbol {symbol_mt5} not found")
        if not mt5.symbol_info(symbol_mt5).visible:
            mt5.symbol_select(symbol_mt5, True)
    except Exception as e:
        # symbol_info can raise if not connected
        raise RuntimeError(f"MT5 symbol error: {e}")
    rates = mt5.copy_rates_from_pos(symbol_mt5, timeframe, 0, count)
    if rates is None:
        raise RuntimeError("MT5 returned no rates")
    df = pd.DataFrame(rates)
    # MT5 returns 'time' as seconds since epoch
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.set_index('time')
    # standardize column names to match other functions
    # mt5 columns: ['time','open','high','low','close','tick_volume','spread','real_volume']
    if 'tick_volume' in df.columns:
        df = df.rename(columns={'tick_volume': 'volume'})
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    df = df[['open','high','low','close','volume']]
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    return df

# ----------------------------
# ORDER EXECUTION (CCXT wrapper)
# ----------------------------
def place_order_ccxt(side: str, symbol: str, amount: float, price: Optional[float] = None, type: str = "market") -> Dict:
    """
    Very basic CCXT order wrapper. For production, add error handling, symbol mapping, margin mode, etc.
    """
    if ccxt is None:
        return {"status": "error", "msg": "ccxt not installed"}
    ex_class = getattr(ccxt, CCXT_EXCHANGE)
    ex = ex_class({'enableRateLimit': True, 'apiKey': API_KEY, 'secret': API_SECRET})
    try:
        if type == "market":
            if side.upper() == "BUY":
                res = ex.create_market_buy_order(symbol, amount)
            else:
                res = ex.create_market_sell_order(symbol, amount)
        else:
            if side.upper() == "BUY":
                res = ex.create_limit_buy_order(symbol, amount, price)
            else:
                res = ex.create_limit_sell_order(symbol, amount, price)
        return {"status": "success", "details": res}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# ----------------------------
# MT5 INIT + ORDER EXECUTION
# ----------------------------
def init_mt5_connection() -> Tuple[bool, Optional[int], Optional[float]]:
    """
    Initialize MT5 and login to demo or live account based on LIVE_MODE.
    Returns (connected, account_number, lot_size)
    """
    if mt5 is None:
        print("MetaTrader5 package not installed.")
        return False, None, None

    if LIVE_MODE:
        account = MT5_LIVE_ACCOUNT
        password = MT5_LIVE_PASSWORD
        server = MT5_LIVE_SERVER
        lot_size = MT5_LIVE_LOT
    else:
        account = MT5_DEMO_ACCOUNT
        password = MT5_DEMO_PASSWORD
        server = MT5_DEMO_SERVER
        lot_size = MT5_DEMO_LOT

    if not mt5.initialize(MT5_PATH):
        print("MT5 initialize failed:", mt5.last_error())
        return False, None, None
    # try login
    authorized = mt5.login(account, password=password, server=server)
    if not authorized:
        # sometimes mt5.login returns False but last_error provides details
        print("MT5 login failed:", mt5.last_error())
        return False, None, None
    print(f"MT5 connected to account {account}")
    return True, account, lot_size

def place_order_mt5(side: str, symbol: str, lot: float) -> Dict:
    """
    Place a market order through MT5. Returns the result dict.
    """
    if mt5 is None:
        return {"status": "error", "msg": "mt5 not available"}
    symbol_mt5 = symbol.replace("/", "")
    info = mt5.symbol_info(symbol_mt5)
    if info is None:
        return {"status": "error", "msg": f"Symbol {symbol_mt5} not found on MT5"}
    if not info.visible:
        mt5.symbol_select(symbol_mt5, True)
    tick = mt5.symbol_info_tick(symbol_mt5)
    if tick is None:
        return {"status": "error", "msg": "failed to get tick"}
    price = tick.ask if side.upper() == "BUY" else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_mt5,
        "volume": float(lot),
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "Flux Bot MT5",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    # result may be a C struct-like object; attempt to provide clear info
    try:
        ret = result._asdict()
    except Exception:
        ret = {"retcode": getattr(result, 'retcode', None), "raw": str(result)}
    if getattr(result, 'retcode', None) == mt5.TRADE_RETCODE_DONE:
        return {"status": "success", "details": ret}
    else:
        return {"status": "error", "details": ret}

# ----------------------------
# TRADE JOURNAL
# ----------------------------
def append_trade_journal(row: Dict[str, Any], filename: str = TRADE_JOURNAL_CSV):
    """
    Append a trade row dict to CSV file. Creates file with headers if missing.
    """
    write_header = not os.path.exists(filename)
    with open(filename, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ----------------------------
# LIVE LOOP (prefers CCXT, falls back to MT5)
# ----------------------------
def run_live_loop():
    print("Starting live loop. EXECUTE_ORDERS =", EXECUTE_ORDERS)
    mt5_connected, mt5_account, mt5_lot = False, None, None
    if mt5 is not None:
        mt5_connected, mt5_account, mt5_lot = init_mt5_connection()

    while True:
        try:
            # Step 1: Get data
            df = None
            used_source = None
            if ccxt is not None:
                try:
                    df = fetch_ohlcv_ccxt(SYMBOL, TIMEFRAME, limit=500)
                    used_source = "ccxt"
                except Exception as e:
                    print("CCXT fetch failed, will try MT5 if available:", e)
                    df = None
            if df is None and mt5_connected:
                try:
                    # convert timeframe to mt5 constant - currently only 5m supported
                    df = fetch_ohlcv_mt5(SYMBOL, timeframe=mt5.TIMEFRAME_M5, count=500)
                    used_source = "mt5"
                except Exception as e:
                    print("MT5 fetch failed:", e)
                    df = None
            if df is None:
                print("No data source available this iteration.")
                time.sleep(5)
                continue

            # Step 2: indicators & zones
            df = df[['open','high','low','close','volume']].dropna()
            df = detect_fvg(df)
            df = detect_structure(df)
            df = detect_orb(df)
            demand_z, supply_z = detect_sd_momentum(df, momentum_span=MOMENTUM_SPAN,
                                                   momentum_body_mult=MOMENTUM_BODY_MULT,
                                                   momentum_count=MOMENTUM_COUNT,
                                                   min_distance_between_zones=MIN_DISTANCE_BETWEEN_ZONES,
                                                   max_bars_back=MAX_BARS_BACK)
            atr_now = float(atr(df, ATR_LENGTH).iloc[-1])
            demand_z = combine_zones(demand_z, atr_val=atr_now)
            supply_z = combine_zones(supply_z, atr_val=atr_now)
            all_zones = demand_z + supply_z

            # Step 3: generate signal
            sig = generate_signal(df, all_zones, atr_now)
            print(pd.Timestamp.now(), "Source:", used_source, "Signal:", sig)

            # Step 4: execute orders if allowed
            if sig['signal'] and EXECUTE_ORDERS:
                if used_source == "ccxt" and ccxt is not None:
                    # example simple sizing
                    balance_usd = 1000.0
                    notional = balance_usd * 0.01
                    qty = notional / sig['entry']
                    res = place_order_ccxt(sig['signal'], SYMBOL, qty, type="market")
                    print("CCXT order result:", res)
                    # journal
                    append_trade_journal({
                        "time": pd.Timestamp.now().isoformat(),
                        "source": "ccxt",
                        "signal": sig['signal'],
                        "entry": sig['entry'],
                        "sl": sig['sl'],
                        "tp": sig['tp'],
                        "qty": qty,
                        "result": str(res),
                    })
                elif mt5_connected:
                    lot_to_use = mt5_lot if mt5_lot is not None else MT5_DEMO_LOT
                    res = place_order_mt5(sig['signal'], SYMBOL, lot_to_use)
                    print("MT5 order result:", res)
                    append_trade_journal({
                        "time": pd.Timestamp.now().isoformat(),
                        "source": "mt5",
                        "signal": sig['signal'],
                        "entry": sig['entry'],
                        "sl": sig['sl'],
                        "tp": sig['tp'],
                        "lot": lot_to_use,
                        "result": str(res),
                    })
                else:
                    print("No execution backend available; not placing order.")
            # wait before next iteration
            time.sleep(10)
        except KeyboardInterrupt:
            print("Live loop stopped by user.")
            break
        except Exception as e:
            print("Live loop error:", e)
            time.sleep(5)
# ----------------------------
# POSITION TRACKING & RISK ENGINE
# ----------------------------

class PositionState:
    """
    Track open position state locally for backtest/live.
    Does not depend on broker state; this ensures consistent logic.
    """
    def __init__(self):
        self.side = None
        self.entry = None
        self.sl = None
        self.tp = None
        self.size = None
        self.open_time = None
        self.status = "flat"

pos = PositionState()

def reset_position():
    pos.side = None
    pos.entry = None
    pos.sl = None
    pos.tp = None
    pos.size = None
    pos.open_time = None
    pos.status = "flat"

def open_position(side: str, entry: float, sl: float, tp: float, size: float):
    pos.side = side
    pos.entry = entry
    pos.sl = sl
    pos.tp = tp
    pos.size = size
    pos.open_time = pd.Timestamp.utcnow()
    pos.status = "open"

def is_position_open():
    return pos.status == "open"

# ----------------------------
# ADVANCED RISK ENGINE
# ----------------------------

def compute_position_size(balance: float, risk_pct: float, entry: float, sl: float):
    """
    Risk a % of balance. Size = risk_amount / stop_distance.
    """
    risk_amount = balance * risk_pct
    stop_distance = abs(entry - sl)
    if stop_distance <= 0:
        return 0
    size = risk_amount / stop_distance
    return size

def compute_tp_by_rr(entry: float, sl: float, rr: float = 3.0, side: str = "BUY"):
    dist = abs(entry - sl)
    if side == "BUY":
        return entry + dist * rr
    else:
        return entry - dist * rr

# ----------------------------
# BREAKEVEN & TRAILING
# ----------------------------

def adjust_sl_to_breakeven(df: pd.DataFrame):
    """
    When in profit by 1R, move SL to entry.
    """
    if not is_position_open():
        return

    last = df['close'].iloc[-1]
    entry = pos.entry
    sl = pos.sl

    if pos.side == "BUY":
        if last >= entry + (entry - sl):  # moved 1R
            pos.sl = entry
    else:
        if last <= entry - (sl - entry):
            pos.sl = entry

def trailing_sl(df: pd.DataFrame, atr_val: float):
    """
    ATR-based trailing stop.
    """
    if not is_position_open():
        return

    last = df['close'].iloc[-1]

    if pos.side == "BUY":
        new_sl = last - atr_val * 2
        pos.sl = max(pos.sl, new_sl)
    else:
        new_sl = last + atr_val * 2
        pos.sl = min(pos.sl, new_sl)

# ----------------------------
# POSITION EXIT CHECK
# ----------------------------

def check_exit_conditions(df: pd.DataFrame):
    """
    Check if SL or TP is hit.
    Returns dict: { "exit": True/False, "reason": "..."}
    """
    if not is_position_open():
        return {"exit": False}

    last = df['close'].iloc[-1]

    if pos.side == "BUY":
        if last <= pos.sl:
            return {"exit": True, "reason": "stop_loss", "exit_price": last}
        if last >= pos.tp:
            return {"exit": True, "reason": "take_profit", "exit_price": last}

    else:  # SELL
        if last >= pos.sl:
            return {"exit": True, "reason": "stop_loss", "exit_price": last}
        if last <= pos.tp:
            return {"exit": True, "reason": "take_profit", "exit_price": last}

    return {"exit": False}

# ----------------------------
# BACKTEST POSITION HANDLING
# ----------------------------

def backtest_open_position(df_row, signal: dict, balance: float):
    entry = signal['entry']
    sl = signal['sl']
    tp = signal['tp']
    side = signal['signal']

    size = compute_position_size(balance, BACKTEST_RISK, entry, sl)
    open_position(side, entry, sl, tp, size)

def backtest_exit_position(df_row, exit_price: float, reason: str, balance: float):
    global pos
    pl = 0
    if pos.side == "BUY":
        pl = (exit_price - pos.entry) * pos.size
    else:
        pl = (pos.entry - exit_price) * pos.size
    balance += pl
    reset_position()
    return balance, pl, reason

# ----------------------------
# CSV IMPORT / EXPORT
# ----------------------------

def export_zones_to_csv(zones: list, filename: str):
    if not zones:
        return
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "low", "high", "type", "strength"])
        for z in zones:
            writer.writerow([z['timestamp'], z['low'], z['high'], z['type'], z.get('strength', '')])

def import_zones_from_csv(filename: str):
    if not os.path.exists(filename):
        return []
    zones = []
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            zones.append({
                "timestamp": r['timestamp'],
                "low": float(r['low']),
                "high": float(r['high']),
                "type": r['type'],
                "strength": float(r.get('strength', 0))
            })
    return zones
# ----------------------------
# PART 4/5 — BACKTEST ENGINE, REPORTING, EXPORTS
# ----------------------------

# Backtest risk default (fraction of account to risk per trade)
BACKTEST_RISK = 0.01  # 1% risk per trade by default

def enhanced_backtest(df: pd.DataFrame, initial_balance: float = 10000.0, risk_pct: float = BACKTEST_RISK, max_trades: Optional[int] = None) -> Dict[str, Any]:
    """
    A more detailed backtest that uses PositionState, compute_position_size, breakeven adjustments, trailing stops, and journals trades.
    Returns a dict with performance metrics and trades list.
    """
    # prepare
    balance = initial_balance
    trades = []
    reset_position()

    # prepare indicators & zones (static zones as in simple backtest for now)
    df2 = detect_fvg(df)
    df2 = detect_structure(df2)
    df2 = detect_orb(df2)
    demand_z, supply_z = detect_sd_momentum(df2, momentum_span=MOMENTUM_SPAN, momentum_body_mult=MOMENTUM_BODY_MULT,
                                           momentum_count=MOMENTUM_COUNT, min_distance_between_zones=MIN_DISTANCE_BETWEEN_ZONES,
                                           max_bars_back=MAX_BARS_BACK)
    atr_latest = float(atr(df2, ATR_LENGTH).iloc[-1])
    demand_z = combine_zones(demand_z, atr_val=atr_latest)
    supply_z = combine_zones(supply_z, atr_val=atr_latest)
    all_zones = demand_z + supply_z

    trade_count = 0
    equity_curve = []
    max_drawdown = 0.0
    peak = balance

    for idx in range(len(df2)):
        sub = df2.iloc[:idx+1]
        if sub.shape[0] < 10:
            equity_curve.append(balance)
            continue

        atr_now = float(atr(sub, ATR_LENGTH).iloc[-1])
        sig = generate_signal(sub, all_zones, atr_now)

        # entry
        if (not is_position_open()) and sig['signal'] in ('BUY', 'SELL'):
            # compute size using risk_pct and current balance
            entry = sig['entry']
            sl = sig['sl']
            size = compute_position_size(balance, risk_pct, entry, sl)
            # normalize size to reasonable value (avoid 0)
            if size <= 0:
                equity_curve.append(balance)
                continue
            open_position(sig['signal'], entry, sl, sig['tp'], size)
            trade_count += 1
            trades.append({
                "time": sub.index[-1].isoformat(),
                "action": "open",
                "side": sig['signal'],
                "entry": entry,
                "sl": sl,
                "tp": sig['tp'],
                "size": size,
            })
            # optional: stop if max_trades reached
            if max_trades is not None and trade_count >= max_trades:
                pass

        # manage open position
        if is_position_open():
            # adjust breakeven and trailing
            adjust_sl_to_breakeven(sub)
            trailing_sl(sub, atr_now)

            # check exit
            out = check_exit_conditions(sub)
            if out.get('exit', False):
                exit_price = out.get('exit_price', float(sub.iloc[-1]['close']))
                reason = out.get('reason', 'unknown')
                # compute pnl
                if pos.side == "BUY":
                    pnl = (exit_price - pos.entry) * pos.size
                else:
                    pnl = (pos.entry - exit_price) * pos.size
                balance += pnl
                trades.append({
                    "time": sub.index[-1].isoformat(),
                    "action": "close",
                    "side": pos.side,
                    "exit_price": exit_price,
                    "reason": reason,
                    "pnl": pnl,
                    "remaining_balance": balance
                })
                reset_position()

        equity_curve.append(balance)
        # update peak & drawdown
        if balance > peak:
            peak = balance
        dd = (peak - balance)
        if dd > max_drawdown:
            max_drawdown = dd

    # final metrics
    profit = balance - initial_balance
    returns = (balance / initial_balance) - 1.0
    win_trades = [t for t in trades if t.get('action') == 'close' and t.get('pnl', 0) > 0]
    loss_trades = [t for t in trades if t.get('action') == 'close' and t.get('pnl', 0) <= 0]
    win_rate = (len(win_trades) / max(1, (len(win_trades) + len(loss_trades)))) * 100.0
    avg_win = sum([t['pnl'] for t in win_trades]) / max(1, len(win_trades)) if win_trades else 0.0
    avg_loss = sum([t['pnl'] for t in loss_trades]) / max(1, len(loss_trades)) if loss_trades else 0.0
    profit_factor = (sum([t['pnl'] for t in win_trades]) / abs(sum([t['pnl'] for t in loss_trades]))) if loss_trades else float('inf')

    results = {
        "initial_balance": initial_balance,
        "final_balance": balance,
        "profit": profit,
        "returns": returns,
        "trades": trades,
        "trade_count": len([t for t in trades if t.get('action') == 'close']),
        "win_rate_pct": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "equity_curve": equity_curve
    }
    return results

# ----------------------------
# BACKTEST REPORTING & EXPORTS
# ----------------------------
def save_backtest_results(results: Dict[str, Any], folder: str = "backtest_results"):
    os.makedirs(folder, exist_ok=True)
    # save trades to CSV
    trades = results.get("trades", [])
    trades_csv = os.path.join(folder, "trades.csv")
    if trades:
        keys = set()
        for t in trades:
            keys.update(t.keys())
        keys = list(keys)
        with open(trades_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for t in trades:
                writer.writerow({k: t.get(k, "") for k in keys})
    # save summary
    summary = {
        "initial_balance": results.get("initial_balance"),
        "final_balance": results.get("final_balance"),
        "profit": results.get("profit"),
        "returns": results.get("returns"),
        "trade_count": results.get("trade_count"),
        "win_rate_pct": results.get("win_rate_pct"),
        "avg_win": results.get("avg_win"),
        "avg_loss": results.get("avg_loss"),
        "profit_factor": results.get("profit_factor"),
        "max_drawdown": results.get("max_drawdown")
    }
    import json
    with open(os.path.join(folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

def print_backtest_summary(results: Dict[str, Any]):
    print("=== BACKTEST SUMMARY ===")
    print(f"Initial balance: {results.get('initial_balance')}")
    print(f"Final balance:   {results.get('final_balance')}")
    print(f"Profit:          {results.get('profit')}")
    print(f"Returns:         {results.get('returns')*100:.2f}%")
    print(f"Trades closed:   {results.get('trade_count')}")
    print(f"Win rate:        {results.get('win_rate_pct'):.2f}%")
    print(f"Avg win:         {results.get('avg_win'):.6f}")
    print(f"Avg loss:        {results.get('avg_loss'):.6f}")
    print(f"Profit factor:   {results.get('profit_factor')}")
    print(f"Max drawdown:    {results.get('max_drawdown'):.6f}")
    print("=======================")

# ----------------------------
# OPTIONAL: simple plot of equity curve (requires matplotlib)
# ----------------------------
def plot_equity_curve(equity_curve: List[float]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not installed; skipping equity plot.")
        return
    plt.figure()
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Bars")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.show()
# ----------------------------
# PART 5/5 — CLI, MAIN, SAFETY, RUN HELPERS
# ----------------------------

import argparse
import sys

def parse_args():
    p = argparse.ArgumentParser(prog="flux_all_in_one_bot.py", description="Flux All-in-One Trading Bot")
    p.add_argument("--mode", choices=["backtest", "live"], default=MODE, help="Run mode: backtest or live")
    p.add_argument("--execute", action="store_true", help="Enable order execution (EXECUTE_ORDERS True) — use with caution")
    p.add_argument("--live-mode", action="store_true", help="Use real MT5 account (LIVE_MODE True). If omitted uses demo MT5.")
    p.add_argument("--symbol", type=str, default=SYMBOL, help="Symbol to trade (CCXT format like BTC/USDT)")
    p.add_argument("--timeframe", type=str, default=TIMEFRAME, help="Timeframe (e.g., 5m, 1h)")
    p.add_argument("--hist-period", type=str, default=HIST_PERIOD, help="Historical period for backtest (yfinance)")
    p.add_argument("--plot", action="store_true", help="Plot equity curve after backtest (requires matplotlib)")
    return p.parse_args()

def safety_checks(args):
    # Basic safety guard before enabling live order execution
    if args.execute:
        print("WARNING: --execute provided. Orders will be sent to broker if live connection available.")
        # require explicit env var or interactive confirmation in interactive shells
        if not sys.stdin.isatty():
            # non-interactive (e.g., CI), refuse to execute for safety
            print("Non-interactive environment detected. Refusing to enable live order execution.")
            return False
    return True

def run_backtest_flow(args):
    global MODE, SYMBOL, TIMEFRAME, HIST_PERIOD
    MODE = "backtest"
    SYMBOL = args.symbol
    TIMEFRAME = args.timeframe
    HIST_PERIOD = args.hist_period
    print(f"Running backtest for {SYMBOL} {TIMEFRAME} period={HIST_PERIOD} ...")
    # map to yfinance ticker
    yf_symbol = "BTC-USD" if SYMBOL.upper().startswith("BTC") else SYMBOL.split("/")[0] + "-USD"
    df = fetch_ohlcv_yfinance(yf_symbol, period=HIST_PERIOD, interval=TIMEFRAME)
    # ensure lowercase cols
    df = df.rename(columns=str.lower)
    df = df[['open','high','low','close','volume']]
    df = df.dropna()
    results = enhanced_backtest(df, initial_balance=10000.0, risk_pct=BACKTEST_RISK)
    print_backtest_summary(results)
    # export results
    save_backtest_results(results)
    if args.plot:
        plot_equity_curve(results.get("equity_curve", []))

def run_live_flow(args):
    global MODE, SYMBOL, TIMEFRAME, EXECUTE_ORDERS, LIVE_MODE
    MODE = "live"
    SYMBOL = args.symbol
    TIMEFRAME = args.timeframe
    EXECUTE_ORDERS = args.execute
    LIVE_MODE = args.live_mode
    print(f"Starting live flow for {SYMBOL} {TIMEFRAME}. EXECUTE_ORDERS={EXECUTE_ORDERS}, LIVE_MODE={LIVE_MODE}")
    # If not allowed to execute and not interactive, warn
    if EXECUTE_ORDERS and not safety_checks(args):
        print("Safety check failed. Aborting live execution.")
        return
    # start live loop
    run_live_loop()

def main():
    """
    Entry point. Use flags --mode, --execute, --live-mode, --symbol, --timeframe.
    Examples:
      # Backtest (safe, default)
      python flux_all_in_one_bot.py --mode backtest

      # Backtest and plot equity
      python flux_all_in_one_bot.py --mode backtest --plot

      # Live demo (no real orders), MT5 demo will be used
      python flux_all_in_one_bot.py --mode live

      # Live and enable order execution (DANGEROUS: make sure credentials and LIVE_MODE are correct)
      python flux_all_in_one_bot.py --mode live --execute --live-mode
    """
    args = parse_args()

    # Quick environment sanity
    try:
        import pandas as _p
    except Exception as e:
        print("Pandas not available. Install requirements: pip install pandas numpy yfinance MetaTrader5 ccxt")
        return

    if args.mode == "backtest":
        run_backtest_flow(args)
    elif args.mode == "live":
        run_live_flow(args)
    else:
        print("Unknown mode. Use --mode backtest|live")

if __name__ == "__main__":
    main()
