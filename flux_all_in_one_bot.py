#!/usr/bin/env python3
"""
flux_all_in_one_bot.py
Combined Flux-style engine with optional MT5 demo/live integration:
 - Indicators: FVG, Supply/Demand (momentum), ORB, BOS/CHoCH, Liquidity
 - Risk: SL/TP calculators
 - Strategy: entry/exit rules using SMC + ORB
 - Modes: backtest (historical) and live (MT5/CCXT)
 - Safe defaults: live trading disabled until EXECUTE_ORDERS = True

Requirements:
  pip install pandas numpy ccxt yfinance MetaTrader5
"""

from __future__ import annotations
import os
import time
import math
import uuid
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
except ImportError:
    mt5 = None

# ----------------------------
# CONFIG
# ----------------------------
EXECUTE_ORDERS = False       # Must set to True to execute real orders!
MODE = "backtest"            # "backtest" or "live"
LIVE_MODE = False            # False = demo MT5, True = live MT5
SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
HIST_PERIOD = "30d"          # only for yfinance demo/backtest
ORB_SESSION_START = "09:00"
ORB_SESSION_END = "09:30"

# ccxt keys (if using live mode) - set as env variables or edit below
CCXT_EXCHANGE = "binance"
API_KEY = os.getenv("CCXT_API_KEY", "")
API_SECRET = os.getenv("CCXT_API_SECRET", "")

# ----------------------------
# MT5 CONFIG
# ----------------------------
# Demo account
MT5_DEMO_ACCOUNT = 10008667090
MT5_DEMO_PASSWORD= “As200479@“
# replace with your demo password
MT5_DEMO_SERVER = "MetaQuotes-Demo"
MT5_DEMO_LOT = 10.25

# Live account
MT5_LIVE_ACCOUNT = 400766810
MT5_LIVE_PASSWORD = "As200479@"  # replace with your live password
MT5_LIVE_SERVER = "XMGlobal-MT5 15"
MT5_LIVE_LOT = 0.01

# Path to your MT5 terminal
MT5_PATH = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

# ----------------------------
# STRATEGY HYPERPARAMS
# ----------------------------
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

# ----------------------------
# MT5 HELPERS
# ----------------------------
def init_mt5():
    """Initialize MT5 based on LIVE_MODE flag"""
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
        print("Failed to initialize MT5:", mt5.last_error())
        return False, None, None
    if not mt5.login(account, password=password, server=server):
        print("Failed to login MT5:", mt5.last_error())
        return False, None, None

    print(f"Connected to MT5 account {account} successfully.")
    return True, account, lot_size

def place_order_mt5(side: str, symbol: str, lot: float) -> Dict:
    """Place a market order in MT5"""
    if mt5 is None:
        return {"status": "error", "msg": "MT5 not available"}

    symbol_mt5 = symbol.replace("/", "")
    symbol_info = mt5.symbol_info(symbol_mt5)
    if symbol_info is None:
        return {"status": "error", "msg": f"Symbol {symbol_mt5} not found in MT5"}

    if not symbol_info.visible:
        mt5.symbol_select(symbol_mt5, True)

    price = mt5.symbol_info_tick(symbol_mt5).ask if side.upper() == "BUY" else mt5.symbol_info_tick(symbol_mt5).bid
    order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol_mt5,
        "volume": lot,
        "type": order_type,
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": "Flux Bot MT5",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return {"status": "success", "msg": "Order executed", "details": result._asdict()}
    else:
        return {"status": "error", "msg": str(result.retcode), "details": result._asdict()}

# ----------------------------
# DATA STRUCTS
# ----------------------------
@dataclass
class SDZone:
    top: float
    bottom: float
    sd_type: str
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
    return highs == highs.rolling(left + right + 1, center=True).max()

def swing_low_bool(df: pd.DataFrame, left=2, right=2) -> pd.Series:
    lows = df['low']
    return lows == lows.rolling(left + right + 1, center=True).min()

def detect_fvg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['bull_fvg'] = ((df['low'].shift(1) > df['high'].shift(3))).astype(int)
    df['bear_fvg'] = ((df['high'].shift(1) < df['low'].shift(3))).astype(int)
    return df

def detect_sd_momentum(df: pd.DataFrame,
                       momentum_span: int = MOMENTUM_SPAN,
                       momentum_body_mult: float = MOMENTUM_BODY_MULT,
                       momentum_count: int = MOMENTUM_COUNT,
                       min_distance_between_zones: int = MIN_DISTANCE_BETWEEN_ZONES,
                       max_bars_back: int = MAX_BARS_BACK) -> Tuple[List[SDZone], List[SDZone]]:
    if df.shape[0] == 0:
        return [], []
    df2 = df.reset_index()
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
            else:
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
        return z.width() * ((z.break_time - z.start_time).total_seconds()*1000 if z.break_time else (pd.Timestamp.now()-z.start_time).total_seconds()*1000)
    union_area = max(1e-9, area(zone1) + area(zone2) - intersection_area)
    overlap_percent = (intersection_area / union_area) * 100.0
    return (overlap_percent > 0), overlap_percent

def combine_zones(zones: List[SDZone], atr_val: float = 0.0, overlap_threshold_percentage: float = 0.0) -> List[SDZone]:
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
# RETEST DETECTION
# ----------------------------
def detect_retests(df: pd.DataFrame, zones: List[SDZone], retest_cooldown_bars: int = RETEST_COOLDOWN_BARS, retests_enabled: bool = True) -> List[Dict]:
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
                if row['high'] > zone.bottom:
                    events.append({'zone_guid': zone.guid, 'time': ts, 'price': zone.top, 'side': 'sell'})
                    last_retest_idx[zone.guid] = idx
            else:
                if row['low'] < zone.top:
                    events.append({'zone_guid': zone.guid, 'time': ts, 'price': zone.bottom, 'side': 'buy'})
                    last_retest_idx[zone.guid] = idx
    return events

# ----------------------------
# ORB DETECTION
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
# STRUCTURE DETECTION
# ----------------------------
def detect_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['swing_high'] = swing_high_bool(df).astype(int)
    df['swing_low'] = swing_low_bool(df).astype(int)
    df['bos_up'] = ((df['close'] > df['high'].shift(1)) & df['swing_high'].shift(1)).astype(int)
    df['bos_down'] = ((df['close'] < df['low'].shift(1)) & df['swing_low'].shift(1)).astype(int)
    df['choch_up'] = (df['bos_up'] & (df['close'] > df['open'])).astype(int)
    df['choch_down'] = (df['bos_down'] & (df['close'] < df['open'])).astype(int)
    return df

# ----------------------------
# RISK CALCULATION
# ----------------------------
def calculate_sl_tp(entry: float, direction: str, atr_val: Optional[float] = None, rr: float = RISK_RR) -> Tuple[float, float]:
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
    last = df.iloc[-1]
    entry_price = float(last['close'])
    buy_reasons = []
    sell_reasons = []

    if last.get('bos_up', 0):
        buy_reasons.append('BOS_UP')
    if last.get('choch_up', 0):
        buy_reasons.append('CHOCH_UP')
    if last.get('orb_break_up', False):
        buy_reasons.append('ORB_BREAK_UP')
    if last.get('bull_fvg', 0):
        buy_reasons.append('BULL_FVG')

    if last.get('bos_down', 0):
        sell_reasons.append('BOS_DOWN')
    if last.get('choch_down', 0):
        sell_reasons.append('CHOCH_DOWN')
    if last.get('orb_break_down', False):
        sell_reasons.append('ORB_BREAK_DOWN')
    if last.get('bear_fvg', 0):
        sell_reasons.append('BEAR_FVG')

    for z in zones:
        if z.break_time is not None:
            continue
        if z.sd_type == "Demand":
            if last['low'] < z.top and last['high'] > z.bottom:
                buy_reasons.append('DEMAND_RETEST')
        else:
            if last['high'] > z.bottom and last['low'] < z.top:
                sell_reasons.append('SUPPLY_RETEST')

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
    return {'signal': None}

# ----------------------------
# BACKTEST
# ----------------------------
def simple_backtest(df: pd.DataFrame, initial_balance=10000.0, position_size_pct=0.02) -> Dict[str, Any]:
    balance = initial_balance
    size_pct = position_size_pct
    trades = []
    df2 = detect_fvg(df)
    df2 = detect_structure(df2)
    df2 = detect_orb(df2)
    demand_z, supply_z = detect_sd_momentum(df2)
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
        sig = generate_signal(sub, all_zones, atr_now)
        price = float(sub.iloc[-1]['close'])
        if open_pos is None and sig['signal'] in ('BUY', 'SELL'):
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
# LIVE LOOP MT5
# ----------------------------
def run_live_mt5():
    connected, account, lot_size = init_mt5()
    if not connected:
        return
    print("Starting live MT5 loop. EXECUTE_ORDERS =", EXECUTE_ORDERS)
    while True:
        try:
            # Fetch latest OHLCV from MT5
            rates = mt5.copy_rates_from_pos("BTCUSD", mt5.TIMEFRAME_M5, 0, 500)
            if rates is None or len(rates) == 0:
                time.sleep(5)
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.set_index('time')
            df = df[['open','high','low','close','tick_volume']]
            df = df.rename(columns={'tick_volume':'volume'})
            df = detect_fvg(df)
            df = detect_structure(df)
            df = detect_orb(df)
            demand_z, supply_z = detect_sd_momentum(df)
            atr_now = float(atr(df, ATR_LENGTH).iloc[-1])
            demand_z = combine_zones(demand_z, atr_val=atr_now)
            supply_z = combine_zones(supply_z, atr_val=atr_now)
            all_zones = demand_z + supply_z
            sig = generate_signal(df, all_zones, atr_now)
            print(pd.Timestamp.now(), "Signal:", sig)
            if sig['signal'] and EXECUTE_ORDERS:
                print(f"Placing {sig['signal']} order in MT5 with lot {lot_size}")
                res = place_order_mt5(sig['signal'], "BTC/USD", lot_size)
                print("Order result:", res)
            time.sleep(10)
        except Exception as e:
            print("MT5 live loop error:", e)
            time.sleep(5)

# ----------------------------
# MAIN
# ----------------------------
def main():
    print("Flux All-in-One Bot - Mode:", MODE)
    if MODE == "backtest":
        import yfinance as yf
        yf_symbol = "BTC-USD" if SYMBOL.startswith("BTC") else SYMBOL.split("/")[0] + "-USD"
        df = yf.download(yf_symbol, period=HIST_PERIOD, interval=TIMEFRAME, progress=False)
        df = df[['Open','High','Low','Close','Volume']].rename(columns=str.lower)
        df = df.dropna()
        result = simple_backtest(df)
        print("Backtest final balance:", result['final_balance'])
        print("Profit:", result['profit'])
        print("Trades count:", len(result['trades']))
    elif MODE == "live":
        run_live_mt5()
    else:
        print("Unknown MODE. Choose 'backtest' or 'live'.")

if __name__ == "__main__":
    main()
