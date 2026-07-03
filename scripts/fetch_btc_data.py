"""
FASE 1: دریافت داده‌های تاریخی قیمت بیت‌کوین
Bitcoin historical price data from multiple free sources
Combines: Bitstamp (2011-2014) + Yahoo Finance (2014-present)
"""

import sys
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))
from config import *


def fetch_bitstamp_early_data() -> pd.DataFrame:
    """
    Fetch BTC/USD daily OHLC from Bitstamp (2011-2014).
    Bitstamp was the earliest reliable exchange.
    """
    url = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
    all_candles = {}
    step = 1000
    
    start_ts = 1313700000  # ~Aug 2011
    end_ts = 1411000000    # ~Sep 2014
    
    print(f"[FETCH] Fetching Bitstamp data (2011-2014)...")
    
    # Fetch in overlapping chunks to ensure coverage
    for chunk_start in range(start_ts, end_ts, int(step * 43200)):  # overlap every 500 days
        params = {
            'step': 86400,
            'limit': step,
            'start': chunk_start,
            'end': min(chunk_start + step * 86400, end_ts)
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.ok:
                data = resp.json()
                if 'data' in data and 'ohlc' in data['data']:
                    for candle in data['data']['ohlc']:
                        ts = int(candle['timestamp'])
                        if ts not in all_candles:
                            all_candles[ts] = {
                                'date': pd.Timestamp(ts, unit='s'),
                                'open': float(candle['open']),
                                'high': float(candle['high']),
                                'low': float(candle['low']),
                                'close': float(candle['close']),
                                'volume': float(candle['volume'])
                            }
            time.sleep(0.3)
        except Exception as e:
            print(f"   [WARN] Chunk error at {chunk_start}: {e}")
    
    if not all_candles:
        print("   [WARN] No Bitstamp data received")
        return None
    
    df = pd.DataFrame(list(all_candles.values()))
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    
    print(f"   ✓ Bitstamp: {len(df)} candles ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_yahoo_data() -> pd.DataFrame:
    """
    Fetch BTC-USD from Yahoo Finance (2014-present).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("[FETCH] yfinance not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
        import yfinance as yf
    
    print(f"[FETCH] Fetching Yahoo Finance BTC-USD...")
    btc = yf.download('BTC-USD', period='max', progress=False)
    
    # yfinance returns MultiIndex columns, flatten
    btc.columns = [col[0].lower() for col in btc.columns]
    btc.index.name = 'date'
    btc.index = pd.to_datetime(btc.index)
    
    print(f"   ✓ Yahoo Finance: {len(btc)} candles ({btc.index[0].date()} → {btc.index[-1].date()})")
    return btc


def combine_sources() -> pd.DataFrame:
    """
    Combine Bitstamp and Yahoo Finance data.
    Fill gaps from Bitstamp, primary source is Yahoo Finance.
    """
    # Get both sources
    bitstamp = fetch_bitstamp_early_data()
    yahoo = fetch_yahoo_data()
    
    if bitstamp is None:
        print("[FETCH] Using Yahoo Finance only")
        return yahoo
    
    # Use Yahoo as primary, fill in earlier data from Bitstamp
    # Bitstamp data is earlier (Yahoo starts 2014-09-17)
    bitstamp_before_yahoo = bitstamp[bitstamp.index < yahoo.index[0]]
    
    combined = pd.concat([bitstamp_before_yahoo, yahoo])
    combined.sort_index(inplace=True)
    combined = combined[~combined.index.duplicated(keep='first')]
    
    print(f"\n[FETCH] ✓ Combined dataset: {len(combined)} candles")
    print(f"   Period: {combined.index[0].date()} → {combined.index[-1].date()}")
    
    return combined


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add returns, volatility, and technical features.
    """
    result = df.copy()
    
    # Daily return
    result['return_1d'] = result['close'].pct_change()
    result['log_return'] = np.log(result['close'] / result['close'].shift(1))
    
    # Forward returns for different horizons (shifted negative = future looking)
    for d in FORWARD_RETURN_DAYS:
        result[f'forward_return_{d}d'] = result['close'].shift(-d) / result['close'] - 1
    
    # Volatility (14-day rolling)
    result['volatility_14d'] = result['return_1d'].rolling(14).std()
    
    # Daily range
    result['daily_range'] = (result['high'] - result['low']) / result['close']
    
    # Simple moving averages
    result['sma_50'] = result['close'].rolling(50).mean()
    result['sma_200'] = result['close'].rolling(200).mean()
    result['sma_ratio_50_200'] = result['sma_50'] / result['sma_200']
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("FASE 1: دریافت داده‌های تاریخی بیت‌کوین")
    print("=" * 60)
    
    df = combine_sources()
    df = add_features(df)
    
    # Save raw
    raw_path = os.path.join(RAW_DATA_DIR, "btc_ohlc_daily.csv")
    df.to_csv(raw_path)
    print(f"\n[SAVE] Raw data: {raw_path}")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   Days: {len(df):,}")
    print(f"   Start: ${df['close'].iloc[0]:,.2f}")
    print(f"   End: ${df['close'].iloc[-1]:,.2f}")
    print(f"   ATH: ${df['close'].max():,.2f}")
    print(f"   Avg daily return: {df['return_1d'].mean()*100:.4f}%")
    print(f"   Daily volatility: {df['return_1d'].std()*100:.2f}%")
    print(f"\n✅ FASE 1 complete.")
