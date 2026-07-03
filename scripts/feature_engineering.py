"""
FASE 3: مهندسی ویژگی — ادغام قیمت و پارامترهای ودیک
Merge BTC price data with Vedic features into a unified dataframe
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import *

# Import Nakshatra/Rashi lists from vedic calculator
from calculate_vedic import NAKSHATRA_LIST, RASHI_LIST, WEEKDAYS

# ----------------------------------------------------------------
def load_merged_data() -> pd.DataFrame:
    """Load and merge BTC price data with Vedic features."""
    
    # Load price data
    price_path = os.path.join(RAW_DATA_DIR, "btc_ohlc_daily.csv")
    btc = pd.read_csv(price_path, parse_dates=['date'], index_col='date')
    
    # Load Vedic features
    vedic_path = os.path.join(PROCESSED_DIR, "vedic_features.parquet")
    vedic = pd.read_parquet(vedic_path)
    
    # Merge on date index
    df = btc.join(vedic, how='inner')
    
    print(f"[MERGE] Merged dataframe:")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Period: {df.index.min().date()} → {df.index.max().date()}")
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical Vedic features for ML:
    - Nakshatra, Rashi, Weekday → ordinal + one-hot
    - Moon phase → ordinal + one-hot
    """
    result = df.copy()
    
    # Ordinal encoding for Nakshatras (27-fold cycle)
    if 'moon_nakshatra' in result.columns:
        result['moon_nakshatra_num'] = result['moon_nakshatra'].map(
            {n: i for i, n in enumerate(NAKSHATRA_LIST)}
        )
    
    if 'sun_nakshatra' in result.columns:
        result['sun_nakshatra_num'] = result['sun_nakshatra'].map(
            {n: i for i, n in enumerate(NAKSHATRA_LIST)}
        )
    
    # Ordinal encoding for Rashi (12-fold cycle)
    if 'moon_rashi' in result.columns:
        result['moon_rashi_num'] = result['moon_rashi'].map(
            {r: i for i, r in enumerate(RASHI_LIST)}
        )
    
    if 'sun_rashi' in result.columns:
        result['sun_rashi_num'] = result['sun_rashi'].map(
            {r: i for i, r in enumerate(RASHI_LIST)}
        )
    
    if 'asc_rashi' in result.columns:
        result['asc_rashi_num'] = result['asc_rashi'].map(
            {r: i for i, r in enumerate(RASHI_LIST)}
        )
    
    # Cyclic encoding for Nakshatra and Rashi (circular!)
    for col in ['moon_nakshatra_num', 'sun_nakshatra_num']:
        if col in result.columns and not result[col].isna().all():
            result[f'{col}_sin'] = np.sin(2 * np.pi * result[col] / 27.0)
            result[f'{col}_cos'] = np.cos(2 * np.pi * result[col] / 27.0)
    
    for col in ['moon_rashi_num', 'sun_rashi_num', 'asc_rashi_num']:
        if col in result.columns and not result[col].isna().all():
            result[f'{col}_sin'] = np.sin(2 * np.pi * result[col] / 12.0)
            result[f'{col}_cos'] = np.cos(2 * np.pi * result[col] / 12.0)
    
    # Moon phase one-hot
    if 'moon_phase' in result.columns:
        phases = result['moon_phase'].unique()
        for phase in phases:
            result[f'moon_phase_{phase}'] = (result['moon_phase'] == phase).astype(int)
    
    # Retrograde flags (already binary)
    # Aspect counts (already numeric)
    
    return result


def add_market_cycle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add market cycle context features.
    """
    result = df.copy()
    
    # Bull/Bear market regime (200-day MA trend)
    result['above_sma_200'] = (result['close'] > result['sma_200']).astype(int)
    
    # Class label: 1 = bullish next week, 0 = bearish next week
    result['target_7d_bull'] = (result['forward_return_7d'] > BULL_THRESHOLD).astype(int)
    result['target_7d_bear'] = (result['forward_return_7d'] < BEAR_THRESHOLD).astype(int)
    result['target_7d_dir'] = np.sign(result['forward_return_7d'])
    
    # Halving cycles (approximate)
    halving_dates = pd.to_datetime([
        '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'
    ])
    result['days_since_halving'] = 0
    for date in halving_dates:
        prev_date = date - pd.Timedelta(days=365)
        mask = (result.index >= prev_date) & (result.index <= date)
        result.loc[mask, 'days_since_halving'] = (result.loc[mask].index - prev_date).days
    
    return result


def build_final_dataset() -> pd.DataFrame:
    """
    End-to-end: load, merge, encode, save.
    """
    df = load_merged_data()
    df = encode_categorical_features(df)
    df = add_market_cycle_features(df)
    
    # Save
    save_path = os.path.join(PROCESSED_DIR, "btc_vedic_unified.parquet")
    df.to_parquet(save_path)
    print(f"\n[SAVE] Unified dataset saved to: {save_path}")
    
    # Stats
    feature_cols = [c for c in df.columns if c not in 
        ['open', 'high', 'low', 'close', 'volume', 'return_1d', 'log_return',
         'forward_return_1d', 'forward_return_3d', 'forward_return_7d',
         'forward_return_14d', 'forward_return_30d',
         'volatility_14d', 'daily_range', 'sma_50', 'sma_200', 'sma_ratio_50_200']]
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total features: {len(feature_cols)}")
    print(f"   Price features: {len(df.columns) - len(feature_cols)}")
    print(f"   Vedic features: {len(feature_cols)}")
    print(f"   Target columns: target_7d_bull, target_7d_bear, target_7d_dir")
    
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("FASE 3: مهندسی ویژگی — ادغام داده‌ها")
    print("=" * 60)
    
    df = build_final_dataset()
    
    # Quick check: correlation preview
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    vedic_num_cols = [c for c in numeric_cols if any(v in c for v in 
        ['moon_', 'sun_', 'retro_', 'aspect_', 'nakshatra_num', 'rashi_num',
         'rahu_', 'ketu_', 'asc_', 'above_', 'sma_', 'vs_', 'venus_saturn'])]
    
    if vedic_num_cols:
        corr = df[vedic_num_cols + ['target_7d_bull']].corr()['target_7d_bull'].drop('target_7d_bull')
        top_corr = corr.abs().sort_values(ascending=False).head(10)
        print(f"\n🔮 Top 10 Vedic correlations with 7d bullish target:")
        for col, val in top_corr.items():
            sign = '+' if corr[col] > 0 else '-'
            print(f"   {col:35s}: {sign}{abs(val):.4f}")
    
    print(f"\n✅ FASE 3 complete.")
