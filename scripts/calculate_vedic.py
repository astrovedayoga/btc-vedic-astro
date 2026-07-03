"""
FASE 2: محاسبه پارامترهای تنجیمی ودیک برای هر روز
Calculate Vedic astrology parameters for each BTC trading day
"""

import sys
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import *

# ----------------------------------------------------------------
# Vedic astrology engine
# ----------------------------------------------------------------
from vedicastro.VedicAstro import VedicHoroscopeData
from flatlib import const

# Nakshatra list (fixed order for feature encoding)
NAKSHATRA_LIST = [
    'Ashwini', 'Bharani', 'Krittika', 'Rohini', 'Mrigashīrsha', 'Ardra',
    'Punarvasu', 'Pushya', 'Āshleshā', 'Maghā', 'PūrvaPhalgunī',
    'UttaraPhalgunī', 'Hasta', 'Chitra', 'Svati', 'Vishakha', 'Anuradha',
    'Jyeshtha', 'Mula', 'PurvaAshadha', 'UttaraAshadha', 'Shravana',
    'Dhanishta', 'Shatabhisha', 'PurvaBhādrapadā', 'UttaraBhādrapadā',
    'Revati'
]

RASHI_LIST = [
    'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
]

PLANETS = ['Sun', 'Moon', 'Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn',
           'Rahu', 'Ketu', 'Uranus', 'Neptune', 'Pluto']

# Days of week (for Hora/daily lords)
WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Tithi mapping (30 tithis)
TITHI_NAMES = [
    'Pratipada', 'Dwitiya', 'Tritiya', 'Chaturthi', 'Panchami',
    'Shashthi', 'Saptami', 'Ashtami', 'Navami', 'Dashami',
    'Ekadashi', 'Dwadashi', 'Trayodashi', 'Chaturdashi', 'Purnima',
    'Pratipada_K', 'Dwitiya_K', 'Tritiya_K', 'Chaturthi_K', 'Panchami_K',
    'Shashthi_K', 'Saptami_K', 'Ashtami_K', 'Navami_K', 'Dashami_K',
    'Ekadashi_K', 'Dwadashi_K', 'Trayodashi_K', 'Chaturdashi_K', 'Amavasya'
]

# ----------------------------------------------------------------
def compute_vedic_for_date(dt: datetime) -> dict:
    """
    Compute Vedic astrology features for a specific date/time.
    Returns a dictionary of features.
    """
    # Use 00:00 UTC each day for consistency
    v = VedicHoroscopeData(
        dt.year, dt.month, dt.day, 0, 0, 0,
        '+00:00',  # UTC
        DEFAULT_LAT, DEFAULT_LON,
        ayanamsa=AYANAMSA,
        house_system=HOUSE_SYSTEM
    )
    chart = v.generate_chart()
    planets_data = v.get_planets_data_from_chart(chart)
    houses_data = v.get_houses_data_from_chart(chart)
    aspects_data = v.get_planetary_aspects(chart)
    
    features = {}
    features['date'] = dt
    
    # --- 1. Moon Nakshatra (critical for market sentiment) ---
    moon_info = None
    for p in planets_data:
        if p.Object == 'Moon':
            moon_info = p
            break
    
    if moon_info:
        features['moon_nakshatra'] = moon_info.Nakshatra
        features['moon_nakshatra_lord'] = moon_info.NakshatraLord
        features['moon_rashi'] = moon_info.Rasi
        features['moon_rashi_lord'] = moon_info.RasiLord
        features['moon_sub_lord'] = moon_info.SubLord
        features['moon_sign_lon'] = moon_info.SignLonDecDeg
    
    # --- 2. Sun position ---
    sun_info = None
    for p in planets_data:
        if p.Object == 'Sun':
            sun_info = p
            break
    if sun_info:
        features['sun_nakshatra'] = sun_info.Nakshatra
        features['sun_rashi'] = sun_info.Rasi
        features['sun_sign_lon'] = sun_info.SignLonDecDeg
    
    # --- 3. Ascendant (Lagna) ---
    asc_info = None
    for p in planets_data:
        if p.Object == 'Asc':
            asc_info = p
            break
    if asc_info:
        features['asc_nakshatra'] = asc_info.Nakshatra
        features['asc_rashi'] = asc_info.Rasi
        features['asc_sign_lon'] = asc_info.SignLonDecDeg
    
    # --- 4. Retrograde planets ---
    for planet in ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn']:
        for p in planets_data:
            if p.Object == planet:
                features[f'retro_{planet.lower()}'] = int(p.isRetroGrade) if p.isRetroGrade is not None else 0
                features[f'{planet.lower()}_rashi'] = p.Rasi
                features[f'{planet.lower()}_nakshatra'] = p.Nakshatra
                features[f'{planet.lower()}_sign_lon'] = p.SignLonDecDeg
    
    # --- 5. Venus-Saturn angular distance (Shukra-Shani separation) ---
    # This is a key Vedic combination for market trends
    venus_info = None
    saturn_info = None
    for p in planets_data:
        if p.Object == 'Venus':
            venus_info = p
        if p.Object == 'Saturn':
            saturn_info = p
    if venus_info and saturn_info:
        venus_lon = venus_info.LonDecDeg
        saturn_lon = saturn_info.LonDecDeg
        sep = abs(venus_lon - saturn_lon) % 360
        sep = min(sep, 360 - sep)
        features['venus_saturn_separation'] = sep
        # Classify the aspect
        if sep < 5:
            features['vs_aspect'] = 'conjunction'  # دقیقاً هم‌درجه
        elif sep < 10:
            features['vs_aspect'] = 'tight'  # نزدیک
        elif 40 < sep < 80:
            features['vs_aspect'] = 'semi_sextile_sextile'
        elif 80 < sep < 100:
            features['vs_aspect'] = 'square'  # تربیع
        elif 110 < sep < 130:
            features['vs_aspect'] = 'trine'  # تأثیر مثبت
        elif 160 < sep < 180:
            features['vs_aspect'] = 'opposition'  # مقابله
        else:
            features['vs_aspect'] = 'other'
        # Sine/cosine encoding for cyclic nature
        features['vs_sep_sin'] = np.sin(2 * np.pi * sep / 360)
        features['vs_sep_cos'] = np.cos(2 * np.pi * sep / 360)

    # --- 6. Rahu & Ketu ---
    for p in planets_data:
        if p.Object == 'Rahu':
            features['rahu_rashi'] = p.Rasi
            features['rahu_nakshatra'] = p.Nakshatra
            features['rahu_retro'] = int(p.isRetroGrade) if p.isRetroGrade is not None else 0
        if p.Object == 'Ketu':
            features['ketu_rashi'] = p.Rasi
            features['ketu_nakshatra'] = p.Nakshatra
            features['ketu_retro'] = int(p.isRetroGrade) if p.isRetroGrade is not None else 0
    
    # --- 7. Day of week (Vedic weekday lord) ---
    features['weekday'] = dt.strftime('%A')
    features['weekday_num'] = dt.weekday()
    
    # --- 8. Aspects (planetary aspects between planets) ---
    # Count major aspects
    aspect_counts = {'conjunction': 0, 'trine': 0, 'square': 0, 'opposition': 0, 'sextile': 0}
    for a in aspects_data:
        aspect_name = a.get('AspectType', str(a))
        if isinstance(aspect_name, str):
            aname = aspect_name.lower()
            if 'conjunction' in aname:
                aspect_counts['conjunction'] += 1
            elif 'trine' in aname:
                aspect_counts['trine'] += 1
            elif 'square' in aname:
                aspect_counts['square'] += 1
            elif 'opposition' in aname:
                aspect_counts['opposition'] += 1
            elif 'sextile' in aname:
                aspect_counts['sextile'] += 1
            elif 'quincunx' in aname:
                aspect_counts['quincunx'] = aspect_counts.get('quincunx', 0) + 1
    
    for k, v in aspect_counts.items():
        features[f'aspect_{k}'] = v
    
    # --- 9. Moon phase approximation ---
    # Sun-Moon angle determines phase
    if sun_info and moon_info:
        angle = abs(moon_info.SignLonDecDeg - sun_info.SignLonDecDeg)
        if angle > 180:
            angle = 360 - angle
        features['moon_sun_angle'] = angle
        features['moon_phase'] = _moon_phase_name(angle)
    
    # --- 10. Eclipse proximity (days to nearest eclipse) ---
    # Simplified: check if Sun and Moon within 15° of Rahu/Ketu
    if sun_info and moon_info:
        for p in planets_data:
            if p.Object == 'Rahu':
                rahu_lon = p.LonDecDeg
                sun_to_rahu = abs(sun_info.LonDecDeg - rahu_lon) % 360
                sun_to_rahu = min(sun_to_rahu, 360 - sun_to_rahu)
                features['sun_rahu_proximity'] = sun_to_rahu
                if sun_to_rahu < 15:
                    features['near_solar_eclipse'] = 1
                else:
                    features['near_solar_eclipse'] = 0
                
                moon_to_rahu = abs(moon_info.LonDecDeg - rahu_lon) % 360
                moon_to_rahu = min(moon_to_rahu, 360 - moon_to_rahu)
                if moon_to_rahu < 15:
                    features['near_lunar_eclipse'] = 1
                else:
                    features['near_lunar_eclipse'] = 0
    
    return features


def _moon_phase_name(angle_deg: float) -> str:
    """Classify moon phase by Sun-Moon angle."""
    if angle_deg < 22.5:
        return 'New_Moon'
    elif angle_deg < 67.5:
        return 'Waxing_Crescent'
    elif angle_deg < 112.5:
        return 'First_Quarter'
    elif angle_deg < 157.5:
        return 'Waxing_Gibbous'
    elif angle_deg < 202.5:
        return 'Full_Moon'
    elif angle_deg < 247.5:
        return 'Waning_Gibbous'
    elif angle_deg < 292.5:
        return 'Last_Quarter'
    elif angle_deg < 337.5:
        return 'Waning_Crescent'
    else:
        return 'New_Moon'


# ----------------------------------------------------------------
def compute_vedic_batch(dates: List[datetime], batch_size: int = 100) -> pd.DataFrame:
    """
    Compute Vedic features for a list of dates.
    Processes in batches with progress reporting.
    """
    all_features = []
    total = len(dates)
    
    print(f"[VEDIC] Computing Vedic parameters for {total:,} days...")
    
    for i, dt in enumerate(dates):
        try:
            features = compute_vedic_for_date(dt)
            all_features.append(features)
        except Exception as e:
            print(f"[VEDIC] Error at {dt.date()}: {e}")
            all_features.append({'date': dt})
        
        if (i + 1) % batch_size == 0:
            pct = (i + 1) / total * 100
            print(f"   {pct:5.1f}% complete ({i+1:,}/{total:,})", flush=True)
    
    df = pd.DataFrame(all_features)
    df.set_index('date', inplace=True)
    
    print(f"[VEDIC] ✓ Computed features for {len(df):,} days")
    print(f"   Features: {len(df.columns)}")
    
    return df


# ----------------------------------------------------------------
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("FASE 2: محاسبه پارامترهای نجومی ودیک")
    print("=" * 60)
    
    # Load BTC price data
    raw_path = os.path.join(RAW_DATA_DIR, "btc_ohlc_daily.csv")
    if not os.path.exists(raw_path):
        print(f"[ERROR] BTC data not found. Run fetch_btc_data.py first.")
        print(f"   Expected: {raw_path}")
        sys.exit(1)
    
    btc_df = pd.read_csv(raw_path, parse_dates=['date'], index_col='date')
    dates = btc_df.index.tolist()
    
    # Compute Vedic features
    vedic_df = compute_vedic_batch(dates)
    
    # Save
    save_path = os.path.join(PROCESSED_DIR, "vedic_features.parquet")
    vedic_df.to_parquet(save_path)
    print(f"\n[SAVE] Vedic features saved to: {save_path}")
    
    # Summary
    print(f"\n📊 Feature Summary:")
    print(f"   Total days: {len(vedic_df):,}")
    print(f"   Categorical columns: {[c for c in vedic_df.columns if vedic_df[c].dtype == 'object']}")
    print(f"   Numeric columns: {[c for c in vedic_df.columns if vedic_df[c].dtype != 'object']}")
    print(f"\n✅ FASE 2 complete.")
