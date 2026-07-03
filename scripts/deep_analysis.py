"""
تحلیل عمیق: تغییرات بزرگ قیمتی BTC × همه پارامترهای ودیک
Deep-dive: Major BTC price moves vs ALL Vedic parameters + Rule extraction
"""

import sys, os, json, math
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from calculate_vedic import NAKSHATRA_LIST, RASHI_LIST, PLANETS

# ================================================================
# LOAD DATA
# ================================================================
df = pd.read_parquet(os.path.join(PROCESSED_DIR, "btc_vedic_unified.parquet"))
print(f"[LOAD] Dataset: {len(df):,} days x {len(df.columns)} features")

# ================================================================
# 1. IDENTIFY MAJOR PRICE MOVEMENTS (top/bottom 10%)
# ================================================================
threshold_up = df['forward_return_7d'].quantile(0.90)
threshold_down = df['forward_return_7d'].quantile(0.10)

print(f"\n{'='*70}")
print(f"MAJOR BTC PRICE MOVES (7d forward)")
print(f"{'='*70}")
print(f"   Bullish threshold (top 10%): > {threshold_up*100:+.2f}%")
print(f"   Bearish threshold (bot 10%): < {threshold_down*100:+.2f}%")

major_bull = df[df['forward_return_7d'] >= threshold_up].copy()
major_bear = df[df['forward_return_7d'] <= threshold_down].copy()

print(f"   Major bullish days: {len(major_bull):,}")
print(f"   Major bearish days: {len(major_bear):,}")

# ================================================================
# 2. PLANETARY LONGITUDE BY RASHI
# ================================================================
print(f"\n{'='*70}")
print(f"ANALYSIS 1: PLANETARY POSITIONS BY RASHI (SIGN)")
print(f"{'='*70}")

planets_rashi_cols = ['sun_rashi', 'moon_rashi', 'mercury_rashi', 'venus_rashi',
                       'mars_rashi', 'jupiter_rashi', 'saturn_rashi',
                       'rahu_rashi', 'ketu_rashi', 'asc_rashi']

rashi_results = []
for col in planets_rashi_cols:
    if col not in df.columns:
        continue
    planet_name = col.replace('_rashi', '')
    
    for rashi in RASHI_LIST:
        total_count = int((df[col] == rashi).sum())
        if total_count < 10:
            continue
        
        bull_count = int((major_bull[col] == rashi).sum())
        bear_count = int((major_bear[col] == rashi).sum())
        
        bull_ratio = bull_count / len(major_bull) * 100
        bear_ratio = bear_count / len(major_bear) * 100
        expected = total_count / len(df) * 100
        bias = bull_ratio - bear_ratio
        
        if abs(bias) > 2.0:
            rashi_results.append({
                'planet': planet_name,
                'rashi': rashi,
                'total_days': total_count,
                'bull_pct': round(bull_ratio, 1),
                'bear_pct': round(bear_ratio, 1),
                'bias': round(bias, 1),
                'expected_pct': round(expected, 1)
            })

rashi_df = pd.DataFrame(rashi_results)
strong_rashi = rashi_df[rashi_df['bias'].abs() >= 3.0].sort_values('bias', ascending=False)
if len(strong_rashi) > 0:
    print(f"\n   Strong Rashi biases (|bias| >= 3%):")
    print(f"   {'Planet':12s} {'Rashi':14s} {'Bull%':>7s} {'Bear%':>7s} {'Bias':>7s}")
    print(f"   {'-'*50}")
    for _, row in strong_rashi.iterrows():
        arrow = 'BULL' if row['bias'] > 0 else 'BEAR'
        print(f"   {row['planet']:12s} {row['rashi']:14s} {row['bull_pct']:>6.1f}% {row['bear_pct']:>6.1f}% {arrow} {abs(row['bias']):>4.1f}%")

# ================================================================
# 3. RETROGRADE ANALYSIS
# ================================================================
print(f"\n{'='*70}")
print(f"ANALYSIS 2: RETROGRADE PLANETS x MAJOR MOVES")
print(f"{'='*70}")

retro_cols = ['retro_mercury', 'retro_venus', 'retro_mars', 
              'retro_jupiter', 'retro_saturn', 'rahu_retro', 'ketu_retro']

for col in retro_cols:
    if col not in df.columns:
        continue
    planet_name = col.replace('retro_', '').replace('_retro', '')
    if col in ['rahu_retro', 'ketu_retro']:
        planet_name = col.replace('_retro', '')
    
    bull_retro = int((major_bull[col] == 1).sum())
    bear_retro = int((major_bear[col] == 1).sum())
    all_retro = int((df[col] == 1).sum())
    
    if all_retro < 5:
        continue
    
    bull_pct = bull_retro / len(major_bull) * 100
    bear_pct = bear_retro / len(major_bear) * 100
    retro_freq = all_retro / len(df) * 100
    avg_ret = df[df[col] == 1]['forward_return_7d'].mean() * 100
    non_ret = df[df[col] == 0]['forward_return_7d'].mean() * 100
    diff = avg_ret - non_ret
    
    marker = 'BULL' if diff > 0.5 else ('BEAR' if diff < -0.5 else 'NEUT')
    win_retro = (df[df[col] == 1]['forward_return_7d'] > 0).mean() * 100
    
    print(f"   {planet_name:12s} freq={retro_freq:>5.1f}%  bull%={bull_pct:>5.1f}%  bear%={bear_pct:>5.1f}%  "
          f"ret_avg={avg_ret:>+5.2f}%  norm_avg={non_ret:>+5.2f}%  diff={marker} {abs(diff):>4.2f}%  win={win_retro:.0f}%")

# ================================================================
# 4. NAKSHATRA ANALYSIS
# ================================================================
print(f"\n{'='*70}")
print(f"ANALYSIS 3: NAKSHATRA POSITIONS x MAJOR MOVES")
print(f"{'='*70}")

for col in ['moon_nakshatra', 'sun_nakshatra', 'asc_nakshatra']:
    if col not in df.columns:
        continue
    name = col.replace('_nakshatra', '').title()
    print(f"\n   --- {name} Nakshatra ---")
    
    nak_rows = []
    for nak in NAKSHATRA_LIST:
        total = int((df[col] == nak).sum())
        if total < 10:
            continue
        bull_share = int((major_bull[col] == nak).sum()) / len(major_bull) * 100
        bear_share = int((major_bear[col] == nak).sum()) / len(major_bear) * 100
        bias = bull_share - bear_share
        win = (df[df[col] == nak]['forward_return_7d'] > 0).mean() * 100
        avg_ret = df[df[col] == nak]['forward_return_7d'].mean() * 100
        
        nak_rows.append({
            'nak': nak, 'total': total, 'bull': round(bull_share,1),
            'bear': round(bear_share,1), 'bias': round(bias,1),
            'win': round(win,1), 'ret': round(avg_ret,2)
        })
    
    nak_df = pd.DataFrame(nak_rows)
    top = nak_df[nak_df['bias'].abs() >= 2.0].sort_values('bias', ascending=False)
    for _, r in top.iterrows():
        arrow = 'BULL' if r['bias'] > 0 else 'BEAR'
        print(f"   {r['nak']:22s} total={r['total']:>4d}  {arrow} bias={abs(r['bias']):.1f}%  "
              f"win={r['win']:.1f}%  ret={r['ret']:+.2f}%")

# ================================================================
# 5. VENUS-SATURN DETAILED
# ================================================================
print(f"\n{'='*70}")
print(f"ANALYSIS 4: VENUS-SATURN DISTANCE (DETAILED 15-degree bins)")
print(f"{'='*70}")

if 'venus_saturn_separation' in df.columns:
    print(f"\n   {'AngleRng':10s} {'Total':>6s} {'Bull%':>7s} {'Bear%':>7s} {'Bias':>7s} {'Win%':>7s} {'AvgRet':>8s}")
    print(f"   {'-'*55}")
    
    for i in range(12):
        lo, hi = i*15, (i+1)*15
        if hi > 180:
            hi = 180
        mask = (df['venus_saturn_separation'] >= lo) & (df['venus_saturn_separation'] < hi)
        total = int(mask.sum())
        if total < 10:
            continue
        
        bull_share = int((major_bull['venus_saturn_separation'].between(lo, hi, inclusive='left')).sum()) / len(major_bull) * 100
        bear_share = int((major_bear['venus_saturn_separation'].between(lo, hi, inclusive='left')).sum()) / len(major_bear) * 100
        bias = bull_share - bear_share
        win = (df.loc[mask, 'forward_return_7d'] > 0).mean() * 100
        avg_ret = df.loc[mask, 'forward_return_7d'].mean() * 100
        
        marker = 'BULL' if bias > 2 else ('BEAR' if bias < -2 else '')
        print(f"   {lo:>3.0f}-{hi:>3.0f} deg   {total:>5d}   {bull_share:>5.1f}%  {bear_share:>5.1f}%  "
              f"{marker:4s}{abs(bias):>4.1f}%  {win:>5.1f}%  {avg_ret:>+6.2f}%")

# ================================================================
# 6. PLANET PAIR CONJUNCTIONS (SAME SIGN)
# ================================================================
print(f"\n{'='*70}")
print(f"ANALYSIS 5: PLANETS IN SAME RASHI (CONJUNCTION)")
print(f"{'='*70}")

planet_rashi_cols = ['sun_rashi', 'moon_rashi', 'mercury_rashi', 'venus_rashi',
                     'mars_rashi', 'jupiter_rashi', 'saturn_rashi']

pair_rows = []
for i in range(len(planet_rashi_cols)):
    for j in range(i+1, len(planet_rashi_cols)):
        col1, col2 = planet_rashi_cols[i], planet_rashi_cols[j]
        if col1 not in df.columns or col2 not in df.columns:
            continue
        p1 = col1.replace('_rashi', '')
        p2 = col2.replace('_rashi', '')
        
        same_mask = df[col1] == df[col2]
        same_count = int(same_mask.sum())
        if same_count < 20:
            continue
        
        bull_same = int((major_bull[col1] == major_bull[col2]).sum()) if len(major_bull) > 0 else 0
        bear_same = int((major_bear[col1] == major_bear[col2]).sum()) if len(major_bear) > 0 else 0
        
        bull_pct = bull_same / len(major_bull) * 100
        bear_pct = bear_same / len(major_bear) * 100
        base_pct = same_count / len(df) * 100
        bias = bull_pct - bear_pct
        
        if abs(bias) >= 2.0:
            win = df.loc[same_mask, 'forward_return_7d'].mean() * 100
            pair_rows.append({
                'pair': f'{p1}+{p2}',
                'count': same_count,
                'freq': round(base_pct,1),
                'bull': round(bull_pct,1),
                'bear': round(bear_pct,1),
                'bias': round(bias,1),
                'ret': round(win,2)
            })

pair_df = pd.DataFrame(pair_rows).sort_values('bias', ascending=False) if pair_rows else pd.DataFrame()
if len(pair_df) > 0:
    print(f"\n   {'Pair':18s} {'Count':>5s} {'Freq%':>6s} {'Bull%':>7s} {'Bear%':>7s} {'Bias':>7s} {'AvgRet':>8s}")
    print(f"   {'-'*55}")
    for _, r in pair_df.iterrows():
        arrow = 'BULL' if r['bias'] > 0 else 'BEAR'
        print(f"   {r['pair']:18s} {r['count']:>5d} {r['freq']:>5.1f}% {r['bull']:>6.1f}% "
              f"{r['bear']:>6.1f}% {arrow} {abs(r['bias']):>4.1f}% {r['ret']:>+6.2f}%")

# ================================================================
# 7. JUPITER-SATURN CYCLES
# ================================================================
print(f"\n{'='*70}")
print(f"ANALYSIS 6: JUPITER-SATURN CYCLES (20-year cycle)")
print(f"{'='*70}")

if all(c in df.columns for c in ['jupiter_sign_lon', 'saturn_sign_lon']):
    df_temp = df.copy()
    df_temp['js_diff'] = abs(df_temp['jupiter_sign_lon'] - df_temp['saturn_sign_lon']) % 360
    df_temp['js_diff'] = df_temp['js_diff'].apply(lambda x: min(x, 360-x))
    
    print(f"\n   {'Aspect':12s} {'Angle':12s} {'Total':>6s} {'Bull%':>7s} {'Bear%':>7s} {'Bias':>7s} {'AvgRet':>8s}")
    print(f"   {'-'*62}")
    
    aspects = [
        (0, 10, 'Conjunction'),
        (55, 75, 'Sextile'),
        (85, 100, 'Square'),
        (110, 130, 'Trine'),
        (170, 180, 'Opposition')
    ]
    
    for lo, hi, label in aspects:
        mask = df_temp['js_diff'].between(lo, hi)
        total = int(mask.sum())
        if total < 10:
            continue
        
        bull_share = int(df_temp.loc[mask].index.isin(major_bull.index).sum()) / len(major_bull) * 100
        bear_share = int(df_temp.loc[mask].index.isin(major_bear.index).sum()) / len(major_bear) * 100
        bias = bull_share - bear_share
        avg_ret = df_temp.loc[mask, 'forward_return_7d'].mean() * 100
        
        marker = 'BULL' if bias > 1.5 else ('BEAR' if bias < -1.5 else '')
        print(f"   {label:12s} {lo:>3.0f}-{hi:>3.0f} deg   {total:>5d}   {bull_share:>5.1f}%  "
              f"{bear_share:>5.1f}%  {marker:4s}{abs(bias):>4.1f}%  {avg_ret:>+6.2f}%")

# ================================================================
# 8. EXTRACT AND SAVE ALL RULES
# ================================================================
print(f"\n{'='*70}")
print(f"EXTRACTED TRADING RULES (based on deep analysis)")
print(f"{'='*70}")

all_rules = []

# Rule 1-4: Venus Retrograde
if 'retro_venus' in df.columns:
    mask = df['retro_venus'] == 1
    ret = df.loc[mask, 'forward_return_7d']
    if len(ret) > 10:
        ret_avg = ret.mean() * 100
        norm_avg = df[df['retro_venus'] == 0]['forward_return_7d'].mean() * 100
        win = (ret > 0).mean() * 100
        all_rules.append({
            'rule': 'RULE: Venus Retrograde = Caution (Bearish bias)',
            'condition': 'When Venus is retrograde',
            'avg_7d_return': f'{ret_avg:+.2f}%',
            'normal_return': f'{norm_avg:+.2f}%',
            'diff': f'{ret_avg - norm_avg:+.2f}%',
            'win_rate': f'{win:.0f}%',
            'samples': len(ret)
        })

# Venus-Saturn conjunction
if 'venus_saturn_separation' in df.columns:
    mask = df['venus_saturn_separation'] < 5
    ret = df.loc[mask, 'forward_return_7d']
    if len(ret) > 5:
        all_rules.append({
            'rule': 'RULE: Venus-Saturn Conjunction (<5 deg) = Strong Bullish',
            'condition': 'Venus and Saturn within 5 degrees',
            'avg_7d_return': f'{ret.mean()*100:+.2f}%',
            'win_rate': f'{(ret>0).mean()*100:.0f}%',
            'samples': len(ret)
        })

# Venus-Saturn Q2 (32-67 deg)
if 'venus_saturn_separation' in df.columns:
    mask = (df['venus_saturn_separation'] > 32) & (df['venus_saturn_separation'] < 67)
    ret = df.loc[mask, 'forward_return_7d']
    if len(ret) > 10:
        all_rules.append({
            'rule': 'RULE: Venus-Saturn 32-67 deg = Strongest Bullish Returns',
            'condition': 'Venus-Saturn separation between 32 and 67 degrees',
            'avg_7d_return': f'{ret.mean()*100:+.2f}%',
            'win_rate': f'{(ret>0).mean()*100:.0f}%',
            'samples': len(ret)
        })

# Venus-Saturn Q4 (102-138 deg)
if 'venus_saturn_separation' in df.columns:
    mask = (df['venus_saturn_separation'] > 102) & (df['venus_saturn_separation'] < 138)
    ret = df.loc[mask, 'forward_return_7d']
    if len(ret) > 10:
        all_rules.append({
            'rule': 'RULE: Venus-Saturn 102-138 deg = High Win Rate Bullish',
            'condition': 'Venus-Saturn separation between 102 and 138 degrees (Trine zone)',
            'avg_7d_return': f'{ret.mean()*100:+.2f}%',
            'win_rate': f'{(ret>0).mean()*100:.0f}%',
            'samples': len(ret)
        })

# Sun-Rahu proximity
if 'sun_rahu_proximity' in df.columns:
    mask = df['sun_rahu_proximity'] < 15
    ret = df.loc[mask, 'forward_return_7d']
    if len(ret) > 5:
        bull_share = int((major_bull['sun_rahu_proximity'] < 15).sum()) / len(major_bull) * 100
        bear_share = int((major_bear['sun_rahu_proximity'] < 15).sum()) / len(major_bear) * 100
        all_rules.append({
            'rule': 'RULE: Sun-Rahu <15 deg (Eclipse Zone) = Bullish Opportunity',
            'condition': 'Sun within 15 degrees of Rahu (north lunar node)',
            'avg_7d_return': f'{ret.mean()*100:+.2f}%',
            'win_rate': f'{(ret>0).mean()*100:.0f}%',
            'bull_vs_bear': f'Major moves: Bullish {bull_share:.0f}% vs Bearish {bear_share:.0f}%',
            'samples': len(ret)
        })

# Waxing Crescent
if 'moon_phase' in df.columns:
    mask = df['moon_phase'] == 'Waxing_Crescent'
    ret = df.loc[mask, 'forward_return_7d']
    if len(ret) > 10:
        all_rules.append({
            'rule': 'RULE: Waxing Crescent Moon = Bullish',
            'condition': 'Moon phase is Waxing Crescent (day 3-7 after New Moon)',
            'avg_7d_return': f'{ret.mean()*100:+.2f}%',
            'win_rate': f'{(ret>0).mean()*100:.0f}%',
            'samples': len(ret)
        })

# Saturn in each sign
if 'saturn_rashi' in df.columns:
    for rashi in RASHI_LIST:
        mask = df['saturn_rashi'] == rashi
        ret = df.loc[mask, 'forward_return_7d']
        total = len(ret)
        if total < 30:
            continue
        win = (ret > 0).mean() * 100
        avg = ret.mean() * 100
        if abs(avg) > 0.3 or win > 57 or win < 48:
            direction = 'Bullish' if avg > 0 else 'Bearish'
            all_rules.append({
                'rule': f'Saturn in {rashi} = {direction} Period',
                'condition': f'Saturn transiting {rashi} (approx 2.5 years)',
                'avg_7d_return': f'{avg:+.2f}%',
                'win_rate': f'{win:.0f}%',
                'samples': total
            })

# Jupiter-Saturn aspects
if all(c in df.columns for c in ['jupiter_sign_lon', 'saturn_sign_lon']):
    df_temp = df.copy()
    df_temp['js_diff'] = abs(df_temp['jupiter_sign_lon'] - df_temp['saturn_sign_lon']) % 360
    df_temp['js_diff'] = df_temp['js_diff'].apply(lambda x: min(x, 360-x))
    
    for lo, hi, label, direction in [(0,10,'Jup-Sat Conjunction','Bullish'), (85,100,'Jup-Sat Square','Bearish'),
                                      (110,130,'Jup-Sat Trine','Bullish'), (170,180,'Jup-Sat Opposition','Bearish')]:
        mask = df_temp['js_diff'].between(lo, hi)
        ret = df_temp.loc[mask, 'forward_return_7d']
        if len(ret) > 10:
            all_rules.append({
                'rule': f'Jupiter-Saturn {label} = {direction}',
                'condition': f'Jupiter-Saturn angle between {lo}-{hi} degrees',
                'avg_7d_return': f'{ret.mean()*100:+.2f}%',
                'win_rate': f'{(ret>0).mean()*100:.0f}%',
                'samples': len(ret)
            })

# Moon Nakshatra top findings
if 'moon_nakshatra' in df.columns:
    for nak in NAKSHATRA_LIST:
        mask = df['moon_nakshatra'] == nak
        ret = df.loc[mask, 'forward_return_7d']
        total = len(ret)
        if total < 20:
            continue
        win = (ret > 0).mean() * 100
        if win > 58:
            all_rules.append({
                'rule': f'Moon in {nak} = Bullish Nakshatra',
                'condition': f'Moon transiting {nak} nakshatra (~13 hours)',
                'avg_7d_return': f'{ret.mean()*100:+.2f}%',
                'win_rate': f'{win:.0f}%',
                'samples': total
            })
        elif win < 48:
            all_rules.append({
                'rule': f'Moon in {nak} = Bearish Nakshatra',
                'condition': f'Moon transiting {nak} nakshatra (~13 hours)',
                'avg_7d_return': f'{ret.mean()*100:+.2f}%',
                'win_rate': f'{win:.0f}%',
                'samples': total
            })

# Print rules
for i, r in enumerate(all_rules):
    print(f"\n   [{i+1}] {r['rule']}")
    print(f"       Condition: {r['condition']}")
    print(f"       Avg 7d return: {r['avg_7d_return']}  |  Win rate: {r['win_rate']}  |  Samples: {r['samples']}")

# Save to JSON
rules_path = os.path.join(REPORTS_DIR, "deep_analysis_rules.json")
with open(rules_path, 'w') as f:
    json.dump(all_rules, f, indent=2, ensure_ascii=False)
print(f"\n[SAVE] Deep analysis rules: {rules_path}")

# Save CSV summary
csv_path = os.path.join(REPORTS_DIR, "deep_analysis_summary.csv")
summary_rows = []
for r in all_rules:
    summary_rows.append({
        'rule': r['rule'],
        'condition': r['condition'],
        'avg_7d_return': r['avg_7d_return'],
        'win_rate': r['win_rate'],
        'samples': r['samples'],
        'extra': r.get('bull_vs_bear', r.get('diff', ''))
    })
pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
print(f"[SAVE] Summary CSV: {csv_path}")

print(f"\n{'='*70}")
print(f"✅ DEEP ANALYSIS COMPLETE — {len(all_rules)} rules extracted")
print(f"{'='*70}")
