"""
FASE 4: تحلیل آماری — همبستگی قیمت با پارامترهای ودیک
Statistical analysis & correlation between BTC returns and Vedic factors
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, f_oneway, chi2_contingency
import json

sys.path.insert(0, str(Path(__file__).parent))
from config import *
from calculate_vedic import NAKSHATRA_LIST, RASHI_LIST, WEEKDAYS

# ----------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "btc_vedic_unified.parquet")
    return pd.read_parquet(path)


def test_nakshatra_effect(df: pd.DataFrame) -> dict:
    """
    Test: Does Moon Nakshatra affect next-week returns?
    ANOVA: returns grouped by each Nakshatra
    """
    print(f"\n{'='*60}")
    print("📊 Analysis 1: Moon Nakshatra → Forward Returns")
    print(f"{'='*60}")
    
    result = []
    groups = df.groupby('moon_nakshatra')['forward_return_7d']
    
    for name, group in groups:
        if len(group) >= MIN_SAMPLES_PER_RULE:
            avg_return = group.mean() * 100
            std_return = group.std() * 100
            win_rate = (group > 0).mean() * 100
            n = len(group)
            
            # t-test against zero
            t_stat, p_val = ttest_ind(
                group, 
                np.zeros_like(group, dtype=float),  # dummy zero array
                equal_var=False
            )
            # Actually, one-sample t-test is better
            t_stat, p_val = stats.ttest_1samp(group.dropna(), 0)
            
            result.append({
                'nakshatra': name,
                'count': n,
                'avg_return_7d_pct': round(avg_return, 3),
                'std_return_pct': round(std_return, 3),
                'win_rate_pct': round(win_rate, 1),
                't_statistic': round(t_stat, 4),
                'p_value': round(p_val, 6),
                'significant': p_val < ALPHA
            })
    
    result_df = pd.DataFrame(result).sort_values('p_value')
    
    # Bonferroni correction
    n_tests = len(result_df)
    result_df['p_bonferroni'] = np.clip(result_df['p_value'] * n_tests, 0, 1)
    result_df['significant_bonf'] = result_df['p_bonferroni'] < ALPHA
    
    print(f"\n   Total Nakshatras tested: {n_tests}")
    print(f"   Significant (raw p<{ALPHA}): {result_df['significant'].sum()}")
    print(f"   Significant (Bonferroni): {result_df['significant_bonf'].sum()}")
    
    sig = result_df[result_df['significant_bonf']].head(10)
    if len(sig) > 0:
        print(f"\n   Top significant Nakshatras (Bonferroni-corrected):")
        for _, row in sig.iterrows():
            print(f"   ⭐ {row['nakshatra']:20s}: {row['avg_return_7d_pct']:+.2f}%  "
                  f"(win: {row['win_rate_pct']:.0f}%, n={row['count']}, p={row['p_bonferroni']:.5f})")
    
    return result_df


def test_moon_phase_effect(df: pd.DataFrame) -> dict:
    """Test Moon phase effect on returns."""
    print(f"\n{'='*60}")
    print("📊 Analysis 2: Moon Phase → Forward Returns")
    print(f"{'='*60}")
    
    result = []
    for phase in sorted(df['moon_phase'].dropna().unique()):
        subset = df[df['moon_phase'] == phase]['forward_return_7d'].dropna()
        if len(subset) >= MIN_SAMPLES_PER_RULE:
            t_stat, p_val = stats.ttest_1samp(subset, 0)
            result.append({
                'moon_phase': phase,
                'count': len(subset),
                'avg_return_7d_pct': round(subset.mean() * 100, 3),
                'win_rate_pct': round((subset > 0).mean() * 100, 1),
                't_statistic': round(t_stat, 4),
                'p_value': round(p_val, 6)
            })
    
    result_df = pd.DataFrame(result)
    print(f"\n   {'Phase':20s} {'Count':>6s} {'Return':>8s} {'WinRate':>8s} {'p-value':>10s}")
    print(f"   {'-'*55}")
    for _, row in result_df.iterrows():
        sig = ' ⭐' if row['p_value'] < ALPHA else ''
        print(f"   {row['moon_phase']:20s} {row['count']:>6d} {row['avg_return_7d_pct']:>+7.2f}% "
              f"{row['win_rate_pct']:>6.1f}% {row['p_value']:>10.5f}{sig}")
    
    return result_df


def test_retrograde_effect(df: pd.DataFrame) -> dict:
    """Test: Is there a return difference when each planet is retrograde?"""
    print(f"\n{'='*60}")
    print("📊 Analysis 3: Retrograde Planets → Forward Returns")
    print(f"{'='*60}")
    
    retro_planets = ['retro_mercury', 'retro_venus', 'retro_mars', 'retro_jupiter', 'retro_saturn']
    
    result = []
    for planet_col in retro_planets:
        retro_mask = df[planet_col] == 1
        normal_mask = df[planet_col] == 0
        
        retro_returns = df.loc[retro_mask, 'forward_return_7d'].dropna()
        normal_returns = df.loc[normal_mask, 'forward_return_7d'].dropna()
        
        if len(retro_returns) >= 5 and len(normal_returns) >= 5:
            t_stat, p_val = ttest_ind(retro_returns, normal_returns, equal_var=False)
            result.append({
                'planet': planet_col.replace('retro_', ''),
                'retro_count': len(retro_returns),
                'normal_count': len(normal_returns),
                'retro_avg_return_pct': round(retro_returns.mean() * 100, 3),
                'normal_avg_return_pct': round(normal_returns.mean() * 100, 3),
                'return_diff_pct': round((retro_returns.mean() - normal_returns.mean()) * 100, 3),
                'retro_win_rate_pct': round((retro_returns > 0).mean() * 100, 1),
                'normal_win_rate_pct': round((normal_returns > 0).mean() * 100, 1),
                't_statistic': round(t_stat, 4),
                'p_value': round(p_val, 6),
                'significant': p_val < ALPHA
            })
    
    result_df = pd.DataFrame(result)
    
    print(f"\n   {'Planet':12s} {'Retro%':>7s} {'RetroRet':>9s} {'NormalRet':>10s} {'Diff':>10s} {'p-value':>10s}")
    print(f"   {'-'*60}")
    for _, row in result_df.iterrows():
        retro_pct = row['retro_count'] / (row['retro_count'] + row['normal_count']) * 100
        sig = ' ⭐' if row['significant'] else ''
        print(f"   {row['planet']:12s} {retro_pct:>6.1f}% {row['retro_avg_return_pct']:>+7.2f}% "
              f"{row['normal_avg_return_pct']:>+7.2f}% {row['return_diff_pct']:>+7.2f}% "
              f"{row['p_value']:>8.5f}{sig}")
    
    return result_df


def test_aspect_combinations(df: pd.DataFrame) -> dict:
    """Test: Is there a relationship between planetary aspects and market direction?"""
    print(f"\n{'='*60}")
    print("📊 Analysis 4: Planetary Aspects → Market Direction")
    print(f"{'='*60}")
    
    aspect_cols = [c for c in df.columns if c.startswith('aspect_')]
    
    result = []
    for col in aspect_cols:
        # Split by aspect count
        high_mask = df[col] > df[col].median()
        low_mask = ~high_mask
        
        high_returns = df.loc[high_mask, 'forward_return_7d'].dropna()
        low_returns = df.loc[low_mask, 'forward_return_7d'].dropna()
        
        if len(high_returns) >= 10 and len(low_returns) >= 10:
            t_stat, p_val = ttest_ind(high_returns, low_returns, equal_var=False)
            result.append({
                'aspect': col.replace('aspect_', ''),
                'high_mean_pct': round(high_returns.mean() * 100, 3),
                'low_mean_pct': round(low_returns.mean() * 100, 3),
                'diff_pct': round((high_returns.mean() - low_returns.mean()) * 100, 3),
                'p_value': round(p_val, 6),
                'significant': p_val < ALPHA
            })
    
    result_df = pd.DataFrame(result).sort_values('p_value')
    
    print(f"\n   {'Aspect':15s} {'HighRet':>8s} {'LowRet':>8s} {'Diff':>8s} {'p-value':>10s}")
    print(f"   {'-'*50}")
    for _, row in result_df.iterrows():
        sig = ' ⭐' if row['significant'] else ''
        print(f"   {row['aspect']:15s} {row['high_mean_pct']:>+7.2f}% "
              f"{row['low_mean_pct']:>+7.2f}% {row['diff_pct']:>+7.2f}% "
              f"{row['p_value']:>8.5f}{sig}")
    
    return result_df


def test_rahu_ketu_effect(df: pd.DataFrame) -> dict:
    """Test: Effect of Rahu/Ketu proximity on BTC volatility."""
    print(f"\n{'='*60}")
    print("📊 Analysis 5: Rahu/Ketu & Eclipse Proximity")
    print(f"{'='*60}")
    
    # Sun-Rahu proximity vs returns
    result = []
    for col in ['sun_rahu_proximity']:
        if col not in df.columns:
            continue
        # Split near vs far
        near_mask = df[col] < 15  # within 15 degrees
        far_mask = df[col] >= 15
        
        near_returns = df.loc[near_mask, 'forward_return_7d'].dropna()
        far_returns = df.loc[far_mask, 'forward_return_7d'].dropna()
        
        if len(near_returns) >= 5 and len(far_returns) >= 5:
            t_stat, p_val = ttest_ind(near_returns, far_returns, equal_var=False)
            result.append({
                'test': 'Sun_Rahu_<15°_vs_other',
                'near_mean_pct': round(near_returns.mean() * 100, 3),
                'far_mean_pct': round(far_returns.mean() * 100, 3),
                'diff_pct': round((near_returns.mean() - far_returns.mean()) * 100, 3),
                'near_n': len(near_returns),
                'near_win_rate': round((near_returns > 0).mean() * 100, 1),
                'p_value': round(p_val, 6)
            })
    
    # Near eclipse vs volatility
    for eclipse_col in ['near_solar_eclipse', 'near_lunar_eclipse']:
        if eclipse_col not in df.columns:
            continue
        near_mask = df[eclipse_col] == 1
        far_mask = df[eclipse_col] == 0
        
        near_vol = df.loc[near_mask, 'volatility_14d'].dropna()
        far_vol = df.loc[far_mask, 'volatility_14d'].dropna()
        
        if len(near_vol) >= 5 and len(far_vol) >= 5:
            t_stat, p_val = ttest_ind(near_vol, far_vol, equal_var=False)
            result.append({
                'test': f'{eclipse_col}_volatility',
                'near_mean_pct': round(near_vol.mean() * 100, 3),
                'far_mean_pct': round(far_vol.mean() * 100, 3),
                'diff_pct': round((near_vol.mean() - far_vol.mean()) * 100, 3),
                'near_n': len(near_vol),
                'near_win_rate': '-',
                'p_value': round(p_val, 6)
            })
    
    result_df = pd.DataFrame(result)
    for _, row in result_df.iterrows():
        sig = ' ⭐' if row['p_value'] < ALPHA else ''
        print(f"   {row['test']:40s} Near={row['near_mean_pct']:+.2f}% Far={row['far_mean_pct']:+.2f}% "
              f"p={row['p_value']:.5f}{sig}")
    
    return result_df


def test_weekday_effect(df: pd.DataFrame) -> dict:
    """Test: Vedic weekday lords effect on returns."""
    print(f"\n{'='*60}")
    print("📊 Analysis 6: Weekday (Vedic Hora Lord) → Returns")
    print(f"{'='*60}")
    
    result = []
    for day in range(7):
        subset = df[df['weekday_num'] == day]['forward_return_7d'].dropna()
        if len(subset) >= MIN_SAMPLES_PER_RULE:
            t_stat, p_val = stats.ttest_1samp(subset, 0)
            result.append({
                'weekday': WEEKDAYS[day],
                'count': len(subset),
                'avg_return_7d_pct': round(subset.mean() * 100, 3),
                'win_rate_pct': round((subset > 0).mean() * 100, 1),
                'p_value': round(p_val, 6),
                'significant': p_val < ALPHA
            })
    
    result_df = pd.DataFrame(result)
    print(f"\n   {'Weekday':15s} {'Count':>6s} {'Return':>9s} {'WinRate':>8s} {'p-value':>10s}")
    print(f"   {'-'*50}")
    for _, row in result_df.iterrows():
        sig = ' ⭐' if row['significant'] else ''
        print(f"   {row['weekday']:15s} {row['count']:>6d} {row['avg_return_7d_pct']:>+7.2f}% "
              f"{row['win_rate_pct']:>6.1f}% {row['p_value']:>8.5f}{sig}")
    
    return result_df


def test_venus_saturn_effect(df: pd.DataFrame) -> dict:
    """
    Test: Effect of Venus-Saturn separation on BTC returns.
    Venus-Saturn (Shukra-Shani) is a major Vedic combination.
    """
    print(f"\n{'='*60}")
    print("📊 Analysis 7: Venus-Saturn Distance → BTC Returns")
    print(f"{'='*60}")
    
    if 'venus_saturn_separation' not in df.columns:
        print("   [SKIP] 'venus_saturn_separation' not in dataset")
        return None
    
    result = []
    
    # 1. Correlation test
    sep = df['venus_saturn_separation'].dropna()
    ret = df['forward_return_7d'].dropna()
    common = sep.index.intersection(ret.index)
    if len(common) > 10:
        from scipy.stats import pearsonr
        r, p_val = pearsonr(sep.loc[common], ret.loc[common])
        result.append({
            'test': 'pearson_correlation',
            'stat': round(r, 4),
            'p_value': round(p_val, 6),
            'significant': p_val < ALPHA
        })
        print(f"\n   📈 Pearson correlation: r={r:.4f}, p={p_val:.5f}")
    
    # 2. Aspect categories vs returns
    if 'vs_aspect' in df.columns:
        print(f"\n   {'Aspect':20s} {'Count':>6s} {'Return':>9s} {'WinRate':>8s} {'p-value':>10s}")
        print(f"   {'-'*55}")
        for aspect in ['conjunction', 'square', 'trine', 'opposition', 'other']:
            subset = df[df['vs_aspect'] == aspect]['forward_return_7d'].dropna()
            if len(subset) >= 5:
                from scipy.stats import ttest_1samp
                t_stat, p_val = ttest_1samp(subset, 0)
                result.append({
                    'test': f'vs_aspect_{aspect}',
                    'count': len(subset),
                    'avg_return_pct': round(subset.mean() * 100, 3),
                    'win_rate_pct': round((subset > 0).mean() * 100, 1),
                    'p_value': round(p_val, 6),
                    'significant': p_val < ALPHA
                })
                sig = ' ⭐' if p_val < ALPHA else ''
                print(f"   {aspect:20s} {len(subset):>6d} {subset.mean()*100:>+7.2f}% "
                      f"{(subset>0).mean()*100:>6.1f}% {p_val:>10.5f}{sig}")
    
    # 3. Quintile analysis: divide into 5 groups by separation angle
    print(f"\n   📊 Quintile analysis (separation angle groups):")
    if 'venus_saturn_separation' in df.columns:
        df_temp = df[['venus_saturn_separation', 'forward_return_7d']].dropna()
        df_temp['quintile'] = pd.qcut(df_temp['venus_saturn_separation'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        print(f"   {'Group':6s} {'AngleRange':>14s} {'Count':>6s} {'Return':>9s} {'WinRate':>8s}")
        print(f"   {'-'*50}")
        for i, (label, group) in enumerate(df_temp.groupby('quintile', observed=True)):
            gr = group['forward_return_7d']
            lo = group['venus_saturn_separation'].min()
            hi = group['venus_saturn_separation'].max()
            print(f"   {label:6s} {lo:>6.1f}-{hi:>5.1f}° {len(gr):>6d} {gr.mean()*100:>+7.2f}% "
                  f"{(gr>0).mean()*100:>6.1f}%")
    
    if result:
        result_df = pd.DataFrame(result)
        csv_path = os.path.join(REPORTS_DIR, "analysis_venus_saturn.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"\n   [SAVE] Venus-Saturn analysis: {csv_path}")
    
    return pd.DataFrame(result) if result else pd.DataFrame()


def run_all_analyses():
    """Run all statistical analyses and compile results."""
    print("=" * 60)
    print("📈 FASE 4: تحلیل آماری جامع")
    print("   بررسی همبستگی قیمت بیت‌کوین با پارامترهای ودیک")
    print("=" * 60)
    
    df = load_dataset()
    print(f"\n[LOAD] Dataset: {len(df):,} days, {len(df.columns)} features")
    
    results = {}
    
    results['nakshatra'] = test_nakshatra_effect(df)
    results['moon_phase'] = test_moon_phase_effect(df)
    results['retrograde'] = test_retrograde_effect(df)
    results['aspects'] = test_aspect_combinations(df)
    results['rahu_ketu'] = test_rahu_ketu_effect(df)
    results['weekday'] = test_weekday_effect(df)
    
    # --- NEW: Venus-Saturn separation analysis ---
    results['venus_saturn'] = test_venus_saturn_effect(df)
    
    # Compile all significant findings
    print(f"\n{'='*60}")
    print("🏆 Significant Findings (Bonferroni corrected)")
    print(f"{'='*60}")
    
    all_significant = []
    
    if 'nakshatra' in results and results['nakshatra'] is not None:
        sig_nak = results['nakshatra'][results['nakshatra']['significant_bonf']]
        for _, row in sig_nak.iterrows():
            all_significant.append({
                'analysis': 'Moon Nakshatra',
                'feature': row['nakshatra'],
                'effect': f"{row['avg_return_7d_pct']:+.2f}% 7d return",
                'win_rate': f"{row['win_rate_pct']:.0f}%",
                'p_value': row['p_bonferroni']
            })
    
    # Save all results
    for name, result_df in results.items():
        if hasattr(result_df, 'shape') and len(result_df) > 0:
            csv_path = os.path.join(REPORTS_DIR, f"analysis_{name}.csv")
            result_df.to_csv(csv_path, index=False)
            print(f"   [SAVE] {csv_path}")
    
    # Save significant findings summary
    if all_significant:
        sig_df = pd.DataFrame(all_significant).sort_values('p_value')
        sig_path = os.path.join(REPORTS_DIR, "significant_findings.csv")
        sig_df.to_csv(sig_path, index=False)
        print(f"   [SAVE] Significant findings: {sig_path}")
    
    if all_significant:
        print(f"\n📌 Total significant findings: {len(all_significant)}")
        for f in all_significant:
            print(f"   ⭐ [{f['analysis']}] {f['feature']}: {f['effect']} "
                  f"(win rate: {f['win_rate']}, p={f['p_value']:.5f})")
    else:
        print(f"\n📌 No significant findings after Bonferroni correction.")
        print(f"   (This is normal — Vedic astrology has many variables and BTC is noisy)")
    
    print(f"\n✅ FASE 4 complete.")
    return results


if __name__ == "__main__":
    run_all_analyses()
