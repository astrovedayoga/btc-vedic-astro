"""
FASE 6: فوروارد تست — Walk-Forward Validation
Forward test: Walk-forward analysis of extracted rules on unseen data
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import *

# ----------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "btc_vedic_unified.parquet")
    return pd.read_parquet(path)


def prepare_features(df: pd.DataFrame) -> tuple:
    """Same feature preparation as rule_extraction."""
    vedic_features = [
        'moon_nakshatra_num', 'moon_nakshatra_num_sin', 'moon_nakshatra_num_cos',
        'moon_rashi_num', 'moon_rashi_num_sin', 'moon_rashi_num_cos',
        'sun_rashi_num', 'sun_rashi_num_sin', 'sun_rashi_num_cos',
        'asc_rashi_num', 'asc_rashi_num_sin', 'asc_rashi_num_cos',
        'moon_sun_angle', 'sun_rahu_proximity',
        'near_solar_eclipse', 'near_lunar_eclipse',
        'retro_mercury', 'retro_venus', 'retro_mars', 'retro_jupiter', 'retro_saturn',
        'rahu_retro', 'ketu_retro',
        'aspect_conjunction', 'aspect_trine', 'aspect_square', 'aspect_opposition', 'aspect_sextile',
        'sun_sign_lon', 'moon_sign_lon',
        'mercury_sign_lon', 'venus_sign_lon', 'mars_sign_lon',
        'jupiter_sign_lon', 'saturn_sign_lon',
        'above_sma_200', 'sma_ratio_50_200', 'volatility_14d',
        'venus_saturn_separation', 'vs_sep_sin', 'vs_sep_cos',
    ]
    moon_phase_cols = [c for c in df.columns if c.startswith('moon_phase_')]
    all_features = vedic_features + moon_phase_cols + ['weekday_num']
    available = [c for c in all_features if c in df.columns]
    
    X = df[available].copy()
    y = df['forward_return_7d'].copy()  # continuous for this
    
    return X, y, available


def walk_forward_test(df: pd.DataFrame, train_window: int = TRAIN_WINDOW,
                      test_window: int = TEST_WINDOW, step_size: int = STEP_SIZE):
    """
    Walk-forward testing of a Decision Tree model.
    
    Train on rolling window → predict next test_window days → evaluate → slide forward.
    """
    print(f"\n{'='*60}")
    print("🔄 Walk-Forward Test")
    print(f"{'='*60}")
    print(f"\n   Train window: {train_window} days ({train_window/365:.1f} years)")
    print(f"   Test window:  {test_window} days ({test_window/365:.1f} years)")
    print(f"   Step size:    {step_size} days")
    
    X, y, feature_names = prepare_features(df)
    
    # Build target: bullish (1) / bearish (-1) / neutral (0)
    y_bull = (y > BULL_THRESHOLD).astype(int)
    y_bear = (y < BEAR_THRESHOLD).astype(int)
    y_direction = y_bull - y_bear  # -1=Bear, 0=Neutral, 1=Bull
    
    # We'll predict bullish vs not-bullish
    target = (y > BULL_THRESHOLD).astype(int)
    
    # Drop NaNs
    valid = ~(X.isna().any(axis=1) | target.isna())
    X_clean = X[valid]
    target_clean = target[valid]
    dates = X_clean.index
    
    n = len(X_clean)
    windows = []
    start = 0
    
    while start + train_window + test_window <= n:
        train_end = start + train_window
        test_end = train_end + test_window
        
        windows.append({
            'train_start': dates[start],
            'train_end': dates[train_end - 1],
            'test_start': dates[train_end],
            'test_end': dates[min(test_end, n) - 1],
            'train_idx': list(range(start, train_end)),
            'test_idx': list(range(train_end, min(test_end, n)))
        })
        
        start += step_size
    
    if not windows:
        print("   ✗ Not enough data for walk-forward. Adjust windows.")
        return None
    
    print(f"\n   Total windows: {len(windows)}\n")
    
    # Run walk-forward
    results = []
    
    for w_idx, win in enumerate(windows):
        train_idx = win['train_idx']
        test_idx = win['test_idx']
        
        X_train = X_clean.iloc[train_idx]
        y_train = target_clean.iloc[train_idx]
        X_test = X_clean.iloc[test_idx]
        y_test = target_clean.iloc[test_idx]
        
        # Train on this window
        dt = DecisionTreeClassifier(
            max_depth=DT_MAX_DEPTH,
            min_samples_leaf=DT_MIN_SAMPLES_LEAF,
            min_impurity_decrease=0.001,
            random_state=42,
            class_weight='balanced'
        )
        dt.fit(X_train, y_train)
        
        # Predict
        y_pred = dt.predict(X_test)
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        
        # Precision/Recall for bullish class
        if y_test.sum() > 0:
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
        else:
            prec, rec = 0, 0
        
        # Bullish prediction ratio (how often model says "up")
        bull_ratio = y_pred.mean()
        
        # Actual return during test period
        actual_returns = y.iloc[test_idx]
        test_returns = y.iloc[test_idx]
        
        # Simulated trading: long when prediction=1, flat when prediction=0
        strategy_returns = y_pred * test_returns
        
        results.append({
            'window': w_idx + 1,
            'test_start': win['test_start'],
            'test_end': win['test_end'],
            'test_days': len(test_idx),
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'bull_pred_ratio': round(bull_ratio, 4),
            'actual_bull_ratio': round(y_test.mean(), 4),
            'avg_return_strategy': round(strategy_returns.mean(), 6),
            'avg_return_buy_hold': round(test_returns.mean(), 6),
            'total_return_strategy': round(strategy_returns.sum(), 4),
            'total_return_buy_hold': round(test_returns.sum(), 4),
            'strategy_win_rate': round((strategy_returns > 0).mean(), 4),
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Walk-Forward Results Summary")
    print(f"{'='*60}")
    
    avg_acc = results_df['accuracy'].mean()
    avg_prec = results_df['precision'].mean()
    avg_rec = results_df['recall'].mean()
    avg_strat_return = results_df['avg_return_strategy'].mean()
    avg_bh_return = results_df['avg_return_buy_hold'].mean()
    
    # Test if strategy outperforms buy-hold statistically
    t_stat, p_val = stats.ttest_rel(
        results_df['avg_return_strategy'],
        results_df['avg_return_buy_hold']
    )
    
    print(f"\n   {'Metric':25s} {'Strategy':>12s} {'Buy & Hold':>12s}")
    print(f"   {'-'*52}")
    print(f"   {'Avg Window Accuracy':25s} {avg_acc:>10.1%} {'':>12s}")
    print(f"   {'Avg Precision (Bull)':25s} {avg_prec:>10.1%} {'':>12s}")
    print(f"   {'Avg Recall (Bull)':25s} {avg_rec:>10.1%} {'':>12s}")
    print(f"   {'Win Rate':25s} {results_df['strategy_win_rate'].mean():>10.1%} "
          f"{results_df['actual_bull_ratio'].mean():>10.1%}")
    print(f"   {'Avg Return/Window':25s} {avg_strat_return:>+10.4f} {avg_bh_return:>+10.4f}")
    print(f"   {'Sharpe-like (mean/std)':25s} "
          f"{results_df['avg_return_strategy'].mean()/results_df['avg_return_strategy'].std():>+10.4f} "
          f"{results_df['avg_return_buy_hold'].mean()/results_df['avg_return_buy_hold'].std():>+10.4f}")
    
    print(f"\n   Statistical Test (strategy vs buy-hold):")
    print(f"   t-statistic: {t_stat:.4f}")
    print(f"   p-value:     {p_val:.5f}")
    print(f"   Outperforms?: {'✅ YES' if p_val < 0.05 else '❌ NO (not statistically significant)'}")
    
    # Best/worst windows
    best = results_df.loc[results_df['avg_return_strategy'].idxmax()]
    worst = results_df.loc[results_df['avg_return_strategy'].idxmin()]
    
    print(f"\n   Best window #{best['window']}: {best['test_start'].date()} → {best['test_end'].date()}")
    print(f"     Strategy return: {best['avg_return_strategy']:+.4f} vs BH: {best['avg_return_buy_hold']:+.4f}")
    print(f"   Worst window #{worst['window']}: {worst['test_start'].date()} → {worst['test_end'].date()}")
    print(f"     Strategy return: {worst['avg_return_strategy']:+.4f} vs BH: {worst['avg_return_buy_hold']:+.4f}")
    
    # Save results
    csv_path = os.path.join(REPORTS_DIR, "walk_forward_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n   [SAVE] Walk-forward results: {csv_path}")
    
    return results_df


def monte_carlo_permutation_test(df: pd.DataFrame, n_permutations: int = 1000):
    """
    Monte Carlo permutation test: shuffle dates to see if Vedic predictions
    are better than random.
    """
    print(f"\n{'='*60}")
    print("🎲 Monte Carlo Permutation Test")
    print(f"{'='*60}")
    print(f"\n   Testing if model accuracy is significantly better than random")
    print(f"   Permutations: {n_permutations:,}")
    
    X, y, feature_names = prepare_features(df)
    target = (y > BULL_THRESHOLD).astype(int)
    
    valid = ~(X.isna().any(axis=1) | target.isna())
    X_clean = X[valid]
    target_clean = target[valid]
    
    # Train on first 80%, test on last 20% (chronological split)
    split = int(len(X_clean) * 0.8)
    X_train = X_clean.iloc[:split]
    y_train = target_clean.iloc[:split]
    X_test = X_clean.iloc[split:]
    y_test = target_clean.iloc[split:]
    
    # Train real model
    dt = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH, min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        random_state=42, class_weight='balanced'
    )
    dt.fit(X_train, y_train)
    real_acc = accuracy_score(y_test, dt.predict(X_test))
    
    print(f"\n   Real accuracy: {real_acc:.4f}")
    
    # Permutations: shuffle y_train labels
    np.random.seed(42)
    perm_accs = []
    for i in range(n_permutations):
        y_shuffled = y_train.sample(frac=1, random_state=i).values
        dt_perm = DecisionTreeClassifier(
            max_depth=DT_MAX_DEPTH, min_samples_leaf=DT_MIN_SAMPLES_LEAF,
            random_state=42, class_weight='balanced'
        )
        dt_perm.fit(X_train, y_shuffled)
        perm_acc = accuracy_score(y_test, dt_perm.predict(X_test))
        perm_accs.append(perm_acc)
    
    perm_accs = np.array(perm_accs)
    p_value = (perm_accs >= real_acc).mean()
    
    print(f"\n   Permutation test results:")
    print(f"   Real accuracy:       {real_acc:.4f}")
    print(f"   Mean perm accuracy:  {perm_accs.mean():.4f}")
    print(f"   Std perm accuracy:   {perm_accs.std():.4f}")
    print(f"   Permutation p-value: {p_value:.5f}")
    print(f"   Model vs random:     {'✅ SIGNIFICANT' if p_value < 0.05 else '❌ Not significant'}")
    
    return {'real_accuracy': real_acc, 'perm_mean': perm_accs.mean(), 
            'p_value': p_value}


def run_all():
    """Run complete forward test pipeline."""
    print("=" * 60)
    print("🧪 FASE 6: فوروارد تست & اعتبارسنجی")
    print("   Forward Testing & Validation")
    print("=" * 60)
    
    df = load_dataset()
    
    # 1. Walk-Forward Test
    wf_results = walk_forward_test(df)
    
    # 2. Monte Carlo Permutation Test
    perm_results = monte_carlo_permutation_test(df)
    
    # 3. Final verdict
    print(f"\n{'='*60}")
    print("🏁 FINAL VERDICT")
    print(f"{'='*60}")
    
    if wf_results is not None:
        avg_strat = wf_results['avg_return_strategy'].mean()
        avg_bh = wf_results['avg_return_buy_hold'].mean()
        outperforms = avg_strat > avg_bh
        
        # Test if outperformance is significant
        t_stat, p_val = stats.ttest_rel(
            wf_results['avg_return_strategy'],
            wf_results['avg_return_buy_hold']
        )
        
        print(f"\n   Walk-Forward:")
        print(f"     Strategy avg return:   {avg_strat:+.6f}")
        print(f"     Buy & Hold avg return: {avg_bh:+.6f}")
        print(f"     Outperformance:        {avg_strat - avg_bh:+.6f}")
        print(f"     Statistically significant: {'✅ YES' if p_val < 0.05 else '❌ NO'}")
    
    print(f"\n   Monte Carlo Permutation:")
    if perm_results['p_value'] < 0.05:
        print(f"     ✅ Model performs better than random (p={perm_results['p_value']:.5f})")
    else:
        print(f"     ❌ Model does NOT outperform random (p={perm_results['p_value']:.5f})")
    
    # Save final verdict
    verdict = {
        'walk_forward_success': bool(avg_strat > avg_bh) if wf_results is not None else None,
        'walk_forward_p_value': p_val if wf_results is not None else None,
        'permutation_p_value': perm_results['p_value'],
        'significant': bool(p_val < 0.05) if wf_results is not None else False
    }
    
    verdict_path = os.path.join(REPORTS_DIR, "final_verdict.json")
    with open(verdict_path, 'w') as f:
        json.dump(verdict, f, indent=2, default=str)
    print(f"\n   [SAVE] Final verdict: {verdict_path}")
    
    print(f"\n✅ FASE 6 complete.")
    return wf_results, perm_results


if __name__ == "__main__":
    run_all()
