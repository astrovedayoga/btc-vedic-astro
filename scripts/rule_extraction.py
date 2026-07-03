"""
FASE 5: استخراج قوانین — Decision Tree + Rule Learner
Extract interpretable trading rules from Vedic + BTC patterns
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import *

# ----------------------------------------------------------------
def load_dataset() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "btc_vedic_unified.parquet")
    return pd.read_parquet(path)


def prepare_ml_features(df: pd.DataFrame) -> tuple:
    """
    Prepare features for ML (select Vedic + market features, drop NAs).
    """
    # Vedic numeric features
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
        'above_sma_200',
        'sma_ratio_50_200',
        'volatility_14d',
        # Venus-Saturn features
        'venus_saturn_separation',
        'vs_sep_sin',
        'vs_sep_cos',
    ]
    
    # One-hot moon phases
    moon_phase_cols = [c for c in df.columns if c.startswith('moon_phase_')]
    
    # Moon phase one-hot + weekday
    weekday_cols = ['weekday_num']
    
    all_features = vedic_features + moon_phase_cols + weekday_cols
    
    # Filter to available columns
    available = [c for c in all_features if c in df.columns]
    missing = [c for c in all_features if c not in df.columns]
    if missing:
        print(f"   [WARN] Missing columns: {missing}")
    
    X = df[available].copy()
    y = df['target_7d_bull'].copy()
    
    # Drop rows with NaN
    mask = X.isna().any(axis=1) | y.isna()
    X = X[~mask]
    y = y[~mask]
    
    print(f"   ML features: {len(available)}")
    print(f"   Samples: {len(X):,}")
    print(f"   Bullish class: {y.mean():.1%}")
    
    return X, y, available


def extract_rules_from_tree(tree, feature_names, class_names=None):
    """Extract human-readable rules from a Decision Tree."""
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]
    
    rules = []
    
    def recurse(node, depth, conditions):
        if tree_.feature[node] != -2:  # not a leaf
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Left (<= threshold)
            left_conditions = conditions.copy()
            left_conditions.append(f"{name} <= {threshold:.4f}")
            recurse(tree_.children_left[node], depth + 1, left_conditions)
            
            # Right (> threshold)
            right_conditions = conditions.copy()
            right_conditions.append(f"{name} > {threshold:.4f}")
            recurse(tree_.children_right[node], depth + 1, right_conditions)
        else:
            # Leaf node
            n_samples = tree_.n_node_samples[node]
            value = tree_.value[node]
            n_class = value[0]
            majority_class = np.argmax(n_class)
            confidence = n_class[majority_class] / n_class.sum()
            
            if n_samples >= MIN_SAMPLES_PER_RULE:
                rules.append({
                    'conditions': ' AND '.join(conditions),
                    'prediction': class_names[majority_class] if class_names else str(majority_class),
                    'samples': int(n_samples),
                    'purity': round(confidence, 3),
                    'class_counts': {class_names[i]: int(n_class[i]) for i in range(len(n_class))} if class_names else str(n_class)
                })
    
    recurse(0, 1, [])
    return rules


def train_decision_tree(X, y, feature_names, max_depth=DT_MAX_DEPTH):
    """Train and extract rules from a Decision Tree."""
    print(f"\n{'='*60}")
    print("🌳 Decision Tree Rule Extraction")
    print(f"{'='*60}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train with pruning
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        min_impurity_decrease=0.001,
        random_state=42,
        class_weight='balanced'
    )
    dt.fit(X_train, y_train)
    
    # Evaluate
    train_score = dt.score(X_train, y_train)
    test_score = dt.score(X_test, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')
    
    print(f"\n   📈 Performance:")
    print(f"   Train accuracy: {train_score:.1%}")
    print(f"   Test accuracy:  {test_score:.1%}")
    print(f"   CV accuracy:    {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")
    print(f"   Tree depth:     {dt.get_depth()}")
    print(f"   Leaf nodes:     {dt.get_n_leaves()}")
    
    # Confusion matrix on test
    y_pred = dt.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\n   Confusion Matrix (test):")
    print(f"   True Neg: {tn:>4d}  False Pos: {fp:>4d}")
    print(f"   False Neg: {fn:>4d}  True Pos:  {tp:>4d}")
    print(f"   Precision: {tp/(tp+fp):.1%}  Recall: {tp/(tp+fn):.1%}  F1: {2*tp/(2*tp+fp+fn):.1%}")
    
    # Extract rules
    class_names = ['Bearish', 'Bullish']
    rules = extract_rules_from_tree(dt, feature_names, class_names)
    
    print(f"\n   📜 Extracted {len(rules)} rules:")
    for i, rule in enumerate(rules):
        if rule['purity'] >= 0.55:  # Only show somewhat meaningful rules
            print(f"\n   Rule #{i+1}: IF {rule['conditions']}")
            print(f"      → {rule['prediction']} (samples: {rule['samples']}, "
                  f"confidence: {rule['purity']:.0%})")
    
    # Save tree
    tree_text = export_text(dt, feature_names=feature_names, show_weights=True)
    tree_path = os.path.join(REPORTS_DIR, "decision_tree_rules.txt")
    with open(tree_path, 'w') as f:
        f.write(tree_text)
    print(f"\n   [SAVE] Tree rules: {tree_path}")
    
    return dt, rules


def train_random_forest(X, y, feature_names):
    """Train Random Forest for feature importance analysis."""
    print(f"\n{'='*60}")
    print("🌲 Random Forest Feature Importance")
    print(f"{'='*60}")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=DT_MAX_DEPTH + 2,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n   Top 15 Most Important Features:")
    print(f"   {'Feature':40s} {'Importance':>10s}")
    print(f"   {'-'*52}")
    for _, row in importance.head(15).iterrows():
        print(f"   {row['feature']:40s} {row['importance']:.5f}")
    
    # Save importance
    imp_path = os.path.join(REPORTS_DIR, "feature_importance.csv")
    importance.to_csv(imp_path, index=False)
    print(f"\n   [SAVE] Feature importance: {imp_path}")
    
    return rf, importance


def run_all():
    """Run complete rule extraction pipeline."""
    print("=" * 60)
    print("🔮 FASE 5: استخراج قوانین معاملاتی")
    print("   Rule Extraction: BTC × Vedic Astrology")
    print("=" * 60)
    
    df = load_dataset()
    print(f"\n[LOAD] Dataset: {len(df):,} rows")
    
    X, y, feature_names = prepare_ml_features(df)
    
    # 1. Decision Tree (interpretable rules)
    dt, rules = train_decision_tree(X, y, feature_names)
    
    # 2. Random Forest (feature importance)
    rf, importance = train_random_forest(X, y, feature_names)
    
    # Save all rules
    rules_path = os.path.join(REPORTS_DIR, "extracted_rules.json")
    with open(rules_path, 'w') as f:
        json.dump(rules, f, indent=2, default=str)
    print(f"   [SAVE] Rules JSON: {rules_path}")
    
    print(f"\n✅ FASE 5 complete.")
    return dt, rules, rf, importance


if __name__ == "__main__":
    run_all()
