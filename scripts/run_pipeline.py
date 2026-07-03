"""
BTC Vedic Astrology Research — Pipeline Runner
Execute all phases sequentially or individually
"""

import sys
import os
import time
from pathlib import Path
import subprocess

SCRIPTS_DIR = Path(__file__).parent

PHASES = [
    ("1", "fetch_btc_data.py", "دریافت داده‌های قیمت BTC"),
    ("2", "calculate_vedic.py", "محاسبه پارامترهای نجومی ودیک"),
    ("3", "feature_engineering.py", "مهندسی ویژگی و ادغام داده‌ها"),
    ("4", "statistical_analysis.py", "تحلیل آماری و همبستگی"),
    ("5", "rule_extraction.py", "استخراج قوانین معاملاتی"),
    ("6", "forward_test.py", "فوروارد تست و اعتبارسنجی"),
]

def run_phase(phase_num: str, script: str, description: str):
    """Run a single phase script."""
    print(f"\n{'#'*70}")
    print(f"# FASE {phase_num}: {description}")
    print(f"# Script: {script}")
    print(f"{'#'*70}\n")
    
    script_path = SCRIPTS_DIR / script
    start = time.time()
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SCRIPTS_DIR.parent,
        capture_output=False,
        text=True
    )
    
    elapsed = time.time() - start
    success = result.returncode == 0
    
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"\n   [{status}] Fase {phase_num}: {elapsed:.1f}s")
    
    return success


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BTC Vedic Astrology Research Pipeline")
    parser.add_argument("--phases", type=str, default="all",
                       help="Phases to run: 'all', '1', '1-3', '1,3,5'")
    parser.add_argument("--skip", type=str, default="",
                       help="Phases to skip: '1', '1,2'")
    
    args = parser.parse_args()
    
    # Parse which phases to run
    phases_to_run = []
    for num, script, desc in PHASES:
        run = False
        if args.phases == "all":
            run = True
        elif "-" in args.phases:
            parts = args.phases.split("-")
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                if start <= int(num) <= end:
                    run = True
        elif "," in args.phases:
            if num in args.phases.split(","):
                run = True
        elif num == args.phases:
            run = True
        
        if run and num not in args.skip.split(","):
            phases_to_run.append((num, script, desc))
    
    if not phases_to_run:
        print("No phases selected. Available:")
        for num, script, desc in PHASES:
            print(f"  {num}: {desc} ({script})")
        sys.exit(1)
    
    print("=" * 70)
    print("🚀 BTC VEDIC ASTROLOGY RESEARCH PIPELINE")
    print("=" * 70)
    print(f"\nRunning: {', '.join(f'Fase {n}' for n,_,_ in phases_to_run)}")
    print()
    
    total_start = time.time()
    successes = 0
    
    for num, script, desc in phases_to_run:
        ok = run_phase(num, script, desc)
        if ok:
            successes += 1
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"🏁 PIPELINE COMPLETE: {successes}/{len(phases_to_run)} phases passed")
    print(f"   Total time: {total_time:.1f}s")
    print(f"{'='*70}")
