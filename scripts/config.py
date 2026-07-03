"""
BTC Vedic Astrology Research - Configuration
All adjustable parameters in one place.
"""

# ----------------------
# Data settings
# ----------------------

# BTC data source: 'coingecko' (free, from 2009) or 'ccxt' (exchange)
BTC_DATA_SOURCE = "coingecko"

# Date range for analysis
BTC_START_DATE = "2010-07-17"  # First BTC price on exchanges (Mt.Gox)
BTC_END_DATE = None  # None = up to today

# CoinGecko API settings (free, no key needed)
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
COINGECKO_RETRY = 3
COINGECKO_DELAY = 1.5  # seconds between calls (rate limit)

# ----------------------
# Vedic Astrology settings
# ----------------------

# Ayanamsa system (Lahiri = most common for Vedic)
AYANAMSA = "Lahiri"

# House system
HOUSE_SYSTEM = "Equal"

# Location for ephemeris calculations (GMT/UTC)
DEFAULT_LAT = 0.0
DEFAULT_LON = 0.0

# ----------------------
# Feature Engineering
# ----------------------

# Target: forward returns to predict
FORWARD_RETURN_DAYS = [1, 3, 7, 14, 30]

# Price change thresholds for classification
BULL_THRESHOLD = 0.02  # 2% gain = bullish
BEAR_THRESHOLD = -0.02  # 2% loss = bearish

# ----------------------
# Rule Extraction
# ----------------------

# Min samples per rule for statistical validity
MIN_SAMPLES_PER_RULE = 10

# Decision Tree params
DT_MAX_DEPTH = 5
DT_MIN_SAMPLES_LEAF = 30

# Significance level for statistical tests
ALPHA = 0.01  # strict (Bonferroni corrected)

# ----------------------
# Walk-Forward Test
# ----------------------

# Training windows (in days)
TRAIN_WINDOW = 1095  # ~3 years
TEST_WINDOW = 180    # ~6 months
STEP_SIZE = 90       # re-train every 3 months

# Minimum trades for statistical significance
MIN_TRADES = 20

# File paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"
CHARTS_DIR = "output/charts"
REPORTS_DIR = "output/reports"
MODELS_DIR = "output/models"
