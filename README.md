# 🪐📈 BTC Vedic Astrology Research

بررسی همبستگی قیمت بیت‌کوین با تغییرات تنجیمی ودیک (Vedic Astrology)

## 🎯 هدف

استخراج قوانین معاملاتی مبتنی بر پارامترهای نجومی ودیک (سیدریال) و فوروارد تست آن‌ها روی داده‌های تاریخی بیت‌کوین.

## 🗂 ساختار پروژه

```
btc-vedic-astro/
├── scripts/
│   ├── config.py              # تنظیمات مرکزی
│   ├── fetch_btc_data.py      # فاز 1: دریافت قیمت BTC
│   ├── calculate_vedic.py     # فاز 2: محاسبات نجومی ودیک
│   ├── feature_engineering.py # فاز 3: ادغام و مهندسی ویژگی
│   ├── statistical_analysis.py# فاز 4: تحلیل آماری
│   ├── rule_extraction.py     # فاز 5: استخراج قوانین
│   ├── forward_test.py        # فاز 6: فوروارد تست
│   └── run_pipeline.py        # اجراکننده pipeline
├── data/
│   ├── raw/                   # داده‌های خام (BTC OHLCV)
│   └── processed/             # داده‌های پردازش‌شده
├── output/
│   ├── reports/               # گزارش‌های تحلیلی
│   ├── charts/                # نمودارها
│   └── models/                # مدل‌های ذخیره‌شده
├── venv/                      # محیط مجازی Python
├── requirements.txt
└── README.md
```

## 🔮 پارامترهای ودیک تحلیل‌شده

| دسته | پارامترها |
|------|-----------|
| **Nakshatra (منزل)** | موقعیت ماه، خورشید، Ascendant در ۲۷ Nakshatra + Pada |
| **Rashi (برج)** | موقعیت ۹ سیاره + Ascendant در ۱۲ برج |
| **Retrograde** | وضعیت ایستایی مریخ، عطارد، مشتری، زهره، زحل |
| **Aspects** | تعداد و نوع تأثیرات سیاره‌ای (تراین، مربع، مقابله،...) |
| **Moon Phase** | ۸ فاز ماه و زاویه خورشید-ماه |
| **Eclipse** | نزدیکی به خورشیدگرفتگی/ماه‌گرفتگی |
| **Weekday** | روز هفته (Vedic Hora lords) |
| **Rahu/Ketu** | موقعیت گره‌های شمالی/جنوبی |

## 📊 خروجی‌ها

- `output/reports/analysis_*.csv` — نتایج آزمون‌های آماری
- `output/reports/significant_findings.csv` — یافته‌های معنی‌دار
- `output/reports/extracted_rules.json` — قوانین استخراج‌شده
- `output/reports/walk_forward_results.csv` — نتایج فوروارد تست
- `output/reports/final_verdict.json` — نتیجه نهایی

## 🚀 اجرا

```bash
source venv/bin/activate

# کل pipeline
python scripts/run_pipeline.py

# یا فاز به فاز:
python scripts/fetch_btc_data.py
python scripts/calculate_vedic.py
python scripts/feature_engineering.py
python scripts/statistical_analysis.py
python scripts/rule_extraction.py
python scripts/forward_test.py
```

## 📌 نکات آماری

- **تصحیح بونفرونی** برای Multiple Comparisons اجرا می‌شود
- **Walk-Forward Validation** با پنجره ۳ سال آموزش + ۶ ماه تست
- **Permutation Test** برای اطمینان از عدم تصادفی بودن نتایج
