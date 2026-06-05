# Week 2 — Market Data & Technical Indicators

## Overview

This week you build the **data foundation** for the entire course.  
By the end you will have a pipeline that downloads raw OHLCV market data,
engineers 20+ technical features from scratch, and saves a clean feature CSV
that the LSTM model will consume directly in a later week.

## Learning Objectives

- Download and store multi-market OHLCV data in a reproducible directory structure
- Understand how technical indicators encode market information mathematically
- Implement 10 indicators from scratch using only `pandas` and `numpy`
- Analyse and compare indicators across symbols and asset classes
- Reason about *why* an indicator reads high or low in a given market context

---

## File Structure

```
week2/
├── data_loading.ipynb          ← Notebook 1: download + explore
├── technical_indicators.ipynb  ← Notebook 2: implement all indicators
├── data/
│   ├── raw/                    ← raw OHLCV CSVs  (auto-created)
│   │   ├── SPY.csv
│   │   ├── AAPL.csv
│   │   └── ...
│   └── processed/              ← feature CSVs    (auto-created)
│       ├── SPY_features.csv
│       └── ...
└── README.md
```

> `data/` is created automatically when you run the notebooks.  
> Do **not** commit the `data/` folder — add it to `.gitignore`.

---

## Notebooks

### Notebook 1 — `data_loading.ipynb`

**What you do:**
- Define project constants: directory paths, date range, and the full symbol universe across four markets (India, US, Crypto, Forex)
- Implement `safe_symbol()` and `download_symbol()` from scratch
- Download all 16 symbols and save raw CSVs to `data/raw/`
- Explore one symbol: return statistics, monthly resampling, annualised volatility
- Build normalised cumulative return plots to compare symbols across markets

**Key skills:** `yfinance`, `pathlib`, `pandas` resampling, MultiIndex handling

---

### Notebook 2 — `technical_indicators.ipynb`

**What you do:**  
Implement **10 indicators** using only `pandas` and `numpy` — no TA libraries.

| # | Indicator | Category | Feature columns |
|---|---|---|---|
| 1 | Simple Moving Average | Trend | `sma_ratio_20`, `sma_ratio_50` |
| 2 | Exponential Moving Average | Trend | `ema_ratio_20` |
| 3 | RSI (14) | Momentum | `rsi_14` |
| 4 | MACD | Momentum | `macd`, `macd_signal` |
| 5 | Bollinger Bands | Volatility | `bb_width`, `bb_pct_b` |
| 6 | Rate of Change | Momentum | `roc_5`, `roc_10` |
| 7 | ATR (14) | Volatility | `atr_ratio` |
| 8 | Stochastic Oscillator | Momentum | `stoch_k`, `stoch_d` |
| 9 | On-Balance Volume | Volume | `obv_change` |
| 10 | VWAP (rolling 20-day) | Volume / Price | `vwap_ratio` |
| — | Z-Score (20-day) | Mean-reversion | `z_score_20` |

The notebook ends by saving a complete feature CSV to `data/processed/<SYMBOL>_features.csv`.

**Important notes:**
- RSI and ATR use **Wilder smoothing** (`ewm(com=n-1, adjust=False)`), not the standard EMA span
- All oscillators are normalised to `[0, 1]` before being stored as model features
- `future_return` and `target_direction` columns are added at the end as prediction targets

---

## Setup

```bash
pip install yfinance pandas numpy matplotlib
```

Run the notebooks in order:

```
1. data_loading.ipynb          → populates data/raw/
2. technical_indicators.ipynb  → populates data/processed/
```

---

## Market Symbols

| Market | Tickers |
|---|---|
| India | `^NSEI`, `^BSESN`, `RELIANCE.NS`, `TCS.NS`, `INFY.NS` |
| US | `^GSPC`, `^IXIC`, `SPY`, `AAPL`, `MSFT`, `NVDA`, `TSLA` |
| Crypto | `BTC-USD`, `ETH-USD` |
| Forex | `EURUSD=X`, `JPY=X` |

Indian equity tickers on the NSE use the `.NS` suffix.  
Indices start with `^`. Crypto pairs use the `-USD` suffix.

---

## Submission Checklist

- [ ] All TODO cells in Notebooks 1 and 2 are executed with visible outputs
- [ ] `data/processed/<SYMBOL>_features.csv` exists for at least **SPY** (more = bonus)
- [ ] Charts have titles, labelled axes, and legends
- [ ] Committed under `week2/` — `data/` folder is gitignored

---

## Bonus

- Complete `analysis.ipynb`
- Explore Hugging Face

---

## References for Week 2

- [yfinance documentation](https://ranaroussi.github.io/yfinance/) — downloading market data from Yahoo Finance
- [Kaggle: Yfinance World Indices Price Data](https://www.kaggle.com/datasets/code1110/yfinance-world-indices-price-data) — example OHLCV dataset fetched using yfinance
- [Investopedia Financial Terms Dictionary](https://www.investopedia.com/financial-term-dictionary-4769738) — quick reference for market and finance terms
- [pandas documentation](https://pandas.pydata.org/docs/) — DataFrame operations, time series, resampling, and rolling windows

---
