# Gold Trading AI - Intelligent Decision Support System for XAU/USD

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/ML-XGBoost-green.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/UI-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Data-25%2B%20Years-gold.svg" alt="Data">
</p>

## Description

End-to-end algorithmic trading system dedicated to gold (XAU/USD). This project combines **Data Engineering**, **Quantitative Analysis**, **Machine Learning** and **Full-Stack Development** to provide real-time trading recommendations.

### Key Features

- **Massive Dataset**: 25+ years of historical data (2000-present)
- **344 Features**: Technical and macroeconomic indicators
- **XGBoost Model**: Optimized binary classification (~90% accuracy)
- **Professional Interface**: Interactive Streamlit dashboard
- **Real-Time**: Live market data via yFinance

---

## Project Architecture

```
GOLD-TRADING-AI/
|
|-- .streamlit/
|   +-- config.toml              # Streamlit theme configuration
|
|-- data/                        # Data directory (auto-generated)
|   |-- raw_market_data.csv      # Raw market data (~7 MB)
|   +-- processed_features.csv   # Processed features (~37 MB)
|
|-- models/                      # Trained models (auto-generated)
|   |-- xgboost_gold_predictor.joblib    # XGBoost model
|   +-- feature_scaler.joblib            # Feature scaler
|
|-- logs/                        # Log files
|
|-- config.py                    # Global configuration
|   |-- SYMBOLS                  # 16 financial assets definitions
|   |-- XGBOOST_PARAMS           # Model hyperparameters
|   +-- Technical indicators settings
|
|-- data_collector.py            # Data collection module
|   |-- MarketDataCollector      # Main class for data fetching
|   |-- fetch_all_data()         # Collect 25+ years of data
|   +-- fetch_realtime_data()    # Real-time market data
|
|-- feature_engineering.py       # Feature engineering module
|   |-- FeatureEngineer          # Main class for feature creation
|   |-- 344 technical features   # RSI, MACD, Bollinger, etc.
|   |-- Macro features           # Correlations, ratios
|   +-- Temporal features        # Seasonality, patterns
|
|-- model_training.py            # Model training module
|   |-- GoldTradingModel         # XGBoost classifier wrapper
|   |-- train()                  # Training with early stopping
|   |-- evaluate()               # Performance metrics
|   +-- predict()                # Generate predictions
|
|-- app.py                       # Streamlit dashboard (1100+ lines)
|   |-- Real-time metrics        # 8 key market indicators
|   |-- Trading signals          # BUY/SELL recommendations
|   |-- Technical charts         # Interactive Plotly charts
|   +-- Correlation analysis     # Cross-asset correlations
|
|-- run_pipeline.py              # Complete pipeline execution
|
|-- requirements.txt             # Python dependencies
|-- README.md                    # Project documentation
|-- LICENSE                      # MIT License
|-- CONTRIBUTING.md              # Contribution guidelines
|-- Makefile                     # Development shortcuts
|-- .gitignore                   # Git ignore rules
+-- .env.example                 # Environment variables template
```

---

## Data Flow

```
[Yahoo Finance API]
        |
        v
+-------------------+
| data_collector.py |  --> Fetches 16 assets, 25+ years
+-------------------+
        |
        v
+-------------------+
| raw_market_data   |  --> 6,344 rows, 80 columns
+-------------------+
        |
        v
+----------------------+
| feature_engineering  |  --> Creates 344 features
+----------------------+
        |
        v
+---------------------+
| processed_features  |  --> 6,084 rows, 431 columns
+---------------------+
        |
        v
+-------------------+
| model_training.py |  --> XGBoost classifier
+-------------------+
        |
        v
+-------------------+
| app.py (Streamlit)|  --> Real-time predictions
+-------------------+
```

---

## Data Sources

### Tracked Assets (16)

| Category | Assets | Symbols |
|----------|--------|---------|
| **Gold** | Gold Futures, GLD ETF, Gold Miners | GC=F, GLD, GDX |
| **Dollar** | Dollar Index | DX-Y.NYB |
| **Rates** | US 10Y, US 2Y | ^TNX, ^IRX |
| **Volatility** | VIX | ^VIX |
| **Indices** | S&P 500, NASDAQ | ^GSPC, ^IXIC |
| **Metals** | Silver, Platinum, Copper | SI=F, PL=F, HG=F |
| **Energy** | Crude Oil | CL=F |
| **Forex** | EUR/USD, USD/JPY | EURUSD=X, JPY=X |
| **Crypto** | Bitcoin | BTC-USD |

### Data Period

- **Start**: January 1, 2000
- **End**: Today
- **Duration**: 25+ years
- **Rows**: ~6,000+ trading days

---

## Features Created (344)

### Price & Returns
- Multi-period returns (1d, 2d, 3d, 5d, 10d, 15d, 20d, 30d, 60d, 90d)
- Log returns, Historical volatility (annualized)
- True Range, ATR (7, 14, 21)
- Opening gap, Distance to extremes

### Moving Averages
- SMA/EMA (5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200, 233)
- Distance to price, MA slopes
- Crossovers (Golden Cross, Death Cross)

### Momentum
- RSI (5, 7, 9, 14, 21, 28) with extreme zones
- MACD standard and alternative
- Stochastic (5, 9, 14, 21), Stochastic RSI
- Williams %R, CCI, ADX, Ultimate Oscillator
- Rate of Change multi-periods

### Volatility
- Bollinger Bands (periods: 10, 20, 50 x std: 1.5, 2, 2.5)
- Position within bands, BB Squeeze
- Breakout signals

### Volume
- Volume ratio, Volume spike detection
- OBV, VPT, MFI, A/D Line

### Macroeconomic
- Ratios: Gold/DXY, Gold/Silver, Gold/Platinum, Gold/Copper
- Rolling correlations (20d, 60d)
- VIX levels, Yield Curve, Oil trends
- S&P 500, NASDAQ, Bitcoin correlations

### Temporal
- Cyclic encoding (day, month)
- Flags: Monday, Friday, start/end of month, quarter
- Gold seasonality (Q1, Q3, Q4)

### Patterns
- Consecutive up/down days
- Candlestick patterns (Doji, Hammer, Engulfing)
- Higher Highs, Lower Lows

---

## Machine Learning Model

### XGBoost Configuration

```python
{
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'gamma': 0.15,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'eval_metric': 'auc'
}
```

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | ~90% |
| **Precision** | ~93% |
| **Recall** | ~91% |
| **F1-Score** | ~92% |
| **ROC-AUC** | ~98% |

---

## Streamlit Dashboard

### Dashboard Sections

1. **Real-Time Data**
   - XAU/USD price with variation
   - DXY, US 10Y, VIX
   - Silver, S&P 500, Oil, Bitcoin

2. **Trading Signal**
   - Recommendation (Strong Buy/Buy/Neutral/Sell/Strong Sell)
   - Probabilities with progress bars
   - Technical indicators (RSI, MACD, Bollinger)

3. **Technical Chart**
   - Japanese candlesticks
   - SMA 20/50/200
   - Bollinger Bands
   - RSI, MACD, Volume

4. **Correlations**
   - Normalized comparative performance
   - Correlation guide

5. **Feature Importance**
   - Top 20 decision factors
   - Interpretation

---

## Installation & Usage

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/IlyasFardaouix/GOLD-TRADING-AI.git
cd GOLD-TRADING-AI

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Execution

```bash
# Option 1: Complete pipeline (recommended for first time)
python run_pipeline.py

# Option 2: Individual steps
python data_collector.py      # Data collection
python feature_engineering.py # Feature creation
python model_training.py      # Model training

# Option 3: Launch application (after training)
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`

---

## Decision Thresholds

| Level | Probability | Signal |
|-------|-------------|--------|
| **Strong** | >= 70% | Strong Buy / Strong Sell |
| **Moderate** | 55-70% | Buy / Sell |
| **Weak** | < 55% | Weak Signal |

---

## Skills Demonstrated

| Domain | Skills |
|--------|--------|
| **Data Engineering** | ETL, Financial APIs, Data pipelines |
| **Quantitative Analysis** | Technical indicators, Macro correlations, Statistics |
| **Machine Learning** | Advanced feature engineering, XGBoost, Temporal validation |
| **Full-Stack Data** | Streamlit, Plotly, UI/UX, Interactive visualizations |

---

## Disclaimer

**This system is developed for educational and demonstration purposes only.**

The predictions provided do not constitute financial advice in any way. Trading involves significant risk of capital loss. Any investment decision should be made after consulting a qualified financial advisor.

---

## Future Improvements

- [ ] Fundamental data (inflation, employment, GDP)
- [ ] Ensemble models (Random Forest + LSTM + Transformer)
- [ ] Backtesting with Sharpe ratio calculation
- [ ] Email/SMS alerts
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] REST API for external integration
- [ ] Sentiment analysis (news, Twitter)

---

## Author

**Ilyas Fardaoui**

---


