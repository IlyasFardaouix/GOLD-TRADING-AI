"""
Configuration du Système de Trading XAU/USD
============================================
Paramètres globaux et clés API

Author: Ilyas Fardaoui
Project: Gold Trading AI - Intelligent Decision Support System
Version: 1.0.0
"""

import os
from datetime import datetime, timedelta

# =====================================
# CONFIGURATION DES DONNÉES
# =====================================

# Symboles des actifs - Dataset étendu
SYMBOLS = {
    'gold': 'GC=F',           # Gold Futures (principal)
    'gold_etf': 'GLD',        # SPDR Gold Trust ETF
    'gold_miners': 'GDX',     # Gold Miners ETF
    'dxy': 'DX-Y.NYB',        # US Dollar Index
    'us10y': '^TNX',          # 10-Year Treasury Yield
    'us2y': '^IRX',           # 2-Year Treasury (courbe des taux)
    'vix': '^VIX',            # Volatility Index
    'sp500': '^GSPC',         # S&P 500
    'nasdaq': '^IXIC',        # NASDAQ Composite
    'silver': 'SI=F',         # Silver Futures
    'platinum': 'PL=F',       # Platinum Futures
    'copper': 'HG=F',         # Copper Futures (indicateur économique)
    'oil': 'CL=F',            # Crude Oil (inflation proxy)
    'eurusd': 'EURUSD=X',     # EUR/USD
    'usdjpy': 'JPY=X',        # USD/JPY
    'btc': 'BTC-USD',         # Bitcoin (corrélation récente)
}

# Période de données historiques - DATASET MASSIF
START_DATE = '2000-01-01'     # 25+ ans de données
END_DATE = datetime.now().strftime('%Y-%m-%d')

# =====================================
# CONFIGURATION DU MODÈLE ML
# =====================================

# Paramètres de prédiction
PREDICTION_HORIZON = 5  # Prédire la direction sur N jours
TRAIN_TEST_SPLIT = 0.85  # Plus de données pour l'entraînement
RANDOM_STATE = 42

# Paramètres XGBoost optimaux pour grand dataset
XGBOOST_PARAMS = {
    'n_estimators': 500,          # Plus d'arbres pour grand dataset
    'max_depth': 8,               # Arbres plus profonds
    'learning_rate': 0.03,        # Learning rate plus bas
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    'min_child_weight': 5,
    'gamma': 0.15,
    'reg_alpha': 0.5,
    'reg_lambda': 2.0,
    'scale_pos_weight': 1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'eval_metric': 'auc',
    'early_stopping_rounds': 50
}

# =====================================
# INDICATEURS TECHNIQUES
# =====================================

# Périodes pour les moyennes mobiles
MA_PERIODS = [5, 10, 20, 50, 100, 200]

# Périodes RSI
RSI_PERIODS = [7, 14, 21]

# Périodes MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Bandes de Bollinger
BB_PERIOD = 20
BB_STD = 2

# =====================================
# CHEMINS DES FICHIERS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Créer les dossiers s'ils n'existent pas
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Fichiers de données
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw_market_data.csv')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed_features.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_gold_predictor.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'feature_scaler.joblib')

# =====================================
# CONFIGURATION STREAMLIT
# =====================================

STREAMLIT_CONFIG = {
    'page_title': '🥇 Gold Trading AI',
    'page_icon': '🥇',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# Intervalle de rafraîchissement (en secondes)
REFRESH_INTERVAL = 60

# =====================================
# SEUILS DE DÉCISION
# =====================================

# Seuil de probabilité pour la recommandation
CONFIDENCE_THRESHOLD_HIGH = 0.70    # Signal fort
CONFIDENCE_THRESHOLD_MEDIUM = 0.55  # Signal modéré
# En dessous = Signal faible / Neutre
