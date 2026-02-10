"""
Module de Feature Engineering - Version Étendue
=================================================
Création de 150+ indicateurs techniques et features pour le ML
Optimisé pour un dataset massif (25+ ans de données)

Author: Ilyas Fardaoui
Project: Gold Trading AI

Technical Indicators:
    - Moving Averages (SMA, EMA) - Multiple periods
    - RSI (Relative Strength Index) - 7, 14, 21 periods
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands - Multiple configurations
    - Stochastic Oscillator, Williams %R, CCI, ADX
    - ATR (Average True Range) for volatility

Macro Features:
    - Dollar correlation, yield spreads
    - Cross-asset ratios (Gold/Silver, Gold/DXY)
    - Pattern detection (higher highs, lower lows)
"""

import pandas as pd
import numpy as np
import warnings
import logging
import os
import sys

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    MA_PERIODS, RSI_PERIODS, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BB_PERIOD, BB_STD, PREDICTION_HORIZON, PROCESSED_DATA_PATH, RAW_DATA_PATH
)

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Classe pour la création de features techniques et macroéconomiques.
    Génère 150+ features pour maximiser la précision du modèle.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise le FeatureEngineer.
        
        Args:
            df: DataFrame avec les données brutes de marché
        """
        self.df = df.copy()
        self.feature_names = []
        self.feature_count = 0
        
    def add_price_features(self, prefix: str = 'gold'):
        """
        Ajoute des features de prix dérivées.
        
        Args:
            prefix: Préfixe de l'actif principal
        """
        logger.info("📈 Création des features de prix...")
        
        close_col = f'{prefix}_close'
        high_col = f'{prefix}_high'
        low_col = f'{prefix}_low'
        open_col = f'{prefix}_open'
        
        if close_col not in self.df.columns:
            logger.warning(f"Colonne {close_col} non trouvée!")
            return
            
        # Returns multi-périodes
        for period in [1, 2, 3, 5, 10, 15, 20, 30, 60, 90]:
            self.df[f'return_{period}d'] = self.df[close_col].pct_change(period)
            
        # Log returns
        self.df['log_return_1d'] = np.log(self.df[close_col] / self.df[close_col].shift(1))
        self.df['log_return_5d'] = np.log(self.df[close_col] / self.df[close_col].shift(5))
        
        # Volatilité historique (rolling std des returns)
        for period in [5, 10, 20, 30, 60]:
            self.df[f'volatility_{period}d'] = self.df['return_1d'].rolling(period).std()
            self.df[f'volatility_{period}d_ann'] = self.df[f'volatility_{period}d'] * np.sqrt(252)
            
        # Range du jour (High-Low)
        if high_col in self.df.columns and low_col in self.df.columns:
            self.df['daily_range'] = (self.df[high_col] - self.df[low_col]) / self.df[close_col]
            self.df['daily_range_ma5'] = self.df['daily_range'].rolling(5).mean()
            self.df['daily_range_ma20'] = self.df['daily_range'].rolling(20).mean()
            
            # True Range & ATR
            self.df['true_range'] = np.maximum(
                self.df[high_col] - self.df[low_col],
                np.maximum(
                    abs(self.df[high_col] - self.df[close_col].shift(1)),
                    abs(self.df[low_col] - self.df[close_col].shift(1))
                )
            )
            for period in [7, 14, 21]:
                self.df[f'atr_{period}'] = self.df['true_range'].rolling(period).mean()
                self.df[f'atr_{period}_pct'] = self.df[f'atr_{period}'] / self.df[close_col]
            
        # Gap d'ouverture
        if open_col in self.df.columns:
            self.df['gap'] = (self.df[open_col] - self.df[close_col].shift(1)) / self.df[close_col].shift(1)
            self.df['gap_ma5'] = self.df['gap'].rolling(5).mean()
            
        # Distance aux extremes
        for period in [20, 50, 100, 200]:
            self.df[f'dist_high_{period}d'] = (self.df[close_col] - self.df[close_col].rolling(period).max()) / self.df[close_col]
            self.df[f'dist_low_{period}d'] = (self.df[close_col] - self.df[close_col].rolling(period).min()) / self.df[close_col]
            
        # Momentum à différentes échelles
        for period in [5, 10, 20, 60]:
            self.df[f'momentum_{period}d'] = self.df[close_col] - self.df[close_col].shift(period)
            
        logger.info("✅ Features de prix créées")
        
    def add_moving_averages(self, prefix: str = 'gold'):
        """
        Ajoute les moyennes mobiles et leurs dérivées.
        
        Args:
            prefix: Préfixe de l'actif
        """
        logger.info("📊 Création des moyennes mobiles...")
        
        close_col = f'{prefix}_close'
        
        if close_col not in self.df.columns:
            return
            
        ma_periods = [5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200, 233]
        
        for period in ma_periods:
            # SMA
            self.df[f'sma_{period}'] = self.df[close_col].rolling(period).mean()
            
            # EMA
            self.df[f'ema_{period}'] = self.df[close_col].ewm(span=period, adjust=False).mean()
            
            # Distance au prix (en %)
            self.df[f'dist_sma_{period}'] = (self.df[close_col] - self.df[f'sma_{period}']) / self.df[f'sma_{period}']
            self.df[f'dist_ema_{period}'] = (self.df[close_col] - self.df[f'ema_{period}']) / self.df[f'ema_{period}']
            
        # Croisements de moyennes mobiles
        ma_pairs = [(5, 20), (8, 21), (10, 50), (20, 50), (50, 100), (50, 200), (100, 200)]
        for fast, slow in ma_pairs:
            if f'sma_{fast}' in self.df.columns and f'sma_{slow}' in self.df.columns:
                self.df[f'ma_cross_{fast}_{slow}'] = (self.df[f'sma_{fast}'] > self.df[f'sma_{slow}']).astype(int)
                # Signal de croisement
                self.df[f'ma_cross_signal_{fast}_{slow}'] = self.df[f'ma_cross_{fast}_{slow}'].diff()
                
        # Golden Cross / Death Cross
        if 'sma_50' in self.df.columns and 'sma_200' in self.df.columns:
            self.df['golden_cross'] = ((self.df['sma_50'] > self.df['sma_200']) & 
                                        (self.df['sma_50'].shift(1) <= self.df['sma_200'].shift(1))).astype(int)
            self.df['death_cross'] = ((self.df['sma_50'] < self.df['sma_200']) & 
                                       (self.df['sma_50'].shift(1) >= self.df['sma_200'].shift(1))).astype(int)
            
        # Pente des moyennes mobiles
        for period in [20, 50, 200]:
            if f'sma_{period}' in self.df.columns:
                self.df[f'sma_{period}_slope'] = self.df[f'sma_{period}'].diff(5) / self.df[f'sma_{period}'].shift(5)
            
        logger.info("✅ Moyennes mobiles créées")
        
    def add_rsi(self, prefix: str = 'gold'):
        """
        Calcule le RSI (Relative Strength Index).
        
        Args:
            prefix: Préfixe de l'actif
        """
        logger.info("📉 Calcul du RSI...")
        
        close_col = f'{prefix}_close'
        
        if close_col not in self.df.columns:
            return
            
        rsi_periods = [5, 7, 9, 14, 21, 28]
        
        for period in rsi_periods:
            delta = self.df[close_col].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            
            avg_gain = gain.ewm(com=period-1, adjust=True).mean()
            avg_loss = loss.ewm(com=period-1, adjust=True).mean()
            
            rs = avg_gain / avg_loss
            self.df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
            
            # Zones de surachat/survente
            self.df[f'rsi_{period}_overbought'] = (self.df[f'rsi_{period}'] > 70).astype(int)
            self.df[f'rsi_{period}_oversold'] = (self.df[f'rsi_{period}'] < 30).astype(int)
            self.df[f'rsi_{period}_extreme_ob'] = (self.df[f'rsi_{period}'] > 80).astype(int)
            self.df[f'rsi_{period}_extreme_os'] = (self.df[f'rsi_{period}'] < 20).astype(int)
            
        # RSI Divergence (simplifié)
        self.df['rsi_14_ma'] = self.df['rsi_14'].rolling(5).mean()
        self.df['rsi_14_divergence'] = self.df['rsi_14'] - self.df['rsi_14_ma']
            
        logger.info("✅ RSI calculé")
        
    def add_macd(self, prefix: str = 'gold'):
        """
        Calcule le MACD (Moving Average Convergence Divergence).
        
        Args:
            prefix: Préfixe de l'actif
        """
        logger.info("📈 Calcul du MACD...")
        
        close_col = f'{prefix}_close'
        
        if close_col not in self.df.columns:
            return
        
        # MACD standard (12, 26, 9)
        ema_fast = self.df[close_col].ewm(span=12, adjust=False).mean()
        ema_slow = self.df[close_col].ewm(span=26, adjust=False).mean()
        
        self.df['macd_line'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd_line'].ewm(span=9, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd_line'] - self.df['macd_signal']
        
        # MACD normalisé
        self.df['macd_normalized'] = self.df['macd_line'] / self.df[close_col] * 100
        
        # Signaux MACD
        self.df['macd_bullish'] = ((self.df['macd_line'] > self.df['macd_signal']) & 
                                    (self.df['macd_line'].shift(1) <= self.df['macd_signal'].shift(1))).astype(int)
        self.df['macd_bearish'] = ((self.df['macd_line'] < self.df['macd_signal']) & 
                                    (self.df['macd_line'].shift(1) >= self.df['macd_signal'].shift(1))).astype(int)
        
        # Histogramme croissant/décroissant
        self.df['macd_hist_increasing'] = (self.df['macd_histogram'] > self.df['macd_histogram'].shift(1)).astype(int)
        
        # MACD alternatif (5, 35, 5) pour swing trading
        ema_fast2 = self.df[close_col].ewm(span=5, adjust=False).mean()
        ema_slow2 = self.df[close_col].ewm(span=35, adjust=False).mean()
        self.df['macd_swing'] = ema_fast2 - ema_slow2
        self.df['macd_swing_signal'] = self.df['macd_swing'].ewm(span=5, adjust=False).mean()
        
        logger.info("✅ MACD calculé")
        
    def add_bollinger_bands(self, prefix: str = 'gold'):
        """
        Calcule les Bandes de Bollinger.
        
        Args:
            prefix: Préfixe de l'actif
        """
        logger.info("📊 Calcul des Bandes de Bollinger...")
        
        close_col = f'{prefix}_close'
        
        if close_col not in self.df.columns:
            return
        
        for period in [10, 20, 50]:
            for std_mult in [1.5, 2, 2.5]:
                suffix = f'{period}_{int(std_mult*10)}'
                
                sma = self.df[close_col].rolling(period).mean()
                std = self.df[close_col].rolling(period).std()
                
                self.df[f'bb_upper_{suffix}'] = sma + (std_mult * std)
                self.df[f'bb_lower_{suffix}'] = sma - (std_mult * std)
                self.df[f'bb_width_{suffix}'] = (self.df[f'bb_upper_{suffix}'] - self.df[f'bb_lower_{suffix}']) / sma
                
                # Position du prix dans les bandes (0 = bas, 1 = haut)
                self.df[f'bb_position_{suffix}'] = ((self.df[close_col] - self.df[f'bb_lower_{suffix}']) / 
                                                     (self.df[f'bb_upper_{suffix}'] - self.df[f'bb_lower_{suffix}']))
        
        # Bandes de Bollinger standard (20, 2)
        sma = self.df[close_col].rolling(20).mean()
        std = self.df[close_col].rolling(20).std()
        self.df['bb_upper'] = sma + (2 * std)
        self.df['bb_lower'] = sma - (2 * std)
        self.df['bb_middle'] = sma
        self.df['bb_width'] = (self.df['bb_upper'] - self.df['bb_lower']) / self.df['bb_middle']
        self.df['bb_position'] = (self.df[close_col] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        # Signaux de breakout
        self.df['bb_breakout_up'] = (self.df[close_col] > self.df['bb_upper']).astype(int)
        self.df['bb_breakout_down'] = (self.df[close_col] < self.df['bb_lower']).astype(int)
        
        # Squeeze (faible volatilité)
        self.df['bb_squeeze'] = (self.df['bb_width'] < self.df['bb_width'].rolling(50).mean() * 0.75).astype(int)
        
        logger.info("✅ Bandes de Bollinger calculées")
        
    def add_momentum_indicators(self, prefix: str = 'gold'):
        """
        Calcule les indicateurs de momentum avancés.
        
        Args:
            prefix: Préfixe de l'actif
        """
        logger.info("⚡ Calcul des indicateurs de momentum...")
        
        close_col = f'{prefix}_close'
        high_col = f'{prefix}_high'
        low_col = f'{prefix}_low'
        
        if close_col not in self.df.columns:
            return
            
        # Rate of Change (ROC)
        for period in [5, 10, 14, 20, 30, 60]:
            self.df[f'roc_{period}'] = (self.df[close_col] - self.df[close_col].shift(period)) / self.df[close_col].shift(period) * 100
            
        # Stochastic Oscillator
        if high_col in self.df.columns and low_col in self.df.columns:
            for period in [5, 9, 14, 21]:
                low_min = self.df[low_col].rolling(period).min()
                high_max = self.df[high_col].rolling(period).max()
                
                self.df[f'stoch_k_{period}'] = 100 * (self.df[close_col] - low_min) / (high_max - low_min)
                self.df[f'stoch_d_{period}'] = self.df[f'stoch_k_{period}'].rolling(3).mean()
                
            # Stochastic RSI
            rsi = self.df['rsi_14']
            rsi_min = rsi.rolling(14).min()
            rsi_max = rsi.rolling(14).max()
            self.df['stoch_rsi'] = (rsi - rsi_min) / (rsi_max - rsi_min)
            
        # Williams %R
        if high_col in self.df.columns and low_col in self.df.columns:
            for period in [10, 14, 20]:
                high_max = self.df[high_col].rolling(period).max()
                low_min = self.df[low_col].rolling(period).min()
                
                self.df[f'williams_r_{period}'] = -100 * (high_max - self.df[close_col]) / (high_max - low_min)
                
        # CCI (Commodity Channel Index)
        if high_col in self.df.columns and low_col in self.df.columns:
            for period in [14, 20]:
                tp = (self.df[high_col] + self.df[low_col] + self.df[close_col]) / 3
                tp_ma = tp.rolling(period).mean()
                tp_md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
                self.df[f'cci_{period}'] = (tp - tp_ma) / (0.015 * tp_md)
                
        # ADX (Average Directional Index)
        if high_col in self.df.columns and low_col in self.df.columns:
            plus_dm = self.df[high_col].diff()
            minus_dm = -self.df[low_col].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr = self.df['true_range'] if 'true_range' in self.df.columns else (self.df[high_col] - self.df[low_col])
            
            for period in [14, 20]:
                atr = tr.rolling(period).mean()
                plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
                minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                self.df[f'adx_{period}'] = dx.rolling(period).mean()
                self.df[f'plus_di_{period}'] = plus_di
                self.df[f'minus_di_{period}'] = minus_di
                
        # Ultimate Oscillator
        if high_col in self.df.columns and low_col in self.df.columns:
            bp = self.df[close_col] - np.minimum(self.df[low_col], self.df[close_col].shift(1))
            tr = np.maximum(self.df[high_col], self.df[close_col].shift(1)) - np.minimum(self.df[low_col], self.df[close_col].shift(1))
            
            avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
            avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
            avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
            
            self.df['ultimate_osc'] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
            
        logger.info("✅ Indicateurs de momentum calculés")
        
    def add_volume_indicators(self, prefix: str = 'gold'):
        """
        Calcule les indicateurs de volume.
        
        Args:
            prefix: Préfixe de l'actif
        """
        logger.info("📊 Calcul des indicateurs de volume...")
        
        close_col = f'{prefix}_close'
        volume_col = f'{prefix}_volume'
        high_col = f'{prefix}_high'
        low_col = f'{prefix}_low'
        
        if volume_col not in self.df.columns:
            logger.warning("Pas de données de volume disponibles")
            return
            
        # Volume moyen
        for period in [5, 10, 20, 50]:
            self.df[f'volume_ma_{period}'] = self.df[volume_col].rolling(period).mean()
            
        # Ratio volume vs moyenne
        self.df['volume_ratio_20'] = self.df[volume_col] / self.df['volume_ma_20']
        self.df['volume_ratio_50'] = self.df[volume_col] / self.df['volume_ma_50']
        
        # Volume spike
        self.df['volume_spike'] = (self.df['volume_ratio_20'] > 2).astype(int)
        
        # On-Balance Volume (OBV)
        direction = np.sign(self.df[close_col].diff())
        self.df['obv'] = (direction * self.df[volume_col]).cumsum()
        self.df['obv_ma_20'] = self.df['obv'].rolling(20).mean()
        self.df['obv_trend'] = (self.df['obv'] > self.df['obv_ma_20']).astype(int)
        
        # Volume Price Trend (VPT)
        price_change = self.df[close_col].pct_change()
        self.df['vpt'] = (price_change * self.df[volume_col]).cumsum()
        
        # Money Flow Index (MFI)
        if high_col in self.df.columns and low_col in self.df.columns:
            tp = (self.df[high_col] + self.df[low_col] + self.df[close_col]) / 3
            mf = tp * self.df[volume_col]
            
            positive_mf = mf.where(tp > tp.shift(1), 0)
            negative_mf = mf.where(tp < tp.shift(1), 0)
            
            for period in [14, 20]:
                mf_ratio = positive_mf.rolling(period).sum() / negative_mf.rolling(period).sum()
                self.df[f'mfi_{period}'] = 100 - (100 / (1 + mf_ratio))
                
        # Accumulation/Distribution Line
        if high_col in self.df.columns and low_col in self.df.columns:
            clv = ((self.df[close_col] - self.df[low_col]) - (self.df[high_col] - self.df[close_col])) / (self.df[high_col] - self.df[low_col])
            clv = clv.fillna(0)
            self.df['ad_line'] = (clv * self.df[volume_col]).cumsum()
            self.df['ad_line_ma'] = self.df['ad_line'].rolling(20).mean()
            
        logger.info("✅ Indicateurs de volume calculés")
        
    def add_macro_features(self):
        """
        Crée des features basées sur les corrélations macro étendues.
        """
        logger.info("🌍 Création des features macroéconomiques...")
        
        gold_close = 'gold_close'
        
        # DXY (Dollar Index)
        if 'dxy_close' in self.df.columns and gold_close in self.df.columns:
            self.df['gold_dxy_ratio'] = self.df[gold_close] / self.df['dxy_close']
            self.df['dxy_return_1d'] = self.df['dxy_close'].pct_change(1)
            self.df['dxy_return_5d'] = self.df['dxy_close'].pct_change(5)
            self.df['dxy_return_20d'] = self.df['dxy_close'].pct_change(20)
            
            # Corrélation roulante
            for period in [20, 60]:
                self.df[f'gold_dxy_corr_{period}d'] = (self.df[gold_close].pct_change()
                                                       .rolling(period)
                                                       .corr(self.df['dxy_close'].pct_change()))
                                                       
        # VIX (indicateur de peur)
        if 'vix_close' in self.df.columns:
            self.df['vix_level'] = self.df['vix_close']
            self.df['vix_ma_20'] = self.df['vix_close'].rolling(20).mean()
            self.df['vix_high'] = (self.df['vix_close'] > 25).astype(int)
            self.df['vix_extreme'] = (self.df['vix_close'] > 35).astype(int)
            self.df['vix_low'] = (self.df['vix_close'] < 15).astype(int)
            self.df['vix_change_1d'] = self.df['vix_close'].pct_change(1)
            self.df['vix_change_5d'] = self.df['vix_close'].pct_change(5)
            self.df['vix_spike'] = (self.df['vix_change_1d'] > 0.1).astype(int)
            
            if gold_close in self.df.columns:
                for period in [20, 60]:
                    self.df[f'gold_vix_corr_{period}d'] = (self.df[gold_close].pct_change()
                                                           .rolling(period)
                                                           .corr(self.df['vix_close'].pct_change()))
            
        # Taux 10 ans US
        if 'us10y_close' in self.df.columns:
            self.df['us10y_level'] = self.df['us10y_close']
            self.df['us10y_change_1d'] = self.df['us10y_close'].diff(1)
            self.df['us10y_change_5d'] = self.df['us10y_close'].diff(5)
            self.df['us10y_change_20d'] = self.df['us10y_close'].diff(20)
            self.df['us10y_ma_20'] = self.df['us10y_close'].rolling(20).mean()
            self.df['us10y_above_ma'] = (self.df['us10y_close'] > self.df['us10y_ma_20']).astype(int)
            self.df['us10y_rising'] = (self.df['us10y_change_5d'] > 0).astype(int)
            
        # Courbe des taux (spread 10Y - 2Y)
        if 'us10y_close' in self.df.columns and 'us2y_close' in self.df.columns:
            self.df['yield_curve'] = self.df['us10y_close'] - self.df['us2y_close']
            self.df['yield_curve_inverted'] = (self.df['yield_curve'] < 0).astype(int)
            
        # Silver (corrélation positive avec l'or)
        if 'silver_close' in self.df.columns and gold_close in self.df.columns:
            self.df['gold_silver_ratio'] = self.df[gold_close] / self.df['silver_close']
            self.df['gold_silver_ratio_ma20'] = self.df['gold_silver_ratio'].rolling(20).mean()
            self.df['gold_silver_ratio_high'] = (self.df['gold_silver_ratio'] > self.df['gold_silver_ratio_ma20']).astype(int)
            self.df['silver_return_1d'] = self.df['silver_close'].pct_change(1)
            self.df['silver_return_5d'] = self.df['silver_close'].pct_change(5)
            
        # Platinum
        if 'platinum_close' in self.df.columns and gold_close in self.df.columns:
            self.df['gold_platinum_ratio'] = self.df[gold_close] / self.df['platinum_close']
            self.df['platinum_return_5d'] = self.df['platinum_close'].pct_change(5)
            
        # Copper (indicateur économique)
        if 'copper_close' in self.df.columns:
            self.df['copper_return_5d'] = self.df['copper_close'].pct_change(5)
            self.df['copper_return_20d'] = self.df['copper_close'].pct_change(20)
            if gold_close in self.df.columns:
                self.df['gold_copper_ratio'] = self.df[gold_close] / self.df['copper_close']
            
        # Oil (proxy inflation)
        if 'oil_close' in self.df.columns:
            self.df['oil_return_5d'] = self.df['oil_close'].pct_change(5)
            self.df['oil_return_20d'] = self.df['oil_close'].pct_change(20)
            self.df['oil_ma_50'] = self.df['oil_close'].rolling(50).mean()
            self.df['oil_above_ma50'] = (self.df['oil_close'] > self.df['oil_ma_50']).astype(int)
            
        # S&P 500 (corrélation variable)
        if 'sp500_close' in self.df.columns:
            self.df['sp500_return_1d'] = self.df['sp500_close'].pct_change(1)
            self.df['sp500_return_5d'] = self.df['sp500_close'].pct_change(5)
            self.df['sp500_return_20d'] = self.df['sp500_close'].pct_change(20)
            self.df['sp500_ma_200'] = self.df['sp500_close'].rolling(200).mean()
            self.df['sp500_above_ma200'] = (self.df['sp500_close'] > self.df['sp500_ma_200']).astype(int)
            
            if gold_close in self.df.columns:
                for period in [20, 60]:
                    self.df[f'gold_sp500_corr_{period}d'] = (self.df[gold_close].pct_change()
                                                              .rolling(period)
                                                              .corr(self.df['sp500_close'].pct_change()))
                                                              
        # NASDAQ
        if 'nasdaq_close' in self.df.columns:
            self.df['nasdaq_return_5d'] = self.df['nasdaq_close'].pct_change(5)
            
        # EUR/USD
        if 'eurusd_close' in self.df.columns:
            self.df['eurusd_return_5d'] = self.df['eurusd_close'].pct_change(5)
            
        # Bitcoin (corrélation récente)
        if 'btc_close' in self.df.columns and gold_close in self.df.columns:
            self.df['btc_return_5d'] = self.df['btc_close'].pct_change(5)
            self.df['gold_btc_corr_60d'] = (self.df[gold_close].pct_change()
                                             .rolling(60)
                                             .corr(self.df['btc_close'].pct_change()))
            
        # Gold Miners ETF (GDX)
        if 'gold_miners_close' in self.df.columns and gold_close in self.df.columns:
            self.df['gdx_gold_ratio'] = self.df['gold_miners_close'] / self.df[gold_close] * 100
            self.df['gdx_return_5d'] = self.df['gold_miners_close'].pct_change(5)
            self.df['gdx_outperform'] = (self.df['gdx_return_5d'] > self.df['return_5d']).astype(int)
            
        logger.info("✅ Features macroéconomiques créées")
        
    def add_temporal_features(self):
        """
        Ajoute des features temporelles détaillées.
        """
        logger.info("📅 Création des features temporelles...")
        
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['day_of_month'] = self.df.index.day
        self.df['week_of_year'] = self.df.index.isocalendar().week.astype(int)
        self.df['month'] = self.df.index.month
        self.df['quarter'] = self.df.index.quarter
        self.df['year'] = self.df.index.year
        
        # Encodage cyclique
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        
        # Flags spéciaux
        self.df['is_monday'] = (self.df['day_of_week'] == 0).astype(int)
        self.df['is_friday'] = (self.df['day_of_week'] == 4).astype(int)
        self.df['is_month_start'] = self.df.index.is_month_start.astype(int)
        self.df['is_month_end'] = self.df.index.is_month_end.astype(int)
        self.df['is_quarter_start'] = self.df.index.is_quarter_start.astype(int)
        self.df['is_quarter_end'] = self.df.index.is_quarter_end.astype(int)
        self.df['is_year_start'] = self.df.index.is_year_start.astype(int)
        self.df['is_year_end'] = self.df.index.is_year_end.astype(int)
        
        # Saisonnalité de l'or
        self.df['is_q1'] = (self.df['quarter'] == 1).astype(int)  # Nouvel an chinois
        self.df['is_q3'] = (self.df['quarter'] == 3).astype(int)  # Été calme
        self.df['is_q4'] = (self.df['quarter'] == 4).astype(int)  # Forte demande
        
        # Mois historiquement forts pour l'or
        self.df['gold_season_strong'] = self.df['month'].isin([1, 2, 8, 9, 11, 12]).astype(int)
        
        logger.info("✅ Features temporelles créées")
        
    def add_lag_features(self, prefix: str = 'gold', max_lag: int = 10):
        """
        Ajoute des features retardées (lags).
        
        Args:
            prefix: Préfixe de l'actif
            max_lag: Nombre maximum de lags
        """
        logger.info("⏰ Création des features de lag...")
        
        close_col = f'{prefix}_close'
        
        if close_col not in self.df.columns:
            return
            
        # Lags des returns
        for lag in range(1, max_lag + 1):
            self.df[f'return_lag_{lag}'] = self.df['return_1d'].shift(lag)
            
        # Lags du RSI
        if 'rsi_14' in self.df.columns:
            for lag in [1, 2, 3, 5]:
                self.df[f'rsi_14_lag_{lag}'] = self.df['rsi_14'].shift(lag)
                
        # Lags du MACD
        if 'macd_histogram' in self.df.columns:
            for lag in [1, 2, 3]:
                self.df[f'macd_hist_lag_{lag}'] = self.df['macd_histogram'].shift(lag)
        
        logger.info("✅ Features de lag créées")
        
    def add_pattern_features(self, prefix: str = 'gold'):
        """
        Détecte des patterns de prix simples.
        
        Args:
            prefix: Préfixe de l'actif
        """
        logger.info("🎯 Détection des patterns...")
        
        close_col = f'{prefix}_close'
        high_col = f'{prefix}_high'
        low_col = f'{prefix}_low'
        open_col = f'{prefix}_open'
        
        if close_col not in self.df.columns:
            return
            
        # Jours consécutifs up/down
        returns = self.df[close_col].pct_change()
        self.df['consecutive_up'] = (returns > 0).astype(int)
        self.df['consecutive_down'] = (returns < 0).astype(int)
        
        # Compteur de jours consécutifs
        self.df['streak_up'] = self.df['consecutive_up'].groupby((self.df['consecutive_up'] != self.df['consecutive_up'].shift()).cumsum()).cumsum()
        self.df['streak_down'] = self.df['consecutive_down'].groupby((self.df['consecutive_down'] != self.df['consecutive_down'].shift()).cumsum()).cumsum()
        
        # Candlestick patterns (simplifié)
        if all(col in self.df.columns for col in [open_col, high_col, low_col]):
            body = abs(self.df[close_col] - self.df[open_col])
            range_hl = self.df[high_col] - self.df[low_col]
            
            # Doji
            self.df['is_doji'] = (body / range_hl < 0.1).astype(int)
            
            # Hammer
            upper_shadow = self.df[high_col] - np.maximum(self.df[close_col], self.df[open_col])
            lower_shadow = np.minimum(self.df[close_col], self.df[open_col]) - self.df[low_col]
            self.df['is_hammer'] = ((lower_shadow > 2 * body) & (upper_shadow < body * 0.5) & (returns.shift(-1) > 0)).astype(int)
            
            # Engulfing
            self.df['bullish_engulfing'] = ((self.df[close_col].shift(1) < self.df[open_col].shift(1)) & 
                                             (self.df[close_col] > self.df[open_col]) &
                                             (self.df[close_col] > self.df[open_col].shift(1)) &
                                             (self.df[open_col] < self.df[close_col].shift(1))).astype(int)
            
        # Higher highs / Lower lows
        if high_col in self.df.columns and low_col in self.df.columns:
            self.df['higher_high'] = (self.df[high_col] > self.df[high_col].shift(1)).astype(int)
            self.df['lower_low'] = (self.df[low_col] < self.df[low_col].shift(1)).astype(int)
            self.df['higher_low'] = (self.df[low_col] > self.df[low_col].shift(1)).astype(int)
            
        logger.info("✅ Patterns détectés")
        
    def create_target(self, prefix: str = 'gold', horizon: int = None):
        """
        Crée la variable cible (direction future du prix).
        
        Args:
            prefix: Préfixe de l'actif
            horizon: Horizon de prédiction en jours
        """
        logger.info("🎯 Création de la variable cible...")
        
        horizon = horizon or PREDICTION_HORIZON
        close_col = f'{prefix}_close'
        
        if close_col not in self.df.columns:
            raise ValueError(f"Colonne {close_col} non trouvée!")
            
        # Calculer le return futur
        future_return = self.df[close_col].shift(-horizon) / self.df[close_col] - 1
        
        # Variable cible binaire: 1 = Achat (prix monte), 0 = Vente (prix baisse)
        self.df['target'] = (future_return > 0).astype(int)
        
        # Magnitude du mouvement (pour analyse)
        self.df['future_return'] = future_return
        
        # Classes multi (optionnel)
        self.df['target_multi'] = pd.cut(future_return, 
                                          bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                                          labels=[0, 1, 2, 3, 4])  # Strong sell, Sell, Neutral, Buy, Strong buy
        
        logger.info(f"✅ Cible créée (horizon: {horizon} jours)")
        logger.info(f"   Distribution: {self.df['target'].value_counts().to_dict()}")
        
    def build_all_features(self, prefix: str = 'gold') -> pd.DataFrame:
        """
        Construit toutes les features.
        
        Args:
            prefix: Préfixe de l'actif principal
            
        Returns:
            DataFrame avec toutes les features
        """
        logger.info("=" * 60)
        logger.info("🚀 DÉBUT DU FEATURE ENGINEERING AVANCÉ")
        logger.info("=" * 60)
        
        self.add_price_features(prefix)
        self.add_moving_averages(prefix)
        self.add_rsi(prefix)
        self.add_macd(prefix)
        self.add_bollinger_bands(prefix)
        self.add_momentum_indicators(prefix)
        self.add_volume_indicators(prefix)
        self.add_macro_features()
        self.add_temporal_features()
        self.add_lag_features(prefix)
        self.add_pattern_features(prefix)
        self.create_target(prefix)
        
        # Nettoyer les valeurs infinies
        initial_len = len(self.df)
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Supprimer seulement les lignes sans cible valide
        if 'target' in self.df.columns:
            self.df = self.df.dropna(subset=['target'])
        
        # Forward-fill puis backward-fill pour les autres colonnes
        self.df = self.df.fillna(method='ffill')
        self.df = self.df.fillna(method='bfill')
        
        # Supprimer les premières lignes qui pourraient avoir des NaN résiduels
        # (typiquement les 250 premières à cause des MA longues)
        max_lookback = 260  # un peu plus que la MA la plus longue (250)
        if len(self.df) > max_lookback:
            self.df = self.df.iloc[max_lookback:]
        
        final_len = len(self.df)
        
        # Compter les features
        feature_cols = self.get_feature_columns()
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ FEATURE ENGINEERING TERMINÉ")
        logger.info("=" * 60)
        logger.info(f"📊 Lignes: {initial_len:,} → {final_len:,} (supprimées: {initial_len - final_len:,})")
        logger.info(f"📈 Features créées: {len(feature_cols)}")
        
        if len(self.df) > 0:
            logger.info(f"📅 Période: {self.df.index.min().strftime('%Y-%m-%d')} → {self.df.index.max().strftime('%Y-%m-%d')}")
        else:
            logger.warning("⚠️ Aucune donnée après nettoyage!")
        
        return self.df
    
    def get_feature_columns(self) -> list:
        """
        Retourne la liste des colonnes de features (sans la cible).
        
        Returns:
            Liste des noms de colonnes
        """
        exclude_cols = ['target', 'future_return', 'target_multi']
        # Exclure aussi les colonnes brutes OHLCV
        exclude_patterns = ['_open', '_high', '_low', '_close', '_volume']
        
        feature_cols = []
        for col in self.df.columns:
            if col in exclude_cols:
                continue
            if any(col.endswith(pattern) for pattern in exclude_patterns):
                continue
            feature_cols.append(col)
            
        return feature_cols
    
    def save_processed_data(self, filepath: str = None):
        """
        Sauvegarde les données traitées.
        
        Args:
            filepath: Chemin du fichier
        """
        filepath = filepath or PROCESSED_DATA_PATH
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.df.to_csv(filepath)
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"💾 Données traitées sauvegardées: {filepath} ({size_mb:.2f} MB)")


def process_raw_data(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Fonction principale pour traiter les données brutes.
    
    Args:
        df: DataFrame brut (optionnel, charge depuis fichier sinon)
        
    Returns:
        DataFrame avec features
    """
    if df is None:
        df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
        
    engineer = FeatureEngineer(df)
    processed_df = engineer.build_all_features()
    engineer.save_processed_data()
    
    return processed_df


if __name__ == "__main__":
    # Test du module
    from data_collector import MarketDataCollector
    
    # Collecter les données
    collector = MarketDataCollector()
    raw_df = collector.fetch_all_data()
    collector.save_data(raw_df)
    
    # Traiter les données
    processed_df = process_raw_data(raw_df)
    
    print("\n" + "=" * 60)
    print("📊 APERÇU DES FEATURES")
    print("=" * 60)
    
    engineer = FeatureEngineer(raw_df)
    feature_cols = engineer.get_feature_columns()
    
    print(f"\n📈 Nombre de features: {len(feature_cols)}")
    print(f"\n📋 Exemples de features:")
    for i, col in enumerate(feature_cols[:30]):
        print(f"   {i+1}. {col}")
    print(f"   ... et {len(feature_cols) - 30} autres")
