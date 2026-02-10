"""
Module de Collecte de Données Financières - Version Étendue
============================================================
Récupération massive des données de marché via APIs (yfinance)
Dataset: 2000-présent pour maximiser la précision du modèle
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
import os
import sys
import time

# Ajouter le répertoire parent au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SYMBOLS, START_DATE, END_DATE, DATA_DIR, RAW_DATA_PATH

warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketDataCollector:
    """
    Collecteur de données de marché multidimensionnel.
    Fusionne les données XAU/USD avec les indicateurs macro.
    Optimisé pour les grands datasets (25+ ans de données).
    """
    
    def __init__(self, symbols: dict = None, start_date: str = None, end_date: str = None):
        self.symbols = symbols or SYMBOLS
        self.start_date = start_date or START_DATE
        self.end_date = end_date or END_DATE
        self.data = {}
        
    def fetch_single_asset(self, symbol: str, name: str, retries: int = 3) -> pd.DataFrame:
        """
        Récupère les données OHLCV pour un actif unique avec retry.
        
        Args:
            symbol: Symbole Yahoo Finance
            name: Nom descriptif de l'actif
            retries: Nombre de tentatives
            
        Returns:
            DataFrame avec les données OHLCV
        """
        for attempt in range(retries):
            try:
                logger.info(f"📥 Téléchargement {name} ({symbol})... [Tentative {attempt + 1}]")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                
                if df.empty:
                    logger.warning(f"⚠️ Aucune donnée pour {symbol}")
                    return None
                    
                # Renommer les colonnes avec le préfixe de l'actif
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.columns = [f'{name}_{col.lower()}' for col in df.columns]
                
                # S'assurer que l'index est en datetime sans timezone
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = 'Date'
                
                # Supprimer les doublons d'index
                df = df[~df.index.duplicated(keep='first')]
                
                logger.info(f"✅ {name}: {len(df):,} lignes ({df.index.min().strftime('%Y-%m-%d')} → {df.index.max().strftime('%Y-%m-%d')})")
                return df
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur {symbol}: {e}")
                if attempt < retries - 1:
                    time.sleep(2)
                    
        return None
    
    def fetch_all_data(self) -> pd.DataFrame:
        """
        Récupère toutes les données de marché et les fusionne.
        
        Returns:
            DataFrame fusionné avec tous les actifs
        """
        logger.info("=" * 60)
        logger.info("🚀 DÉBUT DE LA COLLECTE DES DONNÉES MASSIVES")
        logger.info(f"📅 Période: {self.start_date} → {self.end_date}")
        logger.info("=" * 60)
        
        dataframes = {}
        
        for name, symbol in self.symbols.items():
            df = self.fetch_single_asset(symbol, name)
            if df is not None and len(df) > 100:  # Au moins 100 lignes
                dataframes[name] = df
            time.sleep(0.5)  # Éviter le rate limiting
                
        if not dataframes:
            raise ValueError("Aucune donnée récupérée!")
        
        # Gold doit être présent
        if 'gold' not in dataframes:
            raise ValueError("Les données Gold sont obligatoires!")
        
        # Fusionner tous les DataFrames sur l'index (date)
        logger.info("\n🔄 Fusion des données...")
        merged_df = dataframes['gold'].copy()
        
        for name, df in dataframes.items():
            if name != 'gold':
                merged_df = merged_df.join(df, how='left')
        
        # Interpoler les valeurs manquantes (jours de trading différents)
        logger.info("🔧 Interpolation des valeurs manquantes...")
        merged_df = merged_df.ffill(limit=5).bfill(limit=5)
        
        # Supprimer les lignes avec trop de NaN (> 30%)
        thresh = int(len(merged_df.columns) * 0.7)
        initial_len = len(merged_df)
        merged_df = merged_df.dropna(thresh=thresh)
        
        # Statistiques finales
        logger.info("\n" + "=" * 60)
        logger.info("📊 STATISTIQUES DU DATASET")
        logger.info("=" * 60)
        logger.info(f"📈 Lignes totales: {len(merged_df):,}")
        logger.info(f"📊 Colonnes: {len(merged_df.columns)}")
        logger.info(f"📅 Période: {merged_df.index.min().strftime('%Y-%m-%d')} → {merged_df.index.max().strftime('%Y-%m-%d')}")
        logger.info(f"📆 Durée: {(merged_df.index.max() - merged_df.index.min()).days:,} jours")
        logger.info(f"🗑️ Lignes supprimées: {initial_len - len(merged_df):,}")
        
        # Afficher les actifs disponibles
        logger.info("\n📋 ACTIFS RÉCUPÉRÉS:")
        for name in dataframes.keys():
            col = f'{name}_close'
            if col in merged_df.columns:
                non_null = merged_df[col].notna().sum()
                logger.info(f"   • {name.upper()}: {non_null:,} valeurs")
        
        return merged_df
    
    def fetch_realtime_data(self) -> dict:
        """
        Récupère les dernières données en temps réel.
        
        Returns:
            Dictionnaire avec les dernières valeurs
        """
        realtime_data = {}
        
        for name, symbol in self.symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                
                # Récupérer l'historique récent pour plus de fiabilité
                hist = ticker.history(period='5d')
                
                if not hist.empty:
                    latest = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else latest
                    
                    price = latest['Close']
                    prev_close = prev['Close']
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0
                    
                    realtime_data[name] = {
                        'price': price,
                        'previous_close': prev_close,
                        'day_high': latest['High'],
                        'day_low': latest['Low'],
                        'open': latest['Open'],
                        'volume': latest['Volume'],
                        'change_pct': round(change_pct, 2),
                        'change_abs': round(price - prev_close, 2),
                        'timestamp': hist.index[-1]
                    }
                else:
                    realtime_data[name] = None
                    
            except Exception as e:
                logger.warning(f"⚠️ Temps réel {name}: {e}")
                realtime_data[name] = None
                
        return realtime_data
    
    def get_latest_ohlc(self, period: str = '3mo') -> pd.DataFrame:
        """
        Récupère les dernières données OHLC pour la prédiction.
        
        Args:
            period: Période de données (ex: '5d', '1mo', '3mo')
            
        Returns:
            DataFrame avec les dernières données
        """
        dataframes = {}
        
        for name, symbol in self.symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period)
                
                if not df.empty:
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df.columns = [f'{name}_{col.lower()}' for col in df.columns]
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df = df[~df.index.duplicated(keep='first')]
                    dataframes[name] = df
                    
            except Exception as e:
                logger.warning(f"⚠️ OHLC {name}: {e}")
                
        # Fusionner
        if 'gold' not in dataframes:
            return None
            
        merged_df = dataframes['gold'].copy()
        for name, df in dataframes.items():
            if name != 'gold':
                merged_df = merged_df.join(df, how='left')
                
        merged_df = merged_df.ffill().bfill()
            
        return merged_df
    
    def get_intraday_data(self, symbol: str = 'GC=F', interval: str = '1h', period: str = '5d') -> pd.DataFrame:
        """
        Récupère les données intraday.
        
        Args:
            symbol: Symbole
            interval: Intervalle (1m, 5m, 15m, 1h, etc.)
            period: Période
            
        Returns:
            DataFrame intraday
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            logger.warning(f"⚠️ Intraday: {e}")
            return None
    
    def save_data(self, df: pd.DataFrame, filepath: str = None):
        """
        Sauvegarde les données dans un fichier CSV.
        
        Args:
            df: DataFrame à sauvegarder
            filepath: Chemin du fichier (optionnel)
        """
        filepath = filepath or RAW_DATA_PATH
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath)
        
        # Taille du fichier
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"💾 Données sauvegardées: {filepath} ({size_mb:.2f} MB)")
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.
        
        Args:
            filepath: Chemin du fichier (optionnel)
            
        Returns:
            DataFrame avec les données
        """
        filepath = filepath or RAW_DATA_PATH
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"📂 Données chargées: {len(df):,} lignes")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Retourne un résumé des données.
        
        Args:
            df: DataFrame
            
        Returns:
            Dictionnaire avec les statistiques
        """
        gold_col = 'gold_close'
        
        summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'date_start': df.index.min().strftime('%Y-%m-%d'),
            'date_end': df.index.max().strftime('%Y-%m-%d'),
            'years_covered': round((df.index.max() - df.index.min()).days / 365.25, 1),
            'missing_pct': round(df.isna().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        }
        
        if gold_col in df.columns:
            summary['gold_min'] = df[gold_col].min()
            summary['gold_max'] = df[gold_col].max()
            summary['gold_mean'] = df[gold_col].mean()
            summary['gold_current'] = df[gold_col].iloc[-1]
            
        return summary


def collect_and_save_data():
    """
    Fonction principale pour collecter et sauvegarder les données.
    """
    collector = MarketDataCollector()
    df = collector.fetch_all_data()
    collector.save_data(df)
    
    # Afficher le résumé
    summary = collector.get_data_summary(df)
    logger.info("\n📊 RÉSUMÉ:")
    for key, value in summary.items():
        logger.info(f"   {key}: {value}")
    
    return df


if __name__ == "__main__":
    # Exécution directe du module
    df = collect_and_save_data()
    print("\n" + "=" * 60)
    print("APERÇU DES DONNÉES")
    print("=" * 60)
    print(df.tail(10))
    print("\nColonnes disponibles:")
    print(df.columns.tolist())
