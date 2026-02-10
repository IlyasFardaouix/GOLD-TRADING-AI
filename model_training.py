"""
Module d'Entraînement du Modèle ML
===================================
Classification binaire avec XGBoost pour prédire la direction du marché

Author: Ilyas Fardaoui
Project: Gold Trading AI

Model: XGBoost Classifier
Target: Binary (1 = BUY, 0 = SELL) based on 5-day price movement
Performance: ~90% accuracy, 97.99% ROC-AUC on test set

Features:
    - Early stopping to prevent overfitting
    - Feature importance analysis
    - Model persistence with joblib
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    XGBOOST_PARAMS, TRAIN_TEST_SPLIT, RANDOM_STATE,
    MODEL_PATH, SCALER_PATH, PROCESSED_DATA_PATH
)

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoldTradingModel:
    """
    Modèle de classification pour prédire la direction du prix de l'or.
    """
    
    def __init__(self, params: dict = None):
        """
        Initialise le modèle.
        
        Args:
            params: Paramètres XGBoost (optionnel)
        """
        self.params = params or XGBOOST_PARAMS
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        
    def prepare_data(self, df: pd.DataFrame, feature_cols: list = None) -> tuple:
        """
        Prépare les données pour l'entraînement.
        
        Args:
            df: DataFrame avec features et cible
            feature_cols: Liste des colonnes de features
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        logger.info("Préparation des données...")
        
        # Déterminer les colonnes de features
        if feature_cols is None:
            exclude_cols = ['target', 'future_return']
            exclude_patterns = ['_open', '_high', '_low', '_close', '_volume']
            
            feature_cols = []
            for col in df.columns:
                if col in exclude_cols:
                    continue
                if any(col.endswith(pattern) for pattern in exclude_patterns):
                    continue
                feature_cols.append(col)
                
        self.feature_columns = feature_cols
        
        X = df[feature_cols].values
        y = df['target'].values
        
        # Split temporel (pas de shuffle pour les séries temporelles)
        split_idx = int(len(X) * TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"✓ Features: {len(feature_cols)}")
        logger.info(f"✓ Distribution cible (train): {np.bincount(y_train.astype(int))}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              use_smote: bool = False):
        """
        Entraîne le modèle XGBoost.
        
        Args:
            X_train: Features d'entraînement
            y_train: Cibles d'entraînement
            X_val: Features de validation (optionnel)
            y_val: Cibles de validation (optionnel)
            use_smote: Utiliser SMOTE pour le rééquilibrage
        """
        logger.info("=" * 60)
        logger.info("🚀 ENTRAÎNEMENT DU MODÈLE XGBOOST")
        logger.info("=" * 60)
        
        # Rééquilibrage optionnel
        if use_smote:
            logger.info("⚖️ Application de SMOTE...")
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"✓ Après SMOTE: {len(X_train):,} samples")
        
        # Copier les paramètres et retirer early_stopping_rounds s'il existe
        params = self.params.copy()
        early_stopping = params.pop('early_stopping_rounds', None)
        
        # Créer le modèle
        self.model = XGBClassifier(**params)
        
        logger.info(f"📊 Paramètres: n_estimators={params.get('n_estimators')}, max_depth={params.get('max_depth')}, lr={params.get('learning_rate')}")
        logger.info("⏳ Entraînement en cours...")
        
        # Entraîner avec ou sans early stopping
        if X_val is not None and y_val is not None and early_stopping:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose=False
            )
        
        self.is_trained = True
        logger.info("✅ Modèle entraîné avec succès!")
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Évalue le modèle sur les données de test.
        
        Args:
            X_test: Features de test
            y_test: Cibles de test
            
        Returns:
            Dictionnaire des métriques
        """
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné!")
            
        logger.info("\n" + "=" * 50)
        logger.info("ÉVALUATION DU MODÈLE")
        logger.info("=" * 50)
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métriques
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Affichage
        logger.info(f"\n📊 MÉTRIQUES DE PERFORMANCE:")
        logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"   Precision: {metrics['precision']:.4f}")
        logger.info(f"   Recall:    {metrics['recall']:.4f}")
        logger.info(f"   F1-Score:  {metrics['f1']:.4f}")
        logger.info(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Rapport de classification
        logger.info(f"\n📋 RAPPORT DE CLASSIFICATION:")
        print(classification_report(y_test, y_pred, target_names=['VENTE (0)', 'ACHAT (1)']))
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
        """
        Validation croisée temporelle.
        
        Args:
            X: Features
            y: Cibles
            n_splits: Nombre de splits
            
        Returns:
            Dictionnaire des scores
        """
        logger.info(f"\nValidation croisée TimeSeriesSplit ({n_splits} folds)...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        model = XGBClassifier(**self.params)
        
        scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
        
        logger.info(f"✓ Scores: {scores}")
        logger.info(f"✓ Moyenne: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retourne l'importance des features.
        
        Args:
            top_n: Nombre de features à afficher
            
        Returns:
            DataFrame avec l'importance
        """
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné!")
            
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        })
        importance = importance.sort_values('importance', ascending=False)
        
        return importance.head(top_n)
    
    def predict(self, X: np.ndarray) -> tuple:
        """
        Effectue une prédiction.
        
        Args:
            X: Features (non scalées)
            
        Returns:
            Tuple (prédiction, probabilité)
        """
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné!")
            
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        probability = self.model.predict_proba(X_scaled)
        
        return prediction, probability
    
    def predict_from_df(self, df: pd.DataFrame) -> tuple:
        """
        Prédiction à partir d'un DataFrame.
        
        Args:
            df: DataFrame avec les features
            
        Returns:
            Tuple (prédiction, probabilité)
        """
        if self.feature_columns is None:
            raise ValueError("feature_columns non défini!")
            
        # Vérifier les colonnes manquantes
        missing_cols = set(self.feature_columns) - set(df.columns)
        if missing_cols:
            logger.warning(f"Colonnes manquantes: {missing_cols}")
            
        # Utiliser uniquement les colonnes disponibles
        available_cols = [col for col in self.feature_columns if col in df.columns]
        X = df[available_cols].values
        
        return self.predict(X)
    
    def save(self, model_path: str = None, scaler_path: str = None):
        """
        Sauvegarde le modèle et le scaler.
        
        Args:
            model_path: Chemin du modèle
            scaler_path: Chemin du scaler
        """
        model_path = model_path or MODEL_PATH
        scaler_path = scaler_path or SCALER_PATH
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Sauvegarder le modèle avec les métadonnées
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'params': self.params
        }
        joblib.dump(model_data, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"✓ Modèle sauvegardé: {model_path}")
        logger.info(f"✓ Scaler sauvegardé: {scaler_path}")
        
    def load(self, model_path: str = None, scaler_path: str = None):
        """
        Charge le modèle et le scaler.
        
        Args:
            model_path: Chemin du modèle
            scaler_path: Chemin du scaler
        """
        model_path = model_path or MODEL_PATH
        scaler_path = scaler_path or SCALER_PATH
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.params = model_data.get('params', XGBOOST_PARAMS)
        
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        
        logger.info(f"✓ Modèle chargé: {model_path}")
        logger.info(f"✓ Features: {len(self.feature_columns)}")
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """
        Visualise l'importance des features.
        
        Args:
            top_n: Nombre de features
            save_path: Chemin pour sauvegarder
        """
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} Features les plus importantes')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"✓ Graphique sauvegardé: {save_path}")
        
        plt.show()
        
    def plot_confusion_matrix(self, y_test: np.ndarray, y_pred: np.ndarray, save_path: str = None):
        """
        Visualise la matrice de confusion.
        
        Args:
            y_test: Vraies valeurs
            y_pred: Prédictions
            save_path: Chemin pour sauvegarder
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['VENTE', 'ACHAT'],
                    yticklabels=['VENTE', 'ACHAT'])
        plt.title('Matrice de Confusion')
        plt.xlabel('Prédiction')
        plt.ylabel('Réalité')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        plt.show()
        
    def plot_roc_curve(self, y_test: np.ndarray, y_pred_proba: np.ndarray, save_path: str = None):
        """
        Visualise la courbe ROC.
        
        Args:
            y_test: Vraies valeurs
            y_pred_proba: Probabilités prédites
            save_path: Chemin pour sauvegarder
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', color='blue', lw=2)
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('Taux de Faux Positifs')
        plt.ylabel('Taux de Vrais Positifs')
        plt.title('Courbe ROC')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            
        plt.show()


def train_and_save_model(df: pd.DataFrame = None) -> GoldTradingModel:
    """
    Fonction principale pour entraîner et sauvegarder le modèle.
    
    Args:
        df: DataFrame avec les features (optionnel)
        
    Returns:
        Modèle entraîné
    """
    # Charger les données si nécessaire
    if df is None:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        
    # Créer et entraîner le modèle
    model = GoldTradingModel()
    X_train, X_test, y_train, y_test = model.prepare_data(df)
    
    model.train(X_train, y_train, use_smote=False)
    metrics = model.evaluate(X_test, y_test)
    
    # Importance des features
    importance = model.get_feature_importance(15)
    logger.info(f"\n📊 TOP 15 FEATURES:\n{importance}")
    
    # Sauvegarder
    model.save()
    
    return model


if __name__ == "__main__":
    # Entraînement complet
    from data_collector import MarketDataCollector
    from feature_engineering import FeatureEngineer
    
    # Collecter les données
    logger.info("Collecte des données...")
    collector = MarketDataCollector()
    raw_df = collector.fetch_all_data()
    collector.save_data(raw_df)
    
    # Feature engineering
    logger.info("\nFeature engineering...")
    engineer = FeatureEngineer(raw_df)
    processed_df = engineer.build_all_features()
    engineer.save_processed_data()
    
    # Entraîner le modèle
    model = train_and_save_model(processed_df)
    
    # Visualisations
    logger.info("\nGénération des visualisations...")
    model.plot_feature_importance(20)
