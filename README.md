# 🥇 Gold Trading AI - Système Intelligent d'Aide à la Décision

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/ML-XGBoost-green.svg" alt="XGBoost">
  <img src="https://img.shields.io/badge/UI-Streamlit-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Data-25%2B%20Years-gold.svg" alt="Data">
</p>

## 📋 Description

Système end-to-end de trading algorithmique dédié à l'or (XAU/USD). Ce projet combine **Data Engineering**, **Analyse Quantitative**, **Machine Learning** et **Développement Full-Stack** pour fournir des recommandations de trading en temps réel.

### ✨ Caractéristiques Principales

- 📊 **Dataset Massif** : 25+ ans de données historiques (2000-présent)
- 🤖 **150+ Features** : Indicateurs techniques et macroéconomiques
- 🎯 **Modèle XGBoost** : Classification binaire optimisée
- 🖥️ **Interface Pro** : Dashboard Streamlit interactif et élégant
- ⚡ **Temps Réel** : Données de marché live via yFinance

---

## 🏗️ Architecture

```
Gold/
│
├── 📄 config.py                 # Configuration globale
├── 📄 data_collector.py         # Collecte de données (16 actifs)
├── 📄 feature_engineering.py    # 150+ features techniques
├── 📄 model_training.py         # Entraînement XGBoost
├── 📄 app.py                    # Interface Streamlit Pro
├── 📄 run_pipeline.py           # Script d'exécution
├── 📄 requirements.txt          # Dépendances
│
├── 📁 data/                     # Données (auto-générées)
│   ├── raw_market_data.csv      # ~5-10 MB
│   └── processed_features.csv   # ~20-50 MB
│
└── 📁 models/                   # Modèles sauvegardés
    ├── xgboost_gold_predictor.joblib
    └── feature_scaler.joblib
```

---

## 📊 Sources de Données

### Actifs Suivis (16)

| Catégorie | Actifs | Symboles |
|-----------|--------|----------|
| **Or** | Gold Futures, GLD ETF, Gold Miners | GC=F, GLD, GDX |
| **Dollar** | Dollar Index | DX-Y.NYB |
| **Taux** | US 10Y, US 2Y | ^TNX, ^IRX |
| **Volatilité** | VIX | ^VIX |
| **Indices** | S&P 500, NASDAQ | ^GSPC, ^IXIC |
| **Métaux** | Silver, Platinum, Copper | SI=F, PL=F, HG=F |
| **Énergie** | Crude Oil | CL=F |
| **Forex** | EUR/USD, USD/JPY | EURUSD=X, JPY=X |
| **Crypto** | Bitcoin | BTC-USD |

### Période de Données

- **Début** : 1er Janvier 2000
- **Fin** : Aujourd'hui
- **Durée** : 25+ années
- **Lignes** : ~6,000+ jours de trading

---

## 🔧 Features Créées (150+)

### 📈 Prix & Returns
- Returns multi-périodes (1d, 2d, 3d, 5d, 10d, 15d, 20d, 30d, 60d, 90d)
- Log returns, Volatilité historique (annualisée)
- True Range, ATR (7, 14, 21)
- Gap d'ouverture, Distance aux extremes

### 📊 Moyennes Mobiles
- SMA/EMA (5, 8, 10, 13, 20, 21, 34, 50, 55, 89, 100, 144, 200, 233)
- Distance au prix, Pente des MAs
- Croisements (Golden Cross, Death Cross)

### ⚡ Momentum
- RSI (5, 7, 9, 14, 21, 28) avec zones extrêmes
- MACD standard et alternatif
- Stochastique (5, 9, 14, 21), Stochastic RSI
- Williams %R, CCI, ADX, Ultimate Oscillator
- Rate of Change multi-périodes

### 📉 Volatilité
- Bandes de Bollinger (périodes: 10, 20, 50 × std: 1.5, 2, 2.5)
- Position dans les bandes, BB Squeeze
- Breakout signals

### 📊 Volume
- Volume ratio, Volume spike detection
- OBV, VPT, MFI, A/D Line

### 🌍 Macroéconomique
- Ratios: Gold/DXY, Gold/Silver, Gold/Platinum, Gold/Copper
- Corrélations roulantes (20d, 60d)
- VIX levels, Yield Curve, Oil trends
- S&P 500, NASDAQ, Bitcoin correlations

### 📅 Temporel
- Encodage cyclique (jour, mois)
- Flags: lundi, vendredi, début/fin mois, trimestre
- Saisonnalité or (Q1, Q3, Q4)

### 🎯 Patterns
- Jours consécutifs up/down
- Candlestick patterns (Doji, Hammer, Engulfing)
- Higher Highs, Lower Lows

---

## 🤖 Modèle Machine Learning

### Configuration XGBoost

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

### Métriques

| Métrique | Description |
|----------|-------------|
| **Accuracy** | Précision globale |
| **Precision** | Vrais positifs / Prédictions positives |
| **Recall** | Vrais positifs / Réels positifs |
| **F1-Score** | Moyenne harmonique Precision/Recall |
| **ROC-AUC** | Aire sous la courbe ROC |

---

## 🖥️ Interface Streamlit

### Sections du Dashboard

1. **💰 Données Temps Réel**
   - Prix XAU/USD avec variation
   - DXY, US 10Y, VIX
   - Silver, S&P 500, Oil, Bitcoin

2. **🎯 Signal de Trading**
   - Recommandation (Achat Fort/Achat/Neutre/Vente/Vente Forte)
   - Probabilités avec barres de progression
   - Indicateurs techniques (RSI, MACD, Bollinger)

3. **📈 Graphique Technique**
   - Chandelier japonais
   - SMA 20/50/200
   - Bandes de Bollinger
   - RSI, MACD, Volume

4. **🔗 Corrélations**
   - Performance comparative normalisée
   - Guide des corrélations

5. **🔍 Feature Importance**
   - Top 20 facteurs de décision
   - Interprétation

---

## 🚀 Installation & Utilisation

### Prérequis

- Python 3.9+
- pip

### Installation

```bash
cd "Gold"

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution

```bash
# Option 1: Pipeline complet (recommandé pour la première fois)
python run_pipeline.py

# Option 2: Étapes individuelles
python data_collector.py      # Collecte des données
python feature_engineering.py # Création des features
python model_training.py      # Entraînement du modèle

# Option 3: Lancer l'application (après entraînement)
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

---

## 📈 Seuils de Décision

| Niveau | Probabilité | Signal |
|--------|-------------|--------|
| **Fort** | ≥ 70% | 🟢 Achat Fort / 🔴 Vente Forte |
| **Modéré** | 55-70% | 📈 Achat / 📉 Vente |
| **Faible** | < 55% | 🟡 Signal Faible |

---

## 🎯 Compétences Démontrées

| Domaine | Compétences |
|---------|-------------|
| **Data Engineering** | ETL, APIs financières, pipelines de données |
| **Analyse Quantitative** | Indicateurs techniques, corrélations macro, statistiques |
| **Machine Learning** | Feature engineering avancé, XGBoost, validation temporelle |
| **Full-Stack Data** | Streamlit, Plotly, UI/UX, visualisations interactives |

---

## ⚠️ Avertissement

**Ce système est développé à des fins éducatives et de démonstration uniquement.**

Les prédictions fournies ne constituent en aucun cas des conseils financiers. Le trading comporte des risques significatifs de perte en capital. Toute décision d'investissement doit être prise après consultation d'un conseiller financier qualifié.

---

## 📝 Améliorations Futures

- [ ] Données fondamentales (inflation, emploi, PIB)
- [ ] Modèles ensemble (Random Forest + LSTM + Transformer)
- [ ] Backtesting avec calcul du Sharpe ratio
- [ ] Alertes email/SMS
- [ ] Déploiement cloud (AWS/GCP/Azure)
- [ ] API REST pour intégration externe
- [ ] Analyse de sentiment (news, Twitter)

---

## 📄 Licence

Ce projet est sous licence MIT.

---

<p align="center">
  <strong>Développé avec ❤️ et Python</strong><br>
  <em>Gold Trading AI v2.0</em>
</p>
