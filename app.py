"""
🥇 Gold Trading AI - Interface Professionnelle
================================================
Dashboard interactif avancé pour le trading XAU/USD
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import sys

# Ajouter le répertoire au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    STREAMLIT_CONFIG, REFRESH_INTERVAL, SYMBOLS,
    MODEL_PATH, SCALER_PATH, PROCESSED_DATA_PATH,
    CONFIDENCE_THRESHOLD_HIGH, CONFIDENCE_THRESHOLD_MEDIUM,
    START_DATE
)
from data_collector import MarketDataCollector
from feature_engineering import FeatureEngineer
from model_training import GoldTradingModel

# =====================================
# CONFIGURATION STREAMLIT
# =====================================

st.set_page_config(
    page_title="🥇 Gold Trading AI - Dashboard Pro",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PERSONNALISÉ AVANCÉ
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFD700, #FFA500, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1.5rem 0;
        text-shadow: 0 0 30px rgba(255, 215, 0, 0.3);
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #252538 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(255, 215, 0, 0.15);
    }
    
    /* Signal Boxes */
    .signal-buy-strong {
        background: linear-gradient(135deg, #0d5d2e 0%, #198754 50%, #28a745 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 10px 40px rgba(40, 167, 69, 0.4);
        border: 2px solid rgba(40, 167, 69, 0.5);
        animation: pulse-green 2s infinite;
    }
    
    .signal-sell-strong {
        background: linear-gradient(135deg, #6d1a21 0%, #dc3545 50%, #e74c3c 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 10px 40px rgba(220, 53, 69, 0.4);
        border: 2px solid rgba(220, 53, 69, 0.5);
        animation: pulse-red 2s infinite;
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, #5c4d00 0%, #ffc107 50%, #ffca2c 100%);
        color: #1a1a2e;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 2.2rem;
        font-weight: bold;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 10px 40px rgba(255, 193, 7, 0.3);
        border: 2px solid rgba(255, 193, 7, 0.5);
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 10px 40px rgba(40, 167, 69, 0.4); }
        50% { box-shadow: 0 10px 60px rgba(40, 167, 69, 0.6); }
    }
    
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 10px 40px rgba(220, 53, 69, 0.4); }
        50% { box-shadow: 0 10px 60px rgba(220, 53, 69, 0.6); }
    }
    
    /* Price Display */
    .price-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        color: #FFD700;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.5);
    }
    
    .price-change-positive {
        color: #28a745;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
    }
    
    .price-change-negative {
        color: #dc3545;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
    }
    
    /* Info Boxes */
    .info-box {
        background: rgba(30, 30, 47, 0.8);
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 4px solid #FFD700;
        margin: 0.5rem 0;
    }
    
    .info-box-blue {
        border-left-color: #3498db;
    }
    
    .info-box-green {
        border-left-color: #28a745;
    }
    
    .info-box-red {
        border-left-color: #dc3545;
    }
    
    /* Stats Grid */
    .stats-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFD700;
    }
    
    .stats-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress Bars */
    .confidence-bar {
        background: linear-gradient(90deg, #1e1e2f, #252538);
        border-radius: 10px;
        height: 30px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill-buy {
        background: linear-gradient(90deg, #28a745, #20c997);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .confidence-fill-sell {
        background: linear-gradient(90deg, #dc3545, #e74c3c);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Tables */
    .dataframe {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-family: 'Inter', sans-serif;
        border-top: 1px solid rgba(255, 215, 0, 0.1);
        margin-top: 3rem;
    }
    
    /* Dividers */
    .gold-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
        margin: 2rem 0;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.5rem;
        font-weight: 600;
        color: #fff;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255, 215, 0, 0.3);
    }
    
    /* Indicator Pills */
    .indicator-bullish {
        display: inline-block;
        background: rgba(40, 167, 69, 0.2);
        color: #28a745;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(40, 167, 69, 0.3);
    }
    
    .indicator-bearish {
        display: inline-block;
        background: rgba(220, 53, 69, 0.2);
        color: #dc3545;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(220, 53, 69, 0.3);
    }
    
    .indicator-neutral {
        display: inline-block;
        background: rgba(255, 193, 7, 0.2);
        color: #ffc107;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# =====================================
# FONCTIONS UTILITAIRES
# =====================================

@st.cache_resource
def load_model():
    """Charge le modèle ML."""
    model = GoldTradingModel()
    try:
        model.load()
        return model, True
    except Exception as e:
        return None, False


@st.cache_data(ttl=60)
def fetch_realtime_data():
    """Récupère les données en temps réel."""
    collector = MarketDataCollector()
    return collector.fetch_realtime_data()


@st.cache_data(ttl=300)
def fetch_latest_ohlc(period='3mo'):
    """Récupère les dernières données OHLC."""
    collector = MarketDataCollector()
    return collector.get_latest_ohlc(period)


@st.cache_data(ttl=3600)
def get_dataset_info():
    """Récupère les informations sur le dataset."""
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
        return {
            'rows': len(df),
            'features': len(df.columns),
            'start': df.index.min().strftime('%Y-%m-%d'),
            'end': df.index.max().strftime('%Y-%m-%d'),
            'years': round((df.index.max() - df.index.min()).days / 365.25, 1)
        }
    except:
        return None


def create_main_chart(df: pd.DataFrame):
    """Crée le graphique principal avec tous les indicateurs."""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=('📈 Prix XAU/USD', '📊 RSI (14)', '📉 MACD', '📊 Volume')
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['gold_open'],
            high=df['gold_high'],
            low=df['gold_low'],
            close=df['gold_close'],
            name='XAU/USD',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        ),
        row=1, col=1
    )
    
    # Moyennes mobiles
    if 'sma_20' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', 
                      line=dict(color='#FFD700', width=1.5)),
            row=1, col=1
        )
    if 'sma_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', 
                      line=dict(color='#00bfff', width=1.5)),
            row=1, col=1
        )
    if 'sma_200' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma_200'], name='SMA 200', 
                      line=dict(color='#ff69b4', width=1.5)),
            row=1, col=1
        )
    
    # Bandes de Bollinger
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_upper'], name='BB Upper',
                      line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'),
                      showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['bb_lower'], name='BB Lower',
                      line=dict(color='rgba(150,150,150,0.5)', width=1, dash='dot'),
                      fill='tonexty', fillcolor='rgba(150,150,150,0.1)',
                      showlegend=False),
            row=1, col=1
        )
    
    # RSI
    if 'rsi_14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['rsi_14'], name='RSI 14',
                      line=dict(color='#9b59b6', width=1.5)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="#ff4444", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00ff88", line_width=1, row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", line_width=1, row=2, col=1)
        
        # Zone de surachat/survente
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,68,68,0.1)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,255,136,0.1)", line_width=0, row=2, col=1)
    
    # MACD
    if 'macd_line' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_line'], name='MACD',
                      line=dict(color='#3498db', width=1.5)),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['macd_signal'], name='Signal',
                      line=dict(color='#e74c3c', width=1.5)),
            row=3, col=1
        )
        
        # Histogramme MACD
        colors = ['#00ff88' if val >= 0 else '#ff4444' for val in df['macd_histogram']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['macd_histogram'], name='Histogram',
                  marker_color=colors, opacity=0.7),
            row=3, col=1
        )
    
    # Volume
    if 'gold_volume' in df.columns:
        colors = ['#00ff88' if df['gold_close'].iloc[i] >= df['gold_open'].iloc[i] 
                  else '#ff4444' for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['gold_volume'], name='Volume',
                  marker_color=colors, opacity=0.7),
            row=4, col=1
        )
    
    # Layout
    fig.update_layout(
        template='plotly_dark',
        height=900,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        xaxis_rangeslider_visible=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(family="Inter, sans-serif", color="#fff"),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    # Update axes
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', showgrid=True)
    
    return fig


def create_correlation_chart(df: pd.DataFrame):
    """Crée le graphique de corrélation."""
    
    # Normaliser les prix pour comparaison
    normalized = pd.DataFrame()
    
    if 'gold_close' in df.columns:
        normalized['Gold'] = df['gold_close'] / df['gold_close'].iloc[0] * 100
    if 'dxy_close' in df.columns:
        normalized['DXY'] = df['dxy_close'] / df['dxy_close'].iloc[0] * 100
    if 'vix_close' in df.columns:
        normalized['VIX'] = df['vix_close'] / df['vix_close'].iloc[0] * 100
    if 'sp500_close' in df.columns:
        normalized['S&P 500'] = df['sp500_close'] / df['sp500_close'].iloc[0] * 100
        
    fig = go.Figure()
    
    colors = {'Gold': '#FFD700', 'DXY': '#28a745', 'VIX': '#9b59b6', 'S&P 500': '#3498db'}
    
    for col in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=normalized[col],
                name=col,
                line=dict(color=colors.get(col, '#fff'), width=2)
            )
        )
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        title='📊 Performance Comparative (Base 100)',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def get_signal_html(prediction: int, probability: float) -> tuple:
    """Génère le HTML pour le signal."""
    prob_buy = probability[1] * 100 if prediction == 1 else probability[0] * 100
    
    if prediction == 1:
        if prob_buy >= CONFIDENCE_THRESHOLD_HIGH * 100:
            return "🚀 SIGNAL: ACHAT FORT", "signal-buy-strong"
        elif prob_buy >= CONFIDENCE_THRESHOLD_MEDIUM * 100:
            return "📈 SIGNAL: ACHAT", "signal-buy-strong"
        else:
            return "↗️ SIGNAL: ACHAT FAIBLE", "signal-neutral"
    else:
        prob_sell = probability[0] * 100
        if prob_sell >= CONFIDENCE_THRESHOLD_HIGH * 100:
            return "📉 SIGNAL: VENTE FORTE", "signal-sell-strong"
        elif prob_sell >= CONFIDENCE_THRESHOLD_MEDIUM * 100:
            return "⬇️ SIGNAL: VENTE", "signal-sell-strong"
        else:
            return "↘️ SIGNAL: VENTE FAIBLE", "signal-neutral"


def make_prediction(model: GoldTradingModel, df: pd.DataFrame) -> tuple:
    """Effectue une prédiction avec le modèle."""
    try:
        engineer = FeatureEngineer(df)
        processed_df = engineer.build_all_features()
        
        if processed_df.empty:
            return None, None, None, "Données insuffisantes"
        
        latest = processed_df.iloc[[-1]]
        prediction, probability = model.predict_from_df(latest)
        
        # Récupérer les indicateurs clés
        indicators = {}
        if 'rsi_14' in processed_df.columns:
            indicators['rsi_14'] = processed_df['rsi_14'].iloc[-1]
        if 'macd_histogram' in processed_df.columns:
            indicators['macd'] = processed_df['macd_histogram'].iloc[-1]
        if 'bb_position' in processed_df.columns:
            indicators['bb_position'] = processed_df['bb_position'].iloc[-1]
            
        return prediction[0], probability[0], indicators, None
        
    except Exception as e:
        return None, None, None, str(e)


# =====================================
# INTERFACE PRINCIPALE
# =====================================

def main():
    # En-tête principal
    st.markdown('<h1 class="main-header">🥇 GOLD TRADING AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Système Intelligent d\'Aide à la Décision pour le Trading XAU/USD | Powered by Machine Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        st.markdown("---")
        
        # Statut du modèle
        model, model_loaded = load_model()
        if model_loaded:
            st.success("✅ Modèle chargé")
        else:
            st.error("❌ Modèle non trouvé")
            
        # Info dataset
        dataset_info = get_dataset_info()
        if dataset_info:
            st.markdown("### 📊 Dataset")
            st.markdown(f"""
            - **Lignes:** {dataset_info['rows']:,}
            - **Features:** {dataset_info['features']}
            - **Période:** {dataset_info['years']} ans
            - **De:** {dataset_info['start']}
            - **À:** {dataset_info['end']}
            """)
        
        st.markdown("---")
        
        # Paramètres
        st.markdown("### 📈 Paramètres")
        chart_period = st.selectbox(
            "Période du graphique",
            options=['1mo', '3mo', '6mo', '1y', '2y', '5y'],
            index=1
        )
        
        auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", value=False)
        
        st.markdown("---")
        
        # Actions
        st.markdown("### 🎯 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        
        with col2:
            if st.button("🎯 Train", use_container_width=True):
                with st.spinner("Entraînement..."):
                    try:
                        from model_training import train_and_save_model
                        from data_collector import collect_and_save_data
                        from feature_engineering import process_raw_data
                        
                        raw_df = collect_and_save_data()
                        processed_df = process_raw_data(raw_df)
                        train_and_save_model(processed_df)
                        
                        st.success("✅ Modèle entraîné!")
                        st.cache_resource.clear()
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Erreur: {e}")
        
        st.markdown("---")
        st.markdown("### ℹ️ Légende")
        st.markdown("""
        🟢 **Achat** - Signal haussier  
        🔴 **Vente** - Signal baissier  
        🟡 **Neutre** - Faible confiance
        
        **Confidence:**
        - ≥70% = Fort
        - 55-70% = Modéré
        - <55% = Faible
        """)
    
    # =====================================
    # SECTION 1: PRIX TEMPS RÉEL
    # =====================================
    
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">💰 Données de Marché en Temps Réel</h2>', unsafe_allow_html=True)
    
    realtime_data = fetch_realtime_data()
    
    # Grille de prix
    cols = st.columns(4)
    
    # Gold
    with cols[0]:
        if realtime_data.get('gold'):
            gold = realtime_data['gold']
            price = gold.get('price', 0)
            change = gold.get('change_pct', 0)
            change_class = "price-change-positive" if change >= 0 else "price-change-negative"
            arrow = "▲" if change >= 0 else "▼"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-label">🥇 XAU/USD</div>
                <div class="price-display">${price:,.2f}</div>
                <div class="{change_class}">{arrow} {change:+.2f}%</div>
                <div style="color:#666;font-size:0.8rem;margin-top:0.5rem;">
                    H: ${gold.get('day_high', 0):,.2f} | L: ${gold.get('day_low', 0):,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="stats-label">🥇 XAU/USD</div><div style="color:#666">N/A</div></div>', unsafe_allow_html=True)
    
    # DXY
    with cols[1]:
        if realtime_data.get('dxy'):
            dxy = realtime_data['dxy']
            price = dxy.get('price', 0)
            change = dxy.get('change_pct', 0)
            change_class = "price-change-negative" if change >= 0 else "price-change-positive"  # Inversé pour l'or
            arrow = "▲" if change >= 0 else "▼"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-label">💵 Dollar Index (DXY)</div>
                <div class="stats-value">{price:.2f}</div>
                <div class="{change_class}">{arrow} {change:+.2f}%</div>
                <div style="color:#888;font-size:0.75rem;margin-top:0.5rem;">
                    Corrélation inverse avec l'or
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="stats-label">💵 DXY</div><div style="color:#666">N/A</div></div>', unsafe_allow_html=True)
    
    # US 10Y
    with cols[2]:
        if realtime_data.get('us10y'):
            us10y = realtime_data['us10y']
            price = us10y.get('price', 0)
            change = us10y.get('change_pct', 0)
            change_class = "price-change-negative" if change >= 0 else "price-change-positive"
            arrow = "▲" if change >= 0 else "▼"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-label">📈 Taux US 10 Ans</div>
                <div class="stats-value">{price:.2f}%</div>
                <div class="{change_class}">{arrow} {change:+.2f}%</div>
                <div style="color:#888;font-size:0.75rem;margin-top:0.5rem;">
                    Impact sur les taux réels
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="stats-label">📈 US 10Y</div><div style="color:#666">N/A</div></div>', unsafe_allow_html=True)
    
    # VIX
    with cols[3]:
        if realtime_data.get('vix'):
            vix = realtime_data['vix']
            price = vix.get('price', 0)
            change = vix.get('change_pct', 0)
            
            # Couleur selon niveau VIX
            if price > 30:
                vix_color = "#dc3545"
                vix_label = "⚠️ Peur Extrême"
            elif price > 20:
                vix_color = "#ffc107"
                vix_label = "😰 Peur Modérée"
            else:
                vix_color = "#28a745"
                vix_label = "😊 Faible Volatilité"
            
            change_class = "price-change-positive" if change >= 0 else "price-change-negative"
            arrow = "▲" if change >= 0 else "▼"
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="stats-label">⚡ VIX (Indice de Peur)</div>
                <div class="stats-value" style="color:{vix_color}">{price:.2f}</div>
                <div class="{change_class}">{arrow} {change:+.2f}%</div>
                <div style="color:{vix_color};font-size:0.75rem;margin-top:0.5rem;">
                    {vix_label}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card"><div class="stats-label">⚡ VIX</div><div style="color:#666">N/A</div></div>', unsafe_allow_html=True)
    
    # Deuxième ligne de métriques
    cols2 = st.columns(4)
    
    metrics_to_show = [
        ('silver', '🥈 Argent', 'silver'),
        ('sp500', '📊 S&P 500', 'sp500'),
        ('oil', '🛢️ Pétrole', 'oil'),
        ('btc', '₿ Bitcoin', 'btc')
    ]
    
    for i, (key, label, _) in enumerate(metrics_to_show):
        with cols2[i]:
            if realtime_data.get(key):
                data = realtime_data[key]
                price = data.get('price', 0)
                change = data.get('change_pct', 0)
                change_class = "price-change-positive" if change >= 0 else "price-change-negative"
                arrow = "▲" if change >= 0 else "▼"
                
                st.markdown(f"""
                <div class="metric-card" style="padding:1rem;">
                    <div class="stats-label" style="font-size:0.8rem;">{label}</div>
                    <div class="stats-value" style="font-size:1.3rem;">${price:,.2f}</div>
                    <div class="{change_class}" style="font-size:1rem;">{arrow} {change:+.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card" style="padding:1rem;"><div class="stats-label">{label}</div><div style="color:#666">N/A</div></div>', unsafe_allow_html=True)
    
    # =====================================
    # SECTION 2: SIGNAL DE TRADING
    # =====================================
    
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🎯 Signal de Trading IA</h2>', unsafe_allow_html=True)
    
    if model_loaded:
        with st.spinner("🔮 Analyse en cours..."):
            latest_df = fetch_latest_ohlc('6mo')
            
            if latest_df is not None and not latest_df.empty:
                prediction, probability, indicators, error = make_prediction(model, latest_df)
                
                if error:
                    st.error(f"Erreur de prédiction: {error}")
                else:
                    col_signal, col_details = st.columns([1, 1])
                    
                    with col_signal:
                        signal_text, signal_class = get_signal_html(prediction, probability)
                        
                        st.markdown(f'<div class="{signal_class}">{signal_text}</div>', unsafe_allow_html=True)
                        
                        # Probabilités
                        st.markdown("### 📊 Probabilités")
                        prob_buy = probability[1] * 100
                        prob_sell = probability[0] * 100
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"""
                            <div style="margin:0.5rem 0;">
                                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                                    <span style="color:#28a745;font-weight:600;">🟢 ACHAT</span>
                                    <span style="color:#28a745;font-weight:600;">{prob_buy:.1f}%</span>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill-buy" style="width:{prob_buy}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(f"""
                            <div style="margin:0.5rem 0;">
                                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                                    <span style="color:#dc3545;font-weight:600;">🔴 VENTE</span>
                                    <span style="color:#dc3545;font-weight:600;">{prob_sell:.1f}%</span>
                                </div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill-sell" style="width:{prob_sell}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Niveau de confiance
                        confidence = max(prob_buy, prob_sell)
                        if confidence >= CONFIDENCE_THRESHOLD_HIGH * 100:
                            conf_level = "🟢 ÉLEVÉ"
                            conf_color = "#28a745"
                        elif confidence >= CONFIDENCE_THRESHOLD_MEDIUM * 100:
                            conf_level = "🟡 MODÉRÉ"
                            conf_color = "#ffc107"
                        else:
                            conf_level = "🔴 FAIBLE"
                            conf_color = "#dc3545"
                        
                        st.markdown(f"""
                        <div style="text-align:center;margin-top:1rem;padding:1rem;background:rgba(30,30,47,0.5);border-radius:10px;">
                            <span style="color:#888;">Niveau de confiance:</span>
                            <span style="color:{conf_color};font-weight:700;font-size:1.2rem;margin-left:10px;">{conf_level}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_details:
                        st.markdown("### 📈 Indicateurs Techniques")
                        
                        # RSI
                        if indicators and 'rsi_14' in indicators:
                            rsi = indicators['rsi_14']
                            if rsi > 70:
                                rsi_status = '<span class="indicator-bearish">Surachat ↓</span>'
                            elif rsi < 30:
                                rsi_status = '<span class="indicator-bullish">Survente ↑</span>'
                            else:
                                rsi_status = '<span class="indicator-neutral">Neutre →</span>'
                            
                            st.markdown(f"""
                            <div class="info-box">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <span style="color:#9b59b6;font-weight:600;">RSI (14)</span>
                                    <span style="font-family:'JetBrains Mono';font-size:1.2rem;">{rsi:.1f}</span>
                                </div>
                                <div style="margin-top:5px;">{rsi_status}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # MACD
                        if indicators and 'macd' in indicators:
                            macd = indicators['macd']
                            if macd > 0:
                                macd_status = '<span class="indicator-bullish">Momentum Haussier ↑</span>'
                            else:
                                macd_status = '<span class="indicator-bearish">Momentum Baissier ↓</span>'
                            
                            st.markdown(f"""
                            <div class="info-box info-box-blue">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <span style="color:#3498db;font-weight:600;">MACD Histogram</span>
                                    <span style="font-family:'JetBrains Mono';font-size:1.2rem;">{macd:.2f}</span>
                                </div>
                                <div style="margin-top:5px;">{macd_status}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Bollinger
                        if indicators and 'bb_position' in indicators:
                            bb_pos = indicators['bb_position']
                            if bb_pos > 0.8:
                                bb_status = '<span class="indicator-bearish">Proche bande haute ↓</span>'
                            elif bb_pos < 0.2:
                                bb_status = '<span class="indicator-bullish">Proche bande basse ↑</span>'
                            else:
                                bb_status = '<span class="indicator-neutral">Zone médiane →</span>'
                            
                            st.markdown(f"""
                            <div class="info-box info-box-green">
                                <div style="display:flex;justify-content:space-between;align-items:center;">
                                    <span style="color:#28a745;font-weight:600;">Position Bollinger</span>
                                    <span style="font-family:'JetBrains Mono';font-size:1.2rem;">{bb_pos*100:.0f}%</span>
                                </div>
                                <div style="margin-top:5px;">{bb_status}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Timestamp
                        st.markdown(f"""
                        <div style="text-align:center;margin-top:1rem;color:#666;font-size:0.9rem;">
                            🕐 Dernière analyse: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Impossible de récupérer les données pour la prédiction")
    else:
        st.warning("⚠️ Modèle non chargé. Cliquez sur 'Train' dans la sidebar pour entraîner le modèle.")
        
        if st.button("🚀 Entraîner le modèle maintenant", use_container_width=True):
            with st.spinner("📊 Collecte des données et entraînement..."):
                try:
                    from model_training import train_and_save_model
                    from data_collector import collect_and_save_data
                    from feature_engineering import process_raw_data
                    
                    raw_df = collect_and_save_data()
                    processed_df = process_raw_data(raw_df)
                    train_and_save_model(processed_df)
                    
                    st.success("✅ Modèle entraîné avec succès!")
                    st.cache_resource.clear()
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Erreur: {e}")
    
    # =====================================
    # SECTION 3: GRAPHIQUE PRINCIPAL
    # =====================================
    
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📈 Analyse Technique Avancée</h2>', unsafe_allow_html=True)
    
    with st.spinner("Chargement du graphique..."):
        chart_df = fetch_latest_ohlc(chart_period)
        
        if chart_df is not None and not chart_df.empty:
            # Ajouter les indicateurs
            engineer = FeatureEngineer(chart_df)
            engineer.add_moving_averages()
            engineer.add_rsi()
            engineer.add_macd()
            engineer.add_bollinger_bands()
            
            chart_df_with_indicators = engineer.df
            
            fig = create_main_chart(chart_df_with_indicators)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Données non disponibles")
    
    # =====================================
    # SECTION 4: CORRÉLATIONS
    # =====================================
    
    st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">🔗 Corrélations de Marché</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if chart_df is not None:
            fig_corr = create_correlation_chart(chart_df)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### 📚 Guide des Corrélations
        
        <div class="info-box">
            <strong style="color:#FFD700;">🥇 Or vs 💵 Dollar (DXY)</strong><br>
            Corrélation typiquement <strong style="color:#dc3545;">INVERSE</strong><br>
            <small>Quand le dollar monte, l'or tend à baisser et vice versa.</small>
        </div>
        
        <div class="info-box info-box-blue" style="margin-top:1rem;">
            <strong style="color:#9b59b6;">🥇 Or vs ⚡ VIX</strong><br>
            Corrélation typiquement <strong style="color:#28a745;">POSITIVE</strong><br>
            <small>L'or est une valeur refuge - il monte quand la peur augmente.</small>
        </div>
        
        <div class="info-box info-box-green" style="margin-top:1rem;">
            <strong style="color:#3498db;">🥇 Or vs 📈 Taux Réels</strong><br>
            Corrélation typiquement <strong style="color:#dc3545;">INVERSE</strong><br>
            <small>Des taux réels élevés rendent l'or moins attractif.</small>
        </div>
        """, unsafe_allow_html=True)
    
    # =====================================
    # SECTION 5: IMPORTANCE DES FEATURES
    # =====================================
    
    if model_loaded and model.is_trained:
        st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">🔍 Facteurs de Décision du Modèle</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            importance_df = model.get_feature_importance(20)
            
            fig_imp = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h',
                marker=dict(
                    color=importance_df['importance'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Importance")
                )
            ))
            
            fig_imp.update_layout(
                template='plotly_dark',
                height=600,
                title='🏆 Top 20 Features les Plus Importantes',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.8)',
                yaxis={'categoryorder': 'total ascending'},
                font=dict(family="Inter, sans-serif")
            )
            
            st.plotly_chart(fig_imp, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### 📖 Interprétation
            
            Ce graphique montre les **facteurs les plus influents** dans les décisions du modèle.
            
            **Catégories de features:**
            
            - 📈 **Prix & Returns** - Mouvements de prix historiques
            - 📊 **Indicateurs Techniques** - RSI, MACD, Bollinger...
            - 🌍 **Macro** - DXY, VIX, Taux...
            - 📅 **Temporel** - Saisonnalité, jour de la semaine...
            
            **Plus l'importance est élevée, plus le facteur influence la prédiction.**
            """)
    
    # =====================================
    # FOOTER
    # =====================================
    
    st.markdown("""
    <div class="footer">
        <p>⚠️ <strong>Avertissement:</strong> Ce système est à but éducatif uniquement. 
        Les prédictions ne constituent pas des conseils financiers. Le trading comporte des risques significatifs.</p>
        <p style="margin-top:1rem;">
            🥇 Gold Trading AI | Version 2.0 | Développé avec ❤️ et Python
        </p>
        <p style="color:#444;font-size:0.8rem;">
            Powered by XGBoost • Streamlit • yFinance
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
