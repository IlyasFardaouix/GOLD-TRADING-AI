"""
Script d'exécution complète du pipeline
========================================
Collecte des données → Feature Engineering → Entraînement → Lancement de l'app
"""

import subprocess
import sys
import os

# S'assurer que le répertoire courant est correct
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run_pipeline():
    """Exécute le pipeline complet."""
    
    print("=" * 60)
    print(" GOLD TRADING AI - PIPELINE COMPLET")
    print("=" * 60)
    
    # Étape 1: Collecte des données
    print("\n📥 ÉTAPE 1: Collecte des données...")
    print("-" * 40)
    
    from data_collector import collect_and_save_data
    raw_df = collect_and_save_data()
    print(f"[OK]  Données collectées: {len(raw_df)} lignes")
    
    # Étape 2: Feature Engineering
    print("\n[CONFIG]  ÉTAPE 2: Feature Engineering...")
    print("-" * 40)
    
    from feature_engineering import process_raw_data
    processed_df = process_raw_data(raw_df)
    print(f"[OK]  Features créées: {len(processed_df.columns)} colonnes")
    
    # Étape 3: Entraînement du modèle
    print("\n[AI]  ÉTAPE 3: Entraînement du modèle...")
    print("-" * 40)
    
    from model_training import train_and_save_model
    model = train_and_save_model(processed_df)
    print("[OK]  Modèle entraîné et sauvegardé")
    
    # Étape 4: Lancement de l'application
    print("\n[START]  ÉTAPE 4: Lancement de l'application Streamlit...")
    print("-" * 40)
    print("\nL'application va s'ouvrir dans votre navigateur...")
    print("Pour arrêter: Ctrl+C\n")
    
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    run_pipeline()
