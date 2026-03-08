import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_raw_data(filepath=None):
    if filepath is None:
        filepath = os.path.join(BASE_DIR, 'data', 'raw', 'angiografia_dataset_1000.csv')
    """Carga dataset raw"""
    df = pd.read_csv(filepath)
    print(f"✓ Dataset cargado: {df.shape[0]} registros × {df.shape[1]} columnas")
    return df

def clean_data(df):
    """Limpia valores faltantes y duplicados"""
    print(f"Registros originales: {len(df)}")
    initial_len = len(df)
    df_clean = df.dropna()
    removed = initial_len - len(df_clean)
    print(f"Registros tras limpieza: {len(df_clean)} ({removed} eliminados)")
    df_clean = df_clean.drop_duplicates()
    return df_clean

def calculate_p75_and_target(df):
    """Calcula P75 por tipo y crea variable objetivo"""
    p75_dict = {}
    print("\nNiveles de Referencia (P75) por tipo:")
    for tipo in sorted(df['Tipo_Procedimiento'].unique()):
        subset = df[df['Tipo_Procedimiento'] == tipo]
        p75 = subset['PKA_Gycm2'].quantile(0.75)
        p75_dict[tipo] = p75
        print(f"  {tipo}: {p75:.2f} Gy·cm²")
    
    df['excede_DRL'] = df.apply(
        lambda row: 1 if row['PKA_Gycm2'] > p75_dict[row['Tipo_Procedimiento']] else 0,
        axis=1
    )
    
    print(f"\nDistribución objetivo:")
    print(f"  No excede (0): {(df['excede_DRL'] == 0).sum()} ({(df['excede_DRL'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Excede (1): {(df['excede_DRL'] == 1).sum()} ({(df['excede_DRL'] == 1).sum() / len(df) * 100:.1f}%)")
    
    return df, p75_dict

def encode_and_scale(df):
    """Codifica variables categóricas y escala numéricas"""
    os.makedirs('models', exist_ok=True)
    
    le = LabelEncoder()
    df['Tipo_Procedimiento_encoded'] = le.fit_transform(df['Tipo_Procedimiento'])
    
    scaler = StandardScaler()
    numeric_cols = ['Edad', 'Peso', 'PKA_Gycm2', 'Kar_mGy', 'Tiempo_Fluoroscopia_min']
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    joblib.dump(le, 'models/label_encoder.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n✓ Transformadores guardados en models/")
    
    return df_scaled, le, scaler

def preprocess_pipeline():
    """Pipeline completo de preprocesamiento"""
    print("="*70)
    print("PREPROCESAMIENTO DE DATOS")
    print("="*70)
    
    os.makedirs('data/processed', exist_ok=True)
    
    print("\n[1/4] Cargando datos...")
    df = load_raw_data()
    
    print("\n[2/4] Limpiando datos...")
    df = clean_data(df)
    
    print("\n[3/4] Calculando P75 y variable objetivo...")
    df, _ = calculate_p75_and_target(df)
    
    print("\n[4/4] Codificando y escalando...")
    df_scaled, _, _ = encode_and_scale(df)
    
    df_scaled.to_csv('data/processed/angiografia_clean.csv', index=False)
    print("\n✓ Dataset procesado: data/processed/angiografia_clean.csv")
    print("="*70)
    
    return df_scaled

if __name__ == "__main__":
    preprocess_pipeline()

