import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import time
import json
import os

def load_processed_data():
    """Carga dataset procesado"""
    df = pd.read_csv('data/processed/angiografia_clean.csv')
    
    feature_cols = ['Tipo_Procedimiento_encoded', 'Edad', 'Peso', 
                    'PKA_Gycm2', 'Kar_mGy', 'Tiempo_Fluoroscopia_min']
    
    X = df[feature_cols]
    y = df['excede_DRL']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"✓ Datos cargados:")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def train_and_log_model(model, model_name, X_train, y_train, X_test, y_test):
    """Entrena modelo y registra en MLflow"""
    
    print(f"\n[Entrenando {model_name}]")
    
    with mlflow.start_run(run_name=model_name):
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'training_time': training_time
        }
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        })
        
        if 'XGB' in model_name:
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        print(f"  Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        print(f"  Matriz: TN={tn} FP={fp} FN={fn} TP={tp}")
        
        return model, metrics

def train_all_models():
    """Entrena y compara 3 modelos"""
    
    print("="*70)
    print("ENTRENAMIENTO DE MODELOS CON MLFLOW")
    print("="*70)
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    mlflow.set_experiment("DRL_Angiografia_Prediction")
    
    print("\n[1/3] Cargando datos...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    models = {
        'Logistic_Regression': LogisticRegression(
            random_state=42, max_iter=1000, solver='lbfgs'
        ),
        'Random_Forest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, 
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
    }
    
    print(f"\n[2/3] Entrenando {len(models)} modelos...")
    
    results = {}
    
    for name, model in models.items():
        trained_model, metrics = train_and_log_model(
            model, name, X_train, y_train, X_test, y_test
        )
        results[name] = {'model': trained_model, 'metrics': metrics}
        
        model_filename = f'models/{name.lower()}_model.pkl'
        joblib.dump(trained_model, model_filename)
        print(f"  ✓ Guardado: {model_filename}")
    
    print(f"\n[3/3] Seleccionando mejor modelo...")
    best_name = max(results, key=lambda k: results[k]['metrics']['f1'])
    best_model = results[best_name]['model']
    best_metrics = results[best_name]['metrics']
    
    joblib.dump(best_model, 'models/best_model.pkl')
    
    print(f"\n✅ MEJOR MODELO: {best_name}")
    print(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {best_metrics['f1']:.4f}")
    print(f"   Recall: {best_metrics['recall']:.4f}")
    
    comparison = []
    for name, data in results.items():
        row = {'Modelo': name}
        row.update(data['metrics'])
        comparison.append(row)
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison.to_csv('results/model_comparison.csv', index=False)
    print("\n✓ Comparación guardada: results/model_comparison.csv")
    
    with open('results/metrics.json', 'w') as f:
        json.dump(best_metrics, f, indent=2)
    print("✓ Métricas guardadas: results/metrics.json")
    
    with mlflow.start_run(run_name=f"{best_name}_BEST"):
        if 'XGB' in best_name:
            mlflow.xgboost.log_model(
                best_model, "model", registered_model_name="DRL_Predictor"
            )
        else:
            mlflow.sklearn.log_model(
                best_model, "model", registered_model_name="DRL_Predictor"
            )
    
    print("\n" + "="*70)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*70)
    print(f"\nMLflow UI: mlflow ui --port 5000")
    print(f"Abrir: http://localhost:5000")
    
    return results

if __name__ == "__main__":
    train_all_models()
