from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import unicodedata

app = FastAPI(
    title="DRL Angiografía API",
    description="API para predicción de excedencia de Niveles de Referencia Diagnósticos",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================================
# FUNCIÓN AUXILIAR: NORMALIZAR TEXTO
# ============================================================================
def normalizar_texto(text: str) -> str:
    """
    Normaliza texto eliminando acentos, convirtiendo a minúsculas
    y corrigiendo problemas de codificación UTF-8
    """
    try:
        # Intentar corregir codificación UTF-8 mal interpretada
        text = text.encode('latin1').decode('utf-8')
    except:
        pass
    
    # Normalizar: minúsculas y sin acentos
    text = text.strip().lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return text

# ============================================================================
# CARGAR MODELO Y TRANSFORMADORES AL INICIAR
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

try:
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    label_encoder_original = joblib.load(MODELS_DIR / "label_encoder.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    
    # ========================================================================
    # CORREGIR: NORMALIZAR LAS CLASES DEL ENCODER
    # ========================================================================
    clases_originales = label_encoder_original.classes_
    clases_normalizadas = [normalizar_texto(c) for c in clases_originales]
    
    # Crear un diccionario de mapeo: texto_normalizado -> código
    tipo_to_code = {clases_normalizadas[i]: i for i in range(len(clases_normalizadas))}
    
    print("="*70)
    print("✓ Modelo y transformadores cargados exitosamente")
    print(f"  Modelo: {type(model).__name__}")
    print(f"  Clases originales (con encoding):")
    for clase in clases_originales:
        print(f"    - {clase}")
    print(f"  Clases normalizadas:")
    for clase in clases_normalizadas:
        print(f"    - {clase}")
    print(f"  Features del scaler: {scaler.feature_names_in_}")
    print("="*70)

except Exception as e:
    print(f"❌ Error al cargar modelo: {e}")
    model = None
    label_encoder_original = None
    scaler = None
    tipo_to_code = None

# ============================================================================
# ESQUEMAS PYDANTIC
# ============================================================================
class PredictionRequest(BaseModel):
    tipo: str = Field(..., example="Angiografía Cerebral")
    edad: int = Field(..., ge=18, le=100, example=65)
    peso: float = Field(..., ge=40, le=150, example=75.5)
    pka: float = Field(..., ge=0, le=500, example=145.0)
    kar: float = Field(..., ge=0, le=5000, example=850.0)
    tiempo: float = Field(..., ge=0, le=120, example=22.5)

class PredictionResponse(BaseModel):
    excede_DRL: int
    probabilidad: float
    riesgo: str
    timestamp: str

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", tags=["General"])
def root():
    return {
        "message": "API DRL Angiografía v1.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", tags=["General"])
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predicción"])
def predict(request: PredictionRequest):
    """Realiza una predicción de excedencia de DRL"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        # ====================================================================
        # PASO 1: NORMALIZAR EL TIPO RECIBIDO
        # ====================================================================
        tipo_normalizado = normalizar_texto(request.tipo)
        print(f"\n{'='*70}")
        print(f"NUEVA PREDICCIÓN")
        print(f"{'='*70}")
        print(f"Tipo recibido: {request.tipo}")
        print(f"Tipo normalizado: {tipo_normalizado}")
        print(f"Tipos válidos: {list(tipo_to_code.keys())}")
        
        # ====================================================================
        # PASO 2: BUSCAR EL CÓDIGO EN EL DICCIONARIO
        # ====================================================================
        if tipo_normalizado not in tipo_to_code:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo no reconocido: '{request.tipo}'. Válidos: {list(tipo_to_code.keys())}"
            )
        
        tipo_encoded = tipo_to_code[tipo_normalizado]
        print(f"Tipo codificado: {tipo_encoded}")
        
        # ====================================================================
        # PASO 3: PREPARAR LAS 5 FEATURES NUMÉRICAS
        # ====================================================================
        features_numericas = pd.DataFrame([{
            "Edad": float(request.edad),
            "Peso": float(request.peso),
            "PKA_Gycm2": float(request.pka),
            "Kar_mGy": float(request.kar),
            "Tiempo_Fluoroscopia_min": float(request.tiempo)
        }])
        
        print(f"\nFeatures numéricas (5):")
        print(features_numericas)
        
        # Asegurar orden correcto
        features_numericas = features_numericas[scaler.feature_names_in_]
        
        # ====================================================================
        # PASO 4: ESCALAR LAS FEATURES NUMÉRICAS
        # ====================================================================
        features_escaladas = scaler.transform(features_numericas)
        print(f"\nFeatures escaladas (5): {features_escaladas}")
        print(f"Shape: {features_escaladas.shape}")
        
        # ====================================================================
        # PASO 5: AGREGAR EL TIPO CODIFICADO (6 FEATURES TOTALES)
        # ====================================================================
        features_completas = np.concatenate([
            [[tipo_encoded]],
            features_escaladas
        ], axis=1)
        
        print(f"\nFeatures completas (6): {features_completas}")
        print(f"Shape final: {features_completas.shape}")
        
        # ====================================================================
        # PASO 6: PREDECIR
        # ====================================================================
        pred = int(model.predict(features_completas)[0])
        
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(features_completas)[0][1])
        else:
            prob = 0.5
        
        # Determinar riesgo
        if prob < 0.3:
            riesgo = "BAJO"
        elif prob < 0.7:
            riesgo = "MODERADO"
        else:
            riesgo = "ALTO"
        
        print(f"\nResultado:")
        print(f"  Predicción: {pred} ({'EXCEDE' if pred == 1 else 'NO EXCEDE'})")
        print(f"  Probabilidad: {prob:.4f} ({prob*100:.1f}%)")
        print(f"  Riesgo: {riesgo}")
        print(f"{'='*70}\n")
        
        return PredictionResponse(
            excede_DRL=pred,
            probabilidad=prob,
            riesgo=riesgo,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        print(f"\n❌ Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


