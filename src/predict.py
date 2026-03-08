import joblib
import numpy as np

class DRLPredictor:
    """Clase para realizar predicciones de excedencia de DRL"""
    
    def __init__(self, model_path='models/best_model.pkl'):
        """
        Inicializa el predictor cargando modelo y transformadores
        
        Args:
            model_path: Ruta al modelo serializado
        """
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load('models/label_encoder.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        
        print(f"✓ Modelo cargado desde {model_path}")
    
    def preprocess_input(self, tipo, edad, peso, pka, kar, tiempo):
        """
        Preprocesa entrada del usuario
        
        Args:
            tipo: Tipo de procedimiento (str)
            edad: Edad del paciente (int)
            peso: Peso del paciente (float)
            pka: Producto Kerma-Área (float)
            kar: Kerma de referencia (float)
            tiempo: Tiempo de fluoroscopia (float)
        
        Returns:
            features: Array numpy listo para predicción
        """
        try:
            tipo_encoded = self.label_encoder.transform([tipo])[0]
        except ValueError:
            raise ValueError(f"Tipo de procedimiento '{tipo}' no reconocido")
        
        features = np.array([[tipo_encoded, edad, peso, pka, kar, tiempo]])
        features_scaled = self.scaler.transform(features)
        
        return features_scaled
    
    def predict(self, tipo, edad, peso, pka, kar, tiempo):
        """
        Realiza predicción
        
        Returns:
            dict: {
                'excede_DRL': 0 o 1,
                'probabilidad': float (0-1),
                'riesgo': 'BAJO', 'MODERADO' o 'ALTO'
            }
        """
        features = self.preprocess_input(tipo, edad, peso, pka, kar, tiempo)
        
        pred = self.model.predict(features)[0]
        prob = self.model.predict_proba(features)[0][1]
        
        if prob < 0.3:
            riesgo = "BAJO"
        elif prob < 0.7:
            riesgo = "MODERADO"
        else:
            riesgo = "ALTO"
        
        return {
            'excede_DRL': int(pred),
            'probabilidad': float(prob),
            'riesgo': riesgo
        }

def main():
    """Ejemplo de uso"""
    
    predictor = DRLPredictor()
    
    print("\n" + "="*60)
    print("EJEMPLO 1: Coronariografía con dosis normal")
    print("="*60)
    
    result = predictor.predict(
        tipo="Coronariografía Diagnóstica",
        edad=55,
        peso=75.0,
        pka=42.5,
        kar=320.0,
        tiempo=8.0
    )
    
    print(f"Resultado: {'EXCEDE' if result['excede_DRL'] else 'NO EXCEDE'} DRL")
    print(f"Probabilidad: {result['probabilidad']*100:.1f}%")
    print(f"Nivel de riesgo: {result['riesgo']}")
    
    print("\n" + "="*60)
    print("EJEMPLO 2: Angiografía Cerebral con dosis alta")
    print("="*60)
    
    result = predictor.predict(
        tipo="Angiografía Cerebral",
        edad=68,
        peso=82.0,
        pka=145.0,
        kar=850.0,
        tiempo=22.5
    )
    
    print(f"Resultado: {'EXCEDE' if result['excede_DRL'] else 'NO EXCEDE'} DRL")
    print(f"Probabilidad: {result['probabilidad']*100:.1f}%")
    print(f"Nivel de riesgo: {result['riesgo']}")

if __name__ == "__main__":
    main()
