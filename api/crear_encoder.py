from sklearn.preprocessing import LabelEncoder
import unicodedata
import joblib

# 1️⃣ Definir las clases originales
clases = [
    "Angiografía Aorta Abdominal",
    "Angiografía Cerebral",
    "Angiografía Periférica",
    "Angiografía Renal",
    "Coronariografía Diagnóstica"
]

# 2️⃣ Función para normalizar texto (quita acentos y minúsculas)
def normalize_text(text):
    text = text.strip().lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')
    return text

# 3️⃣ Normalizar todas las clases
clases_norm = [normalize_text(c) for c in clases]

# 4️⃣ Crear y entrenar el LabelEncoder
le = LabelEncoder()
le.fit(clases_norm)

# 5️⃣ Guardar el encoder para usarlo en tu API
joblib.dump(le, "label_encoder.pkl")
print("✅ Nuevo label_encoder guardado con clases normalizadas")