import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'angiografia_clean.csv')


st.set_page_config(
    page_title="DRL Angiografía - Sistema de Predicción",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .danger-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏥 Sistema de Predicción de DRL en Angiografía</h1>', unsafe_allow_html=True)

API_URL = "http://localhost:8000"

st.sidebar.title("🔍 Navegación")
page = st.sidebar.radio(
    "Seleccione una opción:",
    ["🎯 Predicción Individual", "📊 Análisis Exploratorio", "ℹ️ Información"]
)

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

api_status = check_api_health()

if not api_status:
    st.sidebar.error("⚠️ API no disponible")
    st.sidebar.info("Asegúrese de que la API esté ejecutándose en http://localhost:8000")
else:
    st.sidebar.success("✅ API conectada")

if page == "🎯 Predicción Individual":
    st.header("Módulo de Predicción Individual")
    st.write("Ingrese los datos del procedimiento para predecir la probabilidad de exceder el DRL")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos del Procedimiento")
        
        tipo = st.selectbox(
            "Tipo de Procedimiento",
            [
                "Coronariografía Diagnóstica",
                "Angiografía Cerebral",
                "Angiografía Aorta Abdominal",
                "Angiografía Periférica",
                "Angiografía Renal"
            ]
        )
             
        edad = st.number_input("Edad (años)", min_value=18, max_value=100, value=65, step=1)
        peso = st.number_input("Peso (kg)", min_value=40.0, max_value=150.0, value=75.5, step=0.1)
    
    with col2:
        st.subheader("Variables Dosimétricas")
        
        pka = st.number_input("PKA (Gy·cm²)", min_value=0.0, max_value=500.0, value=145.0, step=0.1)
        kar = st.number_input("Ka,r (mGy)", min_value=0.0, max_value=5000.0, value=850.0, step=1.0)
        tiempo = st.number_input("Tiempo Fluoroscopia (min)", min_value=0.0, max_value=120.0, value=22.5, step=0.1)
    
    st.write("")
    if st.button("🔍 Realizar Predicción", type="primary", use_container_width=True):
        
        if not api_status:
            st.error("❌ No se puede conectar con la API. Verifique que esté ejecutándose.")
        else:
            payload = {
                "tipo": tipo,
                "edad": int(edad),
                "peso": float(peso),
                "pka": float(pka),
                "kar": float(kar),
                "tiempo": float(tiempo)
            }
            
            try:
                with st.spinner("Procesando predicción..."):
                    response = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.write("")
                    st.write("---")
                    st.subheader("📊 Resultado de la Predicción")
                    
                    prob_percent = result['probabilidad'] * 100
                    
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob_percent,
                        title={'text': "Probabilidad de Exceder DRL (%)"},
                        number={'suffix': "%"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {
                                'color': "darkred" if prob_percent > 70 else "orange" if prob_percent > 30 else "green"
                            },
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "lightyellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if result['riesgo'] == "BAJO":
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ RIESGO BAJO</h3>
                            <p>El procedimiento tiene <strong>{prob_percent:.1f}%</strong> de probabilidad de exceder el DRL.</p>
                            <p><strong>Recomendación:</strong> El procedimiento puede realizarse con las configuraciones estándar.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    elif result['riesgo'] == "MODERADO":
                        st.warning(f"⚠️ RIESGO MODERADO: {prob_percent:.1f}% de probabilidad")
                    
                    else:
                        st.markdown(f"""
                        <div class="danger-box">
                            <h3>🚨 RIESGO ALTO</h3>
                            <p>El procedimiento tiene <strong>{prob_percent:.1f}%</strong> de probabilidad de exceder el DRL.</p>
                            <p><strong>Recomendación:</strong> Ajustar parámetros antes del procedimiento.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                else:
                    st.error(f"❌ Error en la API: {response.status_code}")
            
            except Exception as e:
                st.error(f"❌ Error: {e}")

elif page == "📊 Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    try:
        df = pd.read_csv(DATA_PATH)
        
        st.subheader("📋 Estadísticas Generales")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Registros", len(df))
        with col2:
            excede_pct = (df['excede_DRL'].sum() / len(df)) * 100
            st.metric("Exceden DRL", f"{excede_pct:.1f}%")
        with col3:
            st.metric("Variables", len(df.columns))
        
        st.subheader("📊 Distribución de PKA")
        fig = px.histogram(df, x='PKA_Gycm2', nbins=30, title="Distribución de PKA")
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("❌ Dataset no encontrado. Ejecute primero el preprocesamiento.")

else:
    st.header("ℹ️ Información del Sistema")
    
    st.subheader("📖 Acerca del Proyecto")
    st.write("""
    Este sistema utiliza Machine Learning para predecir la probabilidad de que
    un procedimiento de angiografía exceda los Niveles de Referencia Diagnósticos (DRL).
    """)
    
    st.subheader("👥 Equipo de Desarrollo")
    st.write("""
    - Carlos Fabián Ospina
    - Mario Alberto Nájar Martínez
    - Cristian David Pérez Ariza
    - Fernanda Paola Campo
    
    **Universidad de los Andes - Maestría en Inteligencia Artificial**
    """)

st.write("")
st.write("---")
st.caption("© 2026 Sistema DRL Angiografía | Universidad de los Andes")

