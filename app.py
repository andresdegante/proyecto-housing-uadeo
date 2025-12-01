import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página (título y diseño ancho)
st.set_page_config(page_title="Predicción de Vivienda - UADEO", layout="wide")

# --- 1. CARGA DE MODELOS (Con caché para velocidad) ---
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load('modelo_housing.pkl')
    scaler = joblib.load('escalador_housing.pkl')
    return modelo, scaler

try:
    modelo, scaler = cargar_modelos()
except FileNotFoundError:
    st.error("⚠️ Error: No se encuentran los archivos .pkl (modelo o escalador). Asegúrate de subirlos al repositorio.")
    st.stop()

# --- 2. ENCABEZADO CON LOGOS (Diseño Académico) ---
# Usamos columnas para alinear logos y texto
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    st.image("https://www.ucol.mx/content/cms/45/image/relaciones-internacionales-sinaloa-2.png", width=100)

with col2:
    st.markdown("""
        <div style="text-align: center;">
            <h2 style="margin-bottom: 0;">Maestría en Inteligencia Artificial Aplicada</h2>
            <h4 style="margin-top: 0;">Universidad Autónoma de Occidente</h4>
            <hr>
            <h3 style="color: #4F8BF9;">🏡 Predicción de Precios de Vivienda (California)</h3>
            <p><b>Profesor:</b> Dr. Raul Oramas Bustillos | <b>Alumno:</b> Psic. Andres Cruz Degante</p>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.image("https://sic.cultura.gob.mx/images/62631", width=100)

st.divider()

# --- 3. INTERFAZ DE USUARIO (INPUTS) ---

st.write("### 📝 Características de la Propiedad")

# Usamos columnas para organizar los inputs limpiamente
col_izq, col_der = st.columns(2)

with col_izq:
    st.info("📍 Ubicación y Antigüedad")
    longitude = st.slider("Longitud", -124.35, -114.31, -118.0)
    latitude = st.slider("Latitud", 32.54, 41.95, 34.0)
    housing_median_age = st.slider("Antigüedad de la casa (Años)", 1, 52, 15)
    
    st.info("💰 Economía")
    median_income = st.slider("Ingreso Medio del Bloque (Deciles)", 0.5, 15.0, 5.0, help="1 unidad = $10k USD anuales aprox.")

with col_der:
    st.info("🏠 Estructura y Población")
    total_rooms = st.number_input("Total de Habitaciones", value=2000, min_value=1)
    total_bedrooms = st.number_input("Total de Dormitorios", value=400, min_value=1)
    population = st.number_input("Población en la zona", value=1000, min_value=1)
    households = st.number_input("Hogares (Familias)", value=300, min_value=1)
    
    ocean_proximity = st.selectbox(
        "Proximidad al Océano",
        ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    )

# --- 4. LÓGICA DE PREDICCIÓN ---

st.markdown("---")
# Botón centrado
_, col_btn, _ = st.columns([1, 1, 1])

if col_btn.button("🔮 Calcular Precio Estimado", type="primary", use_container_width=True):
    
    # 1. Preparar los datos (DataFrame idéntico al entrenamiento)
    input_data = pd.DataFrame([[
        longitude, latitude, housing_median_age, total_rooms, 
        total_bedrooms, population, households, median_income
    ]], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                 'total_bedrooms', 'population', 'households', 'median_income'])
    
    # 2. One-Hot Encoding manual (Misma lógica que antes)
    op_options = ['INLAND', 'ISLAND', '
