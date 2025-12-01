import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Valuación Inmobiliaria | UADEO",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS para apariencia profesional
st.markdown("""
    <style>
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #333333;
    }
    .sub-text {
        color: #555555;
        font-size: 18px;
    }
    .metric-container {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- 1. CARGA DEL MODELO ---
@st.cache_resource
def cargar_pipeline():
    try:
        # Busca el archivo comprimido (el ligero de ~30MB)
        pipeline = joblib.load('pipeline_housing_v2.pkl')
        return pipeline
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

pipeline = cargar_pipeline()

# --- 2. ENCABEZADO ACADÉMICO ---
col_logo, col_titulo = st.columns([1, 6])

with col_logo:
    st.image("https://sic.cultura.gob.mx/images/62631", width=110)

with col_titulo:
    st.markdown("""
    <div class="main-header">
        <h2 style='margin-bottom: 0px;'>Maestría en Inteligencia Artificial Aplicada</h2>
        <h4 style='margin-top: 5px; font-weight: normal;'>Universidad Autónoma de Occidente</h4>
        <hr style='margin-top: 10px; margin-bottom: 10px;'>
        <h3 style='margin-top: 10px; color: #004B93;'>🏡 Sistema de Valuación Inmobiliaria (Modelo California)</h3>
    </div>
    """, unsafe_allow_html=True)

st.markdown("**Profesor:** Dr. Raul Oramas Bustillos &nbsp;&nbsp;|&nbsp;&nbsp; **Alumno:** Psic. Andres Cruz Degante")

if pipeline is None:
    st.error("⚠️ **Error de Sistema:** No se encuentra el archivo `pipeline_housing_v2.pkl` en el repositorio. Por favor, asegúrese de haber subido el modelo entrenado.")
    st.stop()

st.markdown("---")

# --- 3. PANEL DE ENTRADA DE DATOS ---

col_izq, col_der = st.columns(2, gap="large")

with col_izq:
    st.subheader("📍 1. Ubicación y Entorno")
    
    # Mapa interactivo para seleccionar ubicación
    lat = st.slider("Latitud", 32.54, 41.95, 34.05, step=0.01)
    lon = st.slider("Longitud", -124.35, -114.31, -118.24, step=0.01)
    
    # Visualización en mapa
    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_data, zoom=6, use_container_width=True)
    
    # Ingreso económico de la zona
    st.markdown("---")
    st.markdown("**Perfil Económico de la Zona**")
    ingreso_anual = st.number_input(
        "Ingreso Anual Promedio del Vecindario (USD)", 
        min_value=10000, 
        max_value=150000, 
        value=50000, 
        step=1000,
        help="El modelo utiliza el ingreso medio de la zona como un fuerte predictor del valor."
    )

with col_der:
    st.subheader("🏠 2. Características de la Vivienda")
    
    antiguedad = st.slider("Antigüedad del Inmueble (Años)", 1, 52, 15)
    
    st.markdown("**Distribución de Espacios**")
    c1, c2 = st.columns(2)
    
    with c1:
        dormitorios = st.number_input("Número de Dormitorios", min_value=1, max_value=10, value=3)
        banos = st.number_input("Número de Baños Completos", min_value=1, max_value=8, value=2)
        
    with c2:
        otros_cuartos = st.number_input("Otros Espacios (Sala, Cocina, Comedor)", min_value=1, max_value=10, value=2)
        habitantes = st.number_input("Ocupantes Promedio", min_value=1, max_value=10, value=3)

    st.markdown("---")
    st.markdown("**Proximidad a la Costa**")
    
    # Mapeo amigable -> Valor técnico
    ocean_map = {
        "Zona Interior (Lejos de la costa)": "INLAND",
        "Zona Costera (< 1H del Océano)": "<1H OCEAN",
        "En la Bahía (Bay Area)": "NEAR BAY",
        "Frente al Mar": "NEAR OCEAN",
        "Insular (Isla)": "ISLAND"
    }
    
    ocean_selection = st.selectbox("Seleccione la ubicación respecto al mar:", list(ocean_map.keys()))
    ocean_val_model = ocean_map[ocean_selection]

# --- 4. MOTOR DE CÁLCULO ---

st.markdown("<br>", unsafe_allow_html=True)
calc_btn = st.button("CALCULAR TASACIÓN DE MERCADO", type="primary", use_container_width=True)

if calc_btn:
    # A. Ingeniería de Características (Transformar Inputs de Casa -> Inputs de Modelo)
    
    # 1. Total de habitaciones (Suma de todos los espacios)
    total_habitaciones_promedio = dormitorios + banos + otros_cuartos
    
    # 2. Ajuste de Ingreso (El dataset original usa unidades de $10,000)
    median_income_model = ingreso_anual / 10000.0
    
    # 3. Construcción del DataFrame Base
    input_dict = {
        'longitude': lon,
        'latitude': lat,
        'housing_median_age': antiguedad,
        'median_income': median_income_model,
        'rooms_per_household': total_habitaciones_promedio,
        'bedrooms_per_household': dormitorios,
        'population_per_household': habitantes
    }
    
    # 4. One-Hot Encoding Manual (Crítico para que coincida con el entrenamiento)
    # Las columnas deben ser: ocean_proximity_INLAND, ocean_proximity_ISLAND, etc.
    # NOTA: '<1H OCEAN' suele ser la categoría eliminada (drop_first) en el entrenamiento.
    
    opciones_modelo = ['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    for opt in opciones_modelo:
        input_dict[f'ocean_proximity_{opt}'] = 1 if ocean_val_model == opt else 0
        
    df_input = pd.DataFrame([input_dict])
    
    # B. Predicción
    try:
        # El pipeline incluye el StandardScaler, así que no necesitamos escalar manualmente aquí
        prediccion = pipeline.predict(df_input)[0]
        
        # C. Despliegue de Resultados
        st.markdown("---")
        st.subheader("📊 Resultado del Análisis")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-container">
                <p style="margin-bottom: 5px; color: #555;">Valor Estimado</p>
                <h1 style="color: #28a745; margin: 0;">${prediccion:,.2f}</h1>
                <p style="font-size: 12px; color: #888;">USD</p>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.info("ℹ️ **Interpretación del Modelo**")
            st.write(f"""
            El algoritmo ha estimado este valor basándose en una propiedad de **{antiguedad} años** de antigüedad, con **{total_habitaciones_promedio} habitaciones totales**, ubicada en 
            una zona con un ingreso medio de **${ingreso_anual:,.0f}**.
            """)
            
            # Lógica simple de interpretación
            if ocean_val_model == "INLAND":
                st.write("📉 **Factor Geográfico:** La ubicación interior tiende a disminuir el valor comparado con la costa.")
            elif ocean_val_model in ["NEAR OCEAN", "NEAR BAY", "ISLAND"]:
                st.write("📈 **Factor Geográfico:** La proximidad al agua añade una prima significativa al valor.")
                
            if median_income_model > 6.0:
                st.write("📈 **Factor Económico:** El alto poder adquisitivo de la zona impulsa el precio hacia arriba.")

    except Exception as e:
        st.error(f"Error en el proceso de predicción: {str(e)}")
        st.write("Detalle técnico: Verifique que las columnas del Pipeline coincidan con las generadas.")
