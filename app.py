import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Configuración de la página (Diseño ancho y título profesional)
st.set_page_config(page_title="Predicción de Vivienda | UADEO", layout="wide")

# --- 1. CARGA DE RECURSOS ---
@st.cache_resource
def cargar_modelos():
    try:
        # Asegúrate de que estos nombres coincidan exactamente con tus archivos
        modelo = joblib.load('modelo_housing.pkl')
        scaler = joblib.load('escalador_housing.pkl')
        return modelo, scaler
    except Exception as e:
        return None, None

modelo, scaler = cargar_modelos()

# --- 2. ENCABEZADO INSTITUCIONAL ---
col_logo, col_text = st.columns([1, 5])

with col_logo:
    # Logo único solicitado
    st.image("https://sic.cultura.gob.mx/images/62631", width=120)

with col_text:
    st.markdown("""
    <style>
    .title-text {
        font-family: 'Helvetica', sans-serif;
        color: #333333;
    }
    </style>
    <div class="title-text">
        <h3 style='margin-bottom: 0px;'>Maestría en Inteligencia Artificial Aplicada</h3>
        <h4 style='margin-top: 5px; color: #555;'>Universidad Autónoma de Occidente</h4>
        <hr style='margin-top: 5px; margin-bottom: 5px;'>
        <h2 style='margin-top: 10px;'>Estimación de Valor Inmobiliario (Modelo California)</h2>
    </div>
    """, unsafe_allow_html=True)

st.markdown("**Profesor:** Dr. Raul Oramas Bustillos &nbsp;&nbsp;|&nbsp;&nbsp; **Alumno:** Psic. Andres Cruz Degante")
st.markdown("---")

# Verificación de carga
if modelo is None:
    st.error("Error Crítico: No se encontraron los archivos del modelo (.pkl). Por favor verifique el repositorio.")
    st.stop()

# --- 3. PANEL DE CONTROL (INPUTS) ---

col_izq, col_der = st.columns([1, 1], gap="large")

with col_izq:
    st.subheader("1. Ubicación Geográfica")
    
    # Sliders para lat/lon con valores por defecto centrados en zonas habitables
    lat = st.slider("Latitud", 32.54, 41.95, 34.05, step=0.01)
    lon = st.slider("Longitud", -124.35, -114.31, -118.24, step=0.01)
    
    # Mapa interactivo para referencia visual
    map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
    st.map(map_data, zoom=6, use_container_width=True)
    st.caption("El punto rojo indica la zona censal a evaluar.")

    st.subheader("2. Perfil Económico de la Zona")
    # Ingreso convertido a términos reales para el usuario
    # El dataset usa 1.0 = $10k USD. El slider muestra el valor real aproximado.
    ingreso_input = st.slider(
        "Ingreso Medio Anual de los Residentes (USD)", 
        min_value=10000, 
        max_value=150000, 
        value=50000, 
        step=1000,
        format="$%d"
    )
    # Convertimos de vuelta a la escala del modelo (Deciles)
    median_income = ingreso_input / 10000.0

with col_der:
    st.subheader("3. Características de la Propiedad y Zona")
    
    antiguedad = st.slider("Antigüedad Promedio de la Vivienda (Años)", 1, 52, 25)
    
    st.markdown("##### Densidad y Estructura (Totales del Bloque Censal)")
    st.caption("Nota: Los siguientes valores representan los totales acumulados del bloque censal, no de una sola vivienda individual.")
    
    c1, c2 = st.columns(2)
    with c1:
        # Valores por defecto ajustados a medianas más realistas del dataset original
        total_rooms = st.number_input("Total Habitaciones en Bloque", value=2000, step=100)
        population = st.number_input("Población Total en Bloque", value=1200, step=50)
        
    with c2:
        total_bedrooms = st.number_input("Total Dormitorios en Bloque", value=400, step=10)
        households = st.number_input("Total Hogares (Familias)", value=350, step=10)

    st.markdown("##### Entorno")
    # Mapeo de opciones amigables -> valores del modelo
    ocean_map = {
        "Interior (Lejos de la costa)": "INLAND",
        "A menos de 1h del Océano": "<1H OCEAN",
        "Cerca de la Bahía": "NEAR BAY",
        "Cerca del Océano": "NEAR OCEAN",
        "Isla": "ISLAND"
    }
    
    ocean_selection = st.selectbox("Proximidad al Océano", list(ocean_map.keys()))
    ocean_val_model = ocean_map[ocean_selection]

# --- 4. LÓGICA Y RESULTADOS ---

st.markdown("---")
# Botón de cálculo profesional
if st.button("CALCULAR TASACIÓN", type="primary", use_container_width=True):
    
    # 1. Construcción del DataFrame (Idéntico al entrenamiento)
    input_data = pd.DataFrame([[
        lon, lat, antiguedad, total_rooms, 
        total_bedrooms, population, households, median_income
    ]], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                 'total_bedrooms', 'population', 'households', 'median_income'])
    
    # 2. One-Hot Encoding (Manual para garantizar consistencia)
    # Las columnas deben existir aunque sean 0
    op_options = ['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    for opt in op_options:
        input_data[f'ocean_proximity_{opt}'] = 1 if ocean_val_model == opt else 0
        
    # 3. Predicción
    try:
        # Escalar datos
        input_data_scaled = scaler.transform(input_data)
        # Predecir
        prediccion = modelo.predict(input_data_scaled)
        valor_estimado = prediccion[0]
        
        # 4. Presentación de Resultados
        st.markdown("### Resultados del Análisis")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label="Valor Medio Estimado", value=f"${valor_estimado:,.2f} USD")
        
        with res_col2:
            st.info("ℹ️ **Interpretación del Modelo:**")
            
            # Lógica simple de explicación (Heurística basada en EDA)
            factores_pos = []
            factores_neg = []
            
            if median_income > 6.0: # >60k
                factores_pos.append("El alto nivel de ingresos de la zona impulsa el valor.")
            elif median_income < 3.0:
                factores_neg.append("El nivel de ingresos de la zona limita el potencial de valor.")
                
            if ocean_val_model in ["NEAR BAY", "NEAR OCEAN", "ISLAND"]:
                factores_pos.append("La proximidad a la costa añade una prima significativa al precio.")
            elif ocean_val_model == "INLAND":
                factores_neg.append("La ubicación interior tiende a tener precios más accesibles que la costa.")
                
            if antiguedad < 10:
                factores_pos.append("La construcción reciente favorece la valoración.")
            
            # Mostrar explicación
            if factores_pos:
                for f in factores_pos:
                    st.write(f"• 📈 {f}")
            if factores_neg:
                for f in factores_neg:
                    st.write(f"• 📉 {f}")
            
            if not factores_pos and not factores_neg:
                st.write("• El valor se encuentra en el promedio del mercado según las características demográficas estándar.")

    except Exception as e:
        st.error(f"Error en el cálculo: {str(e)}")
