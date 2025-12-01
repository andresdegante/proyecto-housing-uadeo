import gradio as gr
import pandas as pd
import numpy as np
import joblib

# 1. Cargar el "Cerebro" (Modelo y Escalador)
modelo = joblib.load('modelo_housing.pkl')
scaler = joblib.load('escalador_housing.pkl')

# 2. Definir la función de predicción
# Esta función toma los inputs de la web, los procesa y devuelve el precio
def predecir_precio(longitude, latitude, housing_median_age, total_rooms, 
                    total_bedrooms, population, households, median_income, ocean_proximity):
    
    # Crear DataFrame con los datos de entrada
    # OJO: El orden debe ser idéntico al que usaste para entrenar (X_train)
    input_data = pd.DataFrame([[
        longitude, latitude, housing_median_age, total_rooms, 
        total_bedrooms, population, households, median_income
    ]], columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                 'total_bedrooms', 'population', 'households', 'median_income'])
    
    # Manejo de la variable categórica (Ocean Proximity)
    # Recreamos las columnas dummies como en el entrenamiento
    op_options = ['INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    for opt in op_options:
        input_data[f'ocean_proximity_{opt}'] = 1 if ocean_proximity == opt else 0
        
    # IMPORTANTE: Asegurar que todas las columnas del One-Hot estén presentes
    # (Si en el train había '<1H OCEAN', aquí se asume implícito si los otros son 0)
    
    # Escalar los datos (Crucial para que el modelo entienda los números)
    # Nota: Asegúrate de que las columnas coincidan con las del scaler original
    # Si usaste K-Means en el proyecto, habría que regenerar el cluster aquí, 
    # pero para la demo simplificada usaremos las variables base.
    
    try:
        input_data_scaled = scaler.transform(input_data)
        prediction = modelo.predict(input_data_scaled)
        return f"${prediction[0]:,.2f} USD"
    except Exception as e:
        return f"Error en cálculo: Verifica las columnas. Detalle: {str(e)}"

# 3. Diseño de la Interfaz (Minimalista y Académica)
# URLs de los logos
logo_blanco = "https://www.ucol.mx/content/cms/45/image/relaciones-internacionales-sinaloa-2.png" # Usamos el de UADEO/Sinaloa como 'blanco' para contraste
logo_oscuro = "https://sic.cultura.gob.mx/images/62631" # Logo cultura

# HTML para el encabezado
header_html = f"""
<div style="display: flex; justify-content: space-between; align-items:center; padding: 10px; background-color: #0e1117; color: white; border-radius: 10px;">
    <img src="{logo_blanco}" style="height: 60px; filter: invert(0);"> 
    <div style="text-align: center;">
        <h2 style="margin:0;">Maestría en Inteligencia Artificial Aplicada</h2>
        <h4 style="margin:0;">Universidad Autónoma de Occidente</h4>
    </div>
    <img src="{logo_oscuro}" style="height: 60px;">
</div>
<div style="text-align: center; margin-top: 20px;">
    <h3>🏡 Predicción de Precios de Vivienda (California)</h3>
    <p><b>Profesor:</b> Dr. Raul Oramas Bustillos | <b>Alumno:</b> Psic. Andres Cruz Degante</p>
</div>
"""

# Construcción de la App con Blocks (permite diseño flexible)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(header_html)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Características de la Propiedad")
            longitude = gr.Slider(-124.35, -114.31, value=-118.0, label="Longitud")
            latitude = gr.Slider(32.54, 41.95, value=34.0, label="Latitud")
            housing_age = gr.Slider(1, 52, value=15, step=1, label="Antigüedad (Años)")
            income = gr.Slider(0.5, 15.0, value=5.0, label="Ingreso Medio (Deciles)")
            
            with gr.Row():
                rooms = gr.Number(value=2000, label="Total Habitaciones")
                bedrooms = gr.Number(value=400, label="Total Dormitorios")
            
            with gr.Row():
                pop = gr.Number(value=1000, label="Población Zona")
                house = gr.Number(value=300, label="Hogares (Households)")
                
            ocean = gr.Dropdown(
                choices=['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'], 
                value='<1H OCEAN', 
                label="Proximidad al Océano"
            )
            
            btn = gr.Button("🔮 Calcular Precio", variant="primary")

        with gr.Column():
            gr.Markdown("### 💰 Resultado de la Tasación")
            output = gr.Label(label="Precio Estimado")
            gr.Markdown("""
            *Nota: Este modelo utiliza un algoritmo de Machine Learning optimizado.
            Los valores son estimaciones basadas en datos históricos del censo de 1990.*
            """)

    # Conectar el botón con la función
    btn.click(
        fn=predecir_precio, 
        inputs=[longitude, latitude, housing_age, rooms, bedrooms, pop, house, income, ocean], 
        outputs=output
    )

# 4. Lanzar la App
if __name__ == "__main__":
    demo.launch()