# Dashboard Interactivo - Proyecto Final IA I
# Autor: Roberto González Sandoval
# Archivo: Gonzalez_Sandoval_PF.py

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Configuración general del dashboard
st.set_page_config(page_title="Dashboard Auto MPG - IA I", layout="wide")

PRIMARY = "#0A2342"
SECONDARY = "#1D3557"
ACCENT = "#457B9D"

st.markdown(
    f"""
    <style>
    .main {{
        background-color: #F7F9FC;
    }}
    h1, h2, h3 {{
        color: {PRIMARY};
    }}
    
    /* Sidebar - Navegación con letras claras */
    [data-testid="stSidebar"] {{
        background-color: #2C3E50 !important;
    }}
    
    /* Título de Navegación en el sidebar */
    [data-testid="stSidebar"] label {{
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }}
    
    /* Radio buttons del sidebar con texto blanco */
    [data-testid="stSidebar"] [role="radiogroup"] label {{
        color: #FFFFFF !important;
    }}
    
    /* Texto del sidebar */
    [data-testid="stSidebar"] p {{
        color: #E0E0E0 !important;
    }}
    
    /* Cambiar color de etiquetas de sliders a negro */
    .main label[data-testid="stWidgetLabel"] {{
        color: #000000 !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }}
    
    /* Texto de selectbox en blanco - múltiples selectores */
    .main [data-baseweb="select"] {{
        color: #FFFFFF !important;
        background-color: transparent !important;
    }}
    
    .main [data-baseweb="select"] div {{
        color: #FFFFFF !important;
    }}
    
    .main [data-baseweb="select"] span {{
        color: #FFFFFF !important;
    }}
    
    /* Valor del selectbox en blanco */
    .main div[data-baseweb="select"] > div {{
        color: #FFFFFF !important;
    }}
    
    /* Input del selectbox */
    .main [role="button"][aria-haspopup="listbox"] {{
        color: #FFFFFF !important;
    }}
    
    /* Contenedor del selectbox */
    .main .stSelectbox > div > div {{
        color: #FFFFFF !important;
    }}
    
    /* Asegurar que los subtítulos y texto general sean negros */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
        color: {PRIMARY} !important;
    }}
    
    /* Mejorar visibilidad de los valores de los sliders */
    .stSlider {{
        padding-top: 10px !important;
    }}
    
    /* Labels de métricas en negro */
    [data-testid="stMetricLabel"] {{
        color: #000000 !important;
    }}
    
    /* Valores de métricas en negro */
    [data-testid="stMetricValue"] {{
        color: #000000 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


# Carga de datos con limpieza básica
@st.cache_data
def load_data():
    df = pd.read_csv("auto-mpg.csv")
    df = df.replace("?", np.nan)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df = df.dropna(subset=["mpg", "cylinders", "displacement", "horsepower",
                           "weight", "acceleration", "model year", "origin"])
    df["origin"] = df["origin"].astype(int)
    return df


df = load_data()


# Entrenamiento de modelos
X = df[["weight", "cylinders", "displacement", "horsepower",
        "acceleration", "model year", "origin"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo simple (baseline)
model_simple = LinearRegression()
model_simple.fit(X_train[["weight"]], y_train)

# Modelo multivariado
model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

# Modelo polinómico (grado 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)


# Función para métricas
def metrics(model, X_t, y_t):
    preds = model.predict(X_t)
    mse = mean_squared_error(y_t, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_t, preds)
    return mse, rmse, r2


mse_s, rmse_s, r2_s = metrics(model_simple, X_test[["weight"]], y_test)
mse_m, rmse_m, r2_m = metrics(model_multi, X_test, y_test)
mse_p, rmse_p, r2_p = metrics(model_poly, X_test_poly, y_test)


# Menú lateral
section = st.sidebar.radio(
    "Navegación",
    ["1. Contexto", "2. Análisis Exploratorio", "3. Evaluación del Modelo", "4. Simulador Interactivo"]
)


# -------- CONTEXTO --------
if section == "1. Contexto":
    st.title("Predicción de Consumo de Combustible (MPG)")
    st.subheader("Descripción general del proyecto")
    
    st.markdown("")
    
    st.markdown("""
        <div style='background-color: #2C3E50; padding: 25px; border-radius: 10px; color: white;'>
            <p style='color: white !important; font-size: 16px; line-height: 1.8; margin-bottom: 15px;'>
                El objetivo de este proyecto es predecir el <strong>consumo de combustible (MPG - Millas por Galón)</strong> 
                de un vehículo a partir de sus características mecánicas y técnicas. Las variables predictoras incluyen:
            </p>
            <ul style='color: white !important; font-size: 15px; line-height: 1.6;'>
                <li><strong>Peso del vehículo</strong> (weight): Peso total en libras</li>
                <li><strong>Cilindrada del motor</strong> (displacement): Volumen de los cilindros</li>
                <li><strong>Caballos de fuerza</strong> (horsepower): Potencia del motor</li>
                <li><strong>Aceleración</strong> (acceleration): Tiempo de 0 a 60 mph</li>
                <li><strong>Número de cilindros</strong> (cylinders): Cantidad de cilindros</li>
                <li><strong>Año del modelo</strong> (model year): Año de fabricación</li>
                <li><strong>Origen</strong> (origin): País de fabricación (USA, Europa, Japón)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    st.markdown("")
    
    st.markdown("""
        <div style='background-color: #2C3E50; padding: 15px; border-radius: 10px;'>
            <p style='color: white; font-size: 16px; font-weight: 600; margin: 0;'>
                Vista previa del dataset
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown(f"""
        <p style='color: #000000; font-size: 15px; line-height: 1.6;'>
            El dataset contiene <strong>{len(df)} registros</strong> con <strong>{len(df.columns)} variables</strong>. 
            A continuación se muestran las primeras filas para visualizar la estructura de los datos utilizados 
            en el entrenamiento de los modelos predictivos.
        </p>
    """, unsafe_allow_html=True)
    st.dataframe(df.head(750), use_container_width=True, height=450)


# -------- EDA --------
elif section == "2. Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos (EDA)")
    
    st.markdown("""
        <p style='color: #000000; font-size: 16px; line-height: 1.7;'>
            El análisis exploratorio permite comprender la distribución de las variables y las relaciones 
            entre ellas antes de construir los modelos predictivos. Esto ayuda a identificar patrones, 
            valores atípicos y correlaciones relevantes para la predicción del consumo de combustible.
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de MPG")
        st.markdown("""
            <p style='color: #000000; font-size: 14px; line-height: 1.6; margin-bottom: 15px;'>
                Este histograma muestra cómo se distribuyen los valores de MPG en el dataset. 
                Permite identificar si hay vehículos que gastan mucho (MPG bajo) o muy eficientes (MPG alto), 
                y dónde se concentra la mayoría de los vehículos.
            </p>
        """, unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df["mpg"], bins=20, color=ACCENT, ax=ax)
        ax.set_xlabel("MPG (Millas por Galón)")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)
        st.markdown("""
            <p style='color: #555555; font-size: 13px; font-style: italic; margin-top: 10px;'>
                <strong>Observación:</strong> La mayoría de los vehículos se concentran entre 15 y 30 MPG, 
                con una distribución ligeramente sesgada hacia valores más altos, indicando una mayor 
                presencia de vehículos eficientes en el dataset.
            </p>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("Mapa de Correlación")
        st.markdown("""
            <p style='color: #000000; font-size: 14px; line-height: 1.6; margin-bottom: 15px;'>
                El mapa de correlación identifica qué variables están más relacionadas con el MPG. 
                Tonalidades más oscuras indican correlaciones más fuertes.
            </p>
        """, unsafe_allow_html=True)
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, cmap="Blues", annot=False, ax=ax)
        st.pyplot(fig)
        st.markdown("""
            <p style='color: #555555; font-size: 13px; font-style: italic; margin-top: 10px;'>
                <strong>Interpretación:</strong> El peso (weight) y los caballos de fuerza (horsepower) 
                presentan correlación negativa fuerte con MPG (vehículos más pesados y potentes consumen más). 
                El año del modelo (model year) muestra correlación positiva, reflejando mejoras tecnológicas 
                en eficiencia a lo largo del tiempo.
            </p>
        """, unsafe_allow_html=True)


# -------- EVALUACIÓN DE MODELOS --------
elif section == "3. Evaluación del Modelo":
    st.title("Evaluación de los Modelos")
    
    st.markdown("""
        <p style='color: #000000; font-size: 16px; line-height: 1.8; margin-bottom: 20px;'>
            Se comparan <strong>tres enfoques de regresión</strong> para determinar cuál ofrece 
            el mejor desempeño predictivo:
        </p>
    """, unsafe_allow_html=True)
    
    col_desc1, col_desc2, col_desc3 = st.columns(3)
    
    with col_desc1:
        st.markdown("""
            <div style='background-color: #E8F4F8; padding: 15px; border-radius: 8px; height: 180px;'>
                <h4 style='color: #0A2342; margin-top: 0;'>Lineal Simple</h4>
                <p style='color: #000000; font-size: 13px; line-height: 1.5;'>
                    Usa únicamente el <strong>peso del vehículo</strong> como predictor. 
                    Sirve como baseline sencillo para comparación.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_desc2:
        st.markdown("""
            <div style='background-color: #E8F4F8; padding: 15px; border-radius: 8px; height: 180px;'>
                <h4 style='color: #0A2342; margin-top: 0;'>Multivariada</h4>
                <p style='color: #000000; font-size: 13px; line-height: 1.5;'>
                    Incorpora <strong>todas las características</strong> del vehículo 
                    (peso, cilindros, potencia, etc.) para capturar más información.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_desc3:
        st.markdown("""
            <div style='background-color: #E8F4F8; padding: 15px; border-radius: 8px; height: 180px;'>
                <h4 style='color: #0A2342; margin-top: 0;'>Polinómica (Grado 2)</h4>
                <p style='color: #000000; font-size: 13px; line-height: 1.5;'>
                    Permite representar <strong>relaciones no lineales</strong> entre variables, 
                    capturando patrones más complejos.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    st.markdown("")

    st.markdown("""
        <div style='background-color: #2C3E50; padding: 15px; border-radius: 10px;'>
            <p style='color: white; font-size: 16px; font-weight: 600; margin: 0;'>
                Comparación de métricas de desempeño
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")

    metrics_df = pd.DataFrame({
        "Modelo": ["Lineal Simple", "Multivariada", "Polinómica (Grado 2)"],
        "MSE": [mse_s, mse_m, mse_p],
        "RMSE": [rmse_s, rmse_m, rmse_p],
        "R²": [r2_s, r2_m, r2_p]
    })

    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("")
    
    # Interpretación automática
    best_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Modelo']
    best_r2 = metrics_df['R²'].max()
    best_rmse = metrics_df['RMSE'].min()
    
    st.markdown(f"""
        <div style='background-color: #D4EDDA; padding: 20px; border-radius: 10px; border-left: 5px solid #28A745;'>
            <h4 style='color: #155724; margin-top: 0;'>Interpretación de Resultados</h4>
            <p style='color: #000000; font-size: 15px; line-height: 1.7;'>
                El modelo <strong>{best_model}</strong> presenta el <strong>mayor R² ({best_r2:.4f})</strong> y el 
                <strong>menor RMSE ({best_rmse:.4f})</strong>, lo que indica que:
            </p>
            <ul style='color: #000000; font-size: 14px; line-height: 1.6;'>
                <li>Explica mejor la variabilidad del consumo de combustible (MPG)</li>
                <li>Comete menos error promedio en las predicciones</li>
                <li>Es el modelo más adecuado para implementar en el simulador interactivo</li>
            </ul>
            <p style='color: #000000; font-size: 14px; margin-top: 15px;'>
                <strong>Nota:</strong> Un R² cercano a 1 indica un excelente ajuste, mientras que un RMSE bajo 
                significa predicciones más precisas.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("---")
    st.markdown("")

    st.subheader("Gráfica Real vs Predicho (Modelo Polinómico)")
    
    st.markdown("""
        <p style='color: #000000; font-size: 14px; line-height: 1.6; margin-bottom: 15px;'>
            Esta gráfica compara los valores reales de MPG contra las predicciones del modelo. 
            <strong>Mientras más cerca estén los puntos de la diagonal, mejor es el ajuste del modelo.</strong> 
            Una buena predicción implica que los puntos se alinean estrechamente con la línea diagonal ideal.
        </p>
    """, unsafe_allow_html=True)
    
    preds_poly = model_poly.predict(X_test_poly)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=preds_poly, color=PRIMARY, ax=ax, alpha=0.6)
    
    # Línea diagonal de referencia
    min_val = min(y_test.min(), preds_poly.min())
    max_val = max(y_test.max(), preds_poly.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción perfecta')
    
    ax.set_xlabel("Valores Reales (MPG)", fontsize=12)
    ax.set_ylabel("Predicciones (MPG)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("""
        <p style='color: #555555; font-size: 13px; font-style: italic; margin-top: 10px;'>
            <strong>Observación:</strong> Los puntos se distribuyen cerca de la línea diagonal, 
            confirmando que el modelo polinómico captura adecuadamente la relación entre las 
            características del vehículo y su consumo de combustible.
        </p>
    """, unsafe_allow_html=True)


# -------- SIMULADOR --------
elif section == "4. Simulador Interactivo":
    st.title("Simulador Interactivo de MPG")
    st.markdown("")
    st.markdown("---")
    st.markdown("")
    
    # Sección 1: Características Físicas del Vehículo
    st.subheader("Características Físicas del Vehículo")
    st.markdown('<p style="color: #000000; font-size: 14px;">Ajuste las características principales del vehículo</p>', unsafe_allow_html=True)
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.slider(
            label="Peso del Vehículo (lbs)",
            min_value=1500,
            max_value=5000,
            value=3000,
            step=50,
            key="weight",
            help="Peso total del vehículo en libras"
        )
        
        st.markdown("")
        
        displacement = st.slider(
            label="Cilindrada del Motor (cc)",
            min_value=60,
            max_value=500,
            value=200,
            step=10,
            key="displacement",
            help="Volumen total de los cilindros del motor"
        )
        
    with col2:
        horsepower = st.slider(
            label="Caballos de Fuerza (HP)",
            min_value=40,
            max_value=240,
            value=100,
            step=5,
            key="horsepower",
            help="Potencia máxima del motor"
        )
        
        st.markdown("")
        
        acceleration = st.slider(
            label="Aceleración 0-60 mph (segundos)",
            min_value=8,
            max_value=24,
            value=15,
            step=1,
            key="acceleration",
            help="Tiempo en segundos para acelerar de 0 a 60 mph"
        )
    
    st.markdown("---")
    
    # Sección 2: Especificaciones del Modelo
    st.subheader("Especificaciones del Modelo")
    st.markdown('<p style="color: #000000; font-size: 14px;">Configure las especificaciones técnicas</p>', unsafe_allow_html=True)
    st.markdown("")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        cylinders = st.selectbox(
            label="Número de Cilindros",
            options=[3, 4, 5, 6, 8],
            index=1,
            key="cylinders",
            help="Cantidad de cilindros del motor"
        )
        
    with col4:
        model_year = st.slider(
            label="Año del Modelo (19XX)",
            min_value=70,
            max_value=82,
            value=76,
            step=1,
            key="model_year",
            help="Año de fabricación del vehículo",
            format="19%d"
        )
        
    with col5:
        origin = st.selectbox(
            label="Origen del Vehículo",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Estados Unidos", 2: "Europa", 3: "Japón"}[x],
            key="origin",
            help="País o región de fabricación"
        )
    
    st.markdown("---")
    
    # Sección 3: Selección del Modelo de Predicción
    st.subheader("Modelo de Predicción")
    st.markdown('<p style="color: #000000; font-size: 14px;">Seleccione el algoritmo de machine learning a utilizar</p>', unsafe_allow_html=True)
    st.markdown("")
    
    modelo = st.selectbox(
        label="Tipo de Modelo",
        options=["Lineal Simple", "Multivariada", "Polinómica"],
        index=2,
        key="modelo",
        help="Seleccione el algoritmo para la predicción"
    )
    
    # Mostrar descripción del modelo seleccionado
    if modelo == "Lineal Simple":
        st.markdown("""
            <div style='background-color: #D1ECF1; padding: 15px; border-radius: 5px; border-left: 4px solid #0C5460;'>
                <p style='color: #000000; margin: 0; font-size: 14px;'>
                    <strong>Modelo Lineal Simple:</strong> Utiliza únicamente el peso del vehículo como predictor. Es el más simple pero menos preciso.
                </p>
            </div>
        """, unsafe_allow_html=True)
    elif modelo == "Multivariada":
        st.markdown("""
            <div style='background-color: #D1ECF1; padding: 15px; border-radius: 5px; border-left: 4px solid #0C5460;'>
                <p style='color: #000000; margin: 0; font-size: 14px;'>
                    <strong>Modelo Multivariada:</strong> Utiliza todas las características del vehículo (peso, cilindros, potencia, etc.) para una predicción más completa.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style='background-color: #D1ECF1; padding: 15px; border-radius: 5px; border-left: 4px solid #0C5460;'>
                <p style='color: #000000; margin: 0; font-size: 14px;'>
                    <strong>Modelo Polinómica:</strong> Utiliza regresión polinómica de grado 2, capturando relaciones no lineales entre variables. Es el más preciso.
                </p>
            </div>
        """, unsafe_allow_html=True)

    X_input = pd.DataFrame({
        "weight": [weight],
        "cylinders": [cylinders],
        "displacement": [displacement],
        "horsepower": [horsepower],
        "acceleration": [acceleration],
        "model year": [model_year],
        "origin": [origin]
    })

    st.markdown("---")
    
    if modelo == "Lineal Simple":
        pred = model_simple.predict(X_input[["weight"]])[0]
    elif modelo == "Multivariada":
        pred = model_multi.predict(X_input)[0]
    else:
        pred = model_poly.predict(poly.transform(X_input))[0]

    # Mostrar predicción con estilo
    st.markdown("### Resultado de la Predicción")
    
    col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
    with col_pred2:
        st.markdown(
            f"""
            <div style='text-align: center; padding: 30px; background-color: {ACCENT}; 
                        border-radius: 15px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <p style='color: white; margin: 0; font-size: 1.2em; font-weight: 500;'>PREDICCIÓN ESTIMADA</p>
                <h1 style='color: white; font-size: 4em; margin: 20px 0; font-weight: bold;'>{pred:.2f}</h1>
                <p style='color: white; margin: 0; font-size: 1.5em;'>MPG (Millas por Galón)</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("")
    
    # Información adicional
    st.markdown("")
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("Modelo Utilizado", modelo)
    
    with col_info2:
        eficiencia = "Alta" if pred > 25 else "Media" if pred > 18 else "Baja"
        st.metric("Eficiencia", eficiencia)
    
    with col_info3:
        litros_100km = 235.214 / pred  # Conversión de MPG a L/100km
        st.metric("Litros/100km", f"{litros_100km:.2f}")
