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
    st.subheader("Descripción general")
    
    st.markdown("")
    
    st.markdown("""
        <div style='background-color: #2C3E50; padding: 20px; border-radius: 10px; color: white;'>
            <p style='color: white; font-size: 16px; line-height: 1.6;'>
                El objetivo es estimar el rendimiento de combustible (MPG) de un vehículo
                a partir de sus características mecánicas. El dataset Auto MPG incluye
                atributos como cilindros, peso, desplazamiento y potencia, variables
                directamente relacionadas con la eficiencia del motor.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("")
    
    st.markdown("""
        <div style='background-color: #2C3E50; padding: 15px; border-radius: 10px;'>
            <p style='color: white; font-size: 16px; font-weight: 600; margin: 0;'>
                Vista previa del dataset:
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    st.dataframe(df.head(1000), use_container_width=True, height=500)


# -------- EDA --------
elif section == "2. Análisis Exploratorio":
    st.title("Análisis Exploratorio de Datos (EDA)")
    
    st.markdown("")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de MPG")
        fig, ax = plt.subplots(figsize=(5, 3))
        sns.histplot(df["mpg"], bins=20, color=ACCENT, ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Mapa de Correlación")
        corr = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, cmap="Blues", annot=False)
        st.pyplot(fig)


# -------- EVALUACIÓN DE MODELOS --------
elif section == "3. Evaluación del Modelo":
    st.title("Evaluación de los Modelos")
    
    st.markdown("")

    st.markdown("""
        <div style='background-color: #2C3E50; padding: 15px; border-radius: 10px;'>
            <p style='color: white; font-size: 16px; font-weight: 600; margin: 0;'>
                Comparación de métricas para cada modelo:
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
    st.markdown("")

    st.subheader("Real vs Predicho (Modelo Polinómico)")
    preds_poly = model_poly.predict(X_test_poly)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x=y_test, y=preds_poly, color=PRIMARY, ax=ax)
    ax.set_xlabel("Valores Reales (MPG)")
    ax.set_ylabel("Predicciones (MPG)")
    st.pyplot(fig)


# -------- SIMULADOR --------
elif section == "4. Simulador Interactivo":
    st.title("Simulador Interactivo de MPG")
    st.markdown('<p style="color: #000000; font-size: 16px;">Configure las características del vehículo para obtener una predicción de consumo de combustible.</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
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
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.metric("Modelo Utilizado", modelo)
    
    with col_info2:
        eficiencia = "Alta" if pred > 25 else "Media" if pred > 18 else "Baja"
        st.metric("Eficiencia", eficiencia)
    
    with col_info3:
        litros_100km = 235.214 / pred  # Conversión de MPG a L/100km
        st.metric("Litros/100km", f"{litros_100km:.2f}")
