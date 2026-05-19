from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    Prophet = None
    PROPHET_AVAILABLE = False

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*Maximum Likelihood optimization failed.*")


# =========================================================
# Configuración de página
# =========================================================
st.set_page_config(
    page_title="CRISP-DM Ponte Selva",
    page_icon=None,
    layout="wide"
)


# =========================================================
# Estilos simples
# =========================================================
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 800;
        color: #0B2E5F;
        margin-bottom: 0px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        font-weight: 600;
        color: #0B2E5F;
        margin-top: 10px;
        line-height: 1.45;
    }
    .author {
        text-align: center;
        font-size: 18px;
        font-weight: 700;
        color: #1F2937;
        margin-top: 20px;
    }
    .section-box {
        border: 1px solid #D0D7E2;
        border-radius: 10px;
        padding: 18px;
        background-color: #F8FAFC;
        margin-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# Constantes
# =========================================================
REQUIRED_COLUMNS = ["fecha_enc_vent", "score", "canal_venta"]
VALID_CHANNELS = ["WA", "PRE", "TEL", "MAIL"]
DATASET_FILE = Path("dataset_simulado_retroalimentacion_calidad_ventas_canales_codificados.csv")
FREQUENCY_OPTIONS = ["Diaria", "Semanal", "Mensual"]
FREQ_MAP = {"Diaria": "D", "Semanal": "W", "Mensual": "M"}


# =========================================================
# Funciones auxiliares generales
# =========================================================
def show_cover():
    st.markdown('<div class="main-title">PROYECTO DE TITULACIÓN</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="subtitle">
        DISEÑO DE UNA ARQUITECTURA EMPRESARIAL BASADA EN TOGAF QUE INTEGRE LA METODOLOGÍA CRISP-DM
        PARA OPTIMIZACIÓN DE LOS PROCESOS DE UNA EMPRESA DEL SECTOR TEXTIL.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="author">
        AUTOR:<br>
        PABLO GABRIEL JARRÍN MERA
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.write(
        "Esta aplicación web permite desarrollar las fases de CRISP-DM aplicadas al análisis de "
        "retroalimentación post venta de clientes de Ponte Selva. El sistema permite cargar el dataset, "
        "comprender los datos, prepararlos, construir modelos de series de tiempo, evaluar su desempeño "
        "y presentar resultados para apoyar la mejora del proceso comercial."
    )


def read_uploaded_file(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin-1")


def validate_columns(df):
    return [col for col in REQUIRED_COLUMNS if col not in df.columns]


def show_dataset_download_button():
    """Muestra un botón para descargar el dataset incluido en el repositorio."""
    if DATASET_FILE.exists():
        dataset_bytes = DATASET_FILE.read_bytes()
        st.download_button(
            label="Descargar dataset de ejemplo",
            data=dataset_bytes,
            file_name=DATASET_FILE.name,
            mime="text/csv"
        )
    else:
        st.warning(
            "No se encontró el dataset de ejemplo en el servidor. "
            "Verifica que el archivo CSV esté en la misma carpeta que app.py."
        )


def prepare_basic_dataframe(df):
    data = df[["fecha_enc_vent", "score", "canal_venta"]].copy()
    data["fecha_enc_vent"] = pd.to_datetime(data["fecha_enc_vent"], errors="coerce")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["canal_venta"] = data["canal_venta"].astype(str).str.strip().str.upper()
    return data


def get_quality_summary(df):
    return {
        "registros": len(df),
        "columnas": len(df.columns),
        "nulos_totales": int(df.isna().sum().sum()),
        "duplicados": int(df.duplicated().sum()),
        "fechas_invalidas": int(df["fecha_enc_vent"].isna().sum()),
        "scores_invalidos": int(df[(df["score"].notna()) & ((df["score"] < 1) | (df["score"] > 10))].shape[0]),
        "canales_invalidos": int(df[~df["canal_venta"].isin(VALID_CHANNELS)].shape[0])
    }


def clean_for_modeling(df):
    data = df[["fecha_enc_vent", "score", "canal_venta"]].copy()
    data = data.dropna(subset=["fecha_enc_vent", "score"])
    data = data[(data["score"] >= 1) & (data["score"] <= 10)]
    data = data[data["canal_venta"].isin(VALID_CHANNELS)]
    data = data.sort_values("fecha_enc_vent").reset_index(drop=True)
    return data


def detect_outliers_iqr(df_clean):
    q1 = df_clean["score"].quantile(0.25)
    q3 = df_clean["score"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df_clean[(df_clean["score"] < lower) | (df_clean["score"] > upper)].copy()
    return {"q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper, "outliers": outliers}


def create_time_series(df_clean, frequency="D"):
    """
    Crea una tabla auxiliar para series de tiempo.
    Permite trabajar con frecuencia diaria, semanal o mensual.

    Además del score promedio por periodo, se generan variables explicativas externas
    a partir del propio dataset:
    - cantidad de encuestas del periodo
    - proporción de ventas por canal: WA, PRE, TEL, MAIL

    Estas variables se usan como regresores externos en SARIMAX y Prophet.
    """
    freq_map = {
        "Diaria": "D",
        "Semanal": "W",
        "Mensual": "M"
    }

    freq = freq_map.get(frequency, "D")

    data = df_clean.copy()
    data = data.set_index("fecha_enc_vent").sort_index()

    score_series = data["score"].resample(freq).mean()
    count_series = data["score"].resample(freq).count()

    channel_counts = (
        data
        .groupby([pd.Grouper(freq=freq), "canal_venta"])
        .size()
        .unstack(fill_value=0)
    )

    for channel in VALID_CHANNELS:
        if channel not in channel_counts.columns:
            channel_counts[channel] = 0

    channel_counts = channel_counts[VALID_CHANNELS]

    total_by_period = channel_counts.sum(axis=1).replace(0, np.nan)

    channel_props = channel_counts.div(total_by_period, axis=0).fillna(0)
    channel_props = channel_props.rename(columns={
        "WA": "prop_WA",
        "PRE": "prop_PRE",
        "TEL": "prop_TEL",
        "MAIL": "prop_MAIL"
    })

    series = pd.concat(
        [
            score_series.rename("score_original"),
            count_series.rename("encuestas_periodo"),
            channel_props
        ],
        axis=1
    ).reset_index()

    if series.empty:
        return series

    series["tiene_encuesta"] = series["score_original"].notna()

    series["score_preparado"] = (
        series["score_original"]
        .interpolate(method="linear")
        .bfill()
        .ffill()
    )

    # Variables externas. Si no hay encuestas en un periodo, se completan con valores recientes.
    exog_cols = ["encuestas_periodo", "prop_WA", "prop_PRE", "prop_TEL", "prop_MAIL"]

    for col in exog_cols:
        series[col] = pd.to_numeric(series[col], errors="coerce")
        series[col] = series[col].replace([np.inf, -np.inf], np.nan)
        series[col] = series[col].ffill().bfill().fillna(0)

    series["t"] = np.arange(len(series))
    series["mes"] = series["fecha_enc_vent"].dt.month
    series["trimestre"] = series["fecha_enc_vent"].dt.quarter
    series["dia_semana"] = series["fecha_enc_vent"].dt.dayofweek

    series["media_movil_3"] = (
        series["score_preparado"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    series["media_movil_7"] = (
        series["score_preparado"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    series["media_movil_30"] = (
        series["score_preparado"]
        .rolling(window=30, min_periods=1)
        .mean()
    )

    return series



def create_future_dataframe(series, horizon, frequency):
    """
    Crea el dataframe futuro con las variables externas necesarias para SARIMAX y Prophet.
    Como no se conocen las condiciones futuras, se usan los últimos valores observados.
    """
    freq_map = {
        "Diaria": "D",
        "Semanal": "W",
        "Mensual": "M"
    }

    freq = freq_map.get(frequency, "D")

    last_date = series["fecha_enc_vent"].max()
    future_dates = pd.date_range(
        last_date + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq
    )

    future = pd.DataFrame({
        "fecha_enc_vent": future_dates
    })

    exog_cols = get_exog_columns()

    for col in exog_cols:
        if col in series.columns:
            future[col] = float(series[col].iloc[-1])
        else:
            future[col] = 0.0

    return future


def get_exog_columns():
    """
    Variables explicativas externas construidas a partir del propio dataset.
    """
    return ["encuestas_periodo", "prop_WA", "prop_PRE", "prop_TEL", "prop_MAIL"]



def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def get_seasonal_period(frequency):
    """
    Define el periodo estacional sugerido según la frecuencia de la serie.
    """
    if frequency == "Diaria":
        return 7      # estacionalidad semanal
    if frequency == "Semanal":
        return 4      # estacionalidad mensual aproximada
    if frequency == "Mensual":
        return 12     # estacionalidad anual
    return 7


def evaluate_forecast(y_true, y_pred):
    """
    Calcula métricas cuidando que las predicciones estén dentro de la escala 1 a 10.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1, 10)
    return compute_metrics(y_true, y_pred), y_pred


def get_sarimax_exog_sets(series):
    """
    Define combinaciones de variables externas para SARIMAX.
    Se prueban varias opciones porque no siempre todas las variables externas mejoran el modelo.
    """
    available = [c for c in get_exog_columns() if c in series.columns]

    exog_sets = {
        "sin variables externas": [],
        "volumen de encuestas": [c for c in ["encuestas_periodo"] if c in available],
        "canales de venta": [c for c in ["prop_WA", "prop_PRE", "prop_TEL", "prop_MAIL"] if c in available],
        "volumen y canales": available
    }

    # Eliminar combinaciones vacías duplicadas
    cleaned = {}
    seen = set()
    for name, cols in exog_sets.items():
        key = tuple(cols)
        if key not in seen:
            cleaned[name] = cols
            seen.add(key)

    return cleaned


def sarimax_forecast_one_step(train, test, order, seasonal_order, exog_cols):
    """
    Evaluación rolling one-step-ahead para SARIMAX.

    En lugar de pronosticar todo el bloque de prueba de una sola vez, el modelo predice
    un periodo hacia adelante y luego incorpora el valor real observado para continuar.
    Esta forma de evaluación suele ser más adecuada para series de tiempo operativas.
    """
    y_train = train["score_preparado"].astype(float)
    y_test = test["score_preparado"].astype(float)

    if len(exog_cols) > 0:
        exog_train = train[exog_cols].astype(float)
        exog_test = test[exog_cols].astype(float)
    else:
        exog_train = None
        exog_test = None

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False, maxiter=250)

    predictions = []

    for i in range(len(test)):
        if exog_test is not None:
            step_exog = exog_test.iloc[[i]]
            pred = fitted.forecast(steps=1, exog=step_exog)
            fitted = fitted.append(
                endog=pd.Series([y_test.iloc[i]]),
                exog=step_exog,
                refit=False
            )
        else:
            pred = fitted.forecast(steps=1)
            fitted = fitted.append(
                endog=pd.Series([y_test.iloc[i]]),
                refit=False
            )

        predictions.append(float(pred.iloc[0] if hasattr(pred, "iloc") else pred[0]))

    return np.asarray(predictions, dtype=float)


def sarimax_future_forecast(series, future, order, seasonal_order, exog_cols):
    """
    Entrena SARIMAX con toda la serie y genera el pronóstico futuro.
    """
    y_full = series["score_preparado"].astype(float)

    if len(exog_cols) > 0:
        exog_full = series[exog_cols].astype(float)
        exog_future = future[exog_cols].astype(float)
    else:
        exog_full = None
        exog_future = None

    model = SARIMAX(
        y_full,
        exog=exog_full,
        order=order,
        seasonal_order=seasonal_order,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False, maxiter=250)
    future_pred = fitted.forecast(steps=len(future), exog=exog_future)

    return np.asarray(future_pred, dtype=float)


def prophet_test_forecast(train, test, exog_cols, frequency):
    """
    Entrena Prophet con regresores externos y pronostica el conjunto de prueba.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet no está instalado.")

    prophet_train = pd.DataFrame({
        "ds": train["fecha_enc_vent"],
        "y": train["score_preparado"]
    })

    for col in exog_cols:
        prophet_train[col] = train[col].astype(float).values

    model = Prophet(
        weekly_seasonality=(frequency == "Diaria"),
        yearly_seasonality=(frequency in ["Diaria", "Semanal", "Mensual"]),
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5.0
    )

    if frequency == "Mensual":
        model.add_seasonality(name="annual", period=365.25, fourier_order=5)
    elif frequency == "Semanal":
        model.add_seasonality(name="monthly_approx", period=30.5, fourier_order=3)

    for col in exog_cols:
        model.add_regressor(col)

    model.fit(prophet_train)

    future_test = pd.DataFrame({
        "ds": test["fecha_enc_vent"]
    })

    for col in exog_cols:
        future_test[col] = test[col].astype(float).values

    forecast = model.predict(future_test)

    return np.asarray(forecast["yhat"], dtype=float)


def prophet_future_forecast(series, future, exog_cols, frequency):
    """
    Entrena Prophet con toda la serie y genera pronóstico futuro.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet no está instalado.")

    prophet_full = pd.DataFrame({
        "ds": series["fecha_enc_vent"],
        "y": series["score_preparado"]
    })

    for col in exog_cols:
        prophet_full[col] = series[col].astype(float).values

    model = Prophet(
        weekly_seasonality=(frequency == "Diaria"),
        yearly_seasonality=(frequency in ["Diaria", "Semanal", "Mensual"]),
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=5.0
    )

    if frequency == "Mensual":
        model.add_seasonality(name="annual", period=365.25, fourier_order=5)
    elif frequency == "Semanal":
        model.add_seasonality(name="monthly_approx", period=30.5, fourier_order=3)

    for col in exog_cols:
        model.add_regressor(col)

    model.fit(prophet_full)

    future_prophet = pd.DataFrame({
        "ds": future["fecha_enc_vent"]
    })

    for col in exog_cols:
        future_prophet[col] = future[col].astype(float).values

    forecast = model.predict(future_prophet)

    return np.asarray(forecast["yhat"], dtype=float)


def train_and_compare_models(series, frequency="Semanal", horizon=12, include_prophet=False):
    """
    Entrena y compara modelos de series de tiempo de forma optimizada.

    Para evitar que la aplicación se quede procesando demasiado tiempo:
    - Se prueban pocas configuraciones SARIMAX.
    - Se usa una sola predicción del conjunto de prueba, no rolling por cada fila.
    - Prophet queda como opción activable, porque puede demorar bastante en algunos equipos.

    Modelos evaluados:
    - SARIMAX sin estacionalidad
    - SARIMAX con estacionalidad, cuando hay suficientes datos
    - Prophet con regresores externos, solo si el usuario lo activa
    """
    if series.empty or len(series) < 12:
        return None

    split_idx = int(len(series) * 0.8)

    if split_idx < 8 or split_idx >= len(series):
        return None

    train = series.iloc[:split_idx].copy()
    test = series.iloc[split_idx:].copy()

    y_train = train["score_preparado"].astype(float)
    y_test = test["score_preparado"].astype(float).values

    exog_cols = get_exog_columns()
    seasonal_period = get_seasonal_period(frequency)

    results = []

    # Se usan únicamente los regresores externos más útiles.
    # Esto reduce tiempo de entrenamiento y evita problemas de convergencia.
    exog_candidates = {
        "sin variables externas": [],
        "volumen y canales": exog_cols
    }

    # Configuraciones rápidas y estables.
    sarimax_configs = [
        ((1, 0, 1), (0, 0, 0, 0)),
        ((1, 1, 1), (0, 0, 0, 0))
    ]

    if len(train) >= seasonal_period * 3:
        sarimax_configs.append(
            ((1, 0, 1), (1, 0, 0, seasonal_period))
        )

    for exog_name, cols in exog_candidates.items():
        for order, seasonal_order in sarimax_configs:
            model_name = f"SARIMAX{order}x{seasonal_order} - {exog_name}"

            try:
                exog_train = train[cols].astype(float) if len(cols) > 0 else None
                exog_test = test[cols].astype(float) if len(cols) > 0 else None

                model = SARIMAX(
                    y_train,
                    exog=exog_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    simple_differencing=True
                )

                fitted = model.fit(
                    disp=False,
                    maxiter=60,
                    method="lbfgs"
                )

                pred = fitted.forecast(
                    steps=len(test),
                    exog=exog_test
                )

                metrics, pred = evaluate_forecast(y_test, pred)

                results.append({
                    "Modelo": model_name,
                    "Tipo": "SARIMAX",
                    "MAE": metrics["MAE"],
                    "MSE": metrics["MSE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                    "Predicciones": pred,
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "exog_cols": cols
                })

            except Exception:
                pass

    # Prophet se ejecuta solo si está instalado y el usuario lo activa.
    if include_prophet and PROPHET_AVAILABLE:
        try:
            cols = exog_cols
            pred = prophet_test_forecast(
                train=train,
                test=test,
                exog_cols=cols,
                frequency=frequency
            )

            metrics, pred = evaluate_forecast(y_test, pred)

            results.append({
                "Modelo": "Prophet con regresores externos",
                "Tipo": "Prophet",
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Predicciones": pred,
                "exog_cols": cols
            })

        except Exception:
            pass

    if len(results) == 0:
        return None

    metrics_df = pd.DataFrame([
        {
            "Modelo": item["Modelo"],
            "Tipo": item["Tipo"],
            "MAE": item["MAE"],
            "MSE": item["MSE"],
            "RMSE": item["RMSE"],
            "R2": item["R2"]
        }
        for item in results
    ]).sort_values("RMSE").reset_index(drop=True)

    best_model_name = metrics_df.iloc[0]["Modelo"]
    best_result = [item for item in results if item["Modelo"] == best_model_name][0]

    future = create_future_dataframe(
        series,
        horizon,
        frequency
    )

    if best_result["Tipo"] == "SARIMAX":
        future_values = sarimax_future_forecast(
            series=series,
            future=future,
            order=best_result["order"],
            seasonal_order=best_result["seasonal_order"],
            exog_cols=best_result["exog_cols"]
        )

    elif best_result["Tipo"] == "Prophet":
        future_values = prophet_future_forecast(
            series=series,
            future=future,
            exog_cols=best_result["exog_cols"],
            frequency=frequency
        )

    else:
        future_values = np.repeat(series["score_preparado"].mean(), horizon)

    future_values = np.clip(future_values, 1, 10)

    forecast = pd.DataFrame({
        "fecha_enc_vent": future["fecha_enc_vent"],
        "score_predicho": future_values
    })

    last_real = float(series["score_preparado"].iloc[-1])
    last_forecast = float(forecast["score_predicho"].iloc[-1])
    diff = last_forecast - last_real

    if diff > 0.10:
        trend = "Subir"
    elif diff < -0.10:
        trend = "Bajar"
    else:
        trend = "Mantenerse"

    test_comparison = pd.DataFrame({
        "fecha_enc_vent": test["fecha_enc_vent"].values,
        "score_real": y_test,
        "score_predicho": best_result["Predicciones"]
    })

    return {
        "metrics_table": metrics_df,
        "best_model": best_model_name,
        "best_model_type": best_result["Tipo"],
        "test": test_comparison,
        "forecast": forecast,
        "trend": trend,
        "last_real": last_real,
        "last_forecast": last_forecast,
        "frequency": frequency,
        "horizon": horizon,
        "seasonal_period": seasonal_period,
        "exog_cols": best_result.get("exog_cols", []),
        "prophet_available": PROPHET_AVAILABLE,
        "include_prophet": include_prophet
    }



# =========================================================
# Funciones de gráficos
# =========================================================
def get_chart_size(key, default_width=10, default_height=5):
    st.markdown("Ajuste de tamaño del gráfico")
    c1, c2 = st.columns(2)
    width = c1.slider("Ancho", min_value=6, max_value=18, value=default_width, step=1, key=f"{key}_width")
    height = c2.slider("Alto", min_value=3, max_value=12, value=default_height, step=1, key=f"{key}_height")
    return width, height


def show_chart(fig_func, key, *args, default_width=10, default_height=5, **kwargs):
    width, height = get_chart_size(key, default_width=default_width, default_height=default_height)
    fig = fig_func(*args, width=width, height=height, **kwargs)
    st.pyplot(fig, width="content")


def fig_histogram(df, column, title, xlabel, ylabel, width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.hist(df[column].dropna(), bins=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def fig_boxplot(df, column, title, ylabel, width=8, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.boxplot(df[column].dropna(), vert=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def fig_bar(series, title, xlabel, ylabel, width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def fig_line(df, x, y, title, xlabel, ylabel, width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(df[x], df[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_time_series_preparation(series, width=12, height=6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(series["fecha_enc_vent"], series["score_original"], marker="o", linestyle="", label="Score original por periodo")
    ax.plot(series["fecha_enc_vent"], series["score_preparado"], label="Score preparado")
    ax.plot(series["fecha_enc_vent"], series["media_movil_7"], label="Media móvil 7 periodos")
    ax.set_title("Serie de tiempo preparada")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Score")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_forecast(series, forecast, width=12, height=6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(series["fecha_enc_vent"], series["score_preparado"], label="Serie preparada")
    ax.plot(series["fecha_enc_vent"], series["media_movil_7"], label="Media móvil 7 periodos")
    ax.plot(forecast["fecha_enc_vent"], forecast["score_predicho"], label="Pronóstico")
    ax.set_title("Pronóstico de score post venta")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Score")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_evaluation(test_df, width=12, height=6):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(test_df["fecha_enc_vent"], test_df["score_real"], label="Score real")
    ax.plot(test_df["fecha_enc_vent"], test_df["score_predicho"], label="Score predicho")
    ax.set_title("Evaluación del mejor modelo en conjunto de prueba")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Score")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def fig_metrics_comparison(metrics_df, width=10, height=5):
    fig, ax = plt.subplots(figsize=(width, height))
    metrics_df.set_index("Modelo")["RMSE"].plot(kind="bar", ax=ax)
    ax.set_title("Comparación de modelos SARIMAX y Prophet según RMSE")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def build_report(df, df_clean, series, model_results):
    lines = []
    lines.append("REPORTE CRISP-DM")
    lines.append("Proyecto de titulación: Arquitectura empresarial TOGAF integrada con CRISP-DM")
    lines.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("FASE 2: COMPRENSIÓN DE LOS DATOS")
    lines.append(f"Registros cargados: {len(df)}")
    lines.append(f"Columnas del dataset: {', '.join(df.columns)}")
    lines.append(f"Score promedio: {df['score'].mean():.2f}")
    lines.append(f"Días con encuestas reales: {df['fecha_enc_vent'].nunique()}")
    lines.append("")
    lines.append("FASE 3: PREPARACIÓN DE LOS DATOS")
    lines.append(f"Registros después de limpieza: {len(df_clean)}")
    lines.append("El dataset preparado conserva únicamente las columnas originales: fecha_enc_vent, score y canal_venta.")
    lines.append("Para modelado se generó una tabla auxiliar de series de tiempo con frecuencia configurable.")
    lines.append("")
    lines.append("FASE 4: MODELADO")
    if model_results:
        lines.append(f"Frecuencia de análisis: {model_results['frequency']}")
        lines.append(f"Horizonte de pronóstico: {model_results['horizon']}")
        lines.append(f"Mejor modelo de series de tiempo seleccionado: {model_results['best_model']}")
        lines.append(f"Tendencia futura estimada: {model_results['trend']}")
        lines.append(f"Score actual preparado: {model_results['last_real']:.2f}")
        lines.append(f"Score pronosticado al final del horizonte: {model_results['last_forecast']:.2f}")
    else:
        lines.append("No fue posible entrenar el modelo por insuficiencia de datos.")
    lines.append("")
    lines.append("FASE 5: EVALUACIÓN")
    if model_results:
        lines.append("Comparación de métricas SARIMAX y Prophet por modelo:")
        lines.append(model_results["metrics_table"].to_string(index=False))
    else:
        lines.append("No existen métricas disponibles.")
    lines.append("")
    lines.append("FASE 6: DESPLIEGUE")
    lines.append("Los resultados deben utilizarse como insumo para la optimización del proceso comercial y para el diseño TOGAF.")
    return "\n".join(lines)


# =========================================================
# Estado de sesión
# =========================================================
for key, default in {
    "df_original": None,
    "df": None,
    "df_clean": None,
    "series": None,
    "model_results": None,
    "frequency": "Semanal"
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =========================================================
# Menú lateral
# =========================================================
st.sidebar.title("Menú principal")
option = st.sidebar.radio(
    "Selecciona una sección",
    [
        "Portada",
        "Carga de dataset",
        "Fase 1: Comprensión del negocio",
        "Fase 2: Comprensión de los datos",
        "Fase 3: Preparación de los datos",
        "Fase 4: Modelado",
        "Fase 5: Evaluación",
        "Fase 6: Despliegue"
    ]
)


# =========================================================
# Reiniciar scroll al cambiar de pestaña
# =========================================================
if "last_option" not in st.session_state:
    st.session_state.last_option = option

if option != st.session_state.last_option:
    st.session_state.last_option = option
    components.html(
        """
        <script>
        function scrollToTop() {
            window.parent.scrollTo(0, 0);
            const appView = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if (appView) { appView.scrollTo(0, 0); }
            const main = window.parent.document.querySelector('section.main');
            if (main) { main.scrollTo(0, 0); }
            const blockContainer = window.parent.document.querySelector('.block-container');
            if (blockContainer) { blockContainer.scrollIntoView({behavior: "instant", block: "start"}); }
        }
        setTimeout(scrollToTop, 50);
        setTimeout(scrollToTop, 150);
        setTimeout(scrollToTop, 300);
        </script>
        """,
        height=0
    )


# =========================================================
# Portada
# =========================================================
if option == "Portada":
    show_cover()
    st.header("Descarga del dataset de ejemplo")
    st.write("Puedes descargar el dataset simulado usado para probar la aplicación.")
    show_dataset_download_button()
    st.stop()


# =========================================================
# Carga de dataset
# =========================================================
if option == "Carga de dataset":
    st.title("Carga de dataset")
    st.write("Sube el archivo CSV correspondiente a las encuestas post venta. El dataset debe contener las columnas `fecha_enc_vent`, `score` y `canal_venta`.")
    uploaded_file = st.file_uploader("Subir dataset CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df_original = read_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"No se pudo leer el archivo CSV: {e}")
            st.stop()

        missing = validate_columns(df_original)
        if missing:
            st.error(f"El dataset no contiene las columnas requeridas: {missing}")
            st.stop()

        df = prepare_basic_dataframe(df_original)
        st.session_state.df_original = df_original
        st.session_state.df = df
        st.session_state.df_clean = None
        st.session_state.series = None
        st.session_state.model_results = None

        st.success("Dataset cargado correctamente.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{len(df):,}")
        c2.metric("Columnas originales", len(df_original.columns))
        c3.metric("Columnas utilizadas", len(df.columns))
        st.subheader("Vista previa del dataset cargado")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("Aún no se ha cargado ningún dataset.")
    st.stop()


# =========================================================
# Fase 1
# =========================================================
if option == "Fase 1: Comprensión del negocio":
    st.title("Fase 1 CRISP-DM: Comprensión del negocio")
    st.header("Misión")
    st.write("Ofrecer productos textiles de calidad a clientes nacionales, mediante procesos productivos y comerciales eficientes, manteniendo un servicio orientado a la satisfacción del cliente, la confiabilidad en la atención y la mejora continua de sus operaciones.")
    st.header("Visión")
    st.write("Ser líderes en el mercado textil reconocidos por sus productos de calidad, variedad y mejor precio, a través de recursos humanos comprometidos y satisfechos.")
    st.header("Valores")
    st.dataframe(pd.DataFrame({"Valores": ["Compromiso", "Excelencia", "Trabajo en equipo", "Cumplimiento", "Orientación al cliente", "Visión de futuro"]}), use_container_width=True)
    st.header("Objetivo estratégico")
    st.write("Fabricar productos textiles de la más alta calidad mediante procesos de excelencia y mejora continua para satisfacer y exceder las expectativas de sus compradores locales e internacionales.")
    st.header("KPI del objetivo estratégico")
    kpi = pd.DataFrame([
        {"Nombre de KPI": "Tasa de devoluciones por No Conformidad", "Métrica": "(# de pedidos devueltos por fallas de calidad / Total de pedidos entregados) × 100"},
        {"Nombre de KPI": "Nivel de cumplimiento de Entregas", "Métrica": "(Pedidos a tiempo y con cantidad exacta / Total de pedidos despachados) × 100"},
        {"Nombre de KPI": "Índice de Retención de Clientes", "Métrica": "(Clientes activos al final del periodo - Clientes nuevos) / Clientes activos al inicio del periodo × 100"}
    ])
    st.dataframe(kpi, use_container_width=True)
    st.caption("Tabla 2: KPI del objetivo estratégico. Fuente: Elaboración propia.")
    st.header("Objetivo de minería de datos")
    st.write("Analizar las calificaciones de retroalimentación de ventas de Ponte Selva para identificar patrones de satisfacción, comparar los canales de venta y predecir el comportamiento del score mediante modelos de series de tiempo como ARIMA y SARIMA, como apoyo a la mejora del proceso comercial.")
    st.header("Fuente de los datos")
    st.write("Dataset de las encuestas a los clientes posterior a la venta.")
    st.stop()


# =========================================================
# Validación de carga para fases 2 a 6
# =========================================================
if st.session_state.df is None:
    st.warning("Primero debes cargar el dataset en la opción 'Carga de dataset'.")
    st.stop()

df = st.session_state.df


# =========================================================
# Fase 2: Comprensión de los datos
# =========================================================
if option == "Fase 2: Comprensión de los datos":
    st.title("Fase 2 CRISP-DM: Comprensión de los datos")
    st.write("En esta fase se revisa la estructura inicial del dataset, se describen sus variables, se exploran los datos y se identifican problemas de calidad que deberán tratarse en la fase de preparación.")
    summary = get_quality_summary(df)
    st.header("Descripción general del dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{summary['registros']:,}")
    c2.metric("Columnas", summary["columnas"])
    c3.metric("Duplicados exactos", summary["duplicados"])
    c4.metric("Nulos totales", summary["nulos_totales"])
    st.subheader("Vista previa")
    st.dataframe(df.head(30), use_container_width=True)
    st.subheader("Tipos de datos")
    st.dataframe(pd.DataFrame({"columna": df.columns, "tipo_dato": [str(t) for t in df.dtypes]}), use_container_width=True)
    st.header("Diccionario de datos")
    dictionary = pd.DataFrame([
        {"Variable": "fecha_enc_vent", "Tipo": "Fecha", "Descripción": "Fecha de registro de la encuesta post venta."},
        {"Variable": "score", "Tipo": "Numérica", "Descripción": "Calificación otorgada por el cliente sobre 10 puntos."},
        {"Variable": "canal_venta", "Tipo": "Categórica", "Descripción": "Canal utilizado para la venta: WA, PRE, TEL o MAIL."}
    ])
    st.dataframe(dictionary, use_container_width=True)
    st.header("Calidad de datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fechas inválidas", summary["fechas_invalidas"])
    c2.metric("Scores fuera de rango", summary["scores_invalidos"])
    c3.metric("Canales inválidos", summary["canales_invalidos"])
    st.subheader("Valores nulos por columna")
    nulls = df.isna().sum().reset_index()
    nulls.columns = ["columna", "valores_nulos"]
    st.dataframe(nulls, use_container_width=True)
    st.header("Exploración inicial del score")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Promedio", f"{df['score'].mean():.2f}")
    c2.metric("Mediana", f"{df['score'].median():.2f}")
    c3.metric("Mínimo", f"{df['score'].min():.2f}")
    c4.metric("Máximo", f"{df['score'].max():.2f}")
    show_chart(fig_histogram, "fase2_hist_score", df, "score", "Distribución de calificaciones", "Score", "Frecuencia", default_width=10, default_height=5)
    st.header("Exploración por canal")
    show_chart(fig_bar, "fase2_bar_canal", df["canal_venta"].value_counts(), "Cantidad de registros por canal", "Canal", "Cantidad", default_width=10, default_height=5)
    score_channel = df.groupby("canal_venta")["score"].agg(["count", "mean", "min", "max"]).reset_index()
    score_channel.columns = ["canal_venta", "registros", "score_promedio", "score_minimo", "score_maximo"]
    st.dataframe(score_channel, use_container_width=True)
    st.header("Exploración temporal")
    fecha_min = df["fecha_enc_vent"].min()
    fecha_max = df["fecha_enc_vent"].max()
    c1, c2, c3 = st.columns(3)
    c1.metric("Fecha mínima", str(fecha_min.date()) if pd.notna(fecha_min) else "Sin dato")
    c2.metric("Fecha máxima", str(fecha_max.date()) if pd.notna(fecha_max) else "Sin dato")
    c3.metric("Días con registros", int(df["fecha_enc_vent"].nunique()))
    monthly = df.groupby(df["fecha_enc_vent"].dt.to_period("M"))["score"].mean().reset_index()
    monthly["fecha_enc_vent"] = monthly["fecha_enc_vent"].astype(str)
    monthly.columns = ["periodo_mensual", "score_promedio"]
    show_chart(fig_line, "fase2_line_mensual", monthly, "periodo_mensual", "score_promedio", "Score promedio mensual", "Periodo", "Score promedio", default_width=12, default_height=5)
    st.stop()


# =========================================================
# Fase 3: Preparación de los datos
# =========================================================
if option == "Fase 3: Preparación de los datos":
    st.title("Fase 3 CRISP-DM: Preparación de los datos")
    st.write("En esta fase se corrigen formatos, se validan rangos, se filtran registros inconsistentes y se construye una serie auxiliar para el modelado de series de tiempo.")
    df_clean = clean_for_modeling(df)
    outlier_info = detect_outliers_iqr(df_clean)
    st.session_state.df_clean = df_clean
    st.header("Resultado de la limpieza básica")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros originales", f"{len(df):,}")
    c2.metric("Registros preparados", f"{len(df_clean):,}")
    c3.metric("Registros excluidos", f"{len(df) - len(df_clean):,}")
    st.subheader("Reglas aplicadas")
    rules = pd.DataFrame([
        {"Regla": "Conversión de fecha", "Descripción": "La columna fecha_enc_vent se convierte a formato fecha."},
        {"Regla": "Conversión de score", "Descripción": "La columna score se convierte a valor numérico."},
        {"Regla": "Validación de score", "Descripción": "Se conservan únicamente valores entre 1 y 10."},
        {"Regla": "Validación de canal", "Descripción": "Se conservan únicamente los canales WA, PRE, TEL y MAIL."},
        {"Regla": "Conservación de columnas originales", "Descripción": "El dataset preparado mantiene solo fecha_enc_vent, score y canal_venta."},
        {"Regla": "Detección de outliers", "Descripción": "Los outliers del score se identifican mediante IQR, pero no se eliminan automáticamente."},
        {"Regla": "Serie auxiliar", "Descripción": "Para el modelado se genera una serie temporal auxiliar con frecuencia diaria, semanal o mensual."}
    ])
    st.dataframe(rules, use_container_width=True)
    st.header("Dataset preparado")
    st.write("El dataset preparado conserva únicamente las columnas originales entregadas por el archivo.")
    st.dataframe(df_clean.head(30), use_container_width=True)
    st.header("Detección de outliers del score")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Q1", f"{outlier_info['q1']:.2f}")
    c2.metric("Q3", f"{outlier_info['q3']:.2f}")
    c3.metric("IQR", f"{outlier_info['iqr']:.2f}")
    c4.metric("Límite inferior", f"{outlier_info['lower']:.2f}")
    c5.metric("Outliers", len(outlier_info["outliers"]))
    st.write("Los outliers se detectan para análisis, pero no se eliminan automáticamente, ya que un score bajo puede representar una experiencia real de insatisfacción del cliente.")
    show_chart(fig_boxplot, "fase3_boxplot_outliers", df_clean, "score", "Boxplot del score", "Score", default_width=8, default_height=5)
    if len(outlier_info["outliers"]) > 0:
        with st.expander("Ver registros detectados como outliers"):
            st.dataframe(outlier_info["outliers"], use_container_width=True)
    st.header("Distribución del score después de la limpieza")
    show_chart(fig_histogram, "fase3_hist_score_limpio", df_clean, "score", "Distribución de calificaciones después de la limpieza", "Score", "Frecuencia", default_width=10, default_height=5)
    st.header("Preparación de serie de tiempo auxiliar")
    frequency = st.selectbox("Frecuencia de análisis", FREQUENCY_OPTIONS, index=FREQUENCY_OPTIONS.index(st.session_state.frequency))
    st.session_state.frequency = frequency
    series = create_time_series(df_clean, frequency=frequency)
    st.session_state.series = series
    st.session_state.model_results = None
    c1, c2, c3 = st.columns(3)
    c1.metric("Periodos en la serie", f"{len(series):,}")
    c2.metric("Periodos con encuestas", f"{int(series['tiene_encuesta'].sum()):,}")
    c3.metric("Periodos interpolados", f"{int((~series['tiene_encuesta']).sum()):,}")
    st.write("La frecuencia semanal o mensual puede mejorar las métricas porque reduce el ruido diario y evita que la serie dependa excesivamente de días sin encuesta.")
    show_chart(fig_time_series_preparation, "fase3_serie_preparada", series, default_width=12, default_height=6)
    st.subheader("Serie auxiliar preparada")
    st.dataframe(series.head(40), use_container_width=True)
    st.download_button("Descargar dataset preparado", data=df_clean.to_csv(index=False).encode("utf-8"), file_name="dataset_preparado_crisp_dm.csv", mime="text/csv")
    st.download_button("Descargar serie auxiliar", data=series.to_csv(index=False).encode("utf-8"), file_name="serie_auxiliar_crisp_dm.csv", mime="text/csv")
    st.stop()


# =========================================================
# Preparación automática para fases 4 a 6
# =========================================================
if st.session_state.df_clean is None:
    df_clean = clean_for_modeling(df)
    st.session_state.df_clean = df_clean
else:
    df_clean = st.session_state.df_clean

if st.session_state.series is None:
    series = create_time_series(df_clean, frequency=st.session_state.frequency)
    st.session_state.series = series
else:
    series = st.session_state.series


# =========================================================
# Fase 4: Modelado
# =========================================================
if option == "Fase 4: Modelado":
    st.title("Fase 4 CRISP-DM: Modelado")
    st.write(
        "En esta fase se comparan modelos SARIMAX y Prophet para estimar el comportamiento futuro del score post venta. "
        "Para evitar tiempos excesivos, SARIMAX usa una búsqueda reducida de parámetros y Prophet queda como opción activable."
    )

    c1, c2, c3 = st.columns(3)

    frequency = c1.selectbox(
        "Frecuencia de análisis",
        FREQUENCY_OPTIONS,
        index=FREQUENCY_OPTIONS.index(st.session_state.frequency)
    )

    if frequency != st.session_state.frequency:
        st.session_state.frequency = frequency
        series = create_time_series(df_clean, frequency=frequency)
        st.session_state.series = series
        st.session_state.model_results = None

    default_horizon = 30 if frequency == "Diaria" else 12 if frequency == "Semanal" else 6

    horizon = c2.slider(
        "Horizonte de pronóstico",
        min_value=4,
        max_value=60,
        value=default_horizon,
        step=1
    )

    include_prophet = c3.checkbox(
        "Incluir Prophet",
        value=False,
        help="Prophet puede tardar más. Actívalo solo si SARIMAX ya corre correctamente."
    )

    st.info(
        "Recomendación: usa frecuencia semanal o mensual para reducir ruido y mejorar el tiempo de entrenamiento. "
        "La frecuencia diaria puede demorar más si el rango de fechas es largo."
    )

    model_results = train_and_compare_models(
        series,
        frequency=frequency,
        horizon=horizon,
        include_prophet=include_prophet
    )
    st.session_state.model_results = model_results
    if model_results is None:
        st.error("No fue posible entrenar los modelos. Se requieren más datos válidos.")
        st.stop()
    st.header("Modelo seleccionado")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mejor modelo", model_results["best_model"])
    c2.metric("Tendencia estimada", model_results["trend"])
    c3.metric("Último score preparado", f"{model_results['last_real']:.2f}")
    c4.metric("Score futuro estimado", f"{model_results['last_forecast']:.2f}")
    st.header("Comparación de modelos SARIMAX y Prophet")
    st.dataframe(model_results["metrics_table"], use_container_width=True)
    show_chart(fig_metrics_comparison, "fase4_metricas_modelos", model_results["metrics_table"], default_width=11, default_height=5)
    st.header("Pronóstico del mejor modelo")
    show_chart(fig_forecast, "fase4_pronostico", series, model_results["forecast"], default_width=12, default_height=6)
    st.subheader("Tabla de pronóstico")
    st.dataframe(model_results["forecast"], use_container_width=True)
    st.download_button("Descargar pronóstico", data=model_results["forecast"].to_csv(index=False).encode("utf-8"), file_name="pronostico_score_post_venta.csv", mime="text/csv")
    st.header("Interpretación del modelado")
    if model_results["trend"] == "Subir":
        st.success("El mejor modelo estima que las calificaciones post venta tenderán a subir en el horizonte seleccionado.")
    elif model_results["trend"] == "Bajar":
        st.warning("El mejor modelo estima que las calificaciones post venta tenderán a bajar en el horizonte seleccionado.")
    else:
        st.info("El mejor modelo estima que las calificaciones post venta tenderán a mantenerse relativamente estables.")
    st.stop()


# =========================================================
# Modelo automático para fases 5 y 6
# =========================================================
if st.session_state.model_results is None:
    st.session_state.model_results = train_and_compare_models(series, frequency=st.session_state.frequency, horizon=30 if st.session_state.frequency == "Diaria" else 12)
model_results = st.session_state.model_results


# =========================================================
# Fase 5: Evaluación
# =========================================================
if option == "Fase 5: Evaluación":
    st.title("Fase 5 CRISP-DM: Evaluación")
    st.write("En esta fase se comparan las métricas de los modelos entrenados y se evalúa el desempeño del mejor modelo.")
    if model_results is None:
        st.error("No existe un modelo entrenado para evaluar.")
        st.stop()
    st.header("Comparación de métricas SARIMAX y Prophet")
    metrics_df = model_results["metrics_table"]
    st.dataframe(metrics_df, use_container_width=True)
    best_metrics = metrics_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE mejor modelo", f"{best_metrics['MAE']:.4f}")
    c2.metric("MSE mejor modelo", f"{best_metrics['MSE']:.4f}")
    c3.metric("RMSE mejor modelo", f"{best_metrics['RMSE']:.4f}")
    c4.metric("R² mejor modelo", f"{best_metrics['R2']:.4f}")
    st.subheader("Interpretación de métricas")
    st.write("La aplicación selecciona como mejor modelo aquel con menor RMSE. El SARIMAX permite incluir variables externas y se evalúa con una búsqueda reducida para evitar tiempos excesivos. Prophet permite manejar automáticamente estacionalidades, datos faltantes y cambios de tendencia. Si el R² sigue siendo bajo, pero el MAE y RMSE son reducidos, el modelo puede interpretarse como una herramienta de tendencia más que como una explicación completa de la variabilidad del score.")
    st.header("Gráfico comparativo de RMSE")
    show_chart(fig_metrics_comparison, "fase5_metricas_modelos", metrics_df, default_width=11, default_height=5)
    st.header("Comparación entre valores reales y predichos")
    show_chart(fig_evaluation, "fase5_evaluacion", model_results["test"], default_width=12, default_height=6)
    st.subheader("Tabla de evaluación del mejor modelo")
    st.dataframe(model_results["test"], use_container_width=True)
    st.header("Conclusión de evaluación")
    st.write(f"El mejor modelo seleccionado fue: {model_results['best_model']}. Esta selección se realizó con base en el menor RMSE. La comparación con el modelo base permite justificar si el modelo aporta una mejora real frente a predecir únicamente el promedio histórico.")
    st.stop()


# =========================================================
# Fase 6: Despliegue
# =========================================================
if option == "Fase 6: Despliegue":
    st.title("Fase 6 CRISP-DM: Despliegue")
    st.write("En esta fase se presentan los resultados del análisis para que puedan ser utilizados como insumo en la toma de decisiones y posteriormente en el diseño de la arquitectura empresarial basada en TOGAF.")
    if model_results is None:
        st.warning("No existe un modelo entrenado. Ingresa a la Fase 4 para generar el pronóstico.")
    else:
        st.header("Resultado principal")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mejor modelo", model_results["best_model"])
        c2.metric("Tendencia futura", model_results["trend"])
        c3.metric("Score actual", f"{model_results['last_real']:.2f}")
        c4.metric("Score estimado", f"{model_results['last_forecast']:.2f}")
        if model_results["trend"] == "Subir":
            st.write("El resultado sugiere una posible mejora futura en la percepción de calidad post venta.")
        elif model_results["trend"] == "Bajar":
            st.write("El resultado sugiere una posible disminución futura en la percepción de calidad post venta. Se recomienda revisar los canales con menor score y reforzar el seguimiento de satisfacción.")
        else:
            st.write("El resultado sugiere estabilidad en la percepción de calidad post venta.")
    st.header("Recomendaciones para el proceso comercial")
    recommendations = pd.DataFrame([
        {"Recomendación": "Monitorear mensualmente el score promedio post venta."},
        {"Recomendación": "Revisar los canales con menor calificación promedio."},
        {"Recomendación": "Incorporar comentarios textuales en futuras encuestas para conocer causas de insatisfacción."},
        {"Recomendación": "Comparar SARIMAX y Prophet con frecuencia diaria, semanal y mensual antes de tomar decisiones."},
        {"Recomendación": "Crear un tablero de indicadores para seguimiento de satisfacción."},
        {"Recomendación": "Usar los hallazgos como entrada para la arquitectura de negocio, datos, aplicaciones y tecnología en TOGAF."}
    ])
    st.dataframe(recommendations, use_container_width=True)
    st.header("Reporte descargable")
    report = build_report(df, df_clean, series, model_results)
    st.download_button("Descargar reporte CRISP-DM", data=report.encode("utf-8"), file_name="reporte_crisp_dm_ponte_selva.txt", mime="text/plain")
    if model_results is not None:
        st.download_button("Descargar pronóstico final", data=model_results["forecast"].to_csv(index=False).encode("utf-8"), file_name="pronostico_final_score_post_venta.csv", mime="text/csv")
        st.download_button("Descargar métricas de modelos", data=model_results["metrics_table"].to_csv(index=False).encode("utf-8"), file_name="metricas_modelos_crisp_dm.csv", mime="text/csv")
    st.stop()
