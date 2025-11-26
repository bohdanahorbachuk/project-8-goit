import json
import pandas as pd
import streamlit as st
import joblib

from tensorflow.keras.models import load_model

st.set_page_config(page_title="Churn Scoring", page_icon="📉")

st.title("📉 Оцінка відтоку клієнтів за допомогою нейромережв")

st.write(
    """
    Введіть параметри клієнта, і модель передбачить ймовірність відтоку

    """
)

@st.cache_resource
def load_artifacts():
    model = load_model("churn_NNmodel.keras")
    scaler = joblib.load("scalerNN.pkl")
    with open("feature_namesNN.json", "r") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names

try:
    model, scaler, feature_names = load_artifacts()
except Exception as e:
    st.error("❌ Не вдалось завантажити модель")
    st.code(repr(e))
    st.stop()

st.sidebar.header("Параметри клієнта")

input_data = {name: 0.0 for name in feature_names}

if "is_tv_subscriber" in feature_names:
    input_data["is_tv_subscriber"] = int(
        st.sidebar.checkbox("TV subscription (is_tv_subscriber)", value=True)
    )

if "is_movie_package_subscriber" in feature_names:
    input_data["is_movie_package_subscriber"] = int(
        st.sidebar.checkbox("Movie package (is_movie_package_subscriber)", value=False)
    )

if "download_over_limit" in feature_names:
    input_data["download_over_limit"] = int(
        st.sidebar.checkbox("Часто превышает лимит (download_over_limit)", value=False)
    )

if "subscription_age" in feature_names:
    input_data["subscription_age"] = st.sidebar.slider(
        "Стаж підписки, років (subscription_age)",
        min_value=0.0, max_value=20.0, value=2.0, step=0.1,
    )

if "bill_avg" in feature_names:
    input_data["bill_avg"] = st.sidebar.slider(
        "Середній місячний рахунок (bill_avg)",
        min_value=0.0, max_value=500.0, value=20.0, step=1.0,
    )

if "reamining_contract" in feature_names:
    input_data["reamining_contract"] = st.sidebar.slider(
        "Залишок контракта, роки (reamining_contract)",
        min_value=0.0, max_value=5.0, value=1.0, step=0.1,
    )

if "service_failure_count" in feature_names:
    input_data["service_failure_count"] = st.sidebar.slider(
        "Кількість відмов (service_failure_count)",
        min_value=0, max_value=50, value=0, step=1,
    )

if "download_avg" in feature_names:
    input_data["download_avg"] = st.sidebar.slider(
        "Середній трафік завантажень (download_avg)",
        min_value=0.0, max_value=1000.0, value=50.0, step=5.0,
    )

if "upload_avg" in feature_names:
    input_data["upload_avg"] = st.sidebar.slider(
        "Средний трафік передачі",
        min_value=0.0, max_value=100.0, value=5.0, step=1.0,
    )

row = pd.DataFrame([[input_data[col] for col in feature_names]], columns=feature_names)

st.subheader("Введені данні")
st.dataframe(row)

row_scaled = scaler.transform(row)

if st.button("Оцінити риск відтоку"):
    proba = float(model.predict(row_scaled)[0][0])  # число 0..1
    proba_percent = proba * 100

    st.markdown("### 🔮 Результат")
    st.write(f"**Ймовірність відтоку:** `{proba_percent:.2f}%`")

    if proba >= 0.5:
        st.error("Висока ймовірність відтоку")
    else:
        st.success("Низька ймовірність відтоку")

    st.progress(min(max(proba, 0.0), 1.0))
else:
    st.info("Натисни кнопку **«Оцінити риск відтоку»**, щоб отримати прогноз.")
