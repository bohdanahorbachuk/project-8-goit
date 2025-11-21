import pickle
import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn App", layout="wide")

def convert_to_number_or_None(value: str):
    """
    If data has blanks, this method allows user not to enter any value
    and converts it to the correct type
    :param value:
    :return: float or None
    """
    if value == "":
        result = None
    else:
        try:
            result = float(value)
        except ValueError:
            result = None
            st.warning("Введіть число або залиште поле пустим")
    return result


def classification_report_message(model: str, metrics):
    """
    This method displays main metrics of the model
    :param model:
    :param metrics:
    :return:
    """
    st.subheader(f"Про модель {model}:")
    st.markdown(f"💡Точність: **{metrics['accuracy'] * 100:.2f}%**")
    st.write(f"Коли модель каже, що клієнт НЕ піде, "
             f"у {metrics['0']['precision'] * 100:.2f}% випадків це правда. "
             f"З усіх клієнтів, які НЕ йдуть, модель знаходить {metrics['0']['recall'] * 100:.2f}%.")
    st.write(f"Якщо модель каже, що клієнт піде, "
             f"це правда в {metrics['1']['precision'] * 100:.2f}% випадків. "
             f"З усіх реальних клієнтів, які підуть, модель знаходить {metrics['1']['recall'] * 100:.2f}%.")


# ----------------------------
# 1. Load model, medians, metrics
# ----------------------------

# To do other models
with open("random_forest_model.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

# Завантаження медіан
rf_median = None
with open("rf_medians.json", "r") as f:
    rf_medians = json.load(f)

# Завантаження метрик
rf_metrics = None
with open("rf_metrics.json", "r") as f:
    rf_metrics = json.load(f)

st.title("📡 Прогнозування Відтоку Клієнтів для Телекомунікаційної компанії")
st.write("Введіть параметри клієнта, щоб передбачити ймовірність відтоку.")


# ----------------------------------------
# 2. Input fields
# ----------------------------------------

input_mode = st.radio(
    "Оберіть формат вводу даних:",
    ["Прогноз для одного клієнта (ввід даних вручну)", "Прогноз для декількох (завантажити CSV файл)"]
)

if input_mode == "Прогноз для одного клієнта (ввід даних вручну)":
    # --- Перший ряд ---
    cols1 = st.columns(5)

    with cols1[0]:
        is_tv_subscriber = st.selectbox("*Чи підписаний на TV?", ['так', 'ні'])
        is_tv_subscriber = 1 if is_tv_subscriber == "так" else 0
    with cols1[1]:
        is_movie_package_subscriber = st.selectbox("*Чи підписаний на пакет фільмів?", ['так', 'ні'])
        is_movie_package_subscriber = 1 if is_movie_package_subscriber == "так" else 0
    with cols1[2]:
        subscription_age = st.number_input("*Тривалість підписки (міс)", 0.0, 100.0)
    with cols1[3]:
        bill_avg = st.number_input("*Середній чек на місяць", 0.0, 1000.0)
    with cols1[4]:
        # remaining_contract = st.number_input("Залишок контракту (міс)", 0.0, 36.0)
        raw_value_remaining_contract = st.text_input("Залишок контракту (міс) (якщо є)")
        remaining_contract = convert_to_number_or_None(raw_value_remaining_contract)
        if remaining_contract is None:
            remaining_contract = rf_medians['remaining_contract_median']


    # --- Другий ряд ---
    cols2 = st.columns(5)

    with cols2[0]:
        service_failure_count = st.number_input("*Кількість збоїв", 0, 100)
    with cols2[1]:
        # download_avg = st.number_input("Кількість завантаження (GB)", 0.0, 10000.0)
        raw_value_download_avg = st.text_input("Кількість завантаження (GB) (якщо є)")
        download_avg = convert_to_number_or_None(raw_value_download_avg)
        download_avg_missing = 1 if download_avg is None else 0
        if download_avg is None:
            download_avg = rf_medians['download_median']

    with cols2[2]:
        # upload_avg = st.number_input("Кількість вивантаження (GB)", 0.0, 10000.0)
        raw_value_upload_avg = st.text_input("Кількість вивантаження (GB) (якщо є)")
        upload_avg = convert_to_number_or_None(raw_value_upload_avg)
        upload_avg_missing = 1 if upload_avg is None else 0
        if upload_avg is None:
            upload_avg = rf_medians['upload_median']

    with cols2[3]:
        download_over_limit = st.number_input("\n*Завантаження понад межу", 0, 100)
    with cols2[4]:
        pass


elif input_mode == "Прогноз для декількох (завантажити CSV файл)":
    uploaded_file = st.file_uploader("Завантажте CSV файл", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("📄 Завантажені дані:")
        st.dataframe(df)

        # Перевіримо, що всі потрібні колонки є
        required_cols = [
            "is_tv_subscriber",
            "is_movie_package_subscriber",
            "subscription_age",
            "bill_avg",
            "reamining_contract",
            "service_failure_count",
            "download_avg",
            "upload_avg",
            "download_over_limit",
            "download_avg_missing",
            "upload_avg_missing"
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"❌ Відсутні колонки у CSV: {missing}")


# ----------------------------------------
# 3. Prediction
# ----------------------------------------

# ---- Випадаюче меню для вибору моделі ----
model_name = st.selectbox(
    "Оберіть будь-яку модель для передбачення:",
    ['Random Forest', 'SVM', 'Нейронна мережа']
)

if st.button("Передбачити відтік"):
    if input_mode == 'Прогноз для одного клієнта (ввід даних вручну)':
        if model_name == 'Random Forest':
            classification_report_message(model_name, rf_metrics)

            # Prepare input
            X = np.array([[
                is_tv_subscriber,
                is_movie_package_subscriber,
                subscription_age,
                bill_avg,
                remaining_contract,
                service_failure_count,
                download_avg,
                upload_avg,
                download_over_limit,
                download_avg_missing,
                upload_avg_missing
            ]], dtype=float)

            st.subheader("Вхідні дані:")
            st.write(X)

            # Predict
            pred = random_forest_model.predict_proba(X)[0][1] * 100

        elif model_name == 'SVM':
            # To do
            pass

        elif model_name == 'Нейронна мережа':
            # To do
            pass

        cols = st.columns(2)

        with cols[0]:
            st.subheader("Передбачення:")
            st.markdown(f"💔 **Ймовірність, що клієнт піде: {pred:.2f}%**")
            st.markdown(f"👍 **Ймовірність, що клієнт залишиться: {100-pred:.2f}%**")

            if pred > 50:
                st.error("⚠️ Клієнт з високою ймовірністю піде.")
            else:
                st.success("✅ Клієнт, скоріш за все, залишиться.")

        # Візуалізація
        with cols[1]:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(["Клієнт піде"], [pred])
            ax.bar(["Клієнт залишиться"], [100-pred])
            ax.set_ylim(0, 100)
            st.pyplot(fig)


    elif input_mode == "Прогноз для декількох (завантажити CSV файл)":
        if model_name == 'Random Forest':
            preds = random_forest_model.predict_proba(df[required_cols])[:, 1] * 100

        elif model_name == 'SVM':
            # To do
            pass

        elif model_name == 'Нейронна мережа':
            # To do
            pass

        df["churn_probability"] = preds

        st.success("Готово!")
        st.dataframe(df)

        # Збереження результатів
        csv_result = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Завантажити результати",
            csv_result,
            "predictions.csv",
            "text/csv"
        )
