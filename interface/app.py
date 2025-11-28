import joblib
import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

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

# Завантаження моделей
# Random Forest model
random_forest_pipeline = joblib.load("random_forest_pipeline.joblib")

#SVM model
svm_pipeline = joblib.load("churn_svm_model.pkl")

# Neural Network model
nn_model = load_model("churn_NNmodel.keras")
nn_scaler = joblib.load("scalerNN.pkl")
with open("feature_namesNN.json", "r") as f:
    nn_feature_names = json.load(f)

#To do models

# Завантаження медіан
# Random Forest medians
rf_median = None
with open("rf_medians.json", "r") as f:
    rf_medians = json.load(f)

# SVM medians
svm_medians = None
with open("svm_medians.json", "r") as f:
    svm_medians = json.load(f)

# To do medians

#Завантаження метрик
# Random Forest metrics
rf_metrics = None
with open("rf_metrics.json", "r") as f:
    rf_metrics = json.load(f)

# SVM metrics
svm_metrics = None
with open("svm_metrics.json", "r") as f:
    svm_metrics = json.load(f)

# Neural Network metrics
with open("nn_metrics.json", "r") as f:
    nn_metrics = json.load(f)

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
    # Перший ряд
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


    # Другий ряд
    cols2 = st.columns(5)

    with cols2[0]:
        service_failure_count = st.number_input("*Кількість збоїв", 0, 100)
    with cols2[1]:
        # download_avg = st.number_input("Кількість завантаження (GB)", 0.0, 10000.0)
        raw_value_download_avg = st.text_input("Кількість завантаження (GB) (якщо є)")
        download_avg = convert_to_number_or_None(raw_value_download_avg)
        if download_avg is None:
            download_avg = rf_medians['download_median']

    with cols2[2]:
        # upload_avg = st.number_input("Кількість вивантаження (GB)", 0.0, 10000.0)
        raw_value_upload_avg = st.text_input("Кількість вивантаження (GB) (якщо є)")
        upload_avg = convert_to_number_or_None(raw_value_upload_avg)
        if upload_avg is None:
            upload_avg = rf_medians['upload_median']

    with cols2[3]:
        download_over_limit = st.number_input("\n*Завантаження понад межу", 0, 100)
    with cols2[4]:
        pass


elif input_mode == "Прогноз для декількох (завантажити CSV файл)":
    uploaded_file = st.file_uploader("Завантажте CSV файл з колонками "
                                     "is_tv_subscriber,"
                                     "is_movie_package_subscriber,"
                                     "subscription_age,"
                                     "bill_avg,"
                                     "reamining_contract,"
                                     "service_failure_count,"
                                     "download_avg,upload_avg,"
                                     "download_over_limit,"
                                     "download_avg_missing,"
                                     "upload_avg_missing", type=["csv"])

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
            "download_over_limit"
        ]

        missing = [c for c in required_cols if c not in df.columns]

        if missing:
            st.error(f"❌ Відсутні колонки у CSV: {missing}")


# ----------------------------------------
# 3. Prediction
# ----------------------------------------

# Випадаюче меню для вибору моделі
model_name = st.selectbox(
    "Оберіть будь-яку модель для передбачення:",
    ['Random Forest', 'SVM', 'Нейронна мережа']
)


if st.button("Передбачити відтік"):
    if input_mode == 'Прогноз для одного клієнта (ввід даних вручну)':
        # Формуємо вхідні дані
        X = np.array([[
            is_tv_subscriber,
            is_movie_package_subscriber,
            subscription_age,
            bill_avg,
            remaining_contract,
            service_failure_count,
            download_avg,
            upload_avg,
            download_over_limit
        ]], dtype=float)

        st.subheader("Вхідні дані:")
        st.write(X)

        if model_name == 'Random Forest':
            classification_report_message(model_name, rf_metrics)
            #Передбачення
            probability = random_forest_pipeline.predict_proba(X)[0][1] * 100

        elif model_name == 'SVM':
            classification_report_message(model_name, svm_metrics)

            probability = svm_pipeline.predict_proba(X)[0][1] * 100

        elif model_name == 'Нейронна мережа':
            if nn_metrics:
                classification_report_message(model_name, nn_metrics)

                # Формуємо вхід саме під NN: словник з назвами ознак
            nn_input = {
                "is_tv_subscriber": is_tv_subscriber,
                "is_movie_package_subscriber": is_movie_package_subscriber,
                "subscription_age": subscription_age,
                "bill_avg": bill_avg,
                "reamining_contract": remaining_contract,
                "service_failure_count": service_failure_count,
                "download_avg": download_avg,
                "upload_avg": upload_avg,
                "download_over_limit": download_over_limit,
            }

            # DataFrame в правильному порядку колонок
            nn_df = pd.DataFrame([[nn_input[col] for col in nn_feature_names]], columns=nn_feature_names)

            # Масштабування
            nn_scaled = nn_scaler.transform(nn_df)

            # Передбачення нейромережі (ймовірність класу "1" – клієнт піде)
            nn_proba = nn_model.predict(nn_scaled)[0][0]
            probability = nn_proba * 100

        # Відображення результату
        cols = st.columns(2)

        with cols[0]:
            st.subheader("Передбачення:")
            st.markdown(f"💔 **Ймовірність, що клієнт піде: {probability:.2f}%**")
            st.markdown(f"👍 **Ймовірність, що клієнт залишиться: {100-probability:.2f}%**")

            if probability > 50:
                st.error("⚠️ Клієнт з високою ймовірністю піде.")
            else:
                st.success("✅ Клієнт, скоріш за все, залишиться.")

        # Візуалізація
        with cols[1]:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(["Клієнт піде"], [probability])
            ax.bar(["Клієнт залишиться"], [100-probability])
            ax.set_ylim(0, 100)
            st.pyplot(fig)


    elif input_mode == "Прогноз для декількох (завантажити CSV файл)":
        if model_name == 'Random Forest':
            classification_report_message(model_name, rf_metrics)

            probabilities = random_forest_pipeline.predict_proba(df[required_cols])[:, 1] * 100

        elif model_name == 'SVM':
            classification_report_message(model_name, svm_metrics)

            probabilities = svm_pipeline.predict_proba(df[required_cols])[:, 1] * 100

        elif model_name == 'Нейронна мережа':
            if nn_metrics:
                classification_report_message(model_name, nn_metrics)

                # Перевіряємо, що всі фічі для NN є в датафреймі
            missing_nn = [c for c in nn_feature_names if c not in df.columns]
            if missing_nn:
                st.error(f"❌ Відсутні колонки для нейромережі: {missing_nn}")
                st.stop()

            nn_df = df[nn_feature_names].copy()

            # Якщо в даних є пропуски – можна підставити медіани з RF або окремі для NN
            nn_df = nn_df.fillna(nn_df.median(numeric_only=True))

            nn_scaled = nn_scaler.transform(nn_df)
            nn_proba = nn_model.predict(nn_scaled).ravel()
            probabilities = nn_proba * 100


        df["churn_probability"] = probabilities
        df["churn_prediction"] = pd.cut(
            df["churn_probability"],
            bins=[0, 40, 70, 100],
            labels=[
                "Клієнт залишиться",
                "Середній ризик відтоку",
                "Високий ризик відтоку"
            ],
            include_lowest=True
        )

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
