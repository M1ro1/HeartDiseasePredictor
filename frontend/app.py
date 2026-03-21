import streamlit as st
import requests
import base64
import os
from dotenv import load_dotenv

load_dotenv()

# Ініціалізація стану, якщо його ще немає
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000/predict')

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 History")
    if st.session_state.logged_in:
        st.info(f"Вітаємо, {st.session_state.user['username']}!")
        st.write("Ваші останні перевірки:")
        # Тут буде логіка виводу реальних кнопок з бази даних
        st.button("🕒 21.03.2026 - 15.4% Risk")
        st.button("🕒 18.03.2026 - 82.1% Risk")
    else:
        st.warning("Увійдіть в акаунт, щоб зберігати результати та бачити історію.")
        if st.button("🔐 Перейти до входу"):
            st.switch_page("auth.py")

# --- ГОЛОВНИЙ ЕКРАН ---
st.title("🫀 Heart Disease Risk Analyzer")
st.markdown("Введіть медичні показники нижче, щоб отримати прогноз моделі та аналіз факторів впливу.")

# Ввід даних (доступний усім)
with st.expander("📝 Enter Patient Clinical Data", expanded="last_result" not in st.session_state):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
        trestbps = st.number_input("Resting BP (mm Hg)", 90, 200, 120)

    with col_b:
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting sugar > 120 mg/dl", ["True", "False"])
        restecg = st.selectbox("ECG Results", ["normal", "st-t abnormality", "lv hypertrophy"])
        thalch = st.number_input("Max Heart Rate", 60, 220, 150)

    with col_c:
        exang = st.selectbox("Exercise Angina", ["True", "False"])
        oldpeak = st.number_input("ST depression", 0.0, 6.0, 1.0, step=0.1)
        slope = st.selectbox("ST Slope", ["upsloping", "flat", "downsloping"])

    predict_btn = st.button("🚀 Run Analysis", type='primary', use_container_width=True)

# Логіка запиту
if predict_btn:
    patient_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope
    }

    with st.spinner("Analyzing data..."):
        try:
            response = requests.post(API_URL, json=patient_data)
            response.raise_for_status()
            st.session_state.last_result = response.json()
        except Exception as e:
            st.error(f"Помилка з'єднання з сервером: {e}")

# --- ВИВІД РЕЗУЛЬТАТІВ + РЕКОМЕНДАЦІЇ ---
if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.divider()

    col_res1, col_res2 = st.columns([1, 2], gap="large")

    with col_res1:
        st.subheader("Analysis Result")
        prob = res['probability']

        # Вивід результату з рекомендаціями
        if prob >= 70:
            st.error(f"**High Risk!**\n\nProbability: **{prob:.1f}%**")
            st.warning(
                "🔴 **Recommendation:** It is recommended to urgently consult a cardiologist for a complete examination.")
        elif prob >= 45:
            st.warning(f"**Increased Risk.**\n\nProbability: **{prob:.1f}%**")
            st.info(
                "🟡 **Recommendation:** You have certain risk factors. We recommend a routine visit to your doctor and paying attention to your lifestyle.")
        else:
            st.success(f"**Low Risk!**\n\nProbability: **{prob:.1f}%**")
            st.info(
                "🟢 **Recommendation:** Your numbers look good. Continue to lead a healthy lifestyle and do regular check-ups.")

        # Кнопка завантаження
        pdf_bytes = base64.b64decode(res['patient_data_file'])
        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_bytes,
            file_name="heart_disease_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with col_res2:
        st.subheader("Why did the model decide that? (XAI)")
        img_data = base64.b64decode(res['shap_plot'])
        st.image(img_data, use_container_width=True,
                 caption="SHAP Explanation: Red factors increase risk, Blue decrease it.")