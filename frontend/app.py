import streamlit as st
import requests
import base64
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('API_URL', 'http://127.0.0.1:8000/predict')
HISTORY_URL = os.getenv('HISTORY_URL', 'http://127.0.0.1:8000/history')


def fetch_history():
    if st.session_state.get("logged_in"):
        headers = {"X-Token": st.session_state.get("token")}
        try:
            response = requests.get(HISTORY_URL, headers=headers)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.sidebar.error(f"Не вдалося завантажити історію: {e}")
    return []


with st.sidebar:
    st.header("📊 History")
    if st.session_state.logged_in:
        st.info(f"Вітаємо, {st.session_state.user['username']}!")

        history_data = fetch_history()

        if history_data:
            st.write("Ваші останні перевірки:")
            for entry in history_data:
                date_obj = datetime.fromisoformat(entry['created_at'].replace('Z', ''))
                date_str = date_obj.strftime("%d.%m %H:%M")

                if st.button(f"🕒 {date_str} - {entry['probability']:.1f}%", key=f"hist_{entry['id']}",
                             use_container_width=True):
                    st.session_state.last_result = {
                        "probability": entry['probability'],
                        "prediction": entry['prediction'],
                        "shap_plot": entry['shap_image_base64'],
                        "patient_data_file": ""
                    }
                    st.rerun()
        else:
            st.write("Історія поки порожня.")
    else:
        st.warning("Увійдіть в акаунт, щоб зберігати результати.")
        if st.button("🔐 Перейти до входу"):
            st.switch_page("auth.py")

st.title("🫀 Heart Disease Risk Analyzer")

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

if predict_btn:
    patient_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalch': thalch,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope
    }

    with st.spinner("Analyzing data..."):
        try:
            headers = {"X-Token": st.session_state.get("token", "")}
            response = requests.post(API_URL, json=patient_data, headers=headers)
            response.raise_for_status()
            st.session_state.last_result = response.json()
            st.rerun()
        except Exception as e:
            st.error(f"Помилка: {e}")

if "last_result" in st.session_state:
    res = st.session_state.last_result
    st.divider()
    col_res1, col_res2 = st.columns([1, 2], gap="large")
    with col_res1:
        st.subheader("Analysis Result")
        prob = res['probability']
        if prob >= 70:
            st.error(f"**High Risk!**\n\nProbability: **{prob:.1f}%**")
            st.warning("🔴 **Recommendation:** Consult a cardiologist urgently.")
        elif prob >= 45:
            st.warning(f"**Increased Risk.**\n\nProbability: **{prob:.1f}%**")
            st.info("🟡 **Recommendation:** Schedule a routine check-up.")
        else:
            st.success(f"**Low Risk!**\n\nProbability: **{prob:.1f}%**")
            st.info("🟢 **Recommendation:** Keep leading a healthy lifestyle.")

        if res.get('patient_data_file'):
            pdf_bytes = base64.b64decode(res['patient_data_file'])
            st.download_button("📄 Download PDF", pdf_bytes, "report.pdf", "application/pdf", use_container_width=True)
        else:
            st.caption("PDF report not available for historical view.")

    with col_res2:
        st.subheader("Model Explanation (XAI)")
        img_data = base64.b64decode(res['shap_plot'])
        st.image(img_data, use_container_width=True)