import streamlit as st
import requests
import base64

st.set_page_config(page_title='Heart Disease Predictor', page_icon='🫀', layout='wide')
st.title("Heart-disease risk analyzer")
st.markdown("""
Enter your medical data on the left to get the model's prediction and an explanation of which factors most influenced the result..
""")

API_URL = 'http://api:8080/predict'

st.sidebar.header("Patient Data")

age = st.sidebar.slider("Age", 20, 100, 50)
trestbps = st.sidebar.slider("Resting blood pressure (trestbps)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (chol)", 100, 600, 200)
thalch = st.sidebar.slider("Maximum heart rate (thalch)", 60, 220, 150)
oldpeak = st.sidebar.slider("ST depression on ECG (oldpeak)", 0.0, 6.0, 1.0, step=0.1)

sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Type of chest pain (cp)", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
fbs = st.sidebar.selectbox("Sugar in blood > 120 мг/дл (fbs)", ["True", "False"])
restecg = st.sidebar.selectbox("ECG Results (restecg)", ["normal", "st-t abnormality", "lv hypertrophy"])
exang = st.sidebar.selectbox("Angina pectoris during exertion (exang)", ["True", "False"])
slope = st.sidebar.selectbox("Slope the segment ST (slope)", ["upsloping", "flat", "downsloping"])

predict_btn = st.sidebar.button("Get a forecast", type='primary', use_container_width=True)

if predict_btn:
    patient_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalch': thalch,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope
        }


    with st.spinner("Getting a prediction from the model..."):
        try:
            response = requests.post(API_URL, json=patient_data)
            response.raise_for_status()
            result = response.json()
            probability = result['probability']
            img_base64 = result['shap_plot']

            st.divider()
            col1,col2 = st.columns([1,2])

            with col1:
                st.subheader("Analysis result")
                if probability >= 70:
                    st.error(f"**High risk!**\n\nProbability of disease: **{probability:.1f}%**")
                    st.warning("It is recommended to urgently consult a cardiologist.")
                elif probability >= 45:
                    st.warning(f"**Increased risk.**\n\nProbability of disease: **{probability:.1f}%**")
                    st.warning("You have certain risk factors. We recommend that you schedule a routine visit to your doctor, get tested, and pay attention to your diet/activity.")
                else:
                    st.success(f"**Low risk!**\n\nProbability of disease: **{probability:.1f}%**")
                    st.info("Your numbers look good. Continue to lead a healthy lifestyle!")

            with col2:
                st.subheader("Why did the model decide that? (XAI)")
                img_data = base64.b64decode(img_base64)
                st.image(img_data, use_container_width=True)

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to FastAPI server. Make sure it is running!")
        except Exception as e:
            st.error(f"An error occurred while processing: {e}")