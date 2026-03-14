import streamlit as st

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🫀 AI Heart Disease Risk Analyzer</h1>",
            unsafe_allow_html=True)
st.markdown(
    "<h4 style='text-align: center; color: gray;'>A reliable tool for cardiovascular risk analysis based on machine learning</h4>",
    unsafe_allow_html=True)
st.write("---")

col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("About project")
    st.write("""
    This app uses advanced machine learning algorithms to estimate your risk of cardiovascular disease based on your clinical indicators.

    Our tool doesn't just give you raw numbers, it also explains which factors (like cholesterol or blood pressure) have the most impact on your outcome.
    """)

    st.markdown("""
    **Key Features:**
    * 🩸 **Comprehensive Analysis:** Takes into account 11 medical parameters (ECG, pain type, sugar level, etc.).
    * 🧠 **Explainable AI (XAI):** Integrated SHAP graphs allow you to "look under the hood" of artificial intelligence.
    """)

with col2:
    st.info("""
    **💡 Interesting fact:**
    Cardiovascular disease is the leading cause of health problems in the world. Early diagnosis and control of risk factors can significantly improve quality of life.
    """)

    st.warning(""" 
    **⚠️ Medical Disclaimer**\n
    This app is for **educational and informational purposes only**. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult your doctor before making any decisions regarding your health.
    """)

st.write("---")

st.markdown("<br>", unsafe_allow_html=True)
col_empty1, col_cta, col_empty2 = st.columns([1, 2, 1])

with col_cta:
    st.markdown("<h3 style='text-align: center;'>Ready to check the metrics?</h3>", unsafe_allow_html=True)

    if st.button("🚀 Go to the forecast panel", type="primary", use_container_width=True):
        st.switch_page("app.py")