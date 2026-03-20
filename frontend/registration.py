import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_REG_URL = os.getenv('API_REG_URL', 'http://127.0.0.1:8000/registration')

st.markdown("<h2 style='text-align: center;'>📝 Create an Account</h2>", unsafe_allow_html=True)

with st.form("registration_form"):
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    password_confirm = st.text_input("Confirm Password", type="password")

    submit_btn = st.form_submit_button("Sign Up", type="primary", use_container_width=True)

if submit_btn:
    if not username or not email or not password:
        st.error("Please fill in all fields")
    elif password != password_confirm:
        st.error("Passwords do not match")
    else:
        payload = {
            "username": username,
            "email": email,
            "password": password
        }

        try:
            with st.spinner("Creating your account..."):
                response = requests.post(API_REG_URL, json=payload)

                if response.status_code == 200:
                    st.success("✅ Registration successful! Now you can use the analyzer.")
                    st.balloons()
                    if st.button("Go to Prediction"):
                        st.switch_page("app.py")
                else:
                    error_detail = response.json().get("detail", "Registration failed")
                    st.error(f"Error: {error_detail}")

        except requests.exceptions.ConnectionError:
            st.error("Backend server is offline. Please try again later.")