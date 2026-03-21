import streamlit as st
import requests
import os

API_BASE_URL = os.getenv('API_URL', 'http://127.0.0.1:8000')


def show_auth_page():
    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "login"

    st.markdown(
        f"<h2 style='text-align: center;'>{'Log in' if st.session_state.auth_mode == 'login' else 'Registration'}</h2>",
        unsafe_allow_html=True)

    with st.container(border=True):
        if st.session_state.auth_mode == "login":
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Увійти", type="primary", use_container_width=True):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/token",
                        data={"username": username, "password": password}
                    )
                    if response.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.user = {"username": username}
                        st.success("Success!")
                        st.rerun()
                    else:
                        st.error("Wrong login or password")
                except Exception as e:
                    st.error(f"Connection error: {e}")

            st.write("---")
            if st.button("Don't have an account yet? Sign up"):
                st.session_state.auth_mode = "register"
                st.rerun()

        else:
            new_user = st.text_input("Username")
            new_email = st.text_input("Email")
            new_pass = st.text_input("Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")

            if st.button("Create an account", type="primary", use_container_width=True):
                if new_pass != confirm_pass:
                    st.error("Passwords do not match")
                else:
                    response = requests.post(
                        f"{API_BASE_URL}/registration",
                        json={"username": new_user, "email": new_email, "password": new_pass}
                    )
                    if response.status_code == 200:
                        st.success("Registration successful! Please log in.")
                        st.session_state.auth_mode = "login"
                        st.rerun()
                    else:
                        st.error(response.json().get("detail", "Registration failed"))

            st.write("---")
            if st.button("Already have an account? Log in"):
                st.session_state.auth_mode = "login"
                st.rerun()


if __name__ == "__main__":
    show_auth_page()