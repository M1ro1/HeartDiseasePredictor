import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


def show_user_profile():
    if st.session_state.logged_in:
        with st.sidebar:
            st.markdown("---")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write("👤")
            with col2:
                st.write(f"**{st.session_state.user['username']}**")
                st.caption("Active Member")

            if st.button("Logout", icon="🚪", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.user = None
                st.rerun()
            st.markdown("---")


home_page = st.Page("home_page.py", title="Home", icon="🏠")
predict_page = st.Page("app.py", title="Prediction", icon="🫀")
register_page = st.Page("auth.py", title="Sign Up", icon="📝")

if st.session_state.logged_in:
    pg = st.navigation({
        "Home Page": [home_page],
        "Services": [predict_page]
    })
else:
    pg = st.navigation({
        "Main": [home_page],
        "Account": [register_page],
        "Tools": [predict_page]
    })

st.set_page_config(page_title='Heart Disease AI', page_icon='🫀', layout='wide')

show_user_profile()

pg.run()