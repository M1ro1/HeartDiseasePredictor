import streamlit as st

home_page = st.Page("home_page.py", title="Home", icon="🏠")
predict_page = st.Page("app.py", title="Prediction", icon="🫀")
register_page = st.Page("registration.py", title="Sign Up", icon="📝")

pg = st.navigation({
    "Main": [home_page],
    "Account": [register_page],
    "Tools": [predict_page]
})

st.set_page_config(page_title='Heart Disease AI', page_icon='🫀', layout='wide')
pg.run()