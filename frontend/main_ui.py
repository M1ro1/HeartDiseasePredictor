import streamlit as st

home_page = st.Page("home_page.py", title="Home", icon="🏠")
predict_page = st.Page("app.py", title="Prediction", icon="🫀")

pg = st.navigation([home_page, predict_page])

st.set_page_config(page_title='Heart Disease Prediction', layout='wide')
pg.run()
