import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('Pilih Halaman : ', ('EDA', 'Tweets Sentiment Predict'))

if navigation == 'EDA':
    eda.run()
elif navigation == 'Tweets Sentiment Predict':
    prediction.run()