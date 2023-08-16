import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title= 'NVDIA Tweets Sentiment',
    layout= 'wide',
    initial_sidebar_state= 'expanded'
)

def run():

    # Page Title
    st.title('NVDIA Tweets Sentiment EDA')

    # Sub Header
    st.subheader('EDA for NVDIA Tweets Sentiment')

    # Menambahkan Text
    st.write('This page is created by Mehdi')

    # Membuat Garis Lurus
    st.markdown('---')

    # Magic Syntax
    '''
    Page ini, merupakan explorasi sederhana
    Dataset yang digunakan adalah dataset Twitter Training
    Dataset ini berasal dari kaggle
    '''

    # Show Data Frame
    data = pd.read_csv('nvidia_tweets.csv')
    st.dataframe(data)



    # # Membuat Barplot
    st.write('#### Plot Sentiment')
    fig = plt.figure(figsize=(15,5))
    plot = data.plot.pie(y='sentiment', figsize=(15,5))
    st.pyplot(fig)


if __name__ == '__main__':
    run()