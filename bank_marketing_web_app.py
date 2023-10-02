import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

bank_marketing_df = pd.read_csv("https://raw.githubusercontent.com/madhav28/Bank-Marketing-Analysis/main/bank-full.csv", sep=';')

st.title("Bank Marketing Dataset Analysis")

st.markdown("### Histogram Plot Analysis")

st.markdown("#### Here, y tells us the outcome of subscription")

col1, col2 = st.columns([1, 3])

with col1:
    column_options = bank_marketing_df.columns
    column_name = st.selectbox("Select Column", column_options)

with col2:
    fig = sns.histplot(data=bank_marketing_df, x=column_name, hue='y').figure
    st.pyplot(fig)
    plt.close()

st.markdown("### Correlation Heatmap")

col1, col2 = st.columns([1, 3])

with col2:
    fig = sns.heatmap(data=bank_marketing_df.corr(), cmap='coolwarm').figure
    st.pyplot(fig)
    plt.close()





