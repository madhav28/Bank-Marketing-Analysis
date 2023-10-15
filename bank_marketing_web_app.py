import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Bank Marketing Dataset Analysis")

tab1, tab2, tab3 = st.tabs(
    ['About Data', 'Data Visualisation', 'Predictive Modelling'])

bank_marketing_df = pd.read_csv(
    "https://raw.githubusercontent.com/madhav28/Bank-Marketing-Analysis/main/bank-full.csv", sep=';')

with tab1:

    customer_description_df = pd.read_csv(
        "column_description.txt", names=['Type', 'Description'])

    feature_description = {}

    column_name = bank_marketing_df.columns

    feature_description['Column Name'] = column_name
    feature_description['Type'] = customer_description_df['Type']
    feature_description['Description'] = customer_description_df['Description']

    feature_description_df = pd.DataFrame(feature_description)

    idx = np.linspace(1, 17, 17)
    idx = idx.astype(int)

    feature_description_df.index = idx

    st.table(feature_description_df)

with tab2:
    st.markdown("### Histogram Plot Analysis")

    st.markdown("#### Here, y tells us the outcome of subscription")

    col1, col2 = st.columns([1, 3])

    with col1:
        st.markdown("### Control Panel")
        column_options = bank_marketing_df.columns
        column_name = st.selectbox("Select Column", column_options)

    with col2:
        fig = sns.histplot(data=bank_marketing_df,
                           x=column_name, hue='y').figure
        st.pyplot(fig)
        plt.close()

    st.markdown("### Correlation Heatmap")

    col1, col2 = st.columns([1, 3])

    with col2:
        fig = sns.heatmap(bank_marketing_df.corr(
            numeric_only=True), cmap='coolwarm').figure
        st.pyplot(fig)
        plt.close()

with tab3:
    st.text("Work in progress...")
