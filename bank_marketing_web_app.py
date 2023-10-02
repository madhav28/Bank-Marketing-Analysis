import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

bank_marketing_df = pd.read_csv("https://raw.githubusercontent.com/madhav28/Bank-Marketing-Analysis/main/bank-full.csv", sep=';')
customer_description_df = pd.read_csv("column_description.txt", names=['Type', 'Description'])

st.title("Bank Marketing Dataset Analysis")

st.markdown("### Understanding Data: Feature Description")

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
    fig = sns.heatmap(bank_marketing_df.corr(numeric_only=True), cmap='coolwarm').figure
    st.pyplot(fig)
    plt.close()





