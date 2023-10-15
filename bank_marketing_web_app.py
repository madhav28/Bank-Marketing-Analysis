import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

st.title("Bank Marketing Dataset Analysis")

tab1, tab2, tab3 = st.tabs(
    ['About Data', 'Data Visualisation', 'Predictive Modelling'])

bank_marketing_df = pd.read_csv(
    "https://raw.githubusercontent.com/madhav28/Bank-Marketing-Analysis/main/bank-full.csv", sep=';')

with tab1:

    st.markdown("The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. \
                 The classification goal is to predict if the client will subscribe a term deposit (variable y). \
                 In this app, the following two questions are answered: What kind of customers are likely to subscribe \
                 to a term deposit and What are marketing strategies are to be employed by the bank for a successful campaign. \
                 The description of the data used in the application can be found below:")

    st.markdown("**Data Description:**")

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

    tab4, tab5 = st.tabs(['Target Customers', 'Marketing Strategies'])

    with tab4:

        st.markdown("#### Understanding what kind of customers are likely \
                 to subscribe to a term deposit:")

        st.markdown("##### Histogram Plot Analysis")

        target_customers_df = bank_marketing_df[['age', 'marital', 'education', 'job',
                                                 'default', 'balance', 'housing', 'loan', 'y']]

        col1, col2 = st.columns([1, 3])

        with col1:
            column_options = ['age', 'marital', 'education', 'job',
                              'default', 'balance', 'housing', 'loan']
            column_name = st.selectbox("Column Name", column_options)
            bins = st.selectbox("Bins", np.linspace(1, 10, 10).astype(int), 9)

        with col2:
            fig = sns.histplot(data=bank_marketing_df,
                               x=column_name, hue='y', bins=bins).figure
            st.pyplot(fig)
            plt.close()

        st.markdown("##### Scatter Plot Analysis")

        col1, col2 = st.columns([1, 3])

        with col1:

            column_options = ['age', 'marital', 'education', 'job',
                              'default', 'balance', 'housing', 'loan']

            column_x = st.selectbox("Column for x-axis", column_options, 0)
            column_y = st.selectbox("Column for y-axis", column_options, 5)

        with col2:

            chart = alt.Chart(target_customers_df).mark_circle().encode(
                x=column_x, y=column_y, color='y').interactive()

            st.altair_chart(chart, theme="streamlit", use_container_width=True)


with tab3:
    st.text("Work in progress...")
