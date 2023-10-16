import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import hiplot as hip
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

st.title("Bank Marketing Dataset Analysis")

tab1, tab2, tab3 = st.tabs(
    ['About Data', 'Data Visualisation / Analysis', 'Predictive Modelling'])

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
            bins = st.selectbox("Bins", np.linspace(
                1, 10, 10).astype(int), 9, key='target_customers_hist')

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

            column_x = st.selectbox(
                "Column for x-axis", column_options, 0, key='target_customers_sx')
            column_y = st.selectbox(
                "Column for y-axis", column_options, 5, key='target_customers_sy')

        with col2:

            custom_color_scale = alt.Scale(
                domain=['no', 'yes'], range=['red', 'green'])

            chart = alt.Chart(target_customers_df).mark_circle().encode(
                x=column_x, y=column_y, color=alt.Color('y:N',
                                                        scale=custom_color_scale)).interactive()

            st.altair_chart(chart, theme="streamlit", use_container_width=True)

        st.markdown(
            "##### Hiplot Visualisation")

        credit_analysis_df = target_customers_df[[
            'y', 'marital', 'loan', 'balance', 'job', 'education', 'age']]

        exp = hip.Experiment.from_dataframe(credit_analysis_df)

        def save_hiplot_to_html(exp):
            output_file = "hiplot_plot_1.html"
            exp.to_html(output_file)
            return output_file

        hiplot_html_file = save_hiplot_to_html(exp)
        st.components.v1.html(
            open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)

    with tab5:

        st.markdown("#### Understanding successful marketing strategies:")

        st.markdown("##### Histogram Plot Analysis")

        strategies_df = bank_marketing_df[['contact', 'day', 'month', 'duration',
                                           'campaign', 'pdays', 'previous', 'poutcome', 'y']]

        col1, col2 = st.columns([1, 3])

        with col1:
            column_options = ['contact', 'day', 'month', 'duration',
                              'campaign', 'pdays', 'previous', 'poutcome', 'y']
            column_name = st.selectbox("Column Name", column_options)
            bins = st.selectbox("Bins", np.linspace(
                1, 10, 10).astype(int), 9, key='strategies')

        with col2:
            fig = sns.histplot(data=bank_marketing_df,
                               x=column_name, hue='y', bins=bins).figure
            st.pyplot(fig)
            plt.close()

        st.markdown("##### Scatter Plot Analysis")

        col1, col2 = st.columns([1, 3])

        with col1:

            column_options = ['contact', 'day', 'month', 'duration',
                              'campaign', 'pdays', 'previous', 'poutcome', 'y']

            column_x = st.selectbox(
                "Column for x-axis", column_options, 6, key='strategies_sx')
            column_y = st.selectbox(
                "Column for y-axis", column_options, 7, key='strategies_sy')

        with col2:

            custom_color_scale = alt.Scale(
                domain=['no', 'yes'], range=['red', 'green'])

            chart = alt.Chart(strategies_df).mark_circle().encode(
                x=column_x, y=column_y, color=alt.Color('y:N',
                                                        scale=custom_color_scale)).interactive()

            st.altair_chart(chart, theme="streamlit", use_container_width=True)

        st.markdown(
            "##### Hiplot Visualisation")

        strategy_analysis_df = strategies_df[[
            'y', 'campaign', 'poutcome', 'previous']]

        exp = hip.Experiment.from_dataframe(strategy_analysis_df)

        def save_hiplot_to_html(exp):
            output_file = "hiplot_plot_2.html"
            exp.to_html(output_file)
            return output_file

        hiplot_html_file = save_hiplot_to_html(exp)
        st.components.v1.html(
            open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)


with tab3:

    st.markdown(
        "#### Develop a decision tree classifier and predict the subscription outcome")

    st.markdown("##### Model identification:")

    features = st.multiselect(
        "Select features for modelling", bank_marketing_df.columns[0:16],
        default=np.array(bank_marketing_df.columns[0:9]))

    train_percent = st.slider("Select percentage of the data for training",
                              min_value=1, max_value=100, value=75)

    generate_model = st.button("Generate Model", type="primary")

    if generate_model:

        X = bank_marketing_df.drop(columns=['y'])
        y = bank_marketing_df['y']

        le = LabelEncoder()

        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col])

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=1 -
                                                            (float(train_percent)/100),
                                                            random_state=42)

        classifier = DecisionTreeClassifier()

        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        accuracy = round(accuracy*100, 2)

        st.markdown(f"##### Accuracy of the model is {accuracy}%")
