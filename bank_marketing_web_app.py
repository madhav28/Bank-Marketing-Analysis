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

st.title("Bank Marketing Analysis")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ['About Data', 'Objectives', 'Data Visualisation / Analysis', 'Predictive Modelling', 'Conclusion'])

bank_marketing_df = pd.read_csv(
    "https://raw.githubusercontent.com/madhav28/Bank-Marketing-Analysis/main/bank-full.csv", sep=';')
bank_marketing_df.rename(columns={'y': 'outcome'},  inplace=True)

with tab1:

    st.image("overview.jpg")

    st.markdown("**Data Overview:**")

    st.markdown("The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. \
                 The classification goal is to predict if the client will subscribe a term deposit (variable outcome).")

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
    st.markdown("**Objectives of the application:**")

    st.markdown("In this application, the following two questions are answered:")
    st.markdown(
        "   **💡 What kind of customers are likely to subscribe to a term deposit?**")
    st.markdown(
        "   **💡 What marketing strategies are to be employed by the bank for a successful and an economical campaign?**")
    st.markdown(
        "The above two questions are answered in the **Data Visualisation / Analysis** tab of the application. \
        Particularly, first question and second question are answered in the **Target Customers** and **Marketing** tabs respectively.")

with tab3:

    tab6, tab7 = st.tabs(
        ['Target Customers', 'Marketing Strategies'])

    with tab6:

        st.markdown("#### Understanding what kind of customers are likely \
                    to subscribe to a term deposit")

        target_customers_df = bank_marketing_df[['age', 'marital', 'education', 'job',
                                                 'default', 'balance', 'housing', 'loan', 'outcome']]

        feature_analysis_1 = st.checkbox(
            "Feature Analysis", key="feature_analysis_1")

        column_options = ['age', 'marital', 'education', 'job',
                          'default', 'balance', 'housing', 'loan']

        if feature_analysis_1:

            st.markdown("##### Histogram Plot")

            col1, col2 = st.columns([1, 3])

            with col1:
                column_name = st.selectbox("Column Name", column_options)
                bins = st.selectbox("Bins", np.linspace(1, 20, 20).astype(
                    int), 9, key='target_customers_hist')

            with col2:
                if column_name == 'job':
                    plt.figure(figsize=(16, 11))

                fig = sns.histplot(data=target_customers_df,
                                   x=column_name, hue='outcome', bins=bins).figure
                st.pyplot(fig)
                plt.close()

            st.markdown("##### Box Plot")

            num_col_options = ["age", "balance"]

            col1, col2 = st.columns([1, 3])

            with col1:
                box_column_1 = st.selectbox(
                    "Column Name", num_col_options, key="box_column_1")

                min_val = np.min(target_customers_df[box_column_1])
                max_val = np.max(target_customers_df[box_column_1])

                box_column_1_options = np.unique(
                    target_customers_df[box_column_1])

                box_column_1_options = np.sort(box_column_1_options)

                box_column_1_start, box_column_1_end = st.select_slider("Select the range of "+box_column_1,
                                                                        options=box_column_1_options,
                                                                        value=(
                                                                            min_val, max_val),
                                                                        key="box_column_1_range")

            with col2:
                box_column_1_df = target_customers_df[(target_customers_df[box_column_1] >= box_column_1_start) &
                                                      (target_customers_df[box_column_1] <= box_column_1_end)]

                palette = {"no": "#33B5FF", "yes": "#FFB533"}

                fig = sns.boxplot(data=box_column_1_df,
                                  x='outcome', y=box_column_1,
                                  palette=palette).figure
                st.pyplot(fig)
                plt.close()

            st.markdown("##### Violin Plot")

            col1, col2 = st.columns([1, 3])

            with col1:
                violin_column_1 = st.selectbox(
                    "Column Name", num_col_options, key="violin_column_1")

                min_val = np.min(target_customers_df[violin_column_1])
                max_val = np.max(target_customers_df[violin_column_1])

                violin_column_1_options = np.unique(
                    target_customers_df[violin_column_1])

                violin_column_1_options = np.sort(violin_column_1_options)

                violin_column_1_start, violin_column_1_end = st.select_slider("Select the range of "+violin_column_1,
                                                                              options=violin_column_1_options,
                                                                              value=(
                                                                                  min_val, max_val),
                                                                              key="violin_column_1_range")

            with col2:

                violin_column_1_df = target_customers_df[(target_customers_df[violin_column_1] >= violin_column_1_start) &
                                                         (target_customers_df[violin_column_1] <= violin_column_1_end)]

                palette = {"no": "#33B5FF", "yes": "#FFB533"}

                fig = sns.violinplot(data=violin_column_1_df,
                                     x='outcome', y=violin_column_1,
                                     palette=palette).figure
                st.pyplot(fig)
                plt.close()

            st.markdown(
                "**💡 From the above plot analysis, we get the following as the desirable traits of a potential subscriber:**")

            df1 = {'Features': column_options}

            category_with_highest_subscriptions = ["30-40", "married", " > secondary", "management",
                                                   "no", "< 10000", "no", "no"]

            df1["Category with highest subscriptions"] = category_with_highest_subscriptions

            df1 = pd.DataFrame(df1)

            st.table(df1)

        rel_plot_analysis_1 = st.checkbox(
            "Analysing the Relationship between Features", key="rel_plot_analysis_1")

        if rel_plot_analysis_1:

            st.markdown("##### Scatter Plot")

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
                    x=column_x, y=column_y, color=alt.Color('outcome:N',
                                                            scale=custom_color_scale)).interactive()

                st.altair_chart(chart, theme="streamlit",
                                use_container_width=True)

            st.markdown("##### Correlation Heat Map")

            col1, col2 = st.columns([1, 3])

            with col2:

                fig = sns.heatmap(target_customers_df.corr(numeric_only=True),
                                  annot=True, cmap="coolwarm").figure
                st.pyplot(fig)
                plt.close()

            st.markdown("##### KDE Plot")

            col1, col2 = st.columns([1, 3])

            with col1:

                num_col_options = ['age', 'balance']

                box_cx_1 = st.selectbox(
                    "Column for x-axis", num_col_options, 0, key="box_cx_1")
                box_cy_1 = st.selectbox(
                    "Column for y-axis", num_col_options, 1, key="box_cy_1")

            with col2:

                kde_df_1 = target_customers_df[(target_customers_df['balance'] <= 10000) & (
                    target_customers_df['age'] <= 90)]

                fig = sns.kdeplot(kde_df_1,
                                  x=box_cx_1, y=box_cy_1).figure
                st.pyplot(fig)
                plt.close()

            st.markdown("💡 Here, from the plot analysis we can infer that irrespectively of the age, \
                        most of the people have balance less than 5000. We can also validate this from the \
                        Heat Map plot. And most of the subscriptions fall in this balance range.")

        hiplot_visualisation_1 = st.checkbox(
            "Hiplot Visualisation", key="hiplot_visualisation_1")

        if hiplot_visualisation_1:

            st.markdown(
                "##### Hiplot Visualisation")

            credit_analysis_df = target_customers_df[[
                'outcome', 'marital', 'loan', 'balance', 'job', 'education', 'age']]

            exp = hip.Experiment.from_dataframe(credit_analysis_df)

            def save_hiplot_to_html(exp):
                output_file = "hiplot_plot_1.html"
                exp.to_html(output_file)
                return output_file

            hiplot_html_file = save_hiplot_to_html(exp)
            st.components.v1.html(
                open(hiplot_html_file, 'r').read(), height=1200, scrolling=True)

            st.markdown("💡 From the Hiplot, we can get an overview on what kind of clients are \
                        more likely to subscribe to a term deposit. So, the clients of age in \
                        the range of 30-40, with atleast a secondary education, with a balance less \
                        than 5000, with no personal loans, and those who are married are more likely \
                        to subscribe to a term deposit.")

    with tab7:

        st.markdown("#### Understanding successful marketing strategies")

        feature_analysis_2 = st.checkbox(
            "Feature Analysis", key="feature_analysis_2")

        strategies_df = bank_marketing_df[['contact', 'day', 'month', 'duration',
                                           'campaign', 'pdays', 'previous', 'poutcome', 'outcome']]

        if feature_analysis_2:

            st.markdown("##### Histogram Plot")

            col1, col2 = st.columns([1, 3])

            with col1:
                column_options = ['contact', 'day', 'month', 'duration',
                                  'campaign', 'pdays', 'previous', 'poutcome', 'outcome']
                column_name = st.selectbox("Column Name", column_options)
                bins = st.selectbox("Bins", np.linspace(
                    1, 20, 20).astype(int), 9, key='strategies')

            with col2:
                fig = sns.histplot(data=bank_marketing_df,
                                   x=column_name, hue='outcome', bins=bins).figure
                st.pyplot(fig)
                plt.close()

            st.markdown("##### Box Plot")

            num_col_options = ["day", "duration", "campaign",
                               "pdays", "previous"]

            col1, col2 = st.columns([1, 3])

            with col1:
                box_column_2 = st.selectbox(
                    "Column Name", num_col_options, key="box_column_2")

                min_val = np.min(strategies_df[box_column_2])
                max_val = np.max(strategies_df[box_column_2])

                box_column_2_options = np.unique(strategies_df[box_column_2])

                box_column_2_options = np.sort(box_column_2_options)

                box_column_2_start, box_column_2_end = st.select_slider("Select the range of "+box_column_2,
                                                                        options=box_column_2_options,
                                                                        value=(
                                                                            min_val, max_val),
                                                                        key="box_column_2_range")

            with col2:

                box_column_2_df = strategies_df[(strategies_df[box_column_2] >= box_column_2_start) &
                                                (strategies_df[box_column_2] <= box_column_2_end)]

                palette = {"no": "#33B5FF", "yes": "#FFB533"}

                fig = sns.boxplot(data=box_column_2_df,
                                  x='outcome', y=box_column_2,
                                  palette=palette).figure
                st.pyplot(fig)
                plt.close()

            st.markdown("##### Violin Plot")

            col1, col2 = st.columns([1, 3])

            with col1:
                violin_column_2 = st.selectbox(
                    "Column Name", num_col_options, key="violin_column_2")

                min_val = np.min(strategies_df[violin_column_2])
                max_val = np.max(strategies_df[violin_column_2])

                violin_column_2_options = np.unique(
                    strategies_df[violin_column_2])

                violin_column_2_options = np.sort(violin_column_2_options)

                violin_column_2_start, violin_column_2_end = st.select_slider("Select the range of "+violin_column_2,
                                                                              options=violin_column_2_options,
                                                                              value=(
                                                                                  min_val, max_val),
                                                                              key="violin_column_2_range")

            with col2:
                violin_column_2_df = strategies_df[(strategies_df[violin_column_2] >= violin_column_2_start) &
                                                   (strategies_df[violin_column_2] <= violin_column_2_end)]

                palette = {"no": "#33B5FF", "yes": "#FFB533"}

                fig = sns.violinplot(data=violin_column_2_df,
                                     x='outcome', y=violin_column_2,
                                     palette=palette).figure
                st.pyplot(fig)
                plt.close()

            st.markdown(
                "**From the above plots, we get the following inferences:**  ")
            st.markdown("⭐ Last contact day of the week doesn't determine the outcome of the subscription \
                        because the outcomes are fairly randomly distributed.")
            st.markdown("⭐ Percentage of subscriptions are very high when the last contacted months are \
                         March, September, October and December. Client who are last contacted in those \
                         months are more likely to subscribe to a term deposit.")
            st.markdown("⭐ As the duration of call with the client increases, the probability of the client \
                        subscribing to a term deposit increases. Because this shows that client is interested \
                        in the conversation and considering to subcribe.")
            st.markdown("⭐ Optimal number of contacts to be performed for a successful subscription is less than 10. \
                        After 10 contacts the subscription rate is very low.")
            st.markdown("⭐ Majority of the customers decide whether to subscribe to a term deposit or not \
                        within 100 days of the previous campaign.")
            st.markdown("⭐ Clients who subscribed to a term deposit in the previous campaign are highly likely \
                        to subscribe to a term deposit.")

        rel_plot_analysis_2 = st.checkbox(
            "Analysing the Relationship between Features", key="rel_plot_analysis_2")

        if rel_plot_analysis_2:

            st.markdown("##### Scatter Plot")

            col1, col2 = st.columns([1, 3])

            with col1:

                column_options = ['contact', 'day', 'month', 'duration',
                                  'campaign', 'pdays', 'previous', 'poutcome', 'outcome']

                column_x = st.selectbox(
                    "Column for x-axis", column_options, 6, key='strategies_sx')
                column_y = st.selectbox(
                    "Column for y-axis", column_options, 7, key='strategies_sy')

            with col2:

                custom_color_scale = alt.Scale(
                    domain=['no', 'yes'], range=['red', 'green'])

                chart = alt.Chart(strategies_df).mark_circle().encode(
                    x=column_x, y=column_y, color=alt.Color('outcome:N',
                                                            scale=custom_color_scale)).interactive()

                st.altair_chart(chart, theme="streamlit",
                                use_container_width=True)

            st.markdown("##### Correlation Heat Map")

            col1, col2 = st.columns([1, 3])

            with col2:

                fig = sns.heatmap(strategies_df.corr(numeric_only=True),
                                  annot=True, cmap="coolwarm").figure
                st.pyplot(fig)
                plt.close()

            st.markdown("##### KDE Plot")

            col1, col2 = st.columns([1, 3])

            with col1:

                num_col_options = ["day", "duration",
                                   "campaign", "pdays"]

                box_cx_2 = st.selectbox(
                    "Column for x-axis", num_col_options, 0, key="box_cx_2")
                box_cy_2 = st.selectbox(
                    "Column for y-axis", num_col_options, 1, key="box_cy_2")

            with col2:

                fig = sns.kdeplot(strategies_df,
                                  x=box_cx_2, y=box_cy_2).figure
                st.pyplot(fig)
                plt.close()

            st.markdown(
                "We can infer the following from the above plot analysis:")
            st.markdown("⭐ Most of the clients who subscribed in the previous campaign are \
                        mostly going to subscribe for the term deposit in the current campaign.")
            st.markdown("⭐ From the Correlation Heat Map, we can see that most of the features are \
                        not correlated with each other except previous and pdays where the correlation \
                        between these two features is 0.45.")
            st.markdown(
                "⭐ Optimal number of contants for a successful subscribers is less than 10.")

        hiplot_analysis_2 = st.checkbox(
            "Hiplot Visualisation", key="hiplot_analysis_2")

        if hiplot_analysis_2:

            st.markdown(
                "##### Hiplot Visualisation")

            strategy_analysis_df = strategies_df[[
                'outcome', 'campaign', 'poutcome', 'previous']]

            exp = hip.Experiment.from_dataframe(strategy_analysis_df)

            def save_hiplot_to_html(exp):
                output_file = "hiplot_plot_2.html"
                exp.to_html(output_file)
                return output_file

            hiplot_html_file = save_hiplot_to_html(exp)
            st.components.v1.html(
                open(hiplot_html_file, 'r').read(), height=1200, scrolling=True)

            st.markdown("💡 From the Hiplot visualisation, we can see that when the outcome of the previous \
                        campaign is a success then the number of successes is greater than the number of \
                        failures unlike other cases. Looking at the campaign, we can determine that for most of \
                        the sucesses the number of contacts made is less than 10.")


with tab4:

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


with tab5:

    st.markdown("**💡 Desirable traits of the target customers:**  ")

    fea_col = ['age', 'marital', 'education', 'job',
               'default', 'balance', 'housing', 'loan']
    tar_vals = ["30-40", "married", " > secondary", "management",
                "no", "< 10000", "no", "no"]
    traits = {"Features": fea_col, "Target Values": tar_vals}

    traits_df = pd.DataFrame(traits)
    st.table(traits_df)

    st.markdown("**💡 Marketing Strategy:**  ")
    st.markdown("As a first step, the bank should allocate most of its resources for the marketing \
                campaign on the clients with the above traits. Since these clients are the potential \
                subscribers. Later, banks should optimise contacts with the clients based on the duration \
                and number of previous contacts. If the duration of the calls is short and the number of \
                contacts exceeds more than 10 then the client is not a potential subscriber. Instead of \
                contacting this client, bank can focus their resourses on other potential new subscribers.")

    st.markdown("**💡 Final Remarks:**  ")
    st.markdown("Through this application, we performed IDA and EDA on the Bank Marketing Dataset \
                to get valuable insights from the data. These insights helped us to design a \
                optimal marketing campaign for increasing the number of clients subscribing for a \
                term deposit.")
