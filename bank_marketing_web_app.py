import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import altair as alt
import hiplot as hip
import matplotlib.pyplot as plt
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

        st.markdown(
            "The following features of the data are analysed in this tab:")
        st.markdown("**age, marital, education, job, default, \
                    balance, housing, loan, and outcome**")

        target_customers_df = bank_marketing_df[['age', 'marital', 'education', 'job',
                                                 'default', 'balance', 'housing', 'loan', 'outcome']]

        st.markdown("Select the following checkboxes to analyse the data:")

        feature_analysis_1 = st.checkbox(
            "**Feature Analysis**", key="feature_analysis_1")

        column_options = ['age', 'marital', 'education', 'job',
                          'default', 'balance', 'housing', 'loan']

        if feature_analysis_1:

            st.markdown("#### Feature Analysis")

            st.markdown("---")

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

            st.markdown("A histogram plot is a graphical representation of the distribution \
                        of the data. Here, histogram plots are plotted for all the features \
                        in this tab and hue in every histogram is set equal to outcome. Users \
                        can select the feature of their interest, set the number of bins and \
                        visualise the distributions. Through these distributions, users can \
                        get an idea about the population distribution of each feature. \
                        Decisive conclusions can be made by understanding for what values \
                        on the x-axis the count of the outcome on the y-axis is the most.")

            st.markdown("---")

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

                custom_colors = {'yes': '#FFB533', 'no': '#33B5FF'}

                fig = px.box(box_column_1_df,
                             x='outcome', y=box_column_1,
                             color='outcome', color_discrete_map=custom_colors)

                st.plotly_chart(fig)
                plt.close()

            st.markdown(
                "A box plot provide summary of the key statistics of the data like quartile information and outliers. \
                Here, box plots give us information about the spread of the feature data with respect to the \
                campaign outcome.")

            st.markdown("---")

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

                custom_colors = {'yes': '#FFB533', 'no': '#33B5FF'}

                fig = px.violin(violin_column_1_df, box=True,
                                x='outcome', y=violin_column_1,
                                color='outcome', color_discrete_map=custom_colors)

                st.plotly_chart(fig)
                plt.close()

            st.markdown("A violin plot is a data visualization that combines elements of a box \
                        plot with a kernel density estimation plot. Through the violine plots, we \
                        can estimate the most likely value of a feature with respect to a \
                        particular outcome.")

            st.markdown("---")

            st.markdown(
                "**💡 Using the histogram, box, and violin plots, the following are identified \
                as the desirable traits of a potential subscriber:**")

            df1 = {'Features': column_options}

            category_with_highest_subscriptions = ["30-40", "married", " > secondary", "management",
                                                   "no", "< 10000", "no", "no"]

            df1["Category with highest subscriptions"] = category_with_highest_subscriptions

            df1 = pd.DataFrame(df1)

            st.table(df1)

            st.markdown("---")

        rel_plot_analysis_1 = st.checkbox(
            "**Analysing the Relationship between Features**", key="rel_plot_analysis_1")

        if rel_plot_analysis_1:

            st.markdown("#### Analysing the Relationship between Features")

            st.markdown("---")

            st.markdown("##### 2D Scatter Plot")

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

            st.markdown("A 2d scatter plot is a type of data visualisation where \
                        we plot points corresponding to two variables on a \
                        two-dimensional coordinate system. Here, user is given an option \
                        to visualise 2d scatter plots between any two variables and points are \
                        colored either as red or green based on the outcome. Through these plots, \
                        user can understand the relationship between two features of the data.")

            st.markdown("---")

            st.markdown("##### 3D Scatter Plot")

            col1, col2 = st.columns([1, 3])

            with col1:

                column_options = ['age', 'marital', 'education', 'job',
                                  'default', 'balance', 'housing', 'loan']

                column_x = st.selectbox(
                    "Column for x-axis", column_options, 0, key='target_customers_s3x')
                column_y = st.selectbox(
                    "Column for y-axis", column_options, 4, key='target_customers_s3y')
                column_z = st.selectbox(
                    "Column for z-axis", column_options, 7, key='target_customers_s3z')

            with col2:

                custom_colors = {'yes': 'green', 'no': 'red'}

                fig = px.scatter_3d(target_customers_df,
                                    x=column_x, y=column_y, z=column_z,
                                    color='outcome', color_discrete_map=custom_colors)

                st.plotly_chart(fig)

            st.markdown("A scatter 3d plot is a type of data visualisation where \
                        we plot points corresponding to three variables on a \
                        three-dimensional coordinate system. Here, user is given an option \
                        to visualise 3d scatter plots between any three variables and points are \
                        colored either as red or green based on the outcome. Through these plots, \
                        user can understand the relationship between three features of the data.")

            st.markdown("---")

            st.markdown("##### Correlation Heat Map")

            col1, col2 = st.columns([1, 3])

            with col2:

                fig = sns.heatmap(target_customers_df.corr(numeric_only=True),
                                  annot=True, cmap="coolwarm").figure
                st.pyplot(fig)
                plt.close()

            st.markdown("A correlation heat map is a graphical representation of a correlation \
                        matrix that displays the pairwise correlation coefficients between all \
                        the numeric variables. Here, we only see correlation matrix for only \
                        two variables (age and balance) because there are only two numeric \
                        features out of all eight features.")

            st.markdown("---")

            st.markdown(
                "**From the scatter plots and correlation heat map, we can infer the following:**")
            st.markdown("⭐ Irrespective of the age most of the people have balance less than 5000. \
                        Also, there is no correlation between age and balance.")
            st.markdown(
                "⭐ Most of the people who have taken a personal and have credit in default have an \
                    age of greater than or equal to 45. There kind of people rarlely subscribe to a \
                    term deposit.")
            st.markdown("⭐ Out of loan and default features, default is a stronger feature in \
                        deciding the subscription outcome. Irrespective of loan, subscription rate \
                        drastically increases when the person has no credit default.")

        hiplot_visualisation_1 = st.checkbox(
            "**Hiplot Visualisation**", key="hiplot_visualisation_1")

        if hiplot_visualisation_1:

            st.markdown(
                "#### Hiplot Visualisation")

            st.markdown("---")

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

            st.markdown("---")

            st.markdown("💡 **From the Hiplot, we can get an overview on what kind of clients are \
                        more likely to subscribe to a term deposit. So, the clients of age in \
                        the range of 30-40, with atleast a secondary education, with a balance less \
                        than 5000, with no personal loans, and those who are married are more likely \
                        to subscribe to a term deposit.**")

    with tab7:

        st.markdown("#### Understanding successful marketing strategies")
        st.markdown(
            "The following features of the data are analysed in this tab:")
        st.markdown("**contact, day, month, duration, campaign, pdays, \
                    previous, poutcome, and outcome**")

        st.markdown("Select the following checkboxes to analyse the data:")

        feature_analysis_2 = st.checkbox(
            "**Feature Analysis**", key="feature_analysis_2")

        strategies_df = bank_marketing_df[['contact', 'day', 'month', 'duration',
                                           'campaign', 'pdays', 'previous', 'poutcome', 'outcome']]

        if feature_analysis_2:

            st.markdown("#### Feature Analysis")

            st.markdown("---")

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

            st.markdown("A histogram plot is a graphical representation of the distribution \
                        of the data. Here, histogram plots are plotted for all the features \
                        in this tab and hue in every histogram is set equal to outcome. Users \
                        can select the feature of their interest, set the number of bins and \
                        visualise the distributions. Through these distributions, users can \
                        get an idea about the population distribution of each feature. \
                        Decisive conclusions can be made by understanding for what values \
                        on the x-axis the count of the outcome on the y-axis is the most.")

            st.markdown("---")

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

                custom_colors = {'yes': '#FFB533', 'no': '#33B5FF'}

                fig = px.box(box_column_2_df,
                             x='outcome', y=box_column_2,
                             color='outcome', color_discrete_map=custom_colors)

                st.plotly_chart(fig)
                plt.close()

            st.markdown(
                "A box plot provide summary of the key statistics of the data like quartile information and outliers. \
                Here, box plots give us information about the spread of the feature data with respect to the \
                campaign outcome.")

            st.markdown("---")

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

                custom_colors = {'yes': '#FFB533', 'no': '#33B5FF'}

                fig = px.violin(violin_column_2_df, box=True,
                                x='outcome', y=violin_column_2,
                                color='outcome', color_discrete_map=custom_colors)

                st.plotly_chart(fig)
                plt.close()

            st.markdown("A violin plot is a data visualization that combines elements of a box \
                        plot with a kernel density estimation plot. Through the violine plots, we \
                        can estimate the most likely value of a feature with respect to a \
                        particular outcome.")

            st.markdown("---")

            st.markdown(
                "**From the histogram, box, and violin plots, we get the following inferences:**  ")
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
            "**Analysing the Relationship between Features**", key="rel_plot_analysis_2")

        if rel_plot_analysis_2:

            st.markdown("#### Analysing the Relationship between Features")

            st.markdown("---")

            st.markdown("##### 2D Scatter Plot")

            col1, col2 = st.columns([1, 3])

            with col1:

                column_options = ['contact', 'day', 'month', 'duration',
                                  'campaign', 'pdays', 'previous', 'poutcome', 'outcome']

                column_x = st.selectbox(
                    "Column for x-axis", column_options, 3, key='strategies_sx')
                column_y = st.selectbox(
                    "Column for y-axis", column_options, 4, key='strategies_sy')

            with col2:

                custom_color_scale = alt.Scale(
                    domain=['no', 'yes'], range=['red', 'green'])

                chart = alt.Chart(strategies_df).mark_circle().encode(
                    x=column_x, y=column_y, color=alt.Color('outcome:N',
                                                            scale=custom_color_scale)).interactive()

                st.altair_chart(chart, theme="streamlit",
                                use_container_width=True)

            st.markdown("A 2d scatter plot is a type of data visualisation where \
                        we plot points corresponding to two variables on a \
                        two-dimensional coordinate system. Here, user is given an option \
                        to visualise 2d scatter plots between any two variables and points are \
                        colored either as red or green based on the outcome. Through these plots, \
                        user can understand the relationship between two features of the data.")

            st.markdown("---")

            st.markdown("##### 3D Scatter Plot")

            col1, col2 = st.columns([1, 3])

            with col1:

                column_options = ['contact', 'day', 'month', 'duration',
                                  'campaign', 'pdays', 'previous', 'poutcome', 'outcome']

                column_x = st.selectbox(
                    "Column for x-axis", column_options, 1, key='strategies_s3x')
                column_y = st.selectbox(
                    "Column for y-axis", column_options, 2, key='strategies_s3y')
                column_z = st.selectbox(
                    "Column for z-axis", column_options, 0, key='strategies_s3z')

            with col2:

                custom_colors = {'yes': 'green', 'no': 'red'}

                fig = px.scatter_3d(strategies_df,
                                    x=column_x, y=column_y, z=column_z,
                                    color='outcome', color_discrete_map=custom_colors)

                st.plotly_chart(fig)

            st.markdown("A scatter 3d plot is a type of data visualisation where \
                        we plot points corresponding to three variables on a \
                        three-dimensional coordinate system. Here, user is given an option \
                        to visualise 3d scatter plots between any three variables and points are \
                        colored either as red or green based on the outcome. Through these plots, \
                        user can understand the relationship between three features of the data.")

            st.markdown("---")

            st.markdown("##### Correlation Heat Map")

            col1, col2 = st.columns([1, 3])

            with col2:

                fig = sns.heatmap(strategies_df.corr(numeric_only=True),
                                  annot=True, cmap="coolwarm").figure
                st.pyplot(fig)
                plt.close()

            st.markdown("A correlation heat map is a graphical representation of a correlation \
                        matrix that displays the pairwise correlation coefficients between all \
                        the numeric variables. Here, we don't see any strong correlation between \
                        any features. This is because all the numeric features are independent of each other.")

            st.markdown("---")

            st.markdown(
                "**We can infer the following from the above plot analysis:**")
            st.markdown("⭐ Most of the clients who subscribed in the previous campaign are \
                        mostly going to subscribe for the term deposit in the current campaign.")
            st.markdown("⭐ From the Correlation Heat Map, we can see that most of the features are \
                        not correlated with each other except previous and pdays where the correlation \
                        between these two features is 0.45.")
            st.markdown(
                "⭐ Optimal number of contacts for a successful potential subscriber is less than 10.")

        hiplot_analysis_2 = st.checkbox(
            "**Hiplot Visualisation**", key="hiplot_analysis_2")

        if hiplot_analysis_2:

            st.markdown(
                "#### Hiplot Visualisation")

            st.markdown("---")

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

            st.markdown("---")

            st.markdown("💡 **From the Hiplot visualisation, we can see that when the outcome of the previous \
                        campaign is a success then the number of successes is greater than the number of \
                        failures unlike other cases. Looking at the campaign, we can determine that for most of \
                        the sucesses the number of contacts made is less than 10.**")


with tab4:

    st.markdown(
        "#### Develop a decision tree classifier and predict the subscription outcome")

    st.markdown("---")

    st.markdown("#### Model Development")

    features = st.multiselect(
        "Select features for modelling", bank_marketing_df.columns[0:16],
        default=np.array(bank_marketing_df.columns[0:16]))

    train_percent = st.slider("Select percentage of the data for training",
                              min_value=1, max_value=100, value=20)

    max_tree_depth = st.number_input(
        "Maximum Tree Depth:", value=5, step=1, format="%d")

    X = bank_marketing_df[features]
    y = bank_marketing_df['outcome']

    label_encoders = {}

    for col in X.columns:
        label_encoder = LabelEncoder()
        X[col] = label_encoder.fit_transform(X[col])
        label_encoders[col] = label_encoder

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=1 -
                                                        (float(train_percent)/100),
                                                        random_state=42)

    classifier = DecisionTreeClassifier(max_depth=max_tree_depth)

    classifier.fit(X_train, y_train)

    y_train_pred = classifier.predict(X_train)

    y_test_pred = classifier.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)

    test_accuracy = accuracy_score(y_test, y_test_pred)

    train_accuracy = round(train_accuracy*100, 2)

    test_accuracy = round(test_accuracy*100, 2)

    st.markdown(
        f"##### Training accuracy of the model is {train_accuracy}%")

    st.markdown(f"##### Testing accuracy of the model is {test_accuracy}%")

    st.markdown("---")

    st.markdown("#### Predict Subscription Outcome")

    X_pred = {}

    for feature in features:
        if feature == "age":
            age = st.number_input(
                "Enter age:", min_value=18, max_value=100, value=35, step=1, format="%d")
            X_pred["age"] = [age]
        if feature == "job":
            options = ["admin.", "blue-collar", "technician", "services", "management",
                       "retired", "self-employed", "entrepreneur", "housemaid", "student"]
            job = st.selectbox("Select job:", options)
            X_pred["job"] = [job]
        if feature == "marital":
            options = ["married", "single", "divorced"]
            marital = st.selectbox("Select marital:", options)
            X_pred["marital"] = [marital]
        if feature == "education":
            options = ["unknown", "primary", "secondary", "tertiary"]
            education = st.selectbox("Select education:", options)
            X_pred["education"] = [education]
        if feature == "default":
            options = ["no", "yes"]
            default = st.selectbox("Select default:", options)
            X_pred["default"] = [default]
        if feature == "balance":
            balance = st.number_input(
                "Enter balance:", value=2500, step=1, format="%d")
            X_pred["balance"] = [balance]
        if feature == "housing":
            options = ["no", "yes"]
            housing = st.selectbox("Select housing:", options)
            X_pred["housing"] = [housing]
        if feature == "loan":
            options = ["no", "yes"]
            loan = st.selectbox("Select loan:", options)
            X_pred["loan"] = [loan]
        if feature == "contact":
            options = ["unknown", "cellular", "telephone"]
            contact = st.selectbox(
                "Enter contact:", options)
            X_pred["contact"] = [contact]
        if feature == "day":
            day = st.number_input(
                "Enter day:", min_value=1, max_value=31, value=15, step=1, format="%d")
            X_pred["day"] = [day]
        if feature == "month":
            options = ["jan", "feb", "mar", "apr", "may", "jun",
                       "jul", "aug", "sep", "oct", "nov", "dec"]
            month = st.selectbox("Select month:", options)
            X_pred["month"] = [month]
        if feature == "duration":
            duration = st.number_input(
                "Enter duration:", min_value=0, max_value=5000, value=100, step=1, format="%d")
            X_pred["duration"] = [duration]
        if feature == "campaign":
            campaign = st.number_input(
                "Enter campaign:", min_value=0, max_value=250, value=10, step=1, format="%d")
            X_pred["campaign"] = [campaign]
        if feature == "pdays":
            pdays = st.number_input(
                "Enter pdays:", min_value=0, max_value=1000, value=100, step=1, format="%d")
            X_pred["pdays"] = [pdays]
        if feature == "previous":
            previous = st.number_input(
                "Enter previous:", min_value=0, max_value=250, value=10, step=1, format="%d")
            X_pred["previous"] = [previous]
        if feature == "poutcome":
            options = ["success", "failure", "unknown", "other"]
            poutcome = st.selectbox("Select poutcome:", options)
            X_pred["poutcome"] = [poutcome]

    predict_subscription_outcome = st.button(
        "Predict Subscription Outcome", type="primary", key="predict_subscription_outcome")

    if predict_subscription_outcome:
        X_pred = pd.DataFrame(X_pred)

        for col in X_pred.columns:
            X_pred[col] = label_encoders[col].transform(X_pred[col])

        y_pred = classifier.predict(X_pred)

        if y_pred == "yes":
            st.markdown("#### Subscription Outcome: Client will subscribe.")
        else:
            st.markdown(
                "#### Subscription Outcome: Client will not subscribe.")

with tab5:

    st.markdown("**💡 Desirable traits of the target customers:**  ")

    fea_col = ['age', 'marital', 'education', 'job',
               'default', 'balance', 'housing', 'loan']
    tar_vals = ["30-40", "married", " > secondary", "management",
                "no", "< 10000", "no", "no"]
    traits = {"Features": fea_col, "Target Values": tar_vals}

    traits_df = pd.DataFrame(traits)
    st.table(traits_df)

    st.markdown("---")

    st.markdown("**💡 Marketing Strategy:**  ")
    st.markdown("As a first step, the bank should allocate most of its resources for the marketing \
                campaign on the clients with the above traits. Since these clients are the potential \
                subscribers. Later, banks should optimise contacts with the clients based on the duration \
                and number of previous contacts. If the duration of the calls is short and the number of \
                contacts exceeds more than 10 then the client is not a potential subscriber. Instead of \
                contacting this client, bank can focus their resourses on other potential new subscribers.")

    st.markdown("---")

    st.markdown("**💡 Final Remarks:**  ")
    st.markdown("Through this application, we performed IDA and EDA on the Bank Marketing Dataset \
                to get valuable insights from the data. These insights helped us to design a \
                optimal marketing campaign for increasing the number of clients subscribing for a \
                term deposit.")
