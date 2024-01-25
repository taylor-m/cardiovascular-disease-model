import numpy as np
import pandas as pd
# plt.style.use(["dark_background"])
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from category_encoders import LeaveOneOutEncoder
from functions_pkg import predictions_df
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time
st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    st.title("Cardiovascular Disease Patient Classifier")
    st.markdown("Suffering from cardiovascular disease?")

    url = "https://raw.githubusercontent.com/taylor-m/cardiovascular-disease-model/main/cardio_train.csv"
    raw_df = pd.read_csv(url, sep=";", index_col="id")

    #st.set_option('deprecation.showPyplotGlobalUse', False)

    # checkbox for loading data
    @st.cache
    def load_data():
        # dataset file info
        data = pd.read_csv(url, sep=";", index_col="id")

        # new column name mapping
        mapping = {
            "ap_hi": "bp_hi",
            "ap_lo": "bp_lo",
            "gluc": "glucose",
            "alco": "alcohol",
            "cardio": "disease",
        }
        # column renaming
        data = data.rename(columns=mapping)

        # change gender to 0-1 binary
        data.loc[:, "gender"] = data.gender - 1

        # reduce interval in cholesterol & glucose from 1-3 to 0-2
        data.loc[:, "cholesterol"] = data.cholesterol - 1
        data.loc[:, "glucose"] = data.glucose - 1

        # cleaning the data of bp_hi and bp_lo value errors
        # 993 samples with extreme values for bp_hi or bp_lo
        idx = data[(abs(data.bp_hi) > 300) | (abs(data.bp_lo) > 200)].index
        data = data.drop(index=idx)

        # drop samples with negative bp_values
        idx = data[(data.bp_hi < 0) | (data.bp_lo < 0)].index
        data = data.drop(index=idx)
        # drop samples with bp_hi or bp_lo values less than 50; data entry error
        idx = data[(data.bp_lo < 50) | (data.bp_hi < 50)].index
        data = data.drop(index=idx)

        # create column for height in ft
        data["height_ft"] = data.height / 30.48

        # drop samples with heights below 5 feet and above 7 feet
        idx = data[(data.height_ft < 4.5) | (data.height_ft > 7)].index
        data = data.drop(index=idx)

        # added some more common measurement unit columns for better understanding
        data["yrs"] = data.age / 365
        data["height_ft"] = data.height / 30.48
        data["weight_lbs"] = data.weight * 2.205

        # blood pressure difference column
        data["bp_diff"] = data.bp_hi - data.bp_lo

        # BMI column to replace height and weight
        # bmi = weight (kgs) / (height (m))^2
        data["bmi"] = data.weight / (data.height / 100) ** 2

        # drop negative bp_diff samples
        idx = data[data.bp_diff < 0].index
        data = data.drop(index=idx)

        # return cleaned dataset
        return data

    @st.cache
    def xgb_split(df):  # Split the data to 'train and test' sets
        drop_cols = [
            "disease",
            "yrs",
            "height_ft",
            "bp_diff",
            "weight_lbs",
            "active",
            #     "bmi",
            #         "height",
            "weight",
        ]

        X = df.drop(columns=drop_cols)
        y = df.disease

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, 
            # random_state=28, 
            stratify=df.disease
        )
        return X_train, X_test, y_train, y_test

    @st.cache
    def lr_split(df):  # Split the data to 'train and test' sets
        # drop columns for testing sets
        drop_cols = [
            "disease",
            "yrs",
            "height_ft",
            "bp_diff",
            "weight_lbs",
            #             "active",
            #     "bmi",
            #     "height",
                "weight",
        ]

        # train test split of data
        X = df.drop(columns=drop_cols)
        y = df.disease

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, 
            # random_state=28, 
            stratify=df.gender
            # stratification of the data on gender increases the predictive accuracy of the logistic regression model
            # because the data is unbalanced towards women; ~2/3 women 1/3 men
        )
        return X_train, X_test, y_train, y_test

    def plot_metric(metric, prob_pred, y_test, pred_prob):
        if metric == 'Confusion Matrix':
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(pipeline_cv, X_test, y_test, display_labels=class_names)
            st.pyplot()

        if metric == 'Precision-Recall Curve':
            st.subheader('Precision-Recall Curve')
            #         plot_precision_recall_curve(pipeline_cv, X_test, y_test)
            #         st.pyplot()

            fig = go.Figure()

            precision, recall, _ = precision_recall_curve(y_test, pred_prob[:, 1])
            auc_score = average_precision_score(y_test, pred_prob[:, 1])

            name = f"AP={auc_score:.2f})"
            fig.add_trace(go.Scatter(x=recall, y=precision, name=name, mode='lines'))
            fig.update_layout(
                xaxis_title='Recall',
                yaxis_title='Precision',
                #             yaxis=dict(scaleanchor="x", scaleratio=True),
                xaxis=dict(constrain='domain'),
                width=700, height=500,
                yaxis_range=[0.49, 1],
            )
            st.plotly_chart(fig)

        if metric == "Feature Importances":
            st.subheader("Feature Importance")
            st.table(features)

        #     if metric == "Classification Report":
        #         # classification report
        #         st.subheader("Classification Report")
        #         class_report = classification_report(y_test, y_preds)
        #         st.table(class_report)

        if metric == "Prediction Distribution":
            st.subheader("Prediction Probability Distribution")
            #         fig, ax = plt.subplots()
            #         sns.distplot(preds_df.pred_prob)
            #         st.pyplot()

            fig = px.histogram(preds_df.pred_prob, )
            st.plotly_chart(fig)

        if metric == "Calibration Curve":
            #         fig, ax = plt.subplots()
            #         plt.plot(prob_pred, prob_true, "-o")
            #         st.pyplot(fig)

            # Create traces
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=prob_pred, y=prob_true, mode="lines", name="Calibration Curve")
            )
            # Edit the layout
            fig.update_layout(title='Calibration Curve',
                              xaxis_title='Prediction Probability',
                              yaxis_title='True Probability',
                              #                            yaxis_range=[0,1],
                              #                            xaxis_range=[0,1]
                              )
            st.plotly_chart(fig)

    #     if metric == "False Negative Analysis":
    #         st.subheader("False Negative Analysis")
    #         st.table(f_negs)
    #         st.subheader("Sample Means")
    #         st.table(f_negs.mean())

    # cache xgboost model data
    # @st.cache
    def xgboost(X_train, X_test, y_train, y_test):
        # categorical columns to be encoded
        cat_cols = ["cholesterol", "glucose"]

        # data preprocessing
        preprocessing = ColumnTransformer(
            [
                # ("encode_cats", LeaveOneOutEncoder(), cat_cols),
            ],
            remainder="passthrough",
        )

        # preprocessing and model pipeline
        pipeline = Pipeline(
            [
                ("processing", preprocessing),
                ("model", XGBClassifier(use_label_encoder=False)),
            ]
        )
        # grid search values other than optimal hyperparameters removed to lower notebook run time
        # fmt: off
        grid = {
            "model__n_estimators": [1],
            "model__learning_rate": [10, 30],
            #     "model__subsample": [],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
            "model__max_depth": [4, 5, 6],
        }
        # fmt: on
        pipeline_cv = GridSearchCV(pipeline, grid, cv=2, verbose=2, n_jobs=4)
        pipeline_cv.fit(X_train, y_train)

        best_params = pipeline_cv.best_params_
        train_score = pipeline_cv.score(X_train, y_train)
        test_score = pipeline_cv.score(X_test, y_test)

        # feature importances for xgboost model
        feature_importances = pipeline_cv.best_estimator_["model"].feature_importances_
        feature_importances = pd.DataFrame(
            {"feature": X_train.columns, "importance": feature_importances}
        ).sort_values("importance", ascending=False)
        features = feature_importances[feature_importances.importance > 0]
        y_preds = pipeline_cv.predict(X_test)
        preds_df, _ = predictions_df(X_test, y_test, y_preds)

        # classification report
        class_report = classification_report(y_test, y_preds)

        # prediction probabilities
        pred_prob = pipeline_cv.predict_proba(X_test)

        # add prediction probs to preds_df
        preds_df["pred_prob"] = pred_prob[:, 1]

        # classification doesn't require residual information
        preds_df = preds_df.drop(columns=["residuals", "abs_residuals"])

        # dataframe for false negatives sorted by prediction probability descending
        f_negs = preds_df[(preds_df.y_true == 1) & (preds_df.y_preds == 0)].sort_values(
            "pred_prob", ascending=False
        )

        prob_true, prob_pred = calibration_curve(y_test, pred_prob[:, 1], n_bins=10)

        return train_score, test_score, best_params, features, preds_df, class_report, f_negs, pipeline_cv, prob_pred, prob_true, pred_prob

    def lr_model(X_train, X_test, y_train, y_test):
        # categorical columns to be encoded
        cat_cols = ["cholesterol", "glucose"]

        # numeric columns
        num_cols = [
            "age",
                "height",
            # "weight",
            "bp_hi",
            "bp_lo",
        ]

        # data preprocessing; scaling numeric vars, encoding categorical
        preprocessing = ColumnTransformer(
            [
                ("encode_cats", LeaveOneOutEncoder(), cat_cols),
                ("scaler", StandardScaler(), num_cols),
                #         ("scaler", MinMaxScaler(), num_cols),
            ],
            remainder="passthrough",
        )

        # model pipeline
        lr_pipeline = Pipeline(
            [
                ("processing", preprocessing),
                ("model", LogisticRegression(solver="lbfgs", penalty="none", max_iter=1000, random_state=28))
            ]
        )

        # hyperparameter tuning grid
        lr_grid = {
            "model__solver": ['lbfgs'],
            "model__penalty": ["l2", "none"],
            "model__C": [0.75],
        }
        # fmt: on
        # pipeline grid search cv fit
        lr_pipeline_cv = GridSearchCV(lr_pipeline, lr_grid, cv=2, verbose=1, n_jobs=2)
        lr_pipeline_cv.fit(X_train, y_train)

        # best hyperparameters
        lr_best_params = lr_pipeline_cv.best_params_

        # logistic regression train/test scores
        lr_train_score = lr_pipeline_cv.score(X_train, y_train)
        lr_test_score = lr_pipeline_cv.score(X_test, y_test)

        # model prediction probabilities
        lr_pred_prob = lr_pipeline_cv.predict_proba(X_test)

        # model calibration curve
        lr_prob_true, lr_prob_pred = calibration_curve(y_test, lr_pred_prob[:, 1], n_bins=10)

        # prediction percentages
        lr_preds = lr_pipeline_cv.predict(X_test)

        # df created from predictions
        lr_preds_df, _ = predictions_df(X_test, y_test, lr_preds)

        # add prediction probs to preds_df
        lr_preds_df["pred_prob"] = lr_pred_prob[:, 1]

        # classification target, residuals not needed
        lr_preds_df = lr_preds_df.drop(columns=["residuals", "abs_residuals"])

        # classification report
        lr_class_report = classification_report(y_test, lr_preds)

        # dataframe for false negatives sorted by prediction probability descending
        lr_f_negs = lr_preds_df[
            (lr_preds_df.y_true == 1) & (lr_preds_df.y_preds == 0)
            ].sort_values("pred_prob", ascending=False)

        # prediction probability distribution
        lr_pred_hist = lr_preds_df.pred_prob.hist()

        return lr_pipeline_cv, lr_best_params, lr_train_score, lr_test_score, lr_pred_prob, lr_prob_true, lr_prob_pred, lr_preds_df, lr_class_report, lr_f_negs, lr_pred_hist

    # @st.cache(suppress_st_warning=True)
    def num_plot(stat):
        feat = df[stat]
        feat1 = df[df.disease == 1][stat]
        feat0 = df[df.disease == 0][stat]

        fig = go.Figure()
        bins = 150
        # 1st histogram, disease
        fig.add_trace(go.Histogram(
            x=feat1,
            histnorm='',
            name='Disease',
            marker_color='#b51616',
            nbinsx=bins,
        ))
        # 2nd histogram, no disease
        fig.add_trace(go.Histogram(
            x=feat0,
            histnorm='',
            name='No Disease',
            marker_color='#0bbd1a',
            nbinsx=bins,
        ))
        if stat == 'age':
            bins = 30

        # Overlay both histograms
        fig.update_layout(
            barmode='overlay',
            xaxis_title=stat,
            width=1000,
            height=700,
        )
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.75)
        st.plotly_chart(fig)

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # @st.cache(suppress_st_warning=True)
    def cat_plot(stat):

        if stat == 'disease':
            feat = df[stat]
            feat1 = df[df.disease == 1]['gender'].value_counts()
            feat0 = df[df.disease == 0]['gender'].value_counts()
        else:
            feat = df[stat]
            feat1 = df[df.disease == 1][stat].value_counts()
            feat0 = df[df.disease == 0][stat].value_counts()

        intervals = list(feat.unique())

        if stat == 'gender':
            tick_vals = [0, 1]
            tick_text = ["Male", "Female"]
        elif stat == 'disease':
            tick_vals = [0, 1]
            tick_text = ["Female", "Male"]
        else:
            tick_vals = intervals
            for i in range(len(intervals)):
                intervals[i] = str(intervals[i])
            tick_text = intervals


        fig = go.Figure(
            data=[
                go.Bar(
                    name="Disease",
                    x=intervals,
                    y=list(feat1),
                    marker_color='#b51616',
                ),
                go.Bar(
                    name="No Disease",
                    x=intervals,
                    y=list(feat0),
                    marker_color='#0bbd1a',
                )
            ]
        )
        fig.update_traces(
            hovertemplate='%{y:.0f}'
        )
        # Change the bar mode
        fig.update_layout(
            barmode="group",
            title_text=f"{stat} Distribution",
            xaxis_title=f"{stat}",
            width=1000,
            height=700,
        )
        fig.update_xaxes(
            tickvals= tick_vals,
            ticktext= tick_text,
        )
        st.plotly_chart(fig)

    def v_plot(stat):
        fig = go.Figure()

        fig.add_trace(
            go.Violin(
                x=df["gender"][df["disease"] == 0],  # no disease
                y=df[stat][df["disease"] == 0],
                x0='Female',
                legendgroup="No Disease",
                scalegroup="Yes",
                name="No Disease",
                side="negative",
                fillcolor="#09e648",
                line_color='#07591f',
                points=False,
            )
        )
        fig.add_trace(
            go.Violin(
                x=df["gender"][df["disease"] == 1],
                y=df[stat][df["disease"] == 1],
                x0='Male',
                legendgroup="Disease",
                scalegroup="Disease",
                name="Disease",
                side="positive",
                line_color="#e60909",
                points=False,
            )
        )
        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0.01,
                          violinmode="overlay",
                          autosize=True,
                          width=1200,
                          height=700,
                          xaxis=dict(
                              tickvals=[-1, 0, 1, 2],
                              ticktext=["", "Female", "Male", ""],
                              tickmode="array",
                              titlefont=dict(size=30),
                          ),
                          xaxis_title = f"{stat}",
                          )
        fig.update_xaxes(automargin=True, range=(-0.5, 1.5))
        st.plotly_chart(fig)

    # function for saving new runtimes to class_runtimes log
    def save_class_runtime(runtime):
        try:
            runtimes = pd.read_csv("runtime_logs/class_runtimes.csv", header=None, index_col=False)
            # taking column 0 of runtimes as a Series and converting to list
            runtimes = runtimes[0].tolist()
        except:
            runtimes = []
            
        # adding new runtime value to list
        runtimes.append(runtime)
        runtimes = pd.Series(runtimes)
        runtimes.to_csv(path_or_buf='runtime_logs/class_runtimes.csv', index=False, header=False)

    # function for saving new runtimes to class_runtimes log
    def save_pred_runtime(runtime):
        try:
            runtimes = pd.read_csv("runtime_logs/pred_runtimes.csv", header=None, index_col=False)
            # taking column 0 of runtimes as a Series and converting to list
            runtimes = runtimes[0].tolist()
        except:
            runtimes = []
        # adding new runtime value to list
        runtimes.append(runtime)
        runtimes = pd.Series(runtimes)
        runtimes.to_csv(path_or_buf='runtime_logs/pred_runtimes.csv', index=False, header=False)

    # function for loading times from runtime log
    def avg_runtime(model='class'):
        if model == 'class':
            try:
                runtimes = pd.read_csv("runtime_logs/class_runtimes.csv", header=None)
                avg_runtime = runtimes.mean()[0]
            except:
                avg_runtime = 0
        else:
            try:
                runtimes = pd.read_csv("runtime_logs/pred_runtimes.csv", header=None)
                avg_runtime = runtimes.mean()[0]
            except:
                avg_runtime = 0
        
        return avg_runtime

    df = load_data()

    st.sidebar.title("Model")

    class_names = ['Disease', 'No Disease']
    # x_train, x_test, y_train, y_test = split(df)

    option = st.sidebar.selectbox("Model Option", ("Data Info", "Feature Var Plots", "Model"))

    if option == "Data Info":

        # data information
        st.subheader("Types of input features:")
        st.write("   - Objective: factual information")
        st.write("   - Examination: results of medical examination")
        st.write("   - Subjective: information given by the patient")

        if st.sidebar.checkbox("raw data", False):
            st.subheader("Raw Dataset")
            st.write(raw_df.head())

        st.subheader("Clean Data")
        st.write(df.head())
        st.write(f"Number of samples: {df.shape[0]}")
        st.subheader("Variables")
        st.write("1. age (days)")
        st.write("2. gender (0=female|1=male)")
        st.write("3. height (cm)")
        st.write("4. weight (kg)")
        st.write("5. bp_hi [systolic blood pressure]")
        st.write("6. bp_lo [diastolic blood pressure]")
        st.write("7. cholesterol [normal (0) | high (1) | very high(2)]")
        st.write("8. glucose [normal(0) | high(1) | very high(2)]")
        st.write("9. smoke \[smoking?] (0=no|1=yes)")
        st.write("10. alcohol \[drinking?] (0=no|1=yes)")
        st.write("11. active \[physically active?] (0=no|1=yes)")
        st.write("12. disease [presence (1) or absence (0) of cardiovascular disease]")
        st.write("13. bp_diff [bp_hi - bp_lo]")
        st.write("14. bmi [body mass index]")

    num_cols = ["bp_lo", "bp_hi", "bp_diff", "bmi", "height", "weight", "age"]

    if option == "Feature Var Plots":
        st.subheader("Variable/Group Visualization:")
        st.sidebar.subheader("Variables")
        plot_var = st.sidebar.selectbox("Variable", (
            "age", "gender", "height", "weight", "bp_hi", "bp_lo", "cholesterol", "glucose", "smoke", "alcohol",
            "active",
            "bmi", "disease"), key='plot_var')
        if plot_var in num_cols:
            plot_type = st.sidebar.radio("plot type", ("hist", "violin"), key='var_plot_type')
        if st.sidebar.button("Plot Var", False):
            if plot_var in num_cols:
                if plot_type == 'hist':
                    num_plot(plot_var)
                else:
                    v_plot(plot_var)
            else:
                cat_plot(plot_var)

    if option == "Model":
        model = st.sidebar.radio("Model type:", ("Classification", "Prediction"), key="model")
        if model == "Classification":
            rt_avg = avg_runtime()
            
            # multiselect for model results visualization options for the xgboost model
            metrics = st.sidebar.multiselect("Classifier Visualization:", (
                'Calibration Curve', 'Confusion Matrix', 'Precision-Recall Curve', 'Feature Importances',
                'Prediction Distribution'), key="xgb_metric")
            if st.sidebar.button("Run", False):
                # timer for model run time
                tic = time.perf_counter()

                # xgboost train test split
                X_train, X_test, y_train, y_test = xgb_split(df)
                train_score, test_score, best_params, features, preds_df, class_report, f_negs, pipeline_cv, prob_pred, prob_true, pred_prob = xgboost(
                    X_train, X_test, y_train, y_test)
                y_preds = preds_df.y_preds
                st.subheader("XGBoost Classifier Model Results")
                st.write("Accuracy: ", train_score.round(2) * 100, "%")
                st.write("Precision: ", precision_score(preds_df.y_true, preds_df.y_preds, labels=class_names).round(2), "%")
                st.write("Recall: ", recall_score(preds_df.y_true, preds_df.y_preds, labels=class_names).round(2), "%")
                for metric in metrics:
                    fig, ax = plt.subplots()
                    plot_metric(metric, prob_pred, y_test, pred_prob)
                st.write("hyperparameters:")
                st.text(best_params)
                toc = time.perf_counter()
                rt = (toc - tic)
                save_class_runtime(rt)
                st.write(f"Runtime: {rt:0.2f}s")
                rt_avg = avg_runtime()
            st.sidebar.write(f"runtime avg: {rt_avg:0.2f}s")
                
        if model == "Prediction":
            # tic = time.perf_counter()
            mode = st.sidebar.radio("Prediction Model Options", ("model performance", "disease probability"),
                                    key="lr_options")
            X_train, X_test, y_train, y_test = lr_split(df)

            # lr model return vars
            lr_pipeline_cv, lr_best_params, lr_train_score, lr_test_score, lr_pred_prob, prob_true, prob_pred, preds_df, lr_class_report, lr_f_negs, lr_pred_hist = lr_model(
                X_train, X_test, y_train, y_test)
            # toc = time.perf_counter()
            # st.write(f"Runtime: {toc - tic:0.4f}s")
            if mode == "model performance":
                rt_avg = avg_runtime('pred')
                
                metrics = st.sidebar.multiselect("Predictor Performance:", (
                    'Confusion Matrix', 'Precision-Recall Curve', 'Prediction Distribution', 'Calibration Curve'),
                                                 key="lr_metric")
                if st.sidebar.button("Run", False):
                    tic = time.perf_counter()
                    st.subheader("Logistic Regression Model Results")

                    pipeline_cv = lr_pipeline_cv
                    y_preds = preds_df.y_preds
                    f_negs = lr_f_negs
                    st.write("Accuracy: ", lr_train_score.round(2) * 100, "%")
                    st.write("Precision: ",
                             precision_score(preds_df.y_true, preds_df.y_preds, labels=class_names).round(2), "%")
                    st.write("Recall: ", recall_score(preds_df.y_true, preds_df.y_preds, labels=class_names).round(2), "%")

                    for metric in metrics:
                        fig, ax = plt.subplots()
                        plot_metric(metric, prob_pred, y_test, lr_pred_prob)
                    st.write("hyperparameters:")
                    st.text(lr_best_params)
                    toc = time.perf_counter()
                    rt = (toc - tic)
                    save_pred_runtime(rt)
                    st.write(f"Runtime: {rt:0.2f}s")
                    rt_avg = avg_runtime('pred')
                st.sidebar.write(f"runtime avg: {rt_avg:0.2f}s")

            if mode == "disease probability":
                st.subheader('Cardiovascular Disease Probability Prediction')
                # age input years
                age = st.number_input('age')
                # age to days
                age = age * 365

                # gender input
                gender = st.radio('gender', ['male', 'female'])
                if gender == 'male':
                    gender = 1
                else:
                    gender = 0

                # height inches
                ht = st.number_input('height (inches)')
                # height to cm
                ht = ht * 2.54

                # weight lbs
                wt = st.number_input('weight (lbs)')
                # weight to kgs
                wt = wt * 0.453592

                # bp_hi
                bp_hi = st.number_input('systolic bp (high #)')
                # bp_lo
                bp_lo = st.number_input('diastolic bp (low #)')

                # slider scale
                st.write('[0 = normal, 1 = high, 2 = very high]')
                # cholesterol
                chol = st.slider('cholesterol', min_value=0, max_value=2)

                # glucose
                glu = st.slider('glucose', min_value=0, max_value=2)

                # smoke
                smk = st.radio('smoke?', ['no', 'yes'])
                if smk == 'yes':
                    smk = 1
                else:
                    smk = 0

                # alcohol
                alc = st.radio('drink alcohol regularly?', ['no', 'yes'])
                if alc == 'yes':
                    alc = 1
                else:
                    alc = 0
                # alcohol
                act = st.radio('Physically active?', ['no', 'yes'])
                if act == 'yes':
                    act = 1
                else:
                    act = 0

                # bmi
                bmi = wt / ((ht / 100) ** 2)

                X_input = {
                    'id': [0],
                    'age': [age],
                    'gender': [gender],
                    'height': [ht],
                    # 'weight': [wt],
                    'bp_hi': [bp_hi],
                    'bp_lo': [bp_lo],
                    'cholesterol': [chol],
                    'glucose': [glu],
                    'smoke': [smk],
                    'alcohol': [alc],
                    'active': [act],
                    'bmi': [bmi],
                }

                if st.button("Predict", False):
                    tic = time.perf_counter()
                    X_input = pd.DataFrame(X_input)
                    X_input = X_input.set_index('id')
                    input_prob = lr_pipeline_cv.predict_proba(X_input)
                    disease_prob = input_prob[0, 1]
                    st.subheader('Input data')
                    st.dataframe(X_input)
                    st.write("Cardiovascular disease probability: ", disease_prob.round(2) * 100, "%")
                    toc = time.perf_counter()
                    st.write(f"Runtime: {toc - tic:0.4f}s")

if __name__ == '__main__':
    main()
