import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly
import seaborn as sns
from scipy import stats

# plt.style.use(["dark_background"])

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV
from category_encoders import LeaveOneOutEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functions_pkg import print_vif, predictions_df
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_score, recall_score
import streamlit as st



# def main():
st.title("Cardiovascular Disease Patient Classifier")
st.markdown("Suffering from cardiovascular disease?")


path = "cardiovascular_disease_prediction/cardio_train.csv"
raw_df = pd.read_csv(path, sep=";", index_col="id")


st.set_option('deprecation.showPyplotGlobalUse', False)
    
# checkbox for loading data
@st.cache(persist=True)
def load_data():
    # dataset file info
    data = pd.read_csv(path, sep=";", index_col="id")

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
def xgb_split(df):# Split the data to 'train and test' sets
        drop_cols = [
            "disease",
            "yrs",
            "height_ft",
            "bp_diff",
            "weight_lbs",
            "active",
            #     "bmi",
            #         "height",
            #     "weight",
        ]

        X = df.drop(columns=drop_cols)
        y = df.disease

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=28, stratify=df.disease
        )
        return X_train, X_test, y_train, y_test
@st.cache   
def lr_split(df):# Split the data to 'train and test' sets
        # drop columns for testing sets
        drop_cols = [
            "disease",
            "yrs",
            "height_ft",
            "bp_diff",
            "weight_lbs",
            "active",
            #     "bmi",
            #     "height",
            #     "weight",
        ]

        # train test split of data
        X = df.drop(columns=drop_cols)
        y = df.disease

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=28, stratify=df.disease
        )
        return X_train, X_test, y_train, y_test

def plot_metric(metric):
    if metric == 'Confusion Matrix':
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(pipeline_cv, X_test, y_test, display_labels=class_names)
        st.pyplot()

    if metric == 'Precision-Recall Curve':
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(pipeline_cv, X_test, y_test)
        st.pyplot()
        
    if metric == "Feature Importances":
        st.subheader("Feature Importance")
        st.table(features)
        
#     if metric == "Classification Report":
#         # classification report
#         st.subheader("Classification Report")
#         class_report = classification_report(y_test, y_preds)
#         st.table(class_report)
    
    if metric  == "Prediction Distribution":
        st.subheader("Prediction Probability Distribution")
        fig, ax = plt.subplots()
        pred_dist
        st.pyplot()
        
    if metric == "Calibration Curve":
        st.subheader("Calibration Curve")
        fig, ax = plt.subplots()
        ax.plot(lr_prob_pred, lr_prob_true, "-o")
        st.pyplot()
        
#     if metric == "False Negative Analysis":
#         st.subheader("False Negative Analysis")
#         st.table(f_negs)
#         st.subheader("Sample Means")
#         st.table(f_negs.mean())
        
# cache xgboost model data
#@st.cache
def xgboost(X_train, X_test, y_train, y_test):
    # categorical columns to be encoded
    cat_cols = ["cholesterol", "glucose"]

    # data preprocessing
    preprocessing = ColumnTransformer(
        [
            ("encode_cats", LeaveOneOutEncoder(), cat_cols),
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
        "model__n_estimators": np.arange(1, 3),
        "model__learning_rate": np.arange(0, 50, 10),
        #     "model__subsample": [],
        "model__colsample_bytree": np.arange(0.7,1,0.1),
        "model__max_depth": np.arange(4,7),
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

    return train_score, test_score, best_params, features, preds_df, class_report, f_negs, pipeline_cv

def lr_model(X_train, X_test, y_train, y_test):
    # categorical columns to be encoded
    cat_cols = ["cholesterol", "glucose"]
    
    # numeric columns
    num_cols = [
    "age",
    #     "height",
    "weight",
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
    "model__penalty": ["l2","none"],
    "model__C": [0.75],
    }
    # fmt: on
    # pipeline grid search cv fit
    lr_pipeline_cv = GridSearchCV(lr_pipeline, lr_grid, cv=5, verbose=1, n_jobs=2)
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


@st.cache
def num_plot(stat):
    feat = df[stat]
    feat1 = df[df.disease == 1][stat]
    feat0 = df[df.disease == 0][stat]
    fig, ax = plt.subplots()
    sns.distplot(feat1, color='#b51616', label='Disease')
    sns.distplot(feat0, color='#0bbd1a', label='No Disease')
    ax.set_xlabel(stat)
    ax.set_title(f"{stat} Distribution")
    ax.legend()
    st.pyplot(fig)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
@st.cache
def cat_plot(stat):
    feat = df[stat]
    feat1 = df[df.disease == 1][stat].value_counts()
    feat0 = df[df.disease == 0][stat].value_counts()
#     d = {'no_disease':feat0, 'disease':feat1}
#     f = pd.DataFrame(data=d)
    fig, ax = plt.subplots()
#     st.write(f.style.background_gradient())
    fig, ax = plt.subplots()
    width = 0.25
    cd = ax.bar(x=feat1.index-width/2, height=feat1, width=width, color='#e60909', label='Disease')
    no_cd = ax.bar(x=feat0.index+width/2, height=feat0, width=width, color='#09e648', label='No Disease')
    
    # Attach a text label above each bar in *rects*, displaying its height
    for rect in cd:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in no_cd:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
        
    ax.set_xlabel(stat)
    ax.set_xticks(feat.unique())
    ax.set_title(f"{stat}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    
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

num_cols = ["bp_lo", "bp_hi", "bp_diff", "bmi", "height", "weight", "age", "height_ft", "yrs", "weight_lbs"]

if option == "Feature Var Plots":
    st.subheader("By the numbers:")
    st.write("View numbers of samples within each variable and their distributions by disease class and gender")
    st.sidebar.subheader("Variables")
    age = st.sidebar.checkbox("age", False, key="age")
    gender = st.sidebar.checkbox("gender", False)
    height = st.sidebar.checkbox("height", False)
    weight = st.sidebar.checkbox("weight", False)
    bp = st.sidebar.checkbox("blood pressure", False)
    cholesterol = st.sidebar.checkbox("cholesterol", False)
    glucose = st.sidebar.checkbox("glucose", False)
    smoke = st.sidebar.checkbox("smoke", False)
    alcohol = st.sidebar.checkbox("alcohol", False)
    active = st.sidebar.checkbox("active", False)
    bmi = st.sidebar.checkbox("bmi", False)
    disease = st.sidebar.checkbox("disease", False)
    if st.sidebar.button("Plot", key="plot"):                        
        if age:
            num_plot("age")
        if gender:
            cat_plot("gender")
        if height:
            num_plot("height")
        if weight:
            num_plot("weight")    
        if bp:
            fig, axs = plt.subplots(2,1)
            sns.catplot(x="gender", y="bp_hi", hue="disease", kind="violin", split=True, data=df)
            st.pyplot()
            sns.catplot(x="gender", y="bp_lo", hue="disease", kind="violin", split=True, data=df)
            st.pyplot()
            sns.catplot(x="gender", y="bp_diff", hue="disease", kind="violin", split=True, data=df)
            st.pyplot()

        if cholesterol:
            cat_plot("cholesterol") 
        if glucose:
            cat_plot("glucose")    
        if smoke:
            cat_plot("smoke")
        if alcohol:
            cat_plot("alcohol")
        if active:
            cat_plot("active")    
        if bmi:
            num_plot("bmi")    
        if disease:
            cat_plot("disease")     
    
if option == "Model":
    model = st.sidebar.radio("Model type:", ("Classification", "Prediction"), key="model")
    if model == "Classification":
        # multiselect for model results visualization options for the xgboost model
        metrics = st.sidebar.multiselect("Classifier Visualization:", ('Confusion Matrix', 'Precision-Recall Curve', 'Feature Importances', 'Prediction Distribution'), key="xgb_metric")
        if st.sidebar.button("Run", False):
            
            # xgboost train test split
            X_train, X_test, y_train, y_test = xgb_split(df)
            train_score, test_score, best_params, features, preds_df, class_report, f_negs, pipeline_cv = xgboost(X_train, X_test, y_train, y_test)
            y_preds = preds_df.y_preds
            pred_dist = preds_df.pred_prob.hist()
            st.subheader("XGBoost Classifier Model Results")
            st.write("Accuracy: ", train_score.round(2)*100,"%")
            st.write("Precision: ", precision_score(preds_df.y_true, preds_df.y_preds, labels=class_names).round(2))
            st.write("Recall: ", recall_score(preds_df.y_true, preds_df.y_preds, labels=class_names).round(2))
            for metric in metrics:
                fig, ax = plt.subplots()
                plot_metric(metric)
    if model == "Prediction":
        mode = st.sidebar.radio("Prediction Model Options", ("model performance", "disease probability"), key="lr_options")
        X_train, X_test, y_train, y_test = lr_split(df)

        # lr model return vars
        lr_pipeline_cv, lr_best_params, lr_train_score, lr_test_score, lr_pred_prob, lr_prob_true, lr_prob_pred, lr_preds_df, lr_class_report, lr_f_negs, lr_pred_hist = lr_model(X_train, X_test, y_train, y_test)
        if mode == "model performance":
            metrics = st.sidebar.multiselect("Predictor Performance:", ('Confusion Matrix', 'Precision-Recall Curve', 'Prediction Distribution', 'Calibration Curve'), key="lr_metric")
            if st.sidebar.button("Run", False):
                
                st.subheader("Logistic Regression Model Results")
                
                pipeline_cv = lr_pipeline_cv
                y_preds = lr_preds_df.y_preds
                f_negs = lr_f_negs
                st.write("Accuracy: ", lr_train_score.round(2)*100,"%")
                st.write("Precision: ", precision_score(lr_preds_df.y_true, lr_preds_df.y_preds, labels=class_names).round(2))
                st.write("Recall: ", recall_score(lr_preds_df.y_true, lr_preds_df.y_preds, labels=class_names).round(2))
        
                for metric in metrics:
                    fig, ax = plt.subplots()
                    plot_metric(metric)
                    
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
            smk = st.radio('smoke?', ['yes', 'no'])
            if smk == 'yes':
                smk = 1
            else:
                smk = 0
            
            # alcohol
            alc = st.radio('drink alcohol regularly?', ['yes', 'no'])
            if alc == 'yes':
                alc = 1
            else:
                alc = 0
            # bmi
            bmi = wt/((ht/100)**2)

            X_input = {
                'id': [0],
                'age': [age],
                'gender': [gender],
                'height': [ht],
                'weight': [wt],
                'bp_hi': [bp_hi],
                'bp_lo': [bp_lo],
                'cholesterol': [chol],
                'glucose': [glu],
                'smoke': [smk],
                'alcohol': [alc],
                'bmi': [bmi],
            }
            X_input = pd.DataFrame(X_input)
            X_input = X_input.set_index('id')
            input_prob = lr_pipeline_cv.predict_proba(X_input)
            disease_prob = input_prob[0, 1]
            if st.button("Predict", False):
                st.subheader('Input data')
                st.dataframe(X_input)
                st.write("Cardiovascular disease probability: ", disease_prob.round(2)*100,"%")











