```python
import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# LOAD PICKLE MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():

    # Load LightGBM pickle
    with open("lightgbm_bank_marketing.pkl", "rb") as file:
        lightgbm_model = pickle.load(file)

    # Load XGBoost pickle
    with open("xgboost_bank_marketing.pkl", "rb") as file:
        xgb_data = pickle.load(file)

    # Extract objects saved inside XGBoost pickle
    xgb_model = xgb_data["model"]
    preprocessor = xgb_data["preprocessor"]
    selected_features = xgb_data["selected_features"]

    return (
        lightgbm_model,
        xgb_model,
        preprocessor,
        selected_features
    )


# Load models
lightgbm_model, xgb_model, preprocessor, selected_features = load_models()


# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.title("🏦 Bank Marketing Prediction System")

st.write(
    "Predict whether a bank customer is likely to subscribe "
    "to a term deposit using LightGBM and XGBoost."
)

st.divider()


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.header("Customer Information")

age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

balance = st.sidebar.number_input(
    "Balance",
    min_value=-10000,
    max_value=1000000,
    value=1000
)

day = st.sidebar.number_input(
    "Day of Month",
    min_value=1,
    max_value=31,
    value=15
)

pdays = st.sidebar.number_input(
    "Days Since Previous Contact",
    min_value=-1,
    max_value=1000,
    value=-1
)

campaign = st.sidebar.number_input(
    "Number of Contacts",
    min_value=1,
    max_value=100,
    value=1
)

duration = st.sidebar.number_input(
    "Call Duration (seconds)",
    min_value=0,
    max_value=5000,
    value=200
)

month = st.sidebar.selectbox(
    "Month",
    [
        "jan", "feb", "mar", "apr",
        "may", "jun", "jul", "aug",
        "sep", "oct", "nov", "dec"
    ]
)

job = st.sidebar.selectbox(
    "Job",
    [
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown"
    ]
)

contact = st.sidebar.selectbox(
    "Contact Type",
    [
        "cellular",
        "telephone",
        "unknown"
    ]
)

housing = st.sidebar.selectbox(
    "Housing Loan",
    [
        "yes",
        "no"
    ]
)


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------
predict_button = st.button(
    "Predict Customer",
    use_container_width=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame({
        "duration": [duration],
        "balance": [balance],
        "day": [day],
        "age": [age],
        "pdays": [pdays],
        "month": [month],
        "campaign": [campaign],
        "job": [job],
        "contact": [contact],
        "housing": [housing]
    })

    # Make sure feature order is exactly
    # the same as during training
    input_data = input_data[selected_features]


    # --------------------------------------------------
    # LIGHTGBM PREDICTION
    # --------------------------------------------------

    lgbm_input = input_data.copy()

    # LightGBM was trained with categorical columns
    categorical_columns = [
        "month",
        "job",
        "contact",
        "housing"
    ]

    for column in categorical_columns:
        if column in lgbm_input.columns:
            lgbm_input[column] = lgbm_input[column].astype("category")

    lgbm_prediction = lightgbm_model.predict(lgbm_input)[0]

    lgbm_probability = (
        lightgbm_model.predict_proba(lgbm_input)[0][1]
    )


    # --------------------------------------------------
    # XGBOOST PREDICTION
    # --------------------------------------------------

    # IMPORTANT:
    # Use the SAME preprocessor saved inside
    # xgboost_bank_marketing.pkl
    xgb_input = preprocessor.transform(input_data)

    xgb_prediction = xgb_model.predict(xgb_input)[0]

    xgb_probability = (
        xgb_model.predict_proba(xgb_input)[0][1]
    )


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------
    st.divider()

    st.header("Prediction Results")

    col1, col2 = st.columns(2)


    # LightGBM result
    with col1:

        st.subheader("LightGBM")

        if lgbm_prediction == 1:
            st.success(
                "Customer is likely to subscribe"
            )
        else:
            st.error(
                "Customer is unlikely to subscribe"
            )

        st.metric(
            "Subscription Probability",
            f"{lgbm_probability * 100:.2f}%"
        )


    # XGBoost result
    with col2:

        st.subheader("XGBoost")

        if xgb_prediction == 1:
            st.success(
                "Customer is likely to subscribe"
            )
        else:
            st.error(
                "Customer is unlikely to subscribe"
            )

        st.metric(
            "Subscription Probability",
            f"{xgb_probability * 100:.2f}%"
        )


    # --------------------------------------------------
    # MODEL AGREEMENT
    # --------------------------------------------------
    st.divider()

    st.subheader("Model Comparison")

    if lgbm_prediction == xgb_prediction:

        if lgbm_prediction == 1:
            st.success(
                "Both models agree: the customer is likely to subscribe."
            )
        else:
            st.info(
                "Both models agree: the customer is unlikely to subscribe."
            )

    else:

        st.warning(
            "LightGBM and XGBoost give different predictions."
        )


    # --------------------------------------------------
    # INPUT DATA
    # --------------------------------------------------
    with st.expander("View Customer Input"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Bank Marketing Prediction | LightGBM vs XGBoost"
)
```
