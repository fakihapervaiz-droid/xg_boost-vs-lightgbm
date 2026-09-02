import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# --------------------------------------------------
# Load Models
# --------------------------------------------------

@st.cache_resource
def load_models():

    with open("lightgbm_bank_marketing.pkl", "rb") as file:
        lightgbm_model = pickle.load(file)

    with open("xgboost_bank_marketing.pkl", "rb") as file:
        xgb_data = pickle.load(file)

    xgb_model = xgb_data["model"]
    xgb_preprocessor = xgb_data["preprocessor"]
    selected_features = xgb_data["selected_features"]

    return lightgbm_model, xgb_model, xgb_preprocessor, selected_features


lightgbm_model, xgb_model, xgb_preprocessor, selected_features = load_models()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏦 Bank Marketing Prediction System")

st.markdown(
    """
    ### Predict whether a customer is likely to subscribe to a term deposit

    This application compares predictions from **LightGBM** and **XGBoost**
    using the selected features from the Bank Marketing dataset.
    """
)

st.divider()

# --------------------------------------------------
# Sidebar
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
    "Number of Contacts During Campaign",
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
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
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
# Prediction Button
# --------------------------------------------------

predict_button = st.button(
    "🔮 Predict Customer",
    use_container_width=True
)

# --------------------------------------------------
# Prediction
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

    # Make sure columns are in the same order
    input_data = input_data[selected_features]

    # --------------------------------------------------
    # LightGBM Prediction
    # --------------------------------------------------

    # Convert categorical columns to category
    lightgbm_input = input_data.copy()

    for column in lightgbm_input.columns:

        if column in ["month", "job", "contact", "housing"]:

            lightgbm_input[column] = (
                lightgbm_input[column].astype("category")
            )

    lgbm_prediction = lightgbm_model.predict(
        lightgbm_input
    )[0]

    lgbm_probability = lightgbm_model.predict_proba(
        lightgbm_input
    )[0][1]

    # --------------------------------------------------
    # XGBoost Prediction
    # --------------------------------------------------

    xgb_input = xgb_preprocessor.transform(
        input_data
    )

    xgb_prediction = xgb_model.predict(
        xgb_input
    )[0]

    xgb_probability = xgb_model.predict_proba(
        xgb_input
    )[0][1]

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    st.divider()

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    # LightGBM
    with col1:

        st.markdown("### LightGBM")

        if lgbm_prediction == 1:
            st.success("Customer is likely to subscribe")
        else:
            st.error("Customer is unlikely to subscribe")

        st.metric(
            "Probability of Subscription",
            f"{lgbm_probability * 100:.2f}%"
        )

    # XGBoost
    with col2:

        st.markdown("### XGBoost")

        if xgb_prediction == 1:
            st.success("Customer is likely to subscribe")
        else:
            st.error("Customer is unlikely to subscribe")

        st.metric(
            "Probability of Subscription",
            f"{xgb_probability * 100:.2f}%"
        )

    # --------------------------------------------------
    # Model Agreement
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
            "The models disagree on this customer's prediction."
        )

    # --------------------------------------------------
    # Input Summary
    # --------------------------------------------------

    with st.expander("View Input Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Bank Marketing Prediction | LightGBM vs XGBoost"
)