"""
streamlit_page.py - Streamlit application for predicting Total Charges
of NY Hospital Inpatient Discharges (SPARCS) for Respiratory Cancer.

Run with:
    streamlit run streamlit_page.py

The app trains the XGBRegressor model on startup, then provides an
interactive form for users to input patient/discharge features and
receive a predicted Total Charge along with an Out-of-Pocket cost estimate.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from data_loader import (
    load_and_prepare,
    AGE_GROUP_MAP,
    TYPE_OF_ADMISSION_MAP,
    APR_MEDICAL_SURGICAL_MAP,
    PATIENT_DISPOSITION_MAP,
    APR_SEVERITY_MAP,
    APR_DRG_MAP,
    CCSR_PROCEDURE_MAP,
    PAYMENT_TYPOLOGY_MAP,
)
from model import (
    build_model,
    train_model,
    predict,
    evaluate,
    predict_single,
    DEFAULT_INFLATION_FACTOR,
)
from out_of_pocket_cost import calculate_insurance_oop, calculate_self_pay

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NY Respiratory Cancer - Charge Estimator",
    layout="wide",
)

#DATA_PATH = "2018_2021.csv"
DATA_PATH = Path(__file__).resolve().parent / "2018_2021.csv"


# ---------------------------------------------------------------------------
# Cached data loading and model training
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and training model...")
def load_data_and_train():
    """Load data, prepare features, train the model, and return all artifacts."""
    data = load_and_prepare(DATA_PATH)
    mdl = build_model()
    mdl = train_model(mdl, data["X_train"], data["y_train"])

    preds = predict(mdl, data["X_valid"])
    metrics = evaluate(data["y_valid"], preds)

    return mdl, data["ohe_transformer"], metrics, data


model, ohe_ct, val_metrics, prepared_data = load_data_and_train()


# ---------------------------------------------------------------------------
# Sidebar - Model performance
# ---------------------------------------------------------------------------
st.sidebar.header("Model Performance (Validation 2021)")
st.sidebar.metric("R2 Score", f"{val_metrics['R2']:.4f}")
#st.sidebar.metric("RMSE", f"${val_metrics['RMSE']:,.2f}")
st.sidebar.metric("MAE", f"${val_metrics['MAE']:,.2f}")
st.sidebar.metric("MAPE", f"{val_metrics['MAPE']:.2%}")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Model: XGBRegressor (1000 trees, depth 4, lr 0.1). "
    "Target: log(Total Charges + 3000). "
    "Inflation adjustment: 7.17% (IP HCC 2021)."
)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("NY Hospital Inpatient Discharge - Total Charge Estimator")
st.markdown(
    "Predict the estimated Total Charge for a respiratory cancer inpatient "
    "discharge in New York State, using SPARCS data from 2018-2021."
)

st.header("1. Patient and Discharge Information")

col1, col2 = st.columns(2)

with col1:
    age_group = st.selectbox("Age Group", options=list(AGE_GROUP_MAP.keys()))
    type_of_admission = st.selectbox(
        "Type of Admission", options=list(TYPE_OF_ADMISSION_MAP.keys())
    )
    patient_disposition = st.selectbox(
        "Patient Disposition", options=list(PATIENT_DISPOSITION_MAP.keys())
    )
    apr_severity = st.selectbox(
        "APR Severity of Illness", options=list(APR_SEVERITY_MAP.keys())
    )
    apr_med_surg = st.selectbox(
        "APR Medical/Surgical", options=list(APR_MEDICAL_SURGICAL_MAP.keys())
    )

with col2:
    apr_drg = st.selectbox(
        "APR DRG Description", options=list(APR_DRG_MAP.keys())
    )
    ccsr_procedure = st.selectbox(
        "CCSR Procedure Description", options=list(CCSR_PROCEDURE_MAP.keys())
    )
    payment_typology = st.selectbox(
        "Payment Typology", options=list(PAYMENT_TYPOLOGY_MAP.keys())
    )
    length_of_stay = st.number_input(
        "Length of Stay (days)", min_value=0, max_value=365, value=3, step=1
    )
    facility_id = st.number_input(
        "Permanent Facility Id", min_value=1, max_value=9999, value=1176, step=1
    )

inflation = st.slider(
    "Inflation Adjustment (%)",
    min_value=0.0,
    max_value=20.0,
    value=7.17,
    step=0.01,
    help="Annual inflation percentage applied to the prediction. Default is 7.17% (IP HCC 2021).",
)
inflation_factor = 1 + (inflation / 100.0)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
st.header("2. Predicted Total Charge")

if st.button("Estimate Total Charge"):
    input_data = {
        "CCSR Procedure Description": ccsr_procedure,
        "Length of Stay": length_of_stay,
        "Discharge Year": 2021,
        "Age Group": age_group,
        "Type of Admission": type_of_admission,
        "Patient Disposition": patient_disposition,
        "APR DRG Description": apr_drg,
        "APR Severity of Illness Description": apr_severity,
        "APR Medical Surgical Description": apr_med_surg,
        "Payment Typology 1": payment_typology,
        "Permanent Facility Id": facility_id,
    }

    estimated_charge = predict_single(model, ohe_ct, input_data, inflation_factor)
    st.session_state["estimated_charge"] = estimated_charge

if "estimated_charge" in st.session_state:
    estimated_charge = st.session_state["estimated_charge"]
    st.success(f"Estimated Total Charge: ${estimated_charge:,.2f}")

    # ------------------------------------------------------------------
    # Out-of-Pocket Cost
    # ------------------------------------------------------------------
    st.header("3. Out-of-Pocket Cost Estimate")

    pay_option = st.radio(
        "Payment Method", ["Health Insurance", "Self-Pay"], horizontal=True
    )

    if pay_option == "Health Insurance":
        ins_col1, ins_col2, ins_col3 = st.columns(3)
        with ins_col1:
            oop_max = st.number_input(
                "Out-of-Pocket Maximum ($)", min_value=0.0, value=8000.0, step=100.0
            )
        with ins_col2:
            deductible = st.number_input(
                "Deductible ($)", min_value=0.0, value=1500.0, step=100.0
            )
        with ins_col3:
            co_insurance = st.number_input(
                "Co-Insurance (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0
            )

        if st.button("Calculate Out-of-Pocket Cost", key="oop_calc"):
            oop_cost = calculate_insurance_oop(
                estimated_charge, deductible, co_insurance, oop_max
            )
            st.info(f"Estimated Out-of-Pocket Cost: ${oop_cost:,.2f}")

    else:
        discount = st.number_input(
            "Facility Discount / Nonprofit Support (%)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
        )

        if st.button("Calculate Out-of-Pocket Cost", key="oop_calc_sp"):
            oop_cost = calculate_self_pay(estimated_charge, discount)
            st.info(f"Estimated Out-of-Pocket Cost: ${oop_cost:,.2f}")
