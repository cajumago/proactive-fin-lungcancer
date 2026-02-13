"""
model.py - XGBRegressor model for predicting Total Charges of
NY Hospital Inpatient Discharges (SPARCS) for Respiratory Cancer.

This module handles:
- Log-transformed target training with XGBRegressor
- Prediction with inverse log transform and inflation adjustment
- Model evaluation metrics (R2, MSE, RMSE, MAE, MAPE)
- Single-observation prediction for unseen data
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    #root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor

from data_loader import apply_mappings

# Log-transform offset used during training: y_transformed = log(y + OFFSET)
LOG_OFFSET = 3000

# Default inflation factor (IP HCC 2021 Annual Change: 7.167231%)
DEFAULT_INFLATION_FACTOR = 1.0717


def build_model(n_estimators=1000, max_depth=4, learning_rate=0.1):
    """Build an XGBRegressor wrapped in a Pipeline with float64 casting.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Step size shrinkage.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with FunctionTransformer(np.float64) -> XGBRegressor.
    """
    return Pipeline([
        ("transform", FunctionTransformer(np.float64)),
        ("regressor", XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )),
    ])


def train_model(model, X_train, y_train):
    """Train the model on log-transformed target values.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Model pipeline from build_model().
    X_train : pd.DataFrame
        Training features.
    y_train : pd.DataFrame
        Training target (Total Charges, original scale).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted model.
    """
    y_transformed = np.log(y_train + LOG_OFFSET)
    model.fit(X_train, y_transformed)
    return model


def predict(model, X, inflation_factor=DEFAULT_INFLATION_FACTOR):
    """Generate predictions on the original Total Charges scale.

    Applies the inverse log transform and an inflation adjustment.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted model.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    inflation_factor : float
        Multiplicative inflation adjustment (default 1.0717 for 2021).

    Returns
    -------
    np.ndarray
        Predicted Total Charges on the original scale.
    """
    raw_preds = model.predict(X)
    return (np.exp(raw_preds) - LOG_OFFSET) * inflation_factor


def evaluate(y_true, y_pred):
    """Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual Total Charges.
    y_pred : array-like
        Predicted Total Charges.

    Returns
    -------
    dict
        Dictionary with R2, MSE, RMSE, MAE, and MAPE values.
    """
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        #"RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
    }


def predict_single(model, ohe_transformer, input_dict, inflation_factor=DEFAULT_INFLATION_FACTOR):
    """Predict Total Charges for a single new observation.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Fitted model.
    ohe_transformer : sklearn.compose.ColumnTransformer
        Fitted OHE transformer from data_loader.
    input_dict : dict
        Dictionary with raw feature values. Expected keys:
        'CCSR Procedure Description', 'Length of Stay', 'Discharge Year',
        'Age Group', 'Type of Admission', 'Patient Disposition',
        'APR DRG Description', 'APR Severity of Illness Description',
        'APR Medical Surgical Description', 'Payment Typology 1',
        'Permanent Facility Id'.
        Values should be the original string labels (before mapping).
    inflation_factor : float
        Multiplicative inflation adjustment.

    Returns
    -------
    float
        Predicted Total Charge for the observation.
    """
    # Build a single-row DataFrame in the same column order as training
    row = pd.DataFrame([{
        "CCSR Procedure Description": input_dict["CCSR Procedure Description"],
        "Discharge Year": input_dict.get("Discharge Year", 2021),
        "Age Group": input_dict["Age Group"],
        "Length of Stay": input_dict["Length of Stay"],
        "Type of Admission": input_dict["Type of Admission"],
        "Patient Disposition": input_dict["Patient Disposition"],
        "APR DRG Description": input_dict["APR DRG Description"],
        "APR Severity of Illness Description": input_dict["APR Severity of Illness Description"],
        "APR Medical Surgical Description": input_dict["APR Medical Surgical Description"],
        "Payment Typology 1": input_dict["Payment Typology 1"],
        "Total Charges": 0,  # placeholder, will be dropped
        "Permanent Facility Id": input_dict["Permanent Facility Id"],
    }])

    row = apply_mappings(row)
    encoded = ohe_transformer.transform(row)
    col_names = ohe_transformer.get_feature_names_out()
    X = pd.DataFrame(encoded, columns=col_names)

    # Drop Discharge Year and Total Charges (not used as features)
    X = X.drop(["remainder__Discharge Year", "remainder__Total Charges"], axis=1)

    raw_pred = model.predict(X)
    estimated_charge = (np.exp(raw_pred) - LOG_OFFSET) * inflation_factor
    return float(estimated_charge[0])
