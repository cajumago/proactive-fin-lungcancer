# NY Hospital Inpatient Discharges - Respiratory Cancer Total Charge Prediction

Predicts Total Charges for respiratory cancer inpatient discharges in New York State using SPARCS data (2018-2021) and an XGBRegressor model.

## Project Structure

```
ny_rc/
  2018_2021.csv                      # SPARCS dataset (2018-2021)
  data_loader.py                     # Data loading, mapping, imputation, OHE
  model.py                           # XGBRegressor training, prediction, evaluation
  out_of_pocket_cost.py              # Insurance and self-pay OOP cost calculation
  streamlit_page.py                  # Streamlit web application
  model_with_los_Refactored.ipynb    # Refactored analysis notebook
  model_with_los.ipynb               # Original notebook (reference)
  requirements.txt                   # Python dependencies
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook

Open and run `model_with_los_Refactored.ipynb` to walk through the full analysis pipeline: data loading, transformation, feature engineering, model training, evaluation, and out-of-pocket cost estimation.

### Streamlit Application

```bash
streamlit run streamlit_page.py
```

This launches a browser-based interface where you can:
1. Select patient and discharge features from dropdown menus.
2. Adjust the inflation percentage.
3. Get a predicted Total Charge.
4. Calculate out-of-pocket cost for Health Insurance or Self-Pay scenarios.

The model trains on startup using the `2018_2021.csv` dataset. Validation metrics (2021 holdout) are displayed in the sidebar.

## Model Details

- **Algorithm**: XGBRegressor (1000 trees, max depth 4, learning rate 0.1)
- **Target transform**: log(Total Charges + 3000)
- **Inflation adjustment**: configurable, default 7.17% (IP HCC 2021 Annual Change)
- **Training data**: 2018-2020 discharges
- **Validation data**: 2021 discharges

## Module Reference

### data_loader.py

- `load_data(filepath)` - Load CSV and select columns
- `apply_mappings(df)` - Ordinal encode all categorical columns
- `impute_facility_id(df)` - XGBClassifier imputation for Permanent Facility Id
- `impute_ccsr_procedure(df)` - XGBClassifier imputation for CCSR Procedure Description
- `apply_ohe(df)` - One-Hot Encode nominal features
- `split_by_year(df, year)` - Temporal train/validation split
- `load_and_prepare(filepath)` - End-to-end pipeline

### model.py

- `build_model(...)` - Create XGBRegressor pipeline
- `train_model(model, X, y)` - Train on log-transformed target
- `predict(model, X, inflation_factor)` - Predict with inverse transform
- `evaluate(y_true, y_pred)` - Compute R2, MSE, RMSE, MAE, MAPE
- `predict_single(model, ohe_transformer, input_dict)` - Single-observation prediction

### out_of_pocket_cost.py

- `calculate_insurance_oop(charge, deductible, co_insurance_pct, oop_max)` - Insurance OOP cost
- `calculate_self_pay(charge, discount_pct)` - Self-pay OOP cost

## Data Source

[Hospital Inpatient Discharges (SPARCS De-Identified)](https://health.data.ny.gov/browse?q=Hospital%20Inpatient%20Discharges%20(SPARCS%20De-Identified)&sortBy=relevance) - New York State Department of Health
