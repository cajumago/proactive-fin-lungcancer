# NY Hospital Inpatient Discharges - Respiratory Cancer Total Charge Prediction

Predicts Total Charges for respiratory cancer inpatient discharges in New York State using SPARCS data (2018-2021) and an XGBRegressor model. Deployed as a Streamlit web application with PySpark-powered data loading and downloadable reports.

## Project Structure

```
appv2_optimized/
  2018_2021.csv                      # SPARCS dataset (source CSV)
  2018_2021.parquet                  # Optimized Parquet format (used by app)
  data_loader.py                     # Data loading (PySpark/Parquet + pandas fallback), mapping, imputation, OHE
  model.py                           # XGBRegressor training, prediction, evaluation
  out_of_pocket_cost.py              # Insurance and self-pay OOP cost calculation
  report_generator.py                # In-memory report generation (CSV, Excel, PDF)
  streamlit_page.py                  # Streamlit web application
  model_with_los_Refactored.ipynb    # Analysis notebook (includes Parquet conversion)
  requirements.txt                   # Python dependencies
  packages.txt                       # System-level dependencies for Streamlit Cloud
```

## Setup

### Local Development

```bash
pip install -r requirements.txt
```

**macOS note**: XGBoost requires OpenMP. If you get `libxgboost.dylib` errors:
```bash
# Anaconda
conda install -c conda-forge llvm-openmp
# Homebrew
brew install libomp
```

### Streamlit Cloud

The app is configured for deployment on [Streamlit Cloud](https://share.streamlit.io/):

- `packages.txt` installs system dependencies (`libgomp1` for XGBoost, `default-jdk-headless` for PySpark)
- `requirements.txt` installs all Python packages
- Data is loaded from `2018_2021.parquet` using dynamic path resolution (`pathlib`)

### Generating the Parquet File

If `2018_2021.parquet` does not exist, run the conversion cell in the notebook:

```bash
jupyter notebook model_with_los_Refactored.ipynb
# Run Section 2.1 "Convert CSV to Parquet"
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
5. Download results as CSV, Excel, or PDF reports.

The model trains on startup using the Parquet dataset via PySpark. Validation metrics (2021 holdout) are displayed in the sidebar.

## Model Details

- **Algorithm**: XGBRegressor (1000 trees, max depth 4, learning rate 0.1)
- **Target transform**: log(Total Charges + 3000)
- **Inflation adjustment**: configurable, default 7.17% (IP HCC 2021 Annual Change)
- **Training data**: 2018-2020 discharges (27,158 records)
- **Validation data**: 2021 discharges (8,428 records)
- **Validation metrics**: R2=0.8344, MAE=$21,203, MAPE=20.28%

## Module Reference

### data_loader.py

- `get_spark_session()` - Create low-memory PySpark session (local mode, 512MB)
- `load_data_spark(filepath, spark)` - Load Parquet via PySpark, convert to pandas
- `load_data(filepath)` - Load CSV via pandas (fallback)
- `apply_mappings(df)` - Ordinal encode all categorical columns
- `impute_facility_id(df)` - XGBClassifier imputation for Permanent Facility Id
- `impute_ccsr_procedure(df)` - XGBClassifier imputation for CCSR Procedure Description
- `apply_ohe(df)` - One-Hot Encode nominal features
- `split_by_year(df, year)` - Temporal train/validation split
- `load_and_prepare(filepath, use_spark=True)` - End-to-end pipeline

### model.py

- `build_model(...)` - Create XGBRegressor pipeline
- `train_model(model, X, y)` - Train on log-transformed target
- `predict(model, X, inflation_factor)` - Predict with inverse transform
- `evaluate(y_true, y_pred)` - Compute R2, MSE, MAE, MAPE
- `predict_single(model, ohe_transformer, input_dict)` - Single-observation prediction

### out_of_pocket_cost.py

- `calculate_insurance_oop(charge, deductible, co_insurance_pct, oop_max)` - Insurance OOP cost
- `calculate_self_pay(charge, discount_pct)` - Self-pay OOP cost

### report_generator.py

- `generate_csv(input_data, estimated_charge, ...)` - CSV report as bytes
- `generate_excel(input_data, estimated_charge, ...)` - Excel report as bytes
- `generate_pdf(input_data, estimated_charge, ...)` - PDF report as bytes

## Data Source

[Hospital Inpatient Discharges (SPARCS De-Identified)](https://health.data.ny.gov/browse?q=Hospital%20Inpatient%20Discharges%20(SPARCS%20De-Identified)&sortBy=relevance) - New York State Department of Health
