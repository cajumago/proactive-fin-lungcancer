# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive model for NY State respiratory cancer inpatient hospital charges using SPARCS discharge data (2018-2021, 35,620 records). Combines XGBoost regression with a Streamlit web UI for interactive cost estimation and out-of-pocket calculations.

## Commands

```bash
# Run the Streamlit app
streamlit run streamlit_page.py

# Install dependencies
pip install -r requirements.txt

# Run the analysis notebook
jupyter notebook model_with_los_Refactored.ipynb
```

**Note on macOS**: XGBoost requires OpenMP. If you get `libxgboost.dylib` errors, install via `conda install -c conda-forge llvm-openmp` (Anaconda) or `brew install libomp`.

## Architecture

### Data Pipeline (`data_loader.py`)

Orchestrated by `load_and_prepare()`:

1. **Load** — `load_data()` reads `2018_2021.csv`, selects 12 columns
2. **Encode** — `apply_mappings()` ordinal-encodes 9 categorical features using hardcoded dictionaries (e.g., `AGE_GROUP_MAP`, `APR_DRG_MAP`, 175 CCSR procedures)
3. **Impute** — Two XGBClassifiers fill missing values: `impute_facility_id()` (8 NaN) and `impute_ccsr_procedure()` (4,301 NaN)
4. **One-Hot Encode** — `apply_ohe()` expands 5 nominal features → 47 total features
5. **Split** — `split_by_year()` creates temporal split: 2018-2020 train (27,158), 2021 validation (8,428)

### Model (`model.py`)

- **Algorithm**: XGBRegressor (1000 trees, max_depth=4, learning_rate=0.1)
- **Target transform**: `log(Total Charges + 3000)` during training; inverse on prediction
- **Inflation adjustment**: default 7.17% (2021 IP HCC), configurable via slider
- **Prediction**: `predict_single()` for single-observation inference used by Streamlit

### Out-of-Pocket Costs (`out_of_pocket_cost.py`)

Two calculation paths:
- `calculate_insurance_oop(charge, deductible, coinsurance%, oop_max)` — applies deductible then coinsurance, capped at OOP max
- `calculate_self_pay(charge, discount%)` — applies facility discount

### Streamlit UI (`streamlit_page.py`)

- Uses `@st.cache_resource` to load data and train model once
- Two-column form: 10 patient/discharge inputs + inflation slider
- Conditional OOP section appears after prediction (stored in `st.session_state`)
- Sidebar shows validation metrics (R²=0.8344, MAE=$21,203, MAPE=20.28%)

## Key Constants

| Constant | Location | Value |
|----------|----------|-------|
| `LOG_OFFSET` | model.py | 3000 |
| `DEFAULT_INFLATION_FACTOR` | model.py | 1.0717 |
| `DATA_PATH` | streamlit_page.py | `2018_2021.csv` |
| `OHE_COLUMNS` | data_loader.py | Age Group, Type of Admission, Patient Disposition, APR Med/Surg, Payment Typology |

## Important Patterns

- All categorical mapping dictionaries live in `data_loader.py`. When adding new categories, update both the mapping dict and the corresponding Streamlit dropdown options.
- The model pipeline uses sklearn's `FunctionTransformer` wrapping numpy log/exp for the target transform.
- `predict_single()` in `model.py` constructs a DataFrame matching the OHE column structure from training — any changes to feature encoding must be reflected there.
