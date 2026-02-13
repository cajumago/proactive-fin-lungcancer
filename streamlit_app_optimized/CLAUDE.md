# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive model for NY State respiratory cancer inpatient hospital charges using SPARCS discharge data (2018-2021, 35,620 records). Combines XGBoost regression with a Streamlit web UI for interactive cost estimation, out-of-pocket calculations, and downloadable reports (CSV, Excel, PDF).

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

**Streamlit Cloud deployment**: System dependencies are defined in `packages.txt` (`libgomp1` for XGBoost, `default-jdk-headless` for PySpark).

## Architecture

### Data Pipeline (`data_loader.py`)

Orchestrated by `load_and_prepare()`:

1. **Load** — `load_data_spark()` reads `2018_2021.parquet` via PySpark (local mode, 512MB driver), selects 12 columns, converts to pandas. SparkSession is stopped immediately after `.toPandas()` to free JVM memory. Falls back to `load_data()` for CSV via pandas when `use_spark=False`.
2. **Encode** — `apply_mappings()` ordinal-encodes 9 categorical features using hardcoded dictionaries (e.g., `AGE_GROUP_MAP`, `APR_DRG_MAP`, 175 CCSR procedures)
3. **Impute** — Two XGBClassifiers fill missing values: `impute_facility_id()` (8 NaN) and `impute_ccsr_procedure()` (4,301 NaN)
4. **One-Hot Encode** — `apply_ohe()` expands 5 nominal features -> 47 total features
5. **Split** — `split_by_year()` creates temporal split: 2018-2020 train (27,158), 2021 validation (8,428)

PySpark session configuration (`get_spark_session()`): `local[1]`, 512MB driver/executor, Spark UI disabled, adaptive query execution disabled.

### Model (`model.py`)

- **Algorithm**: XGBRegressor (1000 trees, max_depth=4, learning_rate=0.1)
- **Target transform**: `log(Total Charges + 3000)` during training; inverse on prediction
- **Inflation adjustment**: default 7.17% (2021 IP HCC), configurable via slider
- **Prediction**: `predict_single()` for single-observation inference used by Streamlit

### Out-of-Pocket Costs (`out_of_pocket_cost.py`)

Two calculation paths:
- `calculate_insurance_oop(charge, deductible, coinsurance%, oop_max)` — applies deductible then coinsurance, capped at OOP max
- `calculate_self_pay(charge, discount%)` — applies facility discount

### Report Generator (`report_generator.py`)

In-memory report generation for download:
- `generate_csv(...)` — CSV via `io.BytesIO`
- `generate_excel(...)` — Excel (.xlsx) via openpyxl
- `generate_pdf(...)` — Professional PDF via fpdf2 with patient info table, highlighted results, disclaimer

### Streamlit UI (`streamlit_page.py`)

- Uses `pathlib.Path(__file__).resolve().parent` for dynamic path resolution (cloud-safe)
- Split caching: `@st.cache_data` for data loading, `@st.cache_resource` for model training
- Two-column form: 10 patient/discharge inputs + inflation slider
- Conditional OOP section appears after prediction (stored in `st.session_state`)
- Download section: CSV, Excel, PDF buttons (appear after prediction)
- Sidebar shows validation metrics (R2=0.8344, MAE=$21,203, MAPE=20.28%)

### Deployment Files

- `packages.txt` — System-level apt packages for Streamlit Cloud (libgomp1, default-jdk-headless)
- `requirements.txt` — Python dependencies including pyspark, pyarrow, fpdf2, openpyxl

## Key Constants

| Constant | Location | Value |
|----------|----------|-------|
| `LOG_OFFSET` | model.py | 3000 |
| `DEFAULT_INFLATION_FACTOR` | model.py | 1.0717 |
| `DATA_PATH` | streamlit_page.py | `2018_2021.parquet` (resolved via pathlib) |
| `OHE_COLUMNS` | data_loader.py | Age Group, Type of Admission, Patient Disposition, APR Med/Surg, Payment Typology |

## Important Patterns

- All categorical mapping dictionaries live in `data_loader.py`. When adding new categories, update both the mapping dict and the corresponding Streamlit dropdown options.
- The model pipeline uses sklearn's `FunctionTransformer` wrapping numpy log/exp for the target transform.
- `predict_single()` in `model.py` constructs a DataFrame matching the OHE column structure from training — any changes to feature encoding must be reflected there.
- Data is stored as Parquet (converted from CSV via PySpark in the notebook). The CSV is kept for reference but the app reads Parquet.
- SparkSession is ephemeral: created only for Parquet I/O, stopped immediately after to conserve memory on Streamlit Cloud (~1GB limit).
