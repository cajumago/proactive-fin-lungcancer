"""
data_loader.py - Data loading, transformation, and feature engineering for
NY Hospital Inpatient Discharges (SPARCS) Respiratory Cancer analysis.

This module handles:
- CSV data loading and column selection
- Categorical feature mapping (ordinal encoding)
- Missing value imputation (XGBClassifier for Permanent Facility Id and CCSR Procedure)
- One-Hot Encoding for nominal categorical features
- Train/validation splitting by Discharge Year
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
#from pyspark.sql import SparkSession


# ---------------------------------------------------------------------------
# Column configuration
# ---------------------------------------------------------------------------

SELECTED_COLUMNS = [
    "CCSR Procedure Description",
    "Discharge Year",
    "Age Group",
    "Length of Stay",
    "Type of Admission",
    "Patient Disposition",
    "APR DRG Description",
    "APR Severity of Illness Description",
    "APR Medical Surgical Description",
    "Payment Typology 1",
    "Total Charges",
    "Permanent Facility Id",
]

OHE_COLUMNS = [
    "Age Group",
    "Type of Admission",
    "Patient Disposition",
    "APR Medical Surgical Description",
    "Payment Typology 1",
]

# Final column order after OHE (Discharge Year first, Total Charges last)
FINAL_COLUMN_ORDER_PREFIX = ["remainder__Discharge Year"]
FINAL_COLUMN_ORDER_SUFFIX = ["remainder__Total Charges"]


# ---------------------------------------------------------------------------
# Categorical mappings
# ---------------------------------------------------------------------------

AGE_GROUP_MAP = {
    "0 to 17": 1,
    "18 to 29": 2,
    "30 to 49": 3,
    "50 to 69": 4,
    "70 or Older": 5,
}

TYPE_OF_ADMISSION_MAP = {
    "Elective": 1,
    "Emergency": 2,
    "Newborn": 3,
    "Not Available": 4,
    "Trauma": 5,
    "Urgent": 6,
}

APR_MEDICAL_SURGICAL_MAP = {
    "Surgical": 1,
    "Medical": 2,
}

PATIENT_DISPOSITION_MAP = {
    "Hospice - Home": 1,
    "Expired": 2,
    "Home w/ Home Health Services": 3,
    "Home or Self Care": 4,
    "Skilled Nursing Home": 5,
    "Left Against Medical Advice": 6,
    "Short-term Hospital": 7,
    "Hospice - Medical Facility": 8,
    "Inpatient Rehabilitation Facility": 9,
    "Cancer Center or Children's Hospital": 10,
    "Court/Law Enforcement": 11,
    "Psychiatric Hospital or Unit of Hosp": 12,
    "Medicare Cert Long Term Care Hospital": 13,
    "Another Type Not Listed": 14,
    "Facility w/ Custodial/Supportive Care": 15,
    "Federal Health Care Facility": 16,
    "Hosp Basd Medicare Approved Swing Bed": 17,
    "Critical Access Hospital": 18,
    "Medicaid Cert Nursing Facility": 19,
}

APR_SEVERITY_MAP = {
    "Minor": 1,
    "Moderate": 2,
    "Major": 3,
    "Extreme": 4,
}

APR_DRG_MAP = {
    "ALLOGENEIC BONE MARROW TRANSPLANT": 1,
    "AUTOLOGOUS BONE MARROW TRANSPLANT OR T-CELL IMMUNOTHERAPY": 2,
    "CYSTIC FIBROSIS - PULMONARY DISEASE": 3,
    "EAR, NOSE, MOUTH, THROAT, CRANIAL/FACIAL MALIGNANCIES": 4,
    "EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS": 5,
    "EXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS": 6,
    "EXTRACORPOREAL MEMBRANE OXYGENATION (ECMO)": 7,
    "MAJOR RESPIRATORY & CHEST PROCEDURES": 8,
    "MAJOR RESPIRATORY AND CHEST PROCEDURES": 9,
    "MODERATELY EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS": 10,
    "MODERATELY EXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS": 11,
    "NON-EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS": 12,
    "NONEXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS": 13,
    "OTHER RESPIRATORY & CHEST PROCEDURES": 14,
    "OTHER RESPIRATORY AND CHEST PROCEDURES": 15,
    "OTHER RESPIRATORY DIAGNOSES EXCEPT SIGNS, SYMPTOMS & MINOR DIAGNOSES": 16,
    "OTHER RESPIRATORY DIAGNOSES EXCEPT SIGNS, SYMPTOMS AND MISCELLANEOUS DIAGNOSES": 17,
    "RESPIRATORY MALIGNANCY": 18,
    "RESPIRATORY SYSTEM DIAGNOSIS W VENTILATOR SUPPORT 96+ HOURS": 19,
    "RESPIRATORY SYSTEM DIAGNOSIS WITH VENTILATOR SUPPORT > 96 HOURS": 20,
    "TRACHEOSTOMY W MV 96+ HOURS W EXTENSIVE PROCEDURE": 21,
    "TRACHEOSTOMY W MV 96+ HOURS W/O EXTENSIVE PROCEDURE": 22,
    "TRACHEOSTOMY WITH MV >96 HOURS WITH EXTENSIVE PROCEDURE": 23,
    "TRACHEOSTOMY WITH MV >96 HOURS WITHOUT EXTENSIVE PROCEDURE": 24,
}

CCSR_PROCEDURE_MAP = {
    "ABDOMINAL WALL PROCEDURES, NEC": 1,
    "ADMINISTRATION AND TRANSFUSION OF BONE MARROW, STEM CELLS, PANCREATIC ISLET CELLS, AND T-CELLS": 2,
    "ADMINISTRATION OF ALBUMIN AND GLOBULIN": 3,
    "ADMINISTRATION OF ANTI-INFLAMMATORY AGENTS": 4,
    "ADMINISTRATION OF ANTIBIOTICS": 5,
    "ADMINISTRATION OF DIAGNOSTIC SUBSTANCES, NEC": 6,
    "ADMINISTRATION OF NUTRITIONAL AND ELECTROLYTIC SUBSTANCES": 7,
    "ADMINISTRATION OF THERAPEUTIC SUBSTANCES, NEC": 8,
    "ADMINISTRATION OF THROMBOLYTICS AND PLATELET INHIBITORS": 9,
    "ADRENALECTOMY": 10,
    "AIRWAY INTUBATION": 11,
    "ANEURYSM REPAIR PROCEDURES": 12,
    "ANGIOPLASTY AND RELATED VESSEL PROCEDURES (ENDOVASCULAR; EXCLUDING CAROTID)": 13,
    "ARTERIAL OXYGEN SATURATION MONITORING": 14,
    "ARTERY, VEIN, AND GREAT VESSEL PROCEDURES, NEC": 15,
    "ARTHROCENTESIS": 16,
    "BEAM RADIATION": 17,
    "BILIARY AND PANCREATIC CALCULUS REMOVAL": 18,
    "BLADDER CATHETERIZATION AND DRAINAGE": 19,
    "BONE AND JOINT BIOPSY": 20,
    "BONE EXCISION": 21,
    "BONE FIXATION (EXCLUDING EXTREMITIES)": 22,
    "BONE MARROW BIOPSY": 23,
    "BRACHYTHERAPY": 24,
    "BRONCHOSCOPIC EXCISION AND FULGURATION": 25,
    "BRONCHOSCOPY (DIAGNOSTIC)": 26,
    "BRONCHOSCOPY (THERAPEUTIC)": 27,
    "CARDIAC AND CORONARY FLUOROSCOPY": 28,
    "CARDIAC CHEST COMPRESSION": 29,
    "CARDIAC MONITORING": 30,
    "CARDIAC STRESS TESTS": 31,
    "CARDIOVASCULAR DEVICE PROCEDURES, NEC": 32,
    "CARDIOVERSION": 33,
    "CAROTID ENDARTERECTOMY AND STENTING": 34,
    "CHEMOTHERAPY": 35,
    "CHEST TUBE PLACEMENT AND THERAPEUTIC THORACENTESIS": 36,
    "CHEST WALL PROCEDURES, NEC": 37,
    "CLOSED REDUCTION OF BONES AND JOINTS": 38,
    "CNS EXCISION PROCEDURES": 39,
    "COLONOSCOPY AND PROCTOSCOPY WITH BIOPSY": 40,
    "COMMON BILE DUCT SPHINCTEROTOMY AND STENTING": 41,
    "COMPUTERIZED TOMOGRAPHY (CT) WITH CONTRAST": 42,
    "COMPUTERIZED TOMOGRAPHY (CT) WITHOUT CONTRAST": 43,
    "CONTROL OF BLEEDING (NON-ENDOSCOPIC)": 44,
    "CYSTECTOMY (INCLUDING FULGURATION) AND URETHRECTOMY": 45,
    "CYSTOSCOPY AND URETEROSCOPY (INCLUDING BIOPSY)": 46,
    "DENTAL PROCEDURES": 47,
    "DIAGNOSTIC ERCP WITH OR WITHOUT BIOPSY": 48,
    "DIAPHRAGMATIC HERNIA REPAIR": 49,
    "ELECTROCARDIOGRAM (ECG)": 50,
    "ELECTROENCEPHALOGRAM (EEG)": 51,
    "EMBOLECTOMY, ENDARTERECTOMY, AND RELATED VESSEL PROCEDURES (NON-ENDOVASCULAR; EXCLUDING CAROTID)": 52,
    "ENDOCRINE SYSTEM BIOPSY": 53,
    "ENDOSCOPIC CONTROL OF BLEEDING": 54,
    "ENT DIAGNOSTIC ENDOSCOPY (EXCLUDING LARYNGOSCOPY)": 55,
    "ENT DIAGNOSTIC PROCEDURES (NON-ENDOSCOPIC)": 56,
    "ENT DRAINAGE (EXCLUDING MYRINGOTOMY)": 57,
    "ENT EXCISION (EXCLUDING NASAL PASSAGE, SINUSES, TONGUE, SALIVARY GLANDS, LARYNX)": 58,
    "ENT PROCEDURES, NEC": 59,
    "ENT REPAIR": 60,
    "ESOPHAGOGASTRODUODENOSCOPY (EGD) WITH BIOPSY": 61,
    "EXPLORATION OF PERITONEAL CAVITY": 62,
    "EXTRACORPOREAL MEMBRANE OXYGENATION": 63,
    "EYELID PROCEDURES": 64,
    "FEMALE REPRODUCTIVE SYSTEM PROCEDURES, NEC": 65,
    "FEMUR FIXATION": 66,
    "FIXATION OF UPPER EXTREMITY BONES": 67,
    "FLUOROSCOPIC ANGIOGRAPHY (EXCLUDING CORONARY)": 68,
    "FLUOROSCOPIC GUIDANCE FOR CIRCULATORY SYSTEM PROCEDURES": 69,
    "FLUOROSCOPY OF NON-CIRCULATORY ORGANS": 70,
    "GASTRECTOMY": 71,
    "GASTRO-JEJUNAL BYPASS (INCLUDING BARIATRIC)": 72,
    "GASTROSTOMY": 73,
    "GI SYSTEM BIOPSY (NON-ENDOSCOPIC)": 74,
    "GI SYSTEM DRAINAGE (EXCLUDING PARACENTESIS)": 75,
    "GI SYSTEM ENDOSCOPIC THERAPEUTIC PROCEDURES": 76,
    "GI SYSTEM ENDOSCOPY WITHOUT BIOPSY (DIAGNOSTIC)": 77,
    "GI SYSTEM LYSIS OF ADHESIONS": 78,
    "GI SYSTEM REPAIR (EXCLUDING ANORECTAL)": 79,
    "HEART BIOPSY": 80,
    "HEART CONDUCTION MECHANISM PROCEDURES": 81,
    "HEMODIALYSIS": 82,
    "HIP ARTHROPLASTY": 83,
    "HYSTERECTOMY": 84,
    "ILEOSTOMY AND COLOSTOMY": 85,
    "IMMOBILIZATION BY SPLINT OR OTHER EXTERNAL DEVICE": 86,
    "INCISION AND DRAINAGE OF SKIN": 87,
    "INCISION AND DRAINAGE OF SUBCUTANEOUS TISSUE AND FASCIA": 88,
    "INFERIOR VENA CAVA (IVC) FILTER PROCEDURES": 89,
    "INFUSION OF VASOPRESSOR": 90,
    "INTRACRANIAL EPIDURAL AND SUBDURAL SPACE DRAINAGE": 91,
    "INTRAVENOUS INDUCTION OF LABOR": 92,
    "IRRIGATION (DIAGNOSTIC AND THERAPEUTIC)": 93,
    "ISOLATION PROCEDURES": 94,
    "JOINT TISSUE EXCISION (EXCLUDING DISCECTOMY)": 95,
    "KIDNEY AND OTHER URINARY TRACT BIOPSY (NON-ENDOSCOPIC)": 96,
    "LARYNGECTOMY": 97,
    "LARYNGOSCOPY (DIAGNOSTIC)": 98,
    "LIGATION AND EMBOLIZATION OF VESSELS": 99,
    "LIVER BIOPSY": 100,
    "LOWER GI THERAPEUTIC PROCEDURES, NEC (EXCLUDING OPEN AND LAPAROSCOPIC)": 101,
    "LUMBAR PUNCTURE": 102,
    "LUNG, PLEURA, OR DIAPHRAGM BIOPSY (NON-ENDOSCOPIC)": 103,
    "LUNG, PLEURA, OR DIAPHRAGM RESECTION (OPEN AND THORACOSCOPIC)": 104,
    "LYMPH NODE BIOPSY": 105,
    "LYMPH NODE DISSECTION": 106,
    "LYMPH NODE EXCISION (THERAPEUTIC)": 107,
    "MAGNETIC RESONANCE IMAGING (MRI)": 108,
    "MASTECTOMY AND LUMPECTOMY": 109,
    "MEASUREMENT AND MONITORING, NEC": 110,
    "MEASUREMENT DURING CARDIAC CATHETERIZATION": 111,
    "MECHANICAL VENTILATION": 112,
    "MEDIASTINAL PROCEDURES, NEC": 113,
    "MENTAL HEALTH PROCEDURES, NEC": 114,
    "MINIMALLY INVASIVE CNS BIOPSY": 115,
    "MUSCLE, TENDON, BURSA, AND LIGAMENT EXCISION": 116,
    "MUSCULOSKELETAL DEVICE PROCEDURES, NEC": 117,
    "NAIL PROCEDURES": 118,
    "NASAL AND SINUS EXCISION": 119,
    "NON-INVASIVE VENTILATION": 120,
    "OPEN AND THORACOSCOPIC PLEURAL DRAINAGE": 121,
    "OTHER CARDIOVASCULAR SYSTEM MEASUREMENT AND MONITORING": 122,
    "OTHER GI SYSTEM DEVICE PROCEDURES": 123,
    "PACEMAKER AND DEFIBRILLATOR INTERROGATION": 124,
    "PACEMAKER AND DEFIBRILLATOR PROCEDURES": 125,
    "PACKING AND DRESSING PROCEDURES": 126,
    "PANCREATIC AND PROXIMAL BILIARY DILATION AND STENTING": 127,
    "PANCREATICOBILIARY BIOPSY": 128,
    "PARACENTESIS": 129,
    "PERCUTANEOUS CORONARY INTERVENTIONS (PCI)": 130,
    "PERICARDIAL PROCEDURES": 131,
    "PERITONEAL DIALYSIS": 132,
    "PHARMACOTHERAPY FOR MENTAL HEALTH (EXCLUDING SUBSTANCE USE)": 133,
    "PHARMACOTHERAPY FOR SUBSTANCE USE": 134,
    "PHERESIS THERAPY": 135,
    "PHYSICAL, OCCUPATIONAL, AND RESPIRATORY THERAPY TREATMENT": 136,
    "PLACEMENT OF TUNNELED OR IMPLANTABLE PORTION OF A VASCULAR ACCESS DEVICE": 137,
    "PLAIN RADIOGRAPHY": 138,
    "PLANAR NUCLEAR MEDICINE IMAGING": 139,
    "POSITRON EMISSION TOMOGRAPHIC (PET) IMAGING": 140,
    "POTENTIAL COVID-19 THERAPIES": 141,
    "PULMONARY ARTERIAL PRESSURE MONITORING": 142,
    "PULMONARY FUNCTION TESTS": 143,
    "RADIATION THERAPY, NEC": 144,
    "REGIONAL ANESTHESIA": 145,
    "RELEASE OF LUNG AND PLEURA": 146,
    "RESPIRATORY SYSTEM PROCEDURES, NEC": 147,
    "RETROPERITONEAL PROCEDURES, NEC": 148,
    "ROBOTIC-ASSISTED PROCEDURES": 149,
    "SKIN BIOPSY AND DIAGNOSTIC DRAINAGE": 150,
    "SKIN LACERATION REPAIR (EXCLUDING PERINEUM)": 151,
    "SMALL BOWEL RESECTION": 152,
    "SPINAL CORD DECOMPRESSION": 153,
    "SPINAL EPIDURAL CATHETER PLACEMENT": 154,
    "SPINE FUSION": 155,
    "SUBCUTANEOUS TISSUE AND FASCIA EXCISION": 156,
    "SUBCUTANEOUS TISSUE AND FASCIA PROCEDURES, NEC": 157,
    "SUBCUTANEOUS TISSUE, FASCIA, AND MUSCLE BIOPSY": 158,
    "SUBSTANCE USE DETOXIFICATION": 159,
    "TENDON, MUSCLE, BURSA, AND LIGAMENT REPAIR (EXCLUDING PERINEAL)": 160,
    "THORACENTESIS (DIAGNOSTIC)": 161,
    "THYMECTOMY": 162,
    "THYROIDECTOMY": 163,
    "TOMOGRAPHIC NUCLEAR MEDICINE IMAGING": 164,
    "TRACHEOSTOMY": 165,
    "TRANSFUSION OF BLOOD AND BLOOD PRODUCTS": 166,
    "TRANSFUSION OF CLOTTING FACTORS": 167,
    "TRANSFUSION OF PLASMA": 168,
    "ULTRASONOGRAPHY": 169,
    "UPPER GI THERAPEUTIC PROCEDURES, NEC (ENDOSCOPIC)": 170,
    "UPPER GI THERAPEUTIC PROCEDURES, NEC (OPEN AND LAPAROSCOPIC)": 171,
    "URETER AND OTHER URINARY TRACT DILATION": 172,
    "VACCINATIONS": 173,
    "VENOUS AND ARTERIAL CATHETER PLACEMENT": 174,
    "VESSEL REPAIR AND REPLACEMENT": 175,
}

PAYMENT_TYPOLOGY_MAP = {
    "Medicare": 1,
    "Self-Pay": 2,
    "Private Health Insurance": 3,
    "Blue Cross/Blue Shield": 4,
    "Medicaid": 5,
    "Federal/State/Local/VA": 6,
    "Department of Corrections": 7,
    "Miscellaneous/Other": 8,
    "Managed Care, Unspecified": 9,
    "Unknown": 10,
}

# Reverse lookups (integer -> label) for Streamlit dropdowns
AGE_GROUP_LABELS = {v: k for k, v in AGE_GROUP_MAP.items()}
TYPE_OF_ADMISSION_LABELS = {v: k for k, v in TYPE_OF_ADMISSION_MAP.items()}
APR_MEDICAL_SURGICAL_LABELS = {v: k for k, v in APR_MEDICAL_SURGICAL_MAP.items()}
PATIENT_DISPOSITION_LABELS = {v: k for k, v in PATIENT_DISPOSITION_MAP.items()}
APR_SEVERITY_LABELS = {v: k for k, v in APR_SEVERITY_MAP.items()}
APR_DRG_LABELS = {v: k for k, v in APR_DRG_MAP.items()}
CCSR_PROCEDURE_LABELS = {v: k for k, v in CCSR_PROCEDURE_MAP.items()}
PAYMENT_TYPOLOGY_LABELS = {v: k for k, v in PAYMENT_TYPOLOGY_MAP.items()}


# ---------------------------------------------------------------------------
# PySpark session and Parquet loading
# ---------------------------------------------------------------------------

def get_spark_session():
    """Create a low-memory PySpark SparkSession configured for local mode.

    Returns
    -------
    pyspark.sql.SparkSession
    """

    from pyspark.sql import SparkSession
    
    return (
        SparkSession.builder
        .appName("NY_RC_LOS")
        .master("local[1]")
        .config("spark.driver.memory", "512m")
        .config("spark.executor.memory", "512m")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.adaptive.enabled", "false")
        .getOrCreate()
    )


def load_data_spark(filepath, spark=None):
    """Load Parquet data using PySpark, select columns, and convert to pandas.

    The SparkSession is stopped after conversion to free JVM memory.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the Parquet file or directory.
    spark : pyspark.sql.SparkSession, optional
        Existing SparkSession. If None, a new one is created.

    Returns
    -------
    pd.DataFrame
        DataFrame with only the columns needed for modeling.
    """
    stop_spark = spark is None
    if spark is None:
        spark = get_spark_session()

    sdf = spark.read.parquet(str(filepath))
    sdf = sdf.select(SELECTED_COLUMNS)
    df = sdf.toPandas()

    if stop_spark:
        spark.stop()

    return df


# ---------------------------------------------------------------------------
# Data loading (CSV fallback)
# ---------------------------------------------------------------------------

def load_data(filepath):
    """Load the SPARCS CSV and select relevant columns.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (e.g. '2018_2021.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame with only the columns needed for modeling.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df = df.reindex(columns=SELECTED_COLUMNS)
    return df


# ---------------------------------------------------------------------------
# Categorical mapping
# ---------------------------------------------------------------------------

def apply_mappings(df):
    """Apply ordinal encoding to all categorical columns.

    Converts string-valued categorical columns to integer codes using
    the predefined mapping dictionaries. Also converts 'Length of Stay'
    values of '120 +' to 121 and casts to float.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with string categorical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with all categorical columns mapped to integers.
    """
    df = df.copy()

    # Length of Stay: replace '120 +' sentinel with 121
    df["Length of Stay"] = (
        df["Length of Stay"].replace({"120 +": "121"}).astype(str).astype(float)
    )

    df["Age Group"] = df["Age Group"].map(AGE_GROUP_MAP)
    df["Type of Admission"] = df["Type of Admission"].map(TYPE_OF_ADMISSION_MAP)
    df["APR Medical Surgical Description"] = df["APR Medical Surgical Description"].map(
        APR_MEDICAL_SURGICAL_MAP
    )
    df["Patient Disposition"] = df["Patient Disposition"].map(PATIENT_DISPOSITION_MAP)
    df["APR Severity of Illness Description"] = df[
        "APR Severity of Illness Description"
    ].map(APR_SEVERITY_MAP)
    df["APR DRG Description"] = df["APR DRG Description"].map(APR_DRG_MAP)
    df["CCSR Procedure Description"] = df["CCSR Procedure Description"].map(
        CCSR_PROCEDURE_MAP
    )
    df["Payment Typology 1"] = df["Payment Typology 1"].map(PAYMENT_TYPOLOGY_MAP)

    return df


# ---------------------------------------------------------------------------
# Missing value imputation
# ---------------------------------------------------------------------------

def impute_facility_id(df):

    train_0 = df[df['Permanent Facility Id'].notnull()]
    test_0  = df[df['Permanent Facility Id'].isnull()]

    # y as 1-D numpy array of ints
    y_series = train_0.iloc[:, 11].astype(int)            # use iloc[:,11] not iloc[:,11:12]
    y = y_series.values                                   # shape (n_samples,)

    # map original labels -> 0..K-1
    unique_labels = np.unique(y)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    y_mapped = np.array([label_to_idx[v] for v in y], dtype=np.int32)

    # features: fit scaler on train, transform both train and test
    scaler = StandardScaler()
    X_train = train_0.iloc[:, :11].values.astype(np.float64)
    X_test  = test_0.iloc[:, :11].values.astype(np.float64)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # classifier: pass numpy arrays (1-D y) to fit
    clssfr = Pipeline([
        ('transform', FunctionTransformer(lambda X: X.astype(np.float64))),
        ('classifier', XGBClassifier(objective='multi:softprob', num_class=len(unique_labels)))
    ])

    clssfr.fit(X_train_scaled, y_mapped)

    preds_mapped = clssfr.predict(X_test_scaled).astype(int)
    preds_original = np.array([idx_to_label[p] for p in preds_mapped], dtype=y_series.dtype)

    # assign predictions back to original dataframe in one vectorized operation
    df.loc[test_0.index, 'Permanent Facility Id'] = preds_original

    return df


def impute_ccsr_procedure(df):
    
    train_0 = df[df['CCSR Procedure Description'].notnull()]
    test_0  = df[df['CCSR Procedure Description'].isnull()]

    # get y as a 1-D Series/array of scalars
    # use iloc[:, 0] or squeeze() to avoid a (n,1) array
    y_series = train_0.iloc[:, 0].astype(int)        # single column as Series
    y = y_series.values.ravel()                      # 1-D numpy array

    # map original labels -> 0..K-1
    unique_labels = np.unique(y)
    label_to_idx = {int(lab): i for i, lab in enumerate(unique_labels)}
    idx_to_label = {i: int(lab) for lab, i in label_to_idx.items()}

    y_mapped = np.array([label_to_idx[int(v)] for v in y], dtype=np.int32)

    # features: fit scaler on train, transform test
    scaler = StandardScaler()
    X_train = train_0.iloc[:, 1:].values.astype(np.float64)
    X_test  = test_0.iloc[:, 1:].values.astype(np.float64)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # classifier
    clssfr = Pipeline([
        ('transform', FunctionTransformer(lambda X: X.astype(np.float64))),
        ('classifier', XGBClassifier(objective='multi:softprob', num_class=len(unique_labels)))
    ])

    clssfr.fit(X_train_scaled, y_mapped)

    preds_mapped = clssfr.predict(X_test_scaled).astype(int)
    preds_original = np.array([idx_to_label[p] for p in preds_mapped], dtype=y_series.dtype)

    # vectorized assignment (no chained assignment)
    df.loc[test_0.index, 'CCSR Procedure Description'] = preds_original

    return df


# ---------------------------------------------------------------------------
# One-Hot Encoding
# ---------------------------------------------------------------------------

def apply_ohe(df, ct=None, fit=True):

    """Create and return the ColumnTransformer for One-Hot Encoding.

    Encodes: Age Group, Type of Admission, Patient Disposition,
    APR Medical Surgical Description, Payment Typology 1.
    All other columns pass through unchanged.

    Returns
    -------
    sklearn.compose.ColumnTransformer
    """
    ct = ColumnTransformer(
        transformers=[
            ("OHE", OneHotEncoder(sparse_output=False), OHE_COLUMNS),
        ],
        remainder="passthrough",
    )

    """Apply One-Hot Encoding and reorder columns.

    Parameters
    ----------
    df : pd.DataFrame
        Fully imputed DataFrame with mapped integer categories.
    ct : ColumnTransformer or None
        Pre-fitted transformer. If None and fit=True, a new one is created.
    fit : bool
        If True, fit the transformer on df. If False, only transform.

    Returns
    -------
    tuple[pd.DataFrame, ColumnTransformer]
        (encoded DataFrame with reordered columns, fitted transformer)
    """
    encoded = ct.fit_transform(df)
    col_names = ct.get_feature_names_out()

    result = pd.DataFrame(encoded, columns=col_names)

    # Reorder: Discharge Year first, then OHE cols, then remainder cols, Total Charges last
    ohe_cols = [c for c in col_names if c.startswith("OHE__")]
    remainder_cols = [
        c
        for c in col_names
        if c.startswith("remainder__")
        and c not in ("remainder__Discharge Year", "remainder__Total Charges")
    ]
    ordered = (
        FINAL_COLUMN_ORDER_PREFIX + ohe_cols + remainder_cols + FINAL_COLUMN_ORDER_SUFFIX
    )
    result = result.reindex(columns=ordered)
    result.drop_duplicates(keep="first", inplace=True)

    return result, ct


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_by_year(df, validation_year=2021):
    """Split data into training and validation sets by Discharge Year.

    Training set: all years before *validation_year*.
    Validation set: rows matching *validation_year*.

    Parameters
    ----------
    df : pd.DataFrame
        OHE-encoded DataFrame with 'remainder__Discharge Year' column.
    validation_year : int
        Year to use as the validation set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (X_train, X_valid, y_train, y_valid)
        Features exclude Discharge Year and Total Charges.
        Target is Total Charges.
    """
    train_mask = df["remainder__Discharge Year"] < validation_year
    valid_mask = df["remainder__Discharge Year"] >= validation_year

    feature_cols = [
        c
        for c in df.columns
        if c not in ("remainder__Discharge Year", "remainder__Total Charges")
    ]
    target_col = "remainder__Total Charges"

    X_train = df.loc[train_mask, feature_cols]
    X_valid = df.loc[valid_mask, feature_cols]
    y_train = df.loc[train_mask, [target_col]]
    y_valid = df.loc[valid_mask, [target_col]]

    return X_train, X_valid, y_train, y_valid


# ---------------------------------------------------------------------------
# Full pipeline convenience function
# ---------------------------------------------------------------------------

def load_and_prepare(filepath, validation_year=2021, use_spark=True, spark=None):
    """End-to-end data preparation: load, map, impute, encode, split.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the data file (Parquet if use_spark=True, CSV otherwise).
    validation_year : int
        Year to hold out for validation.
    use_spark : bool
        If True, use PySpark to read Parquet. If False, use pandas CSV reader.
    spark : pyspark.sql.SparkSession, optional
        Existing SparkSession (only used if use_spark=True).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'X_train', 'X_valid', 'y_train', 'y_valid': split DataFrames
        - 'ohe_transformer': fitted ColumnTransformer
        - 'encoded_df': full encoded DataFrame (before split)
    """
    if use_spark:
        df = load_data_spark(filepath, spark=spark)
    else:
        df = load_data(filepath)
    df = apply_mappings(df)
    df = impute_facility_id(df)
    df = impute_ccsr_procedure(df)
    encoded_df, ohe_ct = apply_ohe(df)
    X_train, X_valid, y_train, y_valid = split_by_year(encoded_df, validation_year)

    return {
        "X_train": X_train,
        "X_valid": X_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "ohe_transformer": ohe_ct,
        "encoded_df": encoded_df,
    }
