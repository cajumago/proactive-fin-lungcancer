# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session

# Get the current credentials
session = get_active_session()


import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression

from xgboost import XGBClassifier, XGBRegressor


dataframe = session.sql(
    "SELECT * FROM FIN_LUNG_CANCER.PUBLIC.SPARCS_2018_2021;"
)

# Execute the query and convert it into a Pandas dataframe
df = dataframe.to_pandas()

df = df.reindex(columns=['CCSR_PROCED_DESCRIPTION', 'DISCHARGE_YEAR', 'AGE_GROUP', 'TYPE_OF_ADMISSION', 
                         'PATIENT_DISPOSITIONS', 'APR_DRG_DESCRIPTION', 'APR_SEVERITY_DESCRIPTION', 
                         'APR_MEDICAL_SURGICAL', 'PAYMENT_1', 'TOTAL_CHARGES', 
                         'PERMANENT_FACILITY_ID'])

def mapping(df):

    le = LabelEncoder()
    df["AGE_GROUP"] = le.fit_transform(df["AGE_GROUP"])
    df["TYPE_OF_ADMISSION"] = le.fit_transform(df["TYPE_OF_ADMISSION"])
    df["APR_MEDICAL_SURGICAL"] = le.fit_transform(df["APR_MEDICAL_SURGICAL"])
    df['PATIENT_DISPOSITIONS'] = le.fit_transform(df['PATIENT_DISPOSITIONS'])
    df["APR_SEVERITY_DESCRIPTION"] = le.fit_transform(df["APR_SEVERITY_DESCRIPTION"])
    df['APR_DRG_DESCRIPTION'] = le.fit_transform(df['APR_DRG_DESCRIPTION'])
    df['CCSR_PROCED_DESCRIPTION'] = le.fit_transform(df['CCSR_PROCED_DESCRIPTION'])
    df['PAYMENT_1'] = le.fit_transform(df['PAYMENT_1'])

    return df

train_df = mapping(df)

# CCSR Procedure Description
ct = ColumnTransformer(transformers = [('CCSR', SimpleImputer(strategy='most_frequent'), 
                                        ['CCSR_PROCED_DESCRIPTION'])], remainder='passthrough')
ct_CCSR = ct.fit_transform(train_df)
train_df_ccsr = pd.DataFrame(ct_CCSR, columns = train_df.columns)

# Permanent Facility Id
def facilityId_imputation(df):
    
    train_0 = df[df['PERMANENT_FACILITY_ID'].notnull()] # df == train_df
    y_0 = train_0.iloc[:,10:11].values.ravel()
    test_0 = df[df['PERMANENT_FACILITY_ID'].isnull()]   # df == train_df

    scaler = StandardScaler()
    train_0_norm = scaler.fit_transform(train_0.iloc[:,:10])
    train_0_norm = pd.DataFrame(train_0_norm, columns = train_0.iloc[:,:10].columns)
    test_0_norm = scaler.fit_transform(test_0.iloc[:,:10])
    test_0_norm = pd.DataFrame(test_0_norm, columns = test_0.iloc[:,:10].columns)
    
    #clssfr = XGBClassifier() # use_label_encoder=False --> deprecated. ValueError: Invalid classes inferred from unique values of `y`
    clssfr = LogisticRegression(max_iter=500)
    clssfr.fit(train_0_norm, y_0)

    for i, j in enumerate(test_0.index[:len(clssfr.predict(test_0_norm))]):
        df['PERMANENT_FACILITY_ID'].loc[j] = clssfr.predict(test_0_norm)[i]
    
    return df

train_df_toOHE = facilityId_imputation(train_df_ccsr)

ct_oh = ColumnTransformer(transformers = [('OHE', OneHotEncoder(sparse_output=False), 
                                           ['AGE_GROUP', 'TYPE_OF_ADMISSION', 'PATIENT_DISPOSITIONS', 
                                            'APR_MEDICAL_SURGICAL', 'PAYMENT_1'])], 
                          remainder='passthrough')

ct_oh_categorical = ct_oh.fit_transform(train_df_toOHE)
oh_columns = ct_oh.get_feature_names_out()
train_df = pd.DataFrame(ct_oh_categorical, columns = oh_columns)

train_df = train_df.reindex(columns=['remainder__DISCHARGE_YEAR', 'OHE__AGE_GROUP_1.0', 'OHE__AGE_GROUP_2.0', 'OHE__AGE_GROUP_3.0',
                                        'OHE__AGE_GROUP_4.0', 'OHE__AGE_GROUP_5.0',
                                        'OHE__TYPE_OF_ADMISSION_1.0', 'OHE__TYPE_OF_ADMISSION_2.0',
                                        'OHE__TYPE_OF_ADMISSION_4.0', 'OHE__TYPE_OF_ADMISSION_5.0',
                                        'OHE__TYPE_OF_ADMISSION_6.0', 'OHE__PATIENT_DISPOSITIONS_1.0',
                                        'OHE__PATIENT_DISPOSITIONS_2.0', 'OHE__PATIENT_DISPOSITIONS_3.0',
                                        'OHE__PATIENT_DISPOSITIONS_4.0', 'OHE__PATIENT_DISPOSITIONS_5.0',
                                        'OHE__PATIENT_DISPOSITIONS_6.0', 'OHE__PATIENT_DISPOSITIONS_7.0',
                                        'OHE__PATIENT_DISPOSITIONS_8.0', 'OHE__PATIENT_DISPOSITIONS_9.0',
                                        'OHE__PATIENT_DISPOSITIONS_10.0', 'OHE__PATIENT_DISPOSITIONS_11.0',
                                        'OHE__PATIENT_DISPOSITIONS_12.0', 'OHE__PATIENT_DISPOSITIONS_13.0',
                                        'OHE__PATIENT_DISPOSITIONS_14.0', 'OHE__PATIENT_DISPOSITIONS_15.0',
                                        'OHE__PATIENT_DISPOSITIONS_16.0', 'OHE__PATIENT_DISPOSITIONS_17.0',
                                        'OHE__PATIENT_DISPOSITIONS_18.0', 'OHE__PATIENT_DISPOSITIONS_19.0',
                                        'OHE__APR_MEDICAL_SURGICAL_1.0',
                                        'OHE__APR_MEDICAL_SURGICAL_2.0',
                                        'OHE__PAYMENT_1_1.0', 'OHE__PAYMENT_1_2.0',
                                        'OHE__PAYMENT_1_3.0', 'OHE__PAYMENT_1_4.0',
                                        'OHE__PAYMENT_1_5.0', 'OHE__PAYMENT_1_6.0',
                                        'OHE__PAYMENT_1_7.0', 'OHE__PAYMENT_1_8.0',
                                        'OHE__PAYMENT_1_9.0',
                                        'remainder__CCSR_PROCED_DESCRIPTION', 'remainder__APR_DRG_DESCRIPTION', 
                                        'remainder__APR_SEVERITY_DESCRIPTION', 
                                        'remainder__PERMANENT_FACILITY_ID',
                                        'remainder__TOTAL_CHARGES'])

train_df.drop_duplicates(keep='first', inplace=True)

X_train = train_df.drop(train_df[train_df['remainder__DISCHARGE_YEAR'] > 2020].index)
X_train = X_train.iloc[:,1:-1]
X_valid = train_df.drop(train_df[train_df['remainder__DISCHARGE_YEAR'] < 2021].index)
X_valid = X_valid.iloc[:,1:-1]
y_train = train_df.drop(train_df[train_df['remainder__DISCHARGE_YEAR'] > 2020].index)
y_train = y_train.iloc[:,-1:]
y_valid = train_df.drop(train_df[train_df['remainder__DISCHARGE_YEAR'] < 2021].index)
y_valid = y_valid.iloc[:,-1:]

y_train_trans = np.log(y_train + 3000)
y_valid_trans = np.log(y_valid + 3000)

model = Pipeline([('transform', FunctionTransformer(np.float64)),('regressor', XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.1))])
#model = XGBRegressor(n_estimators=1000, max_depth=4, learning_rate=0.1)
#model = LinearRegression()
model.fit(X_train, y_train_trans)

preds = model.predict(X_valid)
# Let's increase our predictions by 7.167231% according to the Inflation Annual Change - IP (HCC 2021)
preds_ = (np.exp(preds) - 3000) * 1.0717