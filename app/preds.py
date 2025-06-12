import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from xgboost import XGBClassifier, XGBRegressor


df = pd.read_csv('2018_2021.csv')

df = df.reindex(columns=['CCSR Procedure Description', 'Discharge Year', 'Age Group', 'Type of Admission', 
                         'Patient Disposition', 'APR DRG Description', 'APR Severity of Illness Description', 
                         'APR Medical Surgical Description', 'Payment Typology 1', 'Total Charges', 
                         'Permanent Facility Id'])

def mapping(df):

    mapping0 = {"0 to 17": 1, "18 to 29": 2, "30 to 49": 3, "50 to 69": 4, "70 or Older": 5}
    df["Age Group"] = df["Age Group"].map(mapping0)
    mapping1 = {"Elective": 1, "Emergency": 2, "Newborn": 3, "Not Available": 4, "Trauma": 5, "Urgent": 6}
    df["Type of Admission"] = df["Type of Admission"].map(mapping1)
    mapping2 = {"Surgical": 1, "Medical": 2}
    df["APR Medical Surgical Description"] = df["APR Medical Surgical Description"].map(mapping2)
    
    mapping3 = {
        'Hospice - Home':1, 'Expired':2, 'Home w/ Home Health Services':3,
        'Home or Self Care':4, 'Skilled Nursing Home':5,
        'Left Against Medical Advice':6, 'Short-term Hospital':7,
        'Hospice - Medical Facility':8, 'Inpatient Rehabilitation Facility':9,
        "Cancer Center or Children's Hospital":10, 'Court/Law Enforcement':11,
        'Psychiatric Hospital or Unit of Hosp':12,
        'Medicare Cert Long Term Care Hospital':13, 'Another Type Not Listed':14,
        'Facility w/ Custodial/Supportive Care':15,
        'Federal Health Care Facility':16,
        'Hosp Basd Medicare Approved Swing Bed':17,
        'Critical Access Hospital':18, 'Medicaid Cert Nursing Facility':19
    }
    df['Patient Disposition'] = df['Patient Disposition'].map(mapping3)
    
    mapping4 = {"Minor": 1, "Moderate": 2, "Major": 3, "Extreme": 4}
    df["APR Severity of Illness Description"] = df["APR Severity of Illness Description"].map(mapping4)
    
    mapping5 = {
        'ALLOGENEIC BONE MARROW TRANSPLANT':1,
        'AUTOLOGOUS BONE MARROW TRANSPLANT OR T-CELL IMMUNOTHERAPY':2,
        'CYSTIC FIBROSIS - PULMONARY DISEASE':3,
        'EAR, NOSE, MOUTH, THROAT, CRANIAL/FACIAL MALIGNANCIES':4,
        'EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS':5,
        'EXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS':6,
        'EXTRACORPOREAL MEMBRANE OXYGENATION (ECMO)':7,
        'MAJOR RESPIRATORY & CHEST PROCEDURES':8,
        'MAJOR RESPIRATORY AND CHEST PROCEDURES':9,
        'MODERATELY EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS':10,
        'MODERATELY EXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS':11,
        'NON-EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS':12,
        'NONEXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS':13,
        'OTHER RESPIRATORY & CHEST PROCEDURES':14,
        'OTHER RESPIRATORY AND CHEST PROCEDURES':15,
        'OTHER RESPIRATORY DIAGNOSES EXCEPT SIGNS, SYMPTOMS & MINOR DIAGNOSES':16,
        'OTHER RESPIRATORY DIAGNOSES EXCEPT SIGNS, SYMPTOMS AND MISCELLANEOUS DIAGNOSES':17,
        'RESPIRATORY MALIGNANCY':18,
        'RESPIRATORY SYSTEM DIAGNOSIS W VENTILATOR SUPPORT 96+ HOURS':19,
        'RESPIRATORY SYSTEM DIAGNOSIS WITH VENTILATOR SUPPORT > 96 HOURS':20,
        'TRACHEOSTOMY W MV 96+ HOURS W EXTENSIVE PROCEDURE':21,
        'TRACHEOSTOMY W MV 96+ HOURS W/O EXTENSIVE PROCEDURE':22,
        'TRACHEOSTOMY WITH MV >96 HOURS WITH EXTENSIVE PROCEDURE':23,
        'TRACHEOSTOMY WITH MV >96 HOURS WITHOUT EXTENSIVE PROCEDURE':24,
    }
    df['APR DRG Description'] = df['APR DRG Description'].map(mapping5)
    
    mapping6 = {
        'ABDOMINAL WALL PROCEDURES, NEC':1,
        'ADMINISTRATION AND TRANSFUSION OF BONE MARROW, STEM CELLS, PANCREATIC ISLET CELLS, AND T-CELLS':2,
        'ADMINISTRATION OF ALBUMIN AND GLOBULIN':3,
        'ADMINISTRATION OF ANTI-INFLAMMATORY AGENTS':4,
        'ADMINISTRATION OF ANTIBIOTICS':5,
        'ADMINISTRATION OF DIAGNOSTIC SUBSTANCES, NEC':6,
        'ADMINISTRATION OF NUTRITIONAL AND ELECTROLYTIC SUBSTANCES':7,
        'ADMINISTRATION OF THERAPEUTIC SUBSTANCES, NEC':8,
        'ADMINISTRATION OF THROMBOLYTICS AND PLATELET INHIBITORS':9,
        'ADRENALECTOMY':10,
        'AIRWAY INTUBATION':11,
        'ANEURYSM REPAIR PROCEDURES':12,
        'ANGIOPLASTY AND RELATED VESSEL PROCEDURES (ENDOVASCULAR; EXCLUDING CAROTID)':13,
        'ARTERIAL OXYGEN SATURATION MONITORING':14,
        'ARTERY, VEIN, AND GREAT VESSEL PROCEDURES, NEC':15,
        'ARTHROCENTESIS':16,
        'BEAM RADIATION':17,
        'BILIARY AND PANCREATIC CALCULUS REMOVAL':18,
        'BLADDER CATHETERIZATION AND DRAINAGE':19,
        'BONE AND JOINT BIOPSY':20,
        'BONE EXCISION':21,
        'BONE FIXATION (EXCLUDING EXTREMITIES)':22,
        'BONE MARROW BIOPSY':23,
        'BRACHYTHERAPY':24,
        'BRONCHOSCOPIC EXCISION AND FULGURATION':25,
        'BRONCHOSCOPY (DIAGNOSTIC)':26,
        'BRONCHOSCOPY (THERAPEUTIC)':27,
        'CARDIAC AND CORONARY FLUOROSCOPY':28,
        'CARDIAC CHEST COMPRESSION':29,
        'CARDIAC MONITORING':30,
        'CARDIAC STRESS TESTS':31,
        'CARDIOVASCULAR DEVICE PROCEDURES, NEC':32,
        'CARDIOVERSION':33,
        'CAROTID ENDARTERECTOMY AND STENTING':34,
        'CHEMOTHERAPY':35,
        'CHEST TUBE PLACEMENT AND THERAPEUTIC THORACENTESIS':36,
        'CHEST WALL PROCEDURES, NEC':37,
        'CLOSED REDUCTION OF BONES AND JOINTS':38,
        'CNS EXCISION PROCEDURES':39,
        'COLONOSCOPY AND PROCTOSCOPY WITH BIOPSY':40,
        'COMMON BILE DUCT SPHINCTEROTOMY AND STENTING':41,
        'COMPUTERIZED TOMOGRAPHY (CT) WITH CONTRAST':42,
        'COMPUTERIZED TOMOGRAPHY (CT) WITHOUT CONTRAST':43,
        'CONTROL OF BLEEDING (NON-ENDOSCOPIC)':44,
        'CYSTECTOMY (INCLUDING FULGURATION) AND URETHRECTOMY':45,
        'CYSTOSCOPY AND URETEROSCOPY (INCLUDING BIOPSY)':46,
        'DENTAL PROCEDURES':47,
        'DIAGNOSTIC ERCP WITH OR WITHOUT BIOPSY':48,
        'DIAPHRAGMATIC HERNIA REPAIR':49,
        'ELECTROCARDIOGRAM (ECG)':50,
        'ELECTROENCEPHALOGRAM (EEG)':51,
        'EMBOLECTOMY, ENDARTERECTOMY, AND RELATED VESSEL PROCEDURES (NON-ENDOVASCULAR; EXCLUDING CAROTID)':52,
        'ENDOCRINE SYSTEM BIOPSY':53,
        'ENDOSCOPIC CONTROL OF BLEEDING':54,
        'ENT DIAGNOSTIC ENDOSCOPY (EXCLUDING LARYNGOSCOPY)':55,
        'ENT DIAGNOSTIC PROCEDURES (NON-ENDOSCOPIC)':56,
        'ENT DRAINAGE (EXCLUDING MYRINGOTOMY)':57,
        'ENT EXCISION (EXCLUDING NASAL PASSAGE, SINUSES, TONGUE, SALIVARY GLANDS, LARYNX)':58,
        'ENT PROCEDURES, NEC':59,
        'ENT REPAIR':60,
        'ESOPHAGOGASTRODUODENOSCOPY (EGD) WITH BIOPSY':61,
        'EXPLORATION OF PERITONEAL CAVITY':62,
        'EXTRACORPOREAL MEMBRANE OXYGENATION':63,
        'EYELID PROCEDURES':64,
        'FEMALE REPRODUCTIVE SYSTEM PROCEDURES, NEC':65,
        'FEMUR FIXATION':66,
        'FIXATION OF UPPER EXTREMITY BONES':67,
        'FLUOROSCOPIC ANGIOGRAPHY (EXCLUDING CORONARY)':68,
        'FLUOROSCOPIC GUIDANCE FOR CIRCULATORY SYSTEM PROCEDURES':69,
        'FLUOROSCOPY OF NON-CIRCULATORY ORGANS':70,
        'GASTRECTOMY':71,
        'GASTRO-JEJUNAL BYPASS (INCLUDING BARIATRIC)':72,
        'GASTROSTOMY':73,
        'GI SYSTEM BIOPSY (NON-ENDOSCOPIC)':74,
        'GI SYSTEM DRAINAGE (EXCLUDING PARACENTESIS)':75,
        'GI SYSTEM ENDOSCOPIC THERAPEUTIC PROCEDURES':76,
        'GI SYSTEM ENDOSCOPY WITHOUT BIOPSY (DIAGNOSTIC)':77,
        'GI SYSTEM LYSIS OF ADHESIONS':78,
        'GI SYSTEM REPAIR (EXCLUDING ANORECTAL)':79,
        'HEART BIOPSY':80,
        'HEART CONDUCTION MECHANISM PROCEDURES':81,
        'HEMODIALYSIS':82,
        'HIP ARTHROPLASTY':83,
        'HYSTERECTOMY':84,
        'ILEOSTOMY AND COLOSTOMY':85,
        'IMMOBILIZATION BY SPLINT OR OTHER EXTERNAL DEVICE':86,
        'INCISION AND DRAINAGE OF SKIN':87,
        'INCISION AND DRAINAGE OF SUBCUTANEOUS TISSUE AND FASCIA':88,
        'INFERIOR VENA CAVA (IVC) FILTER PROCEDURES':89,
        'INFUSION OF VASOPRESSOR':90,
        'INTRACRANIAL EPIDURAL AND SUBDURAL SPACE DRAINAGE':91,
        'INTRAVENOUS INDUCTION OF LABOR':92,
        'IRRIGATION (DIAGNOSTIC AND THERAPEUTIC)':93,
        'ISOLATION PROCEDURES':94,
        'JOINT TISSUE EXCISION (EXCLUDING DISCECTOMY)':95,
        'KIDNEY AND OTHER URINARY TRACT BIOPSY (NON-ENDOSCOPIC)':96,
        'LARYNGECTOMY':97,
        'LARYNGOSCOPY (DIAGNOSTIC)':98,
        'LIGATION AND EMBOLIZATION OF VESSELS':99,
        'LIVER BIOPSY':100,
        'LOWER GI THERAPEUTIC PROCEDURES, NEC (EXCLUDING OPEN AND LAPAROSCOPIC)':101,
        'LUMBAR PUNCTURE':102,
        'LUNG, PLEURA, OR DIAPHRAGM BIOPSY (NON-ENDOSCOPIC)':103,
        'LUNG, PLEURA, OR DIAPHRAGM RESECTION (OPEN AND THORACOSCOPIC)':104,
        'LYMPH NODE BIOPSY':105,
        'LYMPH NODE DISSECTION':106,
        'LYMPH NODE EXCISION (THERAPEUTIC)':107,
        'MAGNETIC RESONANCE IMAGING (MRI)':108,
        'MASTECTOMY AND LUMPECTOMY':109,
        'MEASUREMENT AND MONITORING, NEC':110,
        'MEASUREMENT DURING CARDIAC CATHETERIZATION':111,
        'MECHANICAL VENTILATION':112,
        'MEDIASTINAL PROCEDURES, NEC':113,
        'MENTAL HEALTH PROCEDURES, NEC':114,
        'MINIMALLY INVASIVE CNS BIOPSY':115,
        'MUSCLE, TENDON, BURSA, AND LIGAMENT EXCISION':116,
        'MUSCULOSKELETAL DEVICE PROCEDURES, NEC':117,
        'NAIL PROCEDURES':118,
        'NASAL AND SINUS EXCISION':119,
        'NON-INVASIVE VENTILATION':120,
        'OPEN AND THORACOSCOPIC PLEURAL DRAINAGE':121,
        'OTHER CARDIOVASCULAR SYSTEM MEASUREMENT AND MONITORING':122,
        'OTHER GI SYSTEM DEVICE PROCEDURES':123,
        'PACEMAKER AND DEFIBRILLATOR INTERROGATION':124,
        'PACEMAKER AND DEFIBRILLATOR PROCEDURES':125,
        'PACKING AND DRESSING PROCEDURES':126,
        'PANCREATIC AND PROXIMAL BILIARY DILATION AND STENTING':127,
        'PANCREATICOBILIARY BIOPSY':128,
        'PARACENTESIS':129,
        'PERCUTANEOUS CORONARY INTERVENTIONS (PCI)':130,
        'PERICARDIAL PROCEDURES':131,
        'PERITONEAL DIALYSIS':132,
        'PHARMACOTHERAPY FOR MENTAL HEALTH (EXCLUDING SUBSTANCE USE)':133,
        'PHARMACOTHERAPY FOR SUBSTANCE USE':134,
        'PHERESIS THERAPY':135,
        'PHYSICAL, OCCUPATIONAL, AND RESPIRATORY THERAPY TREATMENT':136,
        'PLACEMENT OF TUNNELED OR IMPLANTABLE PORTION OF A VASCULAR ACCESS DEVICE':137,
        'PLAIN RADIOGRAPHY':138,
        'PLANAR NUCLEAR MEDICINE IMAGING':139,
        'POSITRON EMISSION TOMOGRAPHIC (PET) IMAGING':140,
        'POTENTIAL COVID-19 THERAPIES':141,
        'PULMONARY ARTERIAL PRESSURE MONITORING':142,
        'PULMONARY FUNCTION TESTS':143,
        'RADIATION THERAPY, NEC':144,
        'REGIONAL ANESTHESIA':145,
        'RELEASE OF LUNG AND PLEURA':146,
        'RESPIRATORY SYSTEM PROCEDURES, NEC':147,
        'RETROPERITONEAL PROCEDURES, NEC':148,
        'ROBOTIC-ASSISTED PROCEDURES':149,
        'SKIN BIOPSY AND DIAGNOSTIC DRAINAGE':150,
        'SKIN LACERATION REPAIR (EXCLUDING PERINEUM)':151,
        'SMALL BOWEL RESECTION':152,
        'SPINAL CORD DECOMPRESSION':153,
        'SPINAL EPIDURAL CATHETER PLACEMENT':154,
        'SPINE FUSION':155,
        'SUBCUTANEOUS TISSUE AND FASCIA EXCISION':156,
        'SUBCUTANEOUS TISSUE AND FASCIA PROCEDURES, NEC':157,
        'SUBCUTANEOUS TISSUE, FASCIA, AND MUSCLE BIOPSY':158,
        'SUBSTANCE USE DETOXIFICATION':159,
        'TENDON, MUSCLE, BURSA, AND LIGAMENT REPAIR (EXCLUDING PERINEAL)':160,
        'THORACENTESIS (DIAGNOSTIC)':161,
        'THYMECTOMY':162,
        'THYROIDECTOMY':163,
        'TOMOGRAPHIC NUCLEAR MEDICINE IMAGING':164,
        'TRACHEOSTOMY':165,
        'TRANSFUSION OF BLOOD AND BLOOD PRODUCTS':166,
        'TRANSFUSION OF CLOTTING FACTORS':167,
        'TRANSFUSION OF PLASMA':168,
        'ULTRASONOGRAPHY':169,
        'UPPER GI THERAPEUTIC PROCEDURES, NEC (ENDOSCOPIC)':170,
        'UPPER GI THERAPEUTIC PROCEDURES, NEC (OPEN AND LAPAROSCOPIC)':171,
        'URETER AND OTHER URINARY TRACT DILATION':172,
        'VACCINATIONS':173,
        'VENOUS AND ARTERIAL CATHETER PLACEMENT':174,
        'VESSEL REPAIR AND REPLACEMENT':175,
    }
    df['CCSR Procedure Description'] = df['CCSR Procedure Description'].map(mapping6)
    
    mapping7 = {
        'Medicare':1, 'Self-Pay':2, 'Private Health Insurance':3,
        'Blue Cross/Blue Shield':4, 'Medicaid':5, 'Federal/State/Local/VA':6,
        'Department of Corrections':7, 'Miscellaneous/Other':8,
        'Managed Care, Unspecified':9, 'Unknown':10
    }
    df['Payment Typology 1'] = df['Payment Typology 1'].map(mapping7)

    return df

train_df = mapping(df)

# CCSR Procedure Description
ct = ColumnTransformer(transformers = [('CCSR', SimpleImputer(strategy='most_frequent'), 
                                        ['CCSR Procedure Description'])], remainder='passthrough')
ct_CCSR = ct.fit_transform(train_df)
train_df_ccsr = pd.DataFrame(ct_CCSR, columns = train_df.columns)

# Permanent Facility Id
def facilityId_imputation(df):
    
    train_0 = df[df['Permanent Facility Id'].notnull()] # df == train_df
    y_0 = train_0.iloc[:,10:11].values.ravel()
    test_0 = df[df['Permanent Facility Id'].isnull()]   # df == train_df

    scaler = StandardScaler()
    train_0_norm = scaler.fit_transform(train_0.iloc[:,:10])
    train_0_norm = pd.DataFrame(train_0_norm, columns = train_0.iloc[:,:10].columns)
    test_0_norm = scaler.fit_transform(test_0.iloc[:,:10])
    test_0_norm = pd.DataFrame(test_0_norm, columns = test_0.iloc[:,:10].columns)
    
    #clssfr = XGBClassifier() # use_label_encoder=False --> deprecated. ValueError: Invalid classes inferred from unique values of `y`
    clssfr = LogisticRegression(max_iter=500)
    clssfr.fit(train_0_norm, y_0)

    for i, j in enumerate(test_0.index[:len(clssfr.predict(test_0_norm))]):
        df['Permanent Facility Id'].loc[j] = clssfr.predict(test_0_norm)[i]
    
    return df

train_df_toOHE = facilityId_imputation(train_df_ccsr)

ct_oh = ColumnTransformer(transformers = [('OHE', OneHotEncoder(sparse=False), ['Age Group', 'Type of Admission', 
                                                                                'Patient Disposition', 
                                                                                'APR Medical Surgical Description',
                                                                                'Payment Typology 1'])],
                       remainder='passthrough')

ct_oh_categorical = ct_oh.fit_transform(train_df_toOHE)
oh_columns = ct_oh.get_feature_names_out()
train_df = pd.DataFrame(ct_oh_categorical, columns = oh_columns)

train_df = train_df.reindex(columns=['remainder__Discharge Year', 'OHE__Age Group_1.0', 'OHE__Age Group_2.0', 'OHE__Age Group_3.0',
                                        'OHE__Age Group_4.0', 'OHE__Age Group_5.0',
                                        'OHE__Type of Admission_1.0', 'OHE__Type of Admission_2.0',
                                        'OHE__Type of Admission_4.0', 'OHE__Type of Admission_5.0',
                                        'OHE__Type of Admission_6.0', 'OHE__Patient Disposition_1.0',
                                        'OHE__Patient Disposition_2.0', 'OHE__Patient Disposition_3.0',
                                        'OHE__Patient Disposition_4.0', 'OHE__Patient Disposition_5.0',
                                        'OHE__Patient Disposition_6.0', 'OHE__Patient Disposition_7.0',
                                        'OHE__Patient Disposition_8.0', 'OHE__Patient Disposition_9.0',
                                        'OHE__Patient Disposition_10.0', 'OHE__Patient Disposition_11.0',
                                        'OHE__Patient Disposition_12.0', 'OHE__Patient Disposition_13.0',
                                        'OHE__Patient Disposition_14.0', 'OHE__Patient Disposition_15.0',
                                        'OHE__Patient Disposition_16.0', 'OHE__Patient Disposition_17.0',
                                        'OHE__Patient Disposition_18.0', 'OHE__Patient Disposition_19.0',
                                        'OHE__APR Medical Surgical Description_1.0',
                                        'OHE__APR Medical Surgical Description_2.0',
                                        'OHE__Payment Typology 1_1.0', 'OHE__Payment Typology 1_2.0',
                                        'OHE__Payment Typology 1_3.0', 'OHE__Payment Typology 1_4.0',
                                        'OHE__Payment Typology 1_5.0', 'OHE__Payment Typology 1_6.0',
                                        'OHE__Payment Typology 1_7.0', 'OHE__Payment Typology 1_8.0',
                                        'OHE__Payment Typology 1_9.0',
                                        'remainder__CCSR Procedure Description', 'remainder__APR DRG Description', 
                                        'remainder__APR Severity of Illness Description', 
                                        'remainder__Permanent Facility Id',
                                        'remainder__Total Charges'])

train_df.drop_duplicates(keep='first', inplace=True)

X_train = train_df.drop(train_df[train_df['remainder__Discharge Year'] > 2020].index)
X_train = X_train.iloc[:,1:-1]
X_valid = train_df.drop(train_df[train_df['remainder__Discharge Year'] < 2021].index)
X_valid = X_valid.iloc[:,1:-1]
y_train = train_df.drop(train_df[train_df['remainder__Discharge Year'] > 2020].index)
y_train = y_train.iloc[:,-1:]
y_valid = train_df.drop(train_df[train_df['remainder__Discharge Year'] < 2021].index)
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