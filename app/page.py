from preds import *

import streamlit as st
#import pickle


def show_predict_page():
    st.set_page_config(
        page_title="Financial Planning for Lung Cancer",
        page_icon="✅",
    )

    st.title("Proactive Financial Planning for Lung Cancer Patients through Machine Learning-based Cost Prediction")
    st.image("images/home_img.jpeg", caption='', use_column_width=True)
    st.header("The Problem")
    st.write(
        """Cancer treatment can be very expensive, and even with insurance coverage, out-of-pocket costs can quickly accumulate and become 
        unmanageable. This can lead to financial strain, bankruptcy, and other negative outcomes for patients and their families. Research 
        has shown that over 40% of patients diagnosed with cancer will declare personal bankruptcy within four years of their diagnosis. 
        This is a significant problem that has a major impact on the lives of cancer patients and their families. There are many factors that 
        contribute to the high costs of cancer treatment. These include the cost of drugs and other treatments, the cost of hospital stays 
        and other medical services, and the cost of travel and other related expenses. Additionally, patients may need to take time off work 
        or reduce their work hours to undergo treatment, which can further impact their financial situation. The financial burden of cancer 
        treatment can also have negative impacts on patient outcomes. Patients who experience financial stress may be more likely to delay 
        or forego treatment, which can lead to poorer outcomes and higher healthcare costs in the long run. Patients who declare bankruptcy 
        may also experience psychological distress, which can further impact their overall health and well-being. Overall, the financial 
        burden of cancer treatment is a significant problem that affects a large number of patients and their families. Finding ways to 
        reduce this burden and help patients proactively plan for out-of-pocket costs is an important area of research that has the potential 
        to improve patient outcomes and reduce the risk of negative financial outcomes for those undergoing cancer treatment. The development 
        of a predictive model for cancer treatment costs has the potential to significantly reduce the financial burden of a cancer diagnosis 
        on patients."""
    )

    st.header("Want to know more?")
    st.markdown("* [Omdena Page](https://omdena.com/projects/proactive-financial-planning-for-lung-cancer-patients-through-machine-learning/)")
    st.image("images/omdena_website.jpeg", caption='', use_column_width=False)

    st.header("Estimation")

    facility_id = (
        '1453',
        '1175',
        '1176',
        '216',
        '1458',
        '1463',
        '1630',
        '1456',
        '413',
        '1464',
        '541',
    )

    age = (
        "0 to 17", "18 to 29", "30 to 49", "50 to 69", "70 or Older",
    )
    
    type_admission = (
        "Elective", "Emergency", "Newborn", "Not Available", "Trauma", "Urgent",
    )
    
    apr_medical_surgical = ("Surgical", "Medical")
    
    patient_disposition = (
        'Hospice - Home', 'Expired', 'Home w/ Home Health Services',
        'Home or Self Care', 'Skilled Nursing Home',
        'Left Against Medical Advice', 'Short-term Hospital',
        'Hospice - Medical Facility', 'Inpatient Rehabilitation Facility',
        "Cancer Center or Children's Hospital", 'Court/Law Enforcement',
        'Psychiatric Hospital or Unit of Hosp',
        'Medicare Cert Long Term Care Hospital', 'Another Type Not Listed',
        'Facility w/ Custodial/Supportive Care',
        'Federal Health Care Facility',
        'Hosp Basd Medicare Approved Swing Bed',
        'Critical Access Hospital', 'Medicaid Cert Nursing Facility',
    )

    apr_drg = (
        'ALLOGENEIC BONE MARROW TRANSPLANT',
        'AUTOLOGOUS BONE MARROW TRANSPLANT OR T-CELL IMMUNOTHERAPY',
        'CYSTIC FIBROSIS - PULMONARY DISEASE',
        'EAR, NOSE, MOUTH, THROAT, CRANIAL/FACIAL MALIGNANCIES',
        'EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS',
        'EXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS',
        'EXTRACORPOREAL MEMBRANE OXYGENATION (ECMO)',
        'MAJOR RESPIRATORY & CHEST PROCEDURES',
        'MAJOR RESPIRATORY AND CHEST PROCEDURES',
        'MODERATELY EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS',
        'MODERATELY EXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS',
        'NON-EXTENSIVE O.R. PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS',
        'NONEXTENSIVE PROCEDURE UNRELATED TO PRINCIPAL DIAGNOSIS',
        'OTHER RESPIRATORY & CHEST PROCEDURES',
        'OTHER RESPIRATORY AND CHEST PROCEDURES',
        'OTHER RESPIRATORY DIAGNOSES EXCEPT SIGNS, SYMPTOMS & MINOR DIAGNOSES',
        'OTHER RESPIRATORY DIAGNOSES EXCEPT SIGNS, SYMPTOMS AND MISCELLANEOUS DIAGNOSES',
        'RESPIRATORY MALIGNANCY',
        'RESPIRATORY SYSTEM DIAGNOSIS W VENTILATOR SUPPORT 96+ HOURS',
        'RESPIRATORY SYSTEM DIAGNOSIS WITH VENTILATOR SUPPORT > 96 HOURS',
        'TRACHEOSTOMY W MV 96+ HOURS W EXTENSIVE PROCEDURE',
        'TRACHEOSTOMY W MV 96+ HOURS W/O EXTENSIVE PROCEDURE',
        'TRACHEOSTOMY WITH MV >96 HOURS WITH EXTENSIVE PROCEDURE',
        'TRACHEOSTOMY WITH MV >96 HOURS WITHOUT EXTENSIVE PROCEDURE',
    )
    
    discharge_year = ('2021',)
    apr_severity = ("Minor", "Moderate", "Major", "Extreme")
    total_charges = ('0')

    ccsr_procedure = (
        'ABDOMINAL WALL PROCEDURES, NEC',
        'ADMINISTRATION AND TRANSFUSION OF BONE MARROW, STEM CELLS, PANCREATIC ISLET CELLS, AND T-CELLS',
        'ADMINISTRATION OF ALBUMIN AND GLOBULIN',
        'ADMINISTRATION OF ANTI-INFLAMMATORY AGENTS',
        'ADMINISTRATION OF ANTIBIOTICS',
        'ADMINISTRATION OF DIAGNOSTIC SUBSTANCES, NEC',
        'ADMINISTRATION OF NUTRITIONAL AND ELECTROLYTIC SUBSTANCES',
        'ADMINISTRATION OF THERAPEUTIC SUBSTANCES, NEC',
        'ADMINISTRATION OF THROMBOLYTICS AND PLATELET INHIBITORS',
        'ADRENALECTOMY',
        'AIRWAY INTUBATION',
        'ANEURYSM REPAIR PROCEDURES',
        'ANGIOPLASTY AND RELATED VESSEL PROCEDURES (ENDOVASCULAR; EXCLUDING CAROTID)',
        'ARTERIAL OXYGEN SATURATION MONITORING',
        'ARTERY, VEIN, AND GREAT VESSEL PROCEDURES, NEC',
        'ARTHROCENTESIS',
        'BEAM RADIATION',
        'BILIARY AND PANCREATIC CALCULUS REMOVAL',
        'BLADDER CATHETERIZATION AND DRAINAGE',
        'BONE AND JOINT BIOPSY',
        'BONE EXCISION',
        'BONE FIXATION (EXCLUDING EXTREMITIES)',
        'BONE MARROW BIOPSY',
        'BRACHYTHERAPY',
        'BRONCHOSCOPIC EXCISION AND FULGURATION',
        'BRONCHOSCOPY (DIAGNOSTIC)',
        'BRONCHOSCOPY (THERAPEUTIC)',
        'CARDIAC AND CORONARY FLUOROSCOPY',
        'CARDIAC CHEST COMPRESSION',
        'CARDIAC MONITORING',
        'CARDIAC STRESS TESTS',
        'CARDIOVASCULAR DEVICE PROCEDURES, NEC',
        'CARDIOVERSION',
        'CAROTID ENDARTERECTOMY AND STENTING',
        'CHEMOTHERAPY',
        'CHEST TUBE PLACEMENT AND THERAPEUTIC THORACENTESIS',
        'CHEST WALL PROCEDURES, NEC',
        'CLOSED REDUCTION OF BONES AND JOINTS',
        'CNS EXCISION PROCEDURES',
        'COLONOSCOPY AND PROCTOSCOPY WITH BIOPSY',
        'COMMON BILE DUCT SPHINCTEROTOMY AND STENTING',
        'COMPUTERIZED TOMOGRAPHY (CT) WITH CONTRAST',
        'COMPUTERIZED TOMOGRAPHY (CT) WITHOUT CONTRAST',
        'CONTROL OF BLEEDING (NON-ENDOSCOPIC)',
        'CYSTECTOMY (INCLUDING FULGURATION) AND URETHRECTOMY',
        'CYSTOSCOPY AND URETEROSCOPY (INCLUDING BIOPSY)',
        'DENTAL PROCEDURES',
        'DIAGNOSTIC ERCP WITH OR WITHOUT BIOPSY',
        'DIAPHRAGMATIC HERNIA REPAIR',
        'ELECTROCARDIOGRAM (ECG)',
        'ELECTROENCEPHALOGRAM (EEG)',
        'EMBOLECTOMY, ENDARTERECTOMY, AND RELATED VESSEL PROCEDURES (NON-ENDOVASCULAR; EXCLUDING CAROTID)',
        'ENDOCRINE SYSTEM BIOPSY',
        'ENDOSCOPIC CONTROL OF BLEEDING',
        'ENT DIAGNOSTIC ENDOSCOPY (EXCLUDING LARYNGOSCOPY)',
        'ENT DIAGNOSTIC PROCEDURES (NON-ENDOSCOPIC)',
        'ENT DRAINAGE (EXCLUDING MYRINGOTOMY)',
        'ENT EXCISION (EXCLUDING NASAL PASSAGE, SINUSES, TONGUE, SALIVARY GLANDS, LARYNX)',
        'ENT PROCEDURES, NEC',
        'ENT REPAIR',
        'ESOPHAGOGASTRODUODENOSCOPY (EGD) WITH BIOPSY',
        'EXPLORATION OF PERITONEAL CAVITY',
        'EXTRACORPOREAL MEMBRANE OXYGENATION',
        'EYELID PROCEDURES',
        'FEMALE REPRODUCTIVE SYSTEM PROCEDURES, NEC',
        'FEMUR FIXATION',
        'FIXATION OF UPPER EXTREMITY BONES',
        'FLUOROSCOPIC ANGIOGRAPHY (EXCLUDING CORONARY)',
        'FLUOROSCOPIC GUIDANCE FOR CIRCULATORY SYSTEM PROCEDURES',
        'FLUOROSCOPY OF NON-CIRCULATORY ORGANS',
        'GASTRECTOMY',
        'GASTRO-JEJUNAL BYPASS (INCLUDING BARIATRIC)',
        'GASTROSTOMY',
        'GI SYSTEM BIOPSY (NON-ENDOSCOPIC)',
        'GI SYSTEM DRAINAGE (EXCLUDING PARACENTESIS)',
        'GI SYSTEM ENDOSCOPIC THERAPEUTIC PROCEDURES',
        'GI SYSTEM ENDOSCOPY WITHOUT BIOPSY (DIAGNOSTIC)',
        'GI SYSTEM LYSIS OF ADHESIONS',
        'GI SYSTEM REPAIR (EXCLUDING ANORECTAL)',
        'HEART BIOPSY',
        'HEART CONDUCTION MECHANISM PROCEDURES',
        'HEMODIALYSIS',
        'HIP ARTHROPLASTY',
        'HYSTERECTOMY',
        'ILEOSTOMY AND COLOSTOMY',
        'IMMOBILIZATION BY SPLINT OR OTHER EXTERNAL DEVICE',
        'INCISION AND DRAINAGE OF SKIN',
        'INCISION AND DRAINAGE OF SUBCUTANEOUS TISSUE AND FASCIA',
        'INFERIOR VENA CAVA (IVC) FILTER PROCEDURES',
        'INFUSION OF VASOPRESSOR',
        'INTRACRANIAL EPIDURAL AND SUBDURAL SPACE DRAINAGE',
        'INTRAVENOUS INDUCTION OF LABOR',
        'IRRIGATION (DIAGNOSTIC AND THERAPEUTIC)',
        'ISOLATION PROCEDURES',
        'JOINT TISSUE EXCISION (EXCLUDING DISCECTOMY)',
        'KIDNEY AND OTHER URINARY TRACT BIOPSY (NON-ENDOSCOPIC)',
        'LARYNGECTOMY',
        'LARYNGOSCOPY (DIAGNOSTIC)',
        'LIGATION AND EMBOLIZATION OF VESSELS',
        'LIVER BIOPSY',
        'LOWER GI THERAPEUTIC PROCEDURES, NEC (EXCLUDING OPEN AND LAPAROSCOPIC)',
        'LUMBAR PUNCTURE',
        'LUNG, PLEURA, OR DIAPHRAGM BIOPSY (NON-ENDOSCOPIC)',
        'LUNG, PLEURA, OR DIAPHRAGM RESECTION (OPEN AND THORACOSCOPIC)',
        'LYMPH NODE BIOPSY',
        'LYMPH NODE DISSECTION',
        'LYMPH NODE EXCISION (THERAPEUTIC)',
        'MAGNETIC RESONANCE IMAGING (MRI)',
        'MASTECTOMY AND LUMPECTOMY',
        'MEASUREMENT AND MONITORING, NEC',
        'MEASUREMENT DURING CARDIAC CATHETERIZATION',
        'MECHANICAL VENTILATION',
        'MEDIASTINAL PROCEDURES, NEC',
        'MENTAL HEALTH PROCEDURES, NEC',
        'MINIMALLY INVASIVE CNS BIOPSY',
        'MUSCLE, TENDON, BURSA, AND LIGAMENT EXCISION',
        'MUSCULOSKELETAL DEVICE PROCEDURES, NEC',
        'NAIL PROCEDURES',
        'NASAL AND SINUS EXCISION',
        'NON-INVASIVE VENTILATION',
        'OPEN AND THORACOSCOPIC PLEURAL DRAINAGE',
        'OTHER CARDIOVASCULAR SYSTEM MEASUREMENT AND MONITORING',
        'OTHER GI SYSTEM DEVICE PROCEDURES',
        'PACEMAKER AND DEFIBRILLATOR INTERROGATION',
        'PACEMAKER AND DEFIBRILLATOR PROCEDURES',
        'PACKING AND DRESSING PROCEDURES',
        'PANCREATIC AND PROXIMAL BILIARY DILATION AND STENTING',
        'PANCREATICOBILIARY BIOPSY',
        'PARACENTESIS',
        'PERCUTANEOUS CORONARY INTERVENTIONS (PCI)',
        'PERICARDIAL PROCEDURES',
        'PERITONEAL DIALYSIS',
        'PHARMACOTHERAPY FOR MENTAL HEALTH (EXCLUDING SUBSTANCE USE)',
        'PHARMACOTHERAPY FOR SUBSTANCE USE',
        'PHERESIS THERAPY',
        'PHYSICAL, OCCUPATIONAL, AND RESPIRATORY THERAPY TREATMENT',
        'PLACEMENT OF TUNNELED OR IMPLANTABLE PORTION OF A VASCULAR ACCESS DEVICE',
        'PLAIN RADIOGRAPHY',
        'PLANAR NUCLEAR MEDICINE IMAGING',
        'POSITRON EMISSION TOMOGRAPHIC (PET) IMAGING',
        'POTENTIAL COVID-19 THERAPIES',
        'PULMONARY ARTERIAL PRESSURE MONITORING',
        'PULMONARY FUNCTION TESTS',
        'RADIATION THERAPY, NEC',
        'REGIONAL ANESTHESIA',
        'RELEASE OF LUNG AND PLEURA',
        'RESPIRATORY SYSTEM PROCEDURES, NEC',
        'RETROPERITONEAL PROCEDURES, NEC',
        'ROBOTIC-ASSISTED PROCEDURES',
        'SKIN BIOPSY AND DIAGNOSTIC DRAINAGE',
        'SKIN LACERATION REPAIR (EXCLUDING PERINEUM)',
        'SMALL BOWEL RESECTION',
        'SPINAL CORD DECOMPRESSION',
        'SPINAL EPIDURAL CATHETER PLACEMENT',
        'SPINE FUSION',
        'SUBCUTANEOUS TISSUE AND FASCIA EXCISION',
        'SUBCUTANEOUS TISSUE AND FASCIA PROCEDURES, NEC',
        'SUBCUTANEOUS TISSUE, FASCIA, AND MUSCLE BIOPSY',
        'SUBSTANCE USE DETOXIFICATION',
        'TENDON, MUSCLE, BURSA, AND LIGAMENT REPAIR (EXCLUDING PERINEAL)',
        'THORACENTESIS (DIAGNOSTIC)',
        'THYMECTOMY',
        'THYROIDECTOMY',
        'TOMOGRAPHIC NUCLEAR MEDICINE IMAGING',
        'TRACHEOSTOMY',
        'TRANSFUSION OF BLOOD AND BLOOD PRODUCTS',
        'TRANSFUSION OF CLOTTING FACTORS',
        'TRANSFUSION OF PLASMA',
        'ULTRASONOGRAPHY',
        'UPPER GI THERAPEUTIC PROCEDURES, NEC (ENDOSCOPIC)',
        'UPPER GI THERAPEUTIC PROCEDURES, NEC (OPEN AND LAPAROSCOPIC)',
        'URETER AND OTHER URINARY TRACT DILATION',
        'VACCINATIONS',
        'VENOUS AND ARTERIAL CATHETER PLACEMENT',
        'VESSEL REPAIR AND REPLACEMENT',
    )
    
    pay_typology = (
        'Medicare', 'Self-Pay', 'Private Health Insurance',
        'Blue Cross/Blue Shield', 'Medicaid', 'Federal/State/Local/VA',
        'Department of Corrections', 'Miscellaneous/Other',
        'Managed Care, Unspecified', 'Unknown',
    )

    discharge_year = st.selectbox("Current Year", discharge_year)
    facility_id = st.selectbox("Facility Id", facility_id)
    #total_charges = st.selectbox("Total Bill", total_charges)
    age = st.selectbox("Age Group", age)
    type_admission = st.selectbox("Type of Admission", type_admission)
    apr_medical_surgical = st.selectbox("APR Medical Surgical Description", apr_medical_surgical)
    ccsr_procedure = st.selectbox("CCSR Procedure Description", ccsr_procedure)
    patient_disposition = st.selectbox("Patient Disposition", patient_disposition)
    apr_drg = st.selectbox("APR DRG Description", apr_drg)
    apr_severity = st.selectbox("APR Severity of Illness Description", apr_severity)
    pay_typology = st.selectbox("Payment Typology", pay_typology)

    hit = st.button("Total Charge Estimation")
    if hit:
        X = pd.DataFrame([[ccsr_procedure, discharge_year, age, type_admission, 
                   patient_disposition, apr_drg, apr_severity, apr_medical_surgical, pay_typology, total_charges, facility_id]], 
                 columns=['CCSR Procedure Description', 'Discharge Year', 'Age Group', 'Type of Admission', 
                          'Patient Disposition', 'APR DRG Description', 'APR Severity of Illness Description', 
                          'APR Medical Surgical Description', 'Payment Typology 1', 'Total Charges', 
                          'Permanent Facility Id'])
        
        X = mapping(X)
        X = ct_oh.transform(X)
        X = pd.DataFrame(X, columns = oh_columns)
        X = X.drop(['remainder__Discharge Year', 'remainder__Total Charges'], axis=1)
        pred_X = model.predict(X)
        # Let's increase our second predictions by 7.167231% according to the Inflation Annual Change - IP (HCC 2021)
        pred_X = (np.exp(pred_X) - 3000) * 1.0717
        pred_X = pred_X.item()
        st.subheader(f"The Estimated Total Charge is ${pred_X:.2f}")