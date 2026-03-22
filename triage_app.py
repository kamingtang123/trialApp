import streamlit as st
import pandas as pd
import datetime
import pickle
from pathlib import Path
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="KTAS Triage Assistant", layout="centered")

# App title and disclaimer
st.title("🩺 KTAS Triage Assistant")
st.markdown("**AI Suggestion Tool – For Nurse Reference Only**")
st.markdown("---")

# Current time
st.sidebar.write(f"**Date & Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Model loader
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parents[0] / "ktas_model.pkl"
    with model_path.open("rb") as f:
        return pickle.load(f)


# Input form
with st.form("patient_form"):
    st.subheader("Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sex", ["Male", "Female"])
        arrival_mode = st.selectbox("Arrival Mode", 
                                   ["Walking", "Public Ambulance", "Private Ambulance", 
                                    "Private Vehicle", "Other"])
        injury = st.selectbox("Injury?", ["No", "Yes"])
    
    with col2:
        pain = st.selectbox("Pain?", ["No", "Yes"])
        nrs_pain = st.slider("NRS Pain Score (1 = Little pain, 10 = Worst)", 1, 10, 5)
        if pain == "No":
            nrs_pain = 0
        mental = st.selectbox("Mental Status", 
                            ["Alert", "Responds to Verbal", "Responds to Pain", "Unresponsive"])

    st.subheader("Vital Signs")
    
    col3, col4 = st.columns(2)
    
    with col3:
        sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=120)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=80)
        hr = st.number_input("Heart Rate (bpm)", min_value=0, max_value=250, value=80)
    
    with col4:
        rr = st.number_input("Respiratory Rate (/min)", min_value=0, max_value=60, value=16)
        temp = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.6, step=0.1)
    #    spo2 = st.number_input("SpO₂ (%)", min_value=0, max_value=100, value=98)

    # Submit button
    submitted = st.form_submit_button("🔍 Generate Triage Suggestion")

if submitted:
    st.markdown("---")
    st.subheader("📋 Triage Suggestion Report")

    injury_map = {"No": 1, "Yes": 2}
    pain_map = {"No": 0, "Yes": 1}
    mental_map = {
        "Alert": 1,
        "Responds to Verbal": 2,
        "Responds to Pain": 3,
        "Unresponsive": 4,
    }

    input_row = {
        "Age": float(age),
        "Injury": float(injury_map[injury]),
        "Mental": float(mental_map[mental]),
        "Pain": float(pain_map[pain]),
        "NRS_pain": float(nrs_pain),
        "SBP": float(sbp),
        "DBP": float(dbp),
        "HR": float(hr),
        "RR": float(rr),
        "BT": float(temp),
     #   "Saturation": float(spo2),
    }

    model = load_model()
    feature_names = [str(name) for name in getattr(model, "feature_names_in_", list(input_row.keys()))]
    input_df = pd.DataFrame(
        [[float(input_row.get(name, 0.0)) for name in feature_names]],
        columns=feature_names,
    )

    # Proper mapping from maptable
    arrival_map = {'Walking':1, 'Public Ambulance':2, 'Private Vehicle':3, 
               'Private Ambulance':4, 'Other':5}
    #disposition_map = {1: 'Discharge', 2: 'Ward', 3: 'ICU', 4: 'Discharge', 
               #    5: 'Transfer', 6: 'Death', 7: 'Surgery'}
    #mental_map = {1: 'Alert', 2: 'Verbal', 3: 'Pain', 4: 'Unresponsive'}
    ktas_map = { 'Emergency':1, 'Emergency':2, 'Emergency':3, 'Non-Emergency':4, 'Non-Emergency':5}

    if 'Arrival mode' in input_df.columns:
        input_df['Arrival_mode'] = input_df['Arrival mode'].map(arrival_map)
    else:
        input_df['Arrival_mode'] = np.nan

   # if 'Disposition' in input_df.columns:
    #    input_df['Disposition'] = input_df['Disposition'].map(disposition_map)

    #if 'Mental' in input_df.columns:
    #    input_df['Mental'] = input_df['Mental'].map(mental_map)

# === HIGH-IMPACT FEATURES ===
# Use mapped Arrival_mode for string checks (safest)
    if 'Arrival_mode' in input_df.columns:
        input_df['is_ambulance'] = input_df['Arrival_mode'].isin(['Public Ambulance', 'Private Ambulance']).astype(int)
    else:
        input_df['is_ambulance'] = 0

    if 'Injury' in input_df.columns:
        input_df['is_injury'] = (input_df['Injury'] == 'Yes').astype(int)
    else:
        input_df['is_injury'] = 0

    if 'Pain' in input_df.columns:
        input_df['has_pain'] = (input_df['Pain'] == 'Yes').astype(int)
    else:
        input_df['has_pain'] = 0

# high_pain uses numeric NRS_pain
    if 'NRS_pain' in input_df.columns:
        input_df['high_pain'] = (input_df['NRS_pain'] >= 5).astype(int)
    else:
        input_df['high_pain'] = 0

# Vital signs deviations (coerced earlier)
    if 'SBP' in input_df.columns:
        input_df['hypotension'] = (input_df['SBP'] < 90).astype(int)
        input_df['hypertension'] = (input_df['SBP'] > 180).astype(int)
    else:
        input_df['hypotension'] = 0
        input_df['hypertension'] = 0

    if 'HR' in input_df.columns:
        input_df['tachycardia'] = (input_df['HR'] > 100).astype(int)
        input_df['bradycardia'] = (input_df['HR'] < 60).astype(int)
    else:
        input_df['tachycardia'] = 0
        input_df['bradycardia'] = 0

    if 'RR' in input_df.columns:
        input_df['tachypnea'] = (input_df['RR'] > 20).astype(int)
    else:
        input_df['tachypnea'] = 0

    if 'Saturation' in input_df.columns:
        input_df['hypoxia'] = (input_df['Saturation'] < 94).astype(int)
    else:
        input_df['hypoxia'] = 0

    if 'BT' in input_df.columns:
        input_df['fever'] = (input_df['BT'] > 38.0).astype(int)
    else:
        input_df['fever'] = 0

# Shock index & modified shock index (avoid division by zero)
#if 'HR' in df.columns and 'SBP' in df.columns:
#    df['shock_index'] = df['HR'] / df['SBP'].replace({0: np.nan})
#else:
#    df['shock_index'] = np.nan

#if 'HR' in df.columns and 'DBP' in df.columns:
#    df['modified_shock_index'] = df['HR'] / df['DBP'].replace({0: np.nan})
#else:
#    df['modified_shock_index'] = np.nan

# Age groups
    if 'Age' in input_df.columns:
        input_df['elderly'] = (input_df['Age'] >= 65).astype(int)
        input_df['very_elderly'] = (input_df['Age'] >= 80).astype(int)
    else:
        input_df['elderly'] = 0
        input_df['very_elderly'] = 0

# Mental status severity
    if 'Mental' in input_df.columns:
        input_df['altered_mental'] = (input_df['Mental'] != 'Alert').astype(int)
    else:
        input_df['altered_mental'] = 0
    
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_features]

    pred = model.predict(input_df)[0]
    ktas = int(pred) + 1

    if ktas == 1:
        color = "🔴"
        level = "Resuscitation"
    elif ktas == 2:
        color = "🟠"
        level = "Emergency"
    elif ktas == 3:
        color = "🟡"
        level = "Urgent"
    elif ktas == 4:
        color = "🟢"
        level = "Less Urgent"
    else:
        color = "🔵"
        level = "Non-Urgent"

    # Display result
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; color: #000;">
        <h2 style="color: #000;">{color} Predicted KTAS Level: <strong style="color: #000;">{ktas}</strong> (<span style="color: #000;">{level}</span>)</h2>
    </div>
    """, unsafe_allow_html=True)

    #if hasattr(model, "predict_proba"):
    #    proba = model.predict_proba(input_df)[0]
    #    proba_df = pd.DataFrame({
    #        "KTAS": [int(c) + 1 for c in model.classes_],
    #        "Probability": proba,
    #    }).sort_values("KTAS")
    #    st.markdown("### Model Confidence")
    #    st.dataframe(proba_df, use_container_width=True)

    st.info("⚠️ This is a suggestion only. Final triage decision must be made by clinical judgment.")

    # Optional: Show all entered data
    with st.expander("View All Entered Data"):
        data_summary = {
            "Age": age, "Sex": sex, "Arrival": arrival_mode, "Injury": injury,
            "Pain": pain, "NRS Pain": nrs_pain, "Mental": mental,
            "SBP/DBP": f"{sbp}/{dbp}", "HR": hr, "RR": rr, "Temp": temp, #"SpO₂": #spo2,
        }
        st.json(data_summary)

# Footer
st.markdown("---")
st.caption("KTAS Triage Assistant v1.0 | For clinical support only")