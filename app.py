"""
app.py
Lung Cancer Risk Prediction Web App - Vertical Layout & Uniform Cards
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
from model import LungCancerModel
from generate_data import generate_dataset

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS (MODERN BLUE & WHITE)
# ============================================
st.markdown("""
<style>
    /* ======== GLOBAL ======== */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
        margin: 0 auto;
    }

    /* ======== PREDICTION RESULT CARD ======== */
    .result-card {
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
        background: #ffffff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .result-positive {
        border-color: rgba(239,68,68,0.3);
        background: linear-gradient(135deg, rgba(239,68,68,0.03), #ffffff);
        box-shadow: 0 0 20px rgba(239,68,68,0.08);
    }
    .result-negative {
        border-color: rgba(34,197,94,0.3);
        background: linear-gradient(135deg, rgba(34,197,94,0.03), #ffffff);
        box-shadow: 0 0 20px rgba(34,197,94,0.08);
    }

    /* ======== GAUGE CARD ======== */
    .gauge-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        text-align: center;
    }

    /* ======== FACTOR CARD (USED FOR ALL INFO) ======== */
    .factor-card {
        background: #ffffff;
        border-left: 3px solid #1565c0;
        padding: 15px 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        color: #1e293b;
        font-size: 0.95rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        text-align: left;
    }

    /* ======== RADIO BUTTON STYLE ======== */
    .stRadio > div[role="radiogroup"] > label {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 8px 16px;
        margin: 4px;
        transition: all 0.2s ease;
        color: #1e293b;
    }
    .stRadio > div[role="radiogroup"] > label:hover {
        border-color: #1565c0;
        background: rgba(21, 101, 192, 0.05);
    }

    /* ======== BUTTON STYLE ======== */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 32px !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        width: 100%;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(21, 101, 192, 0.3) !important;
    }

    /* ======== SLIDER STYLE ======== */
    .stSlider > div > div > div > div {
        background-color: #1565c0 !important;
    }

    /* ======== SECTION TITLE ======== */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1565c0;
        margin-bottom: 0.8rem;
        margin-top: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# MODEL INITIALIZATION (CACHED)
# ============================================
@st.cache_resource
def init_model():
    model = LungCancerModel()
    dataset_path = os.path.join('dataset', 'lung_cancer.csv')
    if not os.path.exists(dataset_path):
        generate_dataset()
    model.train(dataset_path)
    return model

model = init_model()


# ============================================
# LAYOUT: SIDEBAR (INPUT FORM)
# ============================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size: 4rem; margin-bottom: 10px;">🫁</div>
        <h2 style="font-size:2.2rem; font-weight:800; color:#1565c0; margin-top:0; margin-bottom:0.5rem;">
        Lung Cancer Check
        </h2>
        <p style="color:#64748b; font-size:1rem; margin:0;">
        Prediction System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown('<p class="section-title">Demographics</p>', unsafe_allow_html=True)
    
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    age = st.slider("Age", 1, 120, 50)

    st.markdown("---")
    st.markdown('<p class="section-title">Lifestyle</p>', unsafe_allow_html=True)
    
    col_side1, col_side2 = st.columns(2)
    with col_side1:
        smoking = st.radio("Smoking", ["No", "Yes"], horizontal=True, index=0)
    with col_side2:
        alcohol = st.radio("Alcohol", ["No", "Yes"], horizontal=True, index=0)

    st.markdown("---")
    st.markdown('<p class="section-title">Clinical Symptoms</p>', unsafe_allow_html=True)
    
    g1, g2 = st.columns(2)
    
    with g1:
        yellow_fingers = st.radio("Yellow Fingers", ["No", "Yes"], horizontal=True, index=0)
        anxiety = st.radio("Anxiety", ["No", "Yes"], horizontal=True, index=0)
        peer_pressure = st.radio("Peer Pressure", ["No", "Yes"], horizontal=True, index=0)
        chronic_disease = st.radio("Chronic Disease", ["No", "Yes"], horizontal=True, index=0)
        fatigue = st.radio("Fatigue", ["No", "Yes"], horizontal=True, index=0)
        allergy = st.radio("Allergy", ["No", "Yes"], horizontal=True, index=0)

    with g2:
        coughing = st.radio("Coughing", ["No", "Yes"], horizontal=True, index=0)
        shortness = st.radio("Shortness of Breath", ["No", "Yes"], horizontal=True, index=0)
        swallowing = st.radio("Swallowing Difficulty", ["No", "Yes"], horizontal=True, index=0)
        chest_pain = st.radio("Chest Pain", ["No", "Yes"], horizontal=True, index=0)
        wheezing = st.radio("Wheezing", ["No", "Yes"], horizontal=True, index=0)

    st.markdown("---")
    
    predict_button = st.button("CHECK RISK NOW", use_container_width=True)


# ============================================
# LAYOUT: MAIN PAGE (PREDICTION RESULTS)
# ============================================

# 1. CENTERED TITLE
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size:2.2rem; font-weight:800; color:#1e293b; margin-bottom:0.5rem;">
        Lung Cancer Risk Prediction
    </h1>
    <p style="color:#64748b; font-size:1rem; margin:0 auto; max-width: 600px;">
        Enter your health conditions in the form on the left sidebar, then click <br>
        <b style="color:#1565c0;">CHECK RISK NOW</b> to see the results.
    </p>
</div>
""", unsafe_allow_html=True)


if predict_button:
    # Mapping English UI inputs to Model Numerical Values (1=No, 2=Yes; Male=1, Female=0)
    input_data = {
        'GENDER': 1 if gender == "Male" else 0,
        'AGE': age,
        'SMOKING': 2 if smoking == "Yes" else 1,
        'YELLOW_FINGERS': 2 if yellow_fingers == "Yes" else 1,
        'ANXIETY': 2 if anxiety == "Yes" else 1,
        'PEER_PRESSURE': 2 if peer_pressure == "Yes" else 1,
        'CHRONIC_DISEASE': 2 if chronic_disease == "Yes" else 1,
        'FATIGUE': 2 if fatigue == "Yes" else 1,
        'ALLERGY': 2 if allergy == "Yes" else 1,
        'WHEEZING': 2 if wheezing == "Yes" else 1,
        'ALCOHOL_CONSUMING': 2 if alcohol == "Yes" else 1,
        'COUGHING': 2 if coughing == "Yes" else 1,
        'SHORTNESS_OF_BREATH': 2 if shortness == "Yes" else 1,
        'SWALLOWING_DIFFICULTY': 2 if swallowing == "Yes" else 1,
        'CHEST_PAIN': 2 if chest_pain == "Yes" else 1,
    }

    prediction, probability = model.predict(input_data)
    
    prob_yes = probability[1]
    risk_pct = prob_yes * 100

    if prediction == 1:
        risk_status = "POSITIVE LUNG CANCER RISK"
        risk_emoji = "⚠️"
        risk_class = "result-positive"
        risk_color = "#d32f2f"
        advice = "Please consult a pulmonologist immediately for further examination such as an X-ray or CT Scan, and improve your healthy lifestyle."
    else:
        risk_status = "NEGATIVE / LOW RISK"
        risk_emoji = "✅"
        risk_class = "result-negative"
        risk_color = "#2e7d32"
        advice = "Your condition is detected as low risk. Keep maintaining a healthy lifestyle, avoid smoking, and do regular health check-ups."

    # ============================================
    # CARD 1: PREDICTION RESULT (Centered)
    # ============================================
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div style="font-size:4rem;">{risk_emoji}</div>
        <h2 style="color:{risk_color}; margin:15px 0; font-size:1.8rem;">{risk_status}</h2>
        <p style="font-size:1.1rem; color:#64748b; margin:0;">
            Probability Level: <b style="color:{risk_color}; font-size:2rem;">{risk_pct:.1f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # CARD 2: PROBABILITY SCORE / GAUGE (Centered)
    # ============================================
    st.markdown("""
    <div class="gauge-card">
        <h3 style="color:#1e293b; margin-top:0; margin-bottom:10px;">📊 Probability Score</h3>
    """, unsafe_allow_html=True)

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_pct,
        number={'suffix': '%', 'font': {'size': 40, 'color': risk_color}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#cbd5e1', 'tickfont': {'color': '#64748b'}},
            'bgcolor': 'rgba(255,255,255,0)',
            'steps': [
                {'range': [0, 40], 'color': 'rgba(34,197,94,0.15)'},
                {'range': [40, 70], 'color': 'rgba(245,158,11,0.15)'},
                {'range': [70, 100], 'color': 'rgba(239,68,68,0.15)'}
            ],
            'threshold': {'line': {'color': risk_color, 'width': 4}, 'thickness': 0.8, 'value': risk_pct},
            'bar': {'color': risk_color}
        }
    ))
    fig_gauge.update_layout(
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)    

    # ============================================
    # CARD 3: MEDICAL RECOMMENDATION (Using factor-card)
    # ============================================
    st.markdown(f"""
    <div class="factor-card" style="border-left-color: #1565c0;">
        <h4 style="margin-top:0; margin-bottom:10px; color:#1e293b;">💡 Medical Recommendation</h4>
        <p style="color:#475569; font-size:1rem; line-height:1.6; margin:0;">
            {advice}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # CARD 4: INFLUENCING FACTORS (Using factor-card + nested cards)
    # ============================================
    
    factors_detected = []
    if smoking == "Yes": factors_detected.append(("🚬 Smoking", "#d32f2f"))
    if alcohol == "Yes": factors_detected.append(("🍺 Alcohol Consuming", "#d32f2f"))
    if chronic_disease == "Yes": factors_detected.append(("🏥 Chronic Disease", "#f59e0b"))
    if coughing == "Yes": factors_detected.append(("😷 Chronic Coughing", "#f59e0b"))
    if shortness == "Yes": factors_detected.append(("😮‍💨 Shortness of Breath", "#f59e0b"))
    if wheezing == "Yes": factors_detected.append(("💨 Wheezing", "#f59e0b"))
    if chest_pain == "Yes": factors_detected.append(("💔 Chest Pain", "#f59e0b"))
    if yellow_fingers == "Yes": factors_detected.append(("🤚 Yellow Fingers", "#64748b"))
    if fatigue == "Yes": factors_detected.append(("😴 Fatigue", "#64748b"))
    if swallowing == "Yes": factors_detected.append(("🥴 Swallowing Difficulty", "#64748b"))
    if anxiety == "Yes": factors_detected.append(("😰 Anxiety", "#64748b"))
    if allergy == "Yes": factors_detected.append(("🤧 Allergy", "#64748b"))
    if age > 60: factors_detected.append((f"👤 Age {age} years", "#64748b"))

    # Outer card - Written in 1 line to avoid code block rendering
    factors_html = '<div class="factor-card" style="border-left-color: #1565c0;"><h4 style="margin-top:0; margin-bottom:15px; color:#1e293b;">📋 Factors Influencing Your Prediction</h4><div style="display: flex; flex-wrap: wrap; gap: 10px;">'

    if factors_detected:
        # Inner cards - Written in 1 line
        for factor, color in factors_detected:
            factors_html += f'<div class="factor-card" style="border-left-color: {color}; margin-bottom: 0; padding: 8px 15px; font-size: 0.9rem; box-shadow: none; border: 1px solid #e2e8f0; border-left: 3px solid {color};">{factor}</div>'
    else:
        factors_html += '<div style="color: #64748b; font-size: 0.95rem;">No major risk factors detected in your input.</div>'
    
    factors_html += '</div></div>'
    st.markdown(factors_html, unsafe_allow_html=True)

    # ============================================
    # CARD 5: DISCLAIMER (Using factor-card yellow/orange)
    # ============================================
    st.markdown("""
    <div class="factor-card" style="border-left-color: #f59e0b; margin-top: 10px;">
        <small style="color: #475569;">⚠️ <b>Disclaimer:</b> This application uses a Decision Tree algorithm to predict patterns based on data. The prediction results are <b>not an official medical diagnosis</b>. Always consult your health condition with a professional doctor.</small>
    </div>
    """, unsafe_allow_html=True)

else:
    # Initial display before button is clicked
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <div style="font-size: 5rem; margin-top: 3rem; margin-bottom: 3rem;">🩺</div>
        <h1 style="font-size:2.2rem; font-weight:800; color:#1e293b; margin-bottom:0.5rem;">
            Ready to Check Your Lung Health?
        </h1>
        <p style="color:#64748b; font-size:1rem; margin:0 auto; max-width: 600px;">
            Fill out the form in the left sidebar, then press the 
            <b style="color:#1565c0;">CHECK RISK NOW</b> button.
        </p>
    </div>
    """, unsafe_allow_html=True)