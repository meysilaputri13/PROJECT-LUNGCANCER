"""
app.py
Aplikasi Web Prediksi Kanker Paru - Layout Vertikal & Card Seragam
Jalankan: streamlit run app.py
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
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Prediksi Kanker Paru",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS (UI MODERN BIRU & PUTIH)
# ============================================
st.markdown("""
<style>
    /* ======== GLOBAL ======== */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px; /* Agar konten tengah tidak terlalu lebar */
        margin: 0 auto;
    }

    /* ======== CARD HASIL PREDIKSI ======== */
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

    /* ======== CARD GAUGE ======== */
    .gauge-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.04);
        text-align: center;
    }

    /* ======== FACTOR CARD (DIGUNAKAN UNTUK SEMUA INFO) ======== */
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
# INISIALISASI MODEL (CACHED)
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
# LAYOUT: SIDEBAR (FORM INPUT)
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
    st.markdown('<p class="section-title">Data Demografis</p>', unsafe_allow_html=True)
    
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"], index=0)
    age = st.slider("Usia", 1, 120, 50)

    st.markdown("---")
    st.markdown('<p class="section-title">Gaya Hidup</p>', unsafe_allow_html=True)
    
    col_side1, col_side2 = st.columns(2)
    with col_side1:
        smoking = st.radio("Merokok", ["Tidak", "Ya"], horizontal=True, index=0)
    with col_side2:
        alcohol = st.radio("Alkohol", ["Tidak", "Ya"], horizontal=True, index=0)

    st.markdown("---")
    st.markdown('<p class="section-title">Gejala Klinis</p>', unsafe_allow_html=True)
    
    g1, g2 = st.columns(2)
    
    with g1:
        yellow_fingers = st.radio("Jari Kuning", ["Tidak", "Ya"], horizontal=True, index=0)
        anxiety = st.radio("Kecemasan", ["Tidak", "Ya"], horizontal=True, index=0)
        peer_pressure = st.radio("Tekanan Lingkungan", ["Tidak", "Ya"], horizontal=True, index=0)
        chronic_disease = st.radio("Penyakit Kronis", ["Tidak", "Ya"], horizontal=True, index=0)
        fatigue = st.radio("Kelelahan", ["Tidak", "Ya"], horizontal=True, index=0)
        allergy = st.radio("Alergi", ["Tidak", "Ya"], horizontal=True, index=0)

    with g2:
        coughing = st.radio("Batuk", ["Tidak", "Ya"], horizontal=True, index=0)
        shortness = st.radio("Sesak Napas", ["Tidak", "Ya"], horizontal=True, index=0)
        swallowing = st.radio("Sulit Menelan", ["Tidak", "Ya"], horizontal=True, index=0)
        chest_pain = st.radio("Nyeri Dada", ["Tidak", "Ya"], horizontal=True, index=0)
        wheezing = st.radio("Mengi", ["Tidak", "Ya"], horizontal=True, index=0)

    st.markdown("---")
    
    predict_button = st.button("CEK RISIKO SEKARANG", use_container_width=True)


# ============================================
# LAYOUT: MAIN PAGE (HASIL PREDIKSI)
# ============================================

# 1. JUDUL RATA TENGAH
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="font-size:2.2rem; font-weight:800; color:#1e293b; margin-bottom:0.5rem;">
        Prediksi Risiko Kanker Paru
    </h1>
    <p style="color:#64748b; font-size:1rem; margin:0 auto; max-width: 600px;">
        Masukkan kondisi tubuh Anda pada formulir di sebelah kiri, kemudian klik 
        <b style="color:#1565c0;">CEK RISIKO SEKARANG</b> untuk mengetahui hasilnya.
    </p>
</div>
""", unsafe_allow_html=True)


if predict_button:
    input_data = {
        'GENDER': 1 if gender == "Laki-laki" else 0,
        'AGE': age,
        'SMOKING': 2 if smoking == "Ya" else 1,
        'YELLOW_FINGERS': 2 if yellow_fingers == "Ya" else 1,
        'ANXIETY': 2 if anxiety == "Ya" else 1,
        'PEER_PRESSURE': 2 if peer_pressure == "Ya" else 1,
        'CHRONIC_DISEASE': 2 if chronic_disease == "Ya" else 1,
        'FATIGUE': 2 if fatigue == "Ya" else 1,
        'ALLERGY': 2 if allergy == "Ya" else 1,
        'WHEEZING': 2 if wheezing == "Ya" else 1,
        'ALCOHOL_CONSUMING': 2 if alcohol == "Ya" else 1,
        'COUGHING': 2 if coughing == "Ya" else 1,
        'SHORTNESS_OF_BREATH': 2 if shortness == "Ya" else 1,
        'SWALLOWING_DIFFICULTY': 2 if swallowing == "Ya" else 1,
        'CHEST_PAIN': 2 if chest_pain == "Ya" else 1,
    }

    prediction, probability = model.predict(input_data)
    
    prob_yes = probability[1]
    risk_pct = prob_yes * 100

    if prediction == 1:
        risk_status = "POSITIF RISIKO KANKER PARU"
        risk_emoji = "⚠️"
        risk_class = "result-positive"
        risk_color = "#d32f2f"
        advice = "Segera konsultasikan kondisi Anda ke dokter spesialis paru untuk pemeriksaan lanjutan seperti Rontgen atau CT Scan dan tingkatkan pola hidup sehat."
    else:
        risk_status = "NEGATIF / RENDAH RISIKO"
        risk_emoji = "✅"
        risk_class = "result-negative"
        risk_color = "#2e7d32"
        advice = "Kondisi Anda terdeteksi rendah risiko. Tetap jaga pola hidup sehat, hindari asap rokok, dan lakukan pemeriksaan kesehatan rutin."

    # ============================================
    # CARD 1: HASIL PREDIKSI (Rata Tengah)
    # ============================================
    st.markdown(f"""
    <div class="result-card {risk_class}">
        <div style="font-size:4rem;">{risk_emoji}</div>
        <h2 style="color:{risk_color}; margin:15px 0; font-size:1.8rem;">{risk_status}</h2>
        <p style="font-size:1.1rem; color:#64748b; margin:0;">
            Tingkat Probabilitas: <b style="color:{risk_color}; font-size:2rem;">{risk_pct:.1f}%</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # CARD 2: SKOR RASIO / GAUGE (Rata Tengah)
    # ============================================
    st.markdown("""
    <div class="gauge-card">
        <h3 style="color:#1e293b; margin-top:0; margin-bottom:10px;">📊 Skor Probabilitas</h3>
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
    # CONFUSION MATRIX
    # ============================================

    st.markdown("""
    <div class="gauge-card">
        <h3 style="color:#1e293b; margin-top:0; margin-bottom:10px;">
            📊 Confusion Matrix
        </h3>
    """, unsafe_allow_html=True)

    cm = model.cm

    fig, ax = plt.subplots(figsize=(4,4))

    ax.matshow(cm, cmap="Blues")

    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha='center', va='center', fontsize=14)

    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")

    st.pyplot(fig)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    

    # ============================================
    # CARD 3: REKOMENDASI MEDIS (Pakai factor-card)
    # ============================================
    st.markdown(f"""
    <div class="factor-card" style="border-left-color: #1565c0;">
        <h4 style="margin-top:0; margin-bottom:10px; color:#1e293b;">💡 Rekomendasi Medis</h4>
        <p style="color:#475569; font-size:1rem; line-height:1.6; margin:0;">
            {advice}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # CARD 4: FAKTOR YANG MEMPENGARUHI (Pakai factor-card + nested cards)
    # ============================================
    
    factors_detected = []
    if smoking == "Ya": factors_detected.append(("🚬 Merokok", "#d32f2f"))
    if alcohol == "Ya": factors_detected.append(("🍺 Konsumsi Alkohol", "#d32f2f"))
    if chronic_disease == "Ya": factors_detected.append(("🏥 Penyakit Kronis", "#f59e0b"))
    if coughing == "Ya": factors_detected.append(("😷 Batuk Berkepanjangan", "#f59e0b"))
    if shortness == "Ya": factors_detected.append(("😮‍💨 Sesak Napas", "#f59e0b"))
    if wheezing == "Ya": factors_detected.append(("💨 Mengi", "#f59e0b"))
    if chest_pain == "Ya": factors_detected.append(("💔 Nyeri Dada", "#f59e0b"))
    if yellow_fingers == "Ya": factors_detected.append(("🤚 Jari Kuning", "#64748b"))
    if fatigue == "Ya": factors_detected.append(("😴 Kelelahan", "#64748b"))
    if swallowing == "Ya": factors_detected.append(("🥴 Sulit Menelan", "#64748b"))
    if anxiety == "Ya": factors_detected.append(("😰 Kecemasan", "#64748b"))
    if allergy == "Ya": factors_detected.append(("🤧 Alergi", "#64748b"))
    if age > 60: factors_detected.append((f"👤 Usia {age} tahun", "#64748b"))

    # Kotak induk (outer card) - Ditulis 1 baris agar tidak terdeteksi code block
    factors_html = '<div class="factor-card" style="border-left-color: #1565c0;"><h4 style="margin-top:0; margin-bottom:15px; color:#1e293b;">📋 Faktor yang Mempengaruhi Prediksi Anda</h4><div style="display: flex; flex-wrap: wrap; gap: 10px;">'

    if factors_detected:
        # Kotak anak (inner card) - Ditulis 1 baris agar tidak terdeteksi code block
        for factor, color in factors_detected:
            factors_html += f'<div class="factor-card" style="border-left-color: {color}; margin-bottom: 0; padding: 8px 15px; font-size: 0.9rem; box-shadow: none; border: 1px solid #e2e8f0; border-left: 3px solid {color};">{factor}</div>'
    else:
        factors_html += '<div style="color: #64748b; font-size: 0.95rem;">Tidak ada faktor risiko utama yang terdeteksi pada input Anda.</div>'
    
    factors_html += '</div></div>'
    st.markdown(factors_html, unsafe_allow_html=True)

    # ============================================
    # CARD 5: DISCLAIMER (Pakai factor-card warna kuning/oranye)
    # ============================================
    st.markdown("""
    <div class="factor-card" style="border-left-color: #f59e0b; margin-top: 10px;">
        <small style="color: #475569;">⚠️ <b>Disclaimer:</b> Aplikasi ini menggunakan algoritma Decision Tree untuk memprediksi pola berdasarkan data. Hasil prediksi <b>bukan merupakan diagnosis medis resmi</b>. Selalu konsultasikan kondisi kesehatan Anda dengan dokter profesional.</small>
    </div>
    """, unsafe_allow_html=True)

else:
    # Tampilan awal sebelum tombol diklik
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #64748b;">
        <div style="font-size: 5rem; margin-top: 3rem; margin-bottom: 3rem;">🩺</div>
        <h1 style="font-size:2.2rem; font-weight:800; color:#1e293b; margin-bottom:0.5rem;">
            Siap Untuk Mengecek Kesehatan Paru Anda?
        </h1>
        <p style="color:#64748b; font-size:1rem; margin:0 auto; max-width: 600px;">
            Isi formulir di sidebar sebelah kiri, lalu tekan tombol 
            <b style="color:#1565c0;">CEK RISIKO SEKARANG</b>
        </p>
    </div>
    """, unsafe_allow_html=True)