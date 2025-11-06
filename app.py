import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. LOAD MODEL
# ======================
model = joblib.load('model.pkl')

st.set_page_config(page_title="Dashboard Prediksi Churn", layout="wide")

# ======================
# 2. HEADER DASHBOARD
# ======================
st.title("Dashboard Prediksi Churn Pelanggan")
st.markdown("Prediksi apakah pelanggan akan **Churn** atau **Bertahan** berdasarkan karakteristik mereka.")

st.divider()

# ======================
# 3. INPUT DATA USER
# ======================
st.sidebar.header("Input Data Pelanggan")

usia = st.sidebar.number_input("Usia", min_value=18, max_value=100, value=30)
lama_langganan_bulan = st.sidebar.number_input("Lama Langganan (bulan)", min_value=0, max_value=120, value=12)
jumlah_pengaduan = st.sidebar.number_input("Jumlah Pengaduan", min_value=0, max_value=20, value=2)

# tampilkan data input
input_df = pd.DataFrame({
    'usia': [usia],
    'lama_langganan_bulan': [lama_langganan_bulan],
    'jumlah_pengaduan': [jumlah_pengaduan]
})
st.sidebar.write("Data yang akan diprediksi:")
st.sidebar.dataframe(input_df)

# ======================
# 4. PREDIKSI
# ======================
st.subheader("🔍 Hasil Prediksi")

if st.button("Prediksi Sekarang"):
    # ubah ke numpy array karena model dilatih tanpa kolom
    input_data = np.array([[usia, lama_langganan_bulan, jumlah_pengaduan]])

    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    hasil = "❌ Pelanggan Berpotensi Churn" if pred[0] == 1 else "✅ Pelanggan Bertahan"
    st.markdown(f"### {hasil}")

    if prob is not None:
        st.write(f"**Probabilitas Churn:** {prob:.2%}")

    color = "red" if pred[0] == 1 else "green"
    st.markdown(
        f"<div style='background-color:{color}; padding:10px; border-radius:10px; text-align:center; color:white;'>"
        f"<h4>{hasil}</h4></div>", unsafe_allow_html=True
    )

st.divider()

