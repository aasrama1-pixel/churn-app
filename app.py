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
st.title("📊 Dashboard Prediksi Churn Pelanggan")
st.markdown("Gunakan aplikasi ini untuk memprediksi apakah pelanggan **berpotensi churn** atau **bertahan** berdasarkan data pelanggan.")

st.divider()

# ======================
# 3. INPUT DATA USER
# ======================
st.sidebar.header("🔧 Input Data Pelanggan")

usia = st.sidebar.number_input("Usia", min_value=18, max_value=100, value=30)
lama_langganan_bulan = st.sidebar.number_input("Lama Langganan (bulan)", min_value=0, max_value=120, value=12)
jumlah_pengaduan = st.sidebar.number_input("Jumlah Pengaduan", min_value=0, max_value=20, value=2)

# tampilkan data input
input_df = pd.DataFrame({
    'usia': [usia],
    'lama_langganan_bulan': [lama_langganan_bulan],
    'jumlah_pengaduan': [jumlah_pengaduan]
})
st.sidebar.write("📋 Data yang akan diprediksi:")
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
    warna = "red" if pred[0] == 1 else "green"

    st.markdown(
        f"<div style='background-color:{warna}; padding:15px; border-radius:10px; text-align:center; color:white;'>"
        f"<h3>{hasil}</h3></div>",
        unsafe_allow_html=True
    )

    if prob is not None:
        st.write(f"### 🔢 Probabilitas Churn: **{prob:.2%}**")

        # progress bar probabilitas churn
        st.progress(int(prob * 100))

        # ======================
        # 5. VISUALISASI RISIKO
        # ======================
        st.subheader("📈 Visualisasi Risiko Churn")
        fig, ax = plt.subplots(figsize=(4, 4))
        wedges, texts, autotexts = ax.pie(
            [prob, 1 - prob],
            labels=['Churn', 'Bertahan'],
            autopct='%1.1f%%',
            startangle=90,
            colors=['#FF6B6B', '#4CAF50'],
            textprops={'fontsize': 12}
        )
        ax.set_title("Proporsi Risiko Churn", fontsize=14)
        st.pyplot(fig)

        # ======================
        # 6. INTERPRETASI HASIL
        # ======================
        st.subheader("🧠 Interpretasi Hasil Prediksi")

        if prob > 0.7:
            st.warning("⚠️ Pelanggan memiliki **risiko tinggi** untuk churn. Perlu tindakan pencegahan segera (misalnya penawaran khusus atau peningkatan layanan).")
        elif prob > 0.4:
            st.info("ℹ️ Pelanggan memiliki **risiko sedang**. Perhatikan perilaku penggunaan dan keluhan pelanggan.")
        else:
            st.success("✅ Pelanggan memiliki **risiko rendah** untuk churn. Jaga kepuasan mereka dengan layanan konsisten.")

st.divider()

# ======================
# 7. METRIK MODEL (STATIS)
# ======================
st.subheader("📏 Evaluasi Model (Contoh dari Hasil Uji)")

col1, col2 = st.columns(2)
col1.metric("Akurasi Model", "89%")
col2.metric("F1-Score", "85%")

st.caption("Nilai di atas berasal dari hasil pengujian model di dataset uji saat pelatihan di Google Colab.")

