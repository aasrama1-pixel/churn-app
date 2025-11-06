import streamlit as st
import joblib
import pandas as pd
import numpy as np

# === CONFIGURASI DASAR ===
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="📊",
    layout="centered",
)

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# === SIDEBAR ===
st.sidebar.title("⚙️ Navigasi")
menu = st.sidebar.radio("Pilih Halaman:", ["🏠 Beranda", "📈 Prediksi", "ℹ️ Tentang"])

# === HALAMAN BERANDA ===
if menu == "🏠 Beranda":
    st.title("📊 Dashboard Prediksi Churn Pelanggan")
    st.markdown(
        """
        Aplikasi ini digunakan untuk memprediksi apakah pelanggan **akan churn atau tidak**, 
        berdasarkan beberapa faktor seperti:
        - 🧍 **Usia Pelanggan**
        - 📆 **Lama Berlangganan**
        - 📞 **Jumlah Pengaduan**

        ---
        **Tujuan aplikasi:** membantu perusahaan menjaga loyalitas pelanggan dan 
        mengurangi potensi kehilangan pelanggan.
        """
    )
    st.image(
        "https://cdn-icons-png.flaticon.com/512/1055/1055646.png",
        width=200,
    )
    st.info("Pilih menu **📈 Prediksi** di sidebar untuk mulai melakukan prediksi.")

# === HALAMAN PREDIKSI ===
elif menu == "📈 Prediksi":
    st.title("📈 Prediksi Churn Pelanggan")
    st.markdown("Masukkan data pelanggan di bawah ini:")

    # Input kolom
    usia = st.number_input("🧍 Usia Pelanggan", min_value=18, max_value=100, step=1)
    lama_langganan = st.number_input("📆 Lama Langganan (bulan)", min_value=1, max_value=120, step=1)
    jumlah_pengaduan = st.number_input("📞 Jumlah Pengaduan", min_value=0, max_value=50, step=1)

    if st.button("🔮 Prediksi Sekarang"):
        # Buat dataframe dengan nama kolom yang sama seperti saat training
        data = pd.DataFrame({
            'usia': [usia],
            'lama_langganan': [lama_langganan],
            'jumlah_pengaduan': [jumlah_pengaduan]
        })

        prediksi = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("📊 Hasil Prediksi:")
        if prediksi == 1:
            st.error("⚠️ Pelanggan kemungkinan akan **CHURN**.")
        else:
            st.success("✅ Pelanggan kemungkinan **TIDAK CHURN**.")

        if proba is not None:
            st.metric(label="Probabilitas Churn", value=f"{proba*100:.2f}%")

        # Simpan hasil prediksi ke CSV (opsional)
        hasil = data.copy()
        hasil["prediksi"] = "Churn" if prediksi == 1 else "Tidak Churn"
        hasil["probabilitas_churn"] = proba
        hasil.to_csv("hasil_prediksi.csv", index=False)

        st.download_button(
            label="📥 Unduh Hasil Prediksi",
            data=hasil.to_csv(index=False),
            file_name="hasil_prediksi.csv",
            mime="text/csv",
        )

# === HALAMAN TENTANG ===
elif menu == "ℹ️ Tentang":
    st.title("ℹ️ Tentang Aplikasi")
    st.markdown(
        """
        Aplikasi ini dikembangkan menggunakan:

