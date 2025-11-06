
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

# ======================
# 1. LOAD MODEL & DATA
# ======================
model = joblib.load('model.pkl')  # ganti nama file sesuai model kamu
# opsional: jika punya dataset asli untuk ditampilkan
# df = pd.read_csv('churn.csv')

st.set_page_config(page_title="Dashboard Prediksi Churn", layout="wide")

# ======================
# 2. HEADER DASHBOARD
# ======================
st.title("Dashboard Prediksi Churn Pelanggan")
st.markdown("Aplikasi ini digunakan untuk memprediksi apakah pelanggan **berpotensi churn** atau **tetap bertahan** berdasarkan karakteristiknya.")

st.divider()

# ======================
# 3. INPUT DATA USER
# ======================
st.sidebar.header("Input Data Pelanggan")

usia = st.sidebar.number_input("Usia Pelanggan", min_value=0, max_value=100, value=30)
lama_langganan_bulan = st.sidebar.number_input("Lama Langganan (bulan)", min_value=0, max_value=12, value=12)
jumlah_pengaduan = st.sidebar.number_input("Jumlah Pengaduan", min_value=0, max_value=20, value=2)

input_data = pd.DataFrame({
    'Usia Pelanggan': [usia],
    'Lama Langganan': [lama_langganan_bulan],
    'Jumlah Pengaduan': [jumlah_pengaduan]
})

st.sidebar.write("Data yang akan diprediksi:")
st.sidebar.dataframe(input_data)

# ======================
# 4. PREDIKSI
# ======================
st.subheader("🔍 Hasil Prediksi")

if st.button("Prediksi Sekarang"):
    pred = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None

    hasil = "❌ Pelanggan Berpotensi Churn" if pred[0] == 1 else "✅ Pelanggan Bertahan"
    st.markdown(f"### {hasil}")

    if prob is not None:
        st.write(f"**Probabilitas Churn:** {prob:.2%}")

    # Warna indikator hasil
    color = "red" if pred[0] == 1 else "green"
    st.markdown(
        f"<div style='background-color:{color}; padding:10px; border-radius:10px; text-align:center; color:white;'>"
        f"<h4>{hasil}</h4></div>", unsafe_allow_html=True)

st.divider()

# ======================
# 5. METRIK EVALUASI
# ======================
st.subheader("📈 Evaluasi Model (Contoh Simulasi)")

# Jika kamu punya data uji dan label aslinya, bisa gunakan di sini
# y_true = df['Churn']  # label sebenarnya
# y_pred = model.predict(df.drop('Churn', axis=1))
# acc = accuracy_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)

# Untuk contoh dashboard, kita tampilkan nilai dummy
acc = 0.89
f1 = 0.85

col1, col2 = st.columns(2)
col1.metric("Akurasi Model", f"{acc*100:.2f}%")
col2.metric("F1-Score", f"{f1*100:.2f}%")

# ======================
# 6. CONFUSION MATRIX
# ======================
st.subheader("📊 Confusion Matrix (Contoh Simulasi)")
cm = [[80, 10], [5, 30]]  # contoh dummy, ganti dengan hasil model kamu

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Tidak Churn", "Churn"], yticklabels=["Tidak Churn", "Churn"])
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
st.pyplot(fig)

st.markdown("""
✅ **Interpretasi:**  
- **True Positive (Churn & Churn)** → pelanggan churn yang berhasil diprediksi churn.  
- **False Positive** → pelanggan bertahan tapi salah diprediksi churn.  
- **True Negative** → pelanggan bertahan yang diprediksi bertahan.  
- **False Negative** → pelanggan churn tapi salah diprediksi bertahan.
""")
