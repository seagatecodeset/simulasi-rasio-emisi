import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Load Data ===
file_path = "Data_Rasio_Emisi.xlsx"
sheet_names = ["Kendaraan Roda Dua", "Bensin", "Solar"]
data = {sheet: pd.read_excel(file_path, sheet_name=sheet) for sheet in sheet_names}

st.set_page_config(page_title="Dashboard Rasio Emisi", layout="wide")
st.title("📊 Dashboard Rasio Emisi Kendaraan")
st.markdown("Analisis dan Prediksi Rasio Emisi berdasarkan kategori kendaraan.")

# === Fungsi bantu untuk grafik dan prediksi ===
def tampilkan_grafik(df, x_col, kategori=None):
    # Normalisasi nama kolom agar tidak error
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Temukan nama kolom yang paling mirip dengan 'rasio_emisi'
    possible_cols = [c for c in df.columns if "rasio" in c and "emisi" in c]
    if len(possible_cols) == 0:
        st.error("Kolom rasio emisi tidak ditemukan di data!")
        st.write("Kolom tersedia:", list(df.columns))
        return
    rasio_col = possible_cols[0]

    # Jika kategori tersedia dan kolomnya ada
    if kategori and "kategori" in df.columns and kategori != "Semua":
        df = df[df["kategori"] == kategori]

    # Group berdasarkan sumbu X
    df_mean = df.groupby(df[x_col].name)[rasio_col].mean().reset_index()

    # Plot hasil
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_mean[x_col], df_mean[rasio_col], marker="o", label="Data Aktual")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Rata-Rata Rasio Emisi berdasarkan {x_col}")
    ax.legend()
    st.pyplot(fig)

    return df_mean

def prediksi(df_mean, x_col, tahun_pred):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression

    # Normalisasi kolom untuk menghindari KeyError
    df_mean.columns = df_mean.columns.str.strip().str.lower().str.replace(" ", "_")

    # Cari kolom rasio emisi
    possible_cols = [c for c in df_mean.columns if "rasio" in c and "emisi" in c]
    if not possible_cols:
        st.error("❌ Kolom rasio emisi tidak ditemukan pada data agregat!")
        st.write("Kolom tersedia:", list(df_mean.columns))
        return
    rasio_col = possible_cols[0]

    # Ambil data X dan y
    X = np.array(df_mean[x_col]).reshape(-1, 1)
    y = np.array(df_mean[rasio_col])

    # Latih model regresi linear
    model = LinearRegression()
    model.fit(X, y)

    # Prediksi beberapa tahun ke depan
    x_future = np.arange(X.max() + 1, X.max() + tahun_pred + 1).reshape(-1, 1)
    y_future = model.predict(x_future)

    # Plot aktual + prediksi
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(X, y, "o-", label="Data Aktual")
    ax.plot(x_future, y_future, "r--", label=f"Prediksi {tahun_pred} Tahun ke Depan")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Prediksi Rata-Rata Rasio Emisi berdasarkan {x_col}")
    ax.legend()
    st.pyplot(fig)

    # Tampilkan data prediksi
    pred_df = {
        x_col: x_future.flatten(),
        "Prediksi Rasio Emisi": y_future
    }
    st.dataframe(pred_df)

# === Tab Navigasi ===
tab1, tab2, tab3 = st.tabs(["🚲 Kendaraan Roda Dua", "⛽ Bensin", "🚛 Solar"])

# === TAB 1: Kendaraan Roda Dua ===
with tab1:
    st.subheader("🚲 Kendaraan Roda Dua")
    df = data["Kendaraan Roda Dua"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [col for col in df.columns if "umur" in col or "tahun" in col]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x1")
    if st.button("Tampilkan Grafik", key="g1"):
        df_mean = tampilkan_grafik(df, x_col)
    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p1")
    if st.button("Prediksi", key="pred1"):
        df_mean = tampilkan_grafik(df, x_col)
        prediksi(df_mean, x_col, tahun_pred)

# === TAB 2: Bensin ===
with tab2:
    st.subheader("⛽ Kendaraan Bensin")
    df = data["Bensin"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [col for col in df.columns if "umur" in col or "tahun" in col]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x2")
    kategori = st.selectbox("Pilih kategori:", ["B", "C", "D", "Semua"], key="cat2")
    if st.button("Tampilkan Grafik", key="g2"):
        df_mean = tampilkan_grafik(df, x_col, kategori)
    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p2")
    if st.button("Prediksi", key="pred2"):
        df_mean = tampilkan_grafik(df, x_col, kategori)
        prediksi(df_mean, x_col, tahun_pred)

# === TAB 3: Solar ===
with tab3:
    st.subheader("🚛 Kendaraan Solar")
    df = data["Solar"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [col for col in df.columns if "umur" in col or "tahun" in col]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x3")
    kategori = st.selectbox("Pilih kategori:", ["C", "D", "E", "F", "G", "Semua"], key="cat3")
    if st.button("Tampilkan Grafik", key="g3"):
        df_mean = tampilkan_grafik(df, x_col, kategori)
    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p3")
    if st.button("Prediksi", key="pred3"):
        df_mean = tampilkan_grafik(df, x_col, kategori)
        prediksi(df_mean, x_col, tahun_pred)


