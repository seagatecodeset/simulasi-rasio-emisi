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
def tampilkan_grafik(df, x_col, kategori_opsi=None, kategori_pilihan=None):
    # Normalisasi kolom agar aman
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Identifikasi kolom kategori dan rasio emisi
    kategori_col = None
    for c in df.columns:
        if "kategori" in c:
            kategori_col = c
    rasio_col = None
    for c in df.columns:
        if "rasio" in c and "emisi" in c:
            rasio_col = c

    if rasio_col is None:
        st.error("Kolom 'Rasio Emisi' tidak ditemukan di data!")
        st.write("Kolom yang ada:", list(df.columns))
        return None

    # === Filter data berdasarkan kategori ===
    if kategori_col and kategori_pilihan and kategori_pilihan != "Semua":
        df_plot = df[df[kategori_col].astype(str).str.upper() == kategori_pilihan.upper()]
    elif kategori_col and kategori_opsi:
        df_plot = df[df[kategori_col].astype(str).str.upper().isin(kategori_opsi)]
    else:
        df_plot = df.copy()

    # === Hitung rata-rata rasio per sumbu X dan kategori (jika ada) ===
    if kategori_col and "kategori" in df.columns:
        df_mean = df_plot.groupby([x_col, kategori_col])[rasio_col].mean().reset_index()
    else:
        df_mean = df_plot.groupby(x_col)[rasio_col].mean().reset_index()

    # === Plot ===
    fig, ax = plt.subplots(figsize=(8, 4))
    if kategori_col and "kategori" in df.columns and kategori_pilihan == "Semua":
        for cat in sorted(df_mean[kategori_col].unique()):
            data_cat = df_mean[df_mean[kategori_col] == cat]
            ax.plot(data_cat[x_col], data_cat[rasio_col], marker="o", label=f"Kategori {cat}")
        ax.legend(title="Kategori")
    else:
        ax.plot(df_mean[x_col], df_mean[rasio_col], "o-", label="Data Aktual")
        ax.legend()

    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Rata-Rata Rasio Emisi berdasarkan {x_col}")
    st.pyplot(fig)

    return df_mean


def prediksi(df_mean, x_col, tahun_pred):
    # Normalisasi kolom
    df_mean.columns = df_mean.columns.str.strip().str.lower().str.replace(" ", "_")

    # Deteksi kolom rasio
    rasio_col = None
    for c in df_mean.columns:
        if "rasio" in c and "emisi" in c:
            rasio_col = c
    if rasio_col is None:
        st.error("❌ Kolom rasio emisi tidak ditemukan pada data agregat!")
        return

    # Ambil data
    X = np.array(df_mean[x_col]).reshape(-1, 1)
    y = np.array(df_mean[rasio_col])

    # Model Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Prediksi
    x_future = np.arange(X.max() + 1, X.max() + tahun_pred + 1).reshape(-1, 1)
    y_future = model.predict(x_future)

    # Plot hasil
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(X, y, "o-", label="Data Aktual")
    ax.plot(x_future, y_future, "r--", label=f"Prediksi {tahun_pred} Tahun ke Depan")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Prediksi Rata-Rata Rasio Emisi berdasarkan {x_col}")
    ax.legend()
    st.pyplot(fig)

    # Tampilkan tabel prediksi
    pred_df = pd.DataFrame({x_col: x_future.flatten(), "Prediksi Rasio Emisi": y_future})
    st.dataframe(pred_df, use_container_width=True)


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
        if df_mean is not None:
            prediksi(df_mean, x_col, tahun_pred)

# === TAB 2: Bensin ===
with tab2:
    st.subheader("⛽ Kendaraan Bensin")
    df = data["Bensin"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [col for col in df.columns if "umur" in col or "tahun" in col]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x2")

    kategori_opsi = ["B", "C", "D"]
    kategori_pilihan = st.selectbox("Pilih kategori:", kategori_opsi + ["Semua"], key="cat2")

    if st.button("Tampilkan Grafik", key="g2"):
        df_mean = tampilkan_grafik(df, x_col, kategori_opsi, kategori_pilihan)

    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p2")
    if st.button("Prediksi", key="pred2"):
        df_mean = tampilkan_grafik(df, x_col, kategori_opsi, kategori_pilihan)
        if df_mean is not None:
            prediksi(df_mean, x_col, tahun_pred)

# === TAB 3: Solar ===
with tab3:
    st.subheader("🚛 Kendaraan Solar")
    df = data["Solar"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [col for col in df.columns if "umur" in col or "tahun" in col]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x3")

    kategori_opsi = ["C", "D", "E", "F", "G"]
    kategori_pilihan = st.selectbox("Pilih kategori:", kategori_opsi + ["Semua"], key="cat3")

    if st.button("Tampilkan Grafik", key="g3"):
        df_mean = tampilkan_grafik(df, x_col, kategori_opsi, kategori_pilihan)

    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p3")
    if st.button("Prediksi", key="pred3"):
        df_mean = tampilkan_grafik(df, x_col, kategori_opsi, kategori_pilihan)
        if df_mean is not None:
            prediksi(df_mean, x_col, tahun_pred)
