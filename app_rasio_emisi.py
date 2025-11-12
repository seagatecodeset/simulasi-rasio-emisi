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

# === Fungsi bantu untuk grafik ===
def tampilkan_grafik(df, x_col, kategori=None):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # deteksi kolom rasio emisi
    possible_cols = [c for c in df.columns if "rasio" in c and "emisi" in c]
    if not possible_cols:
        st.error("Kolom 'Rasio Emisi' tidak ditemukan.")
        return
    rasio_col = possible_cols[0]

    has_class = "klasifikasi" in df.columns
    fig, ax = plt.subplots(figsize=(8, 4))

    if kategori and has_class:
        if kategori != "Semua":
            df = df[df["klasifikasi"].str.upper() == kategori.upper()]
            df_mean = df.groupby(df[x_col].name)[rasio_col].mean().reset_index()
            ax.plot(df_mean[x_col], df_mean[rasio_col], marker="o", label=f"Klasifikasi {kategori}")
        else:
            for k in sorted(df["klasifikasi"].dropna().unique()):
                df_sub = df[df["klasifikasi"].str.upper() == k.upper()]
                df_mean = df_sub.groupby(df_sub[x_col].name)[rasio_col].mean().reset_index()
                ax.plot(df_mean[x_col], df_mean[rasio_col], marker="o", label=f"Klasifikasi {k}")
    else:
        df_mean = df.groupby(df[x_col].name)[rasio_col].mean().reset_index()
        ax.plot(df_mean[x_col], df_mean[rasio_col], marker="o", label="Data Aktual")

    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Rata-Rata Rasio Emisi berdasarkan {x_col}")
    ax.legend()
    st.pyplot(fig)

    return df_mean

# === Fungsi prediksi ===
def prediksi(df, x_col, tahun_pred, kategori=None):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    possible_cols = [c for c in df.columns if "rasio" in c and "emisi" in c]
    if not possible_cols:
        st.error("Kolom rasio emisi tidak ditemukan.")
        return
    rasio_col = possible_cols[0]

    has_class = "klasifikasi" in df.columns
    fig, ax = plt.subplots(figsize=(8, 4))

    if has_class and kategori:
        if kategori != "Semua":
            # Prediksi hanya 1 klasifikasi
            df_sub = df[df["klasifikasi"].str.upper() == kategori.upper()]
            df_mean = df_sub.groupby(df_sub[x_col].name)[rasio_col].mean().reset_index()
            if df_mean.empty:
                st.warning(f"Tidak ada data untuk klasifikasi {kategori}.")
                return

            X = np.array(df_mean[x_col]).reshape(-1, 1)
            y = np.array(df_mean[rasio_col])
            model = LinearRegression()
            model.fit(X, y)

            x_future = np.arange(X.max() + 1, X.max() + tahun_pred + 1).reshape(-1, 1)
            y_future = model.predict(x_future)

            ax.plot(X, y, "o-", label=f"Data Aktual {kategori}")
            ax.plot(x_future, y_future, "--", label=f"Prediksi {kategori}")
        else:
            # Prediksi semua klasifikasi
            for k in sorted(df["klasifikasi"].dropna().unique()):
                df_sub = df[df["klasifikasi"].str.upper() == k.upper()]
                df_mean = df_sub.groupby(df_sub[x_col].name)[rasio_col].mean().reset_index()
                if df_mean.empty:
                    continue

                X = np.array(df_mean[x_col]).reshape(-1, 1)
                y = np.array(df_mean[rasio_col])
                model = LinearRegression()
                model.fit(X, y)

                x_future = np.arange(X.max() + 1, X.max() + tahun_pred + 1).reshape(-1, 1)
                y_future = model.predict(x_future)

                ax.plot(X, y, "o-", label=f"Data Aktual {k}")
                ax.plot(x_future, y_future, "--", label=f"Prediksi {k}")
    else:
        # Prediksi umum tanpa klasifikasi
        df_mean = df.groupby(df[x_col].name)[rasio_col].mean().reset_index()
        X = np.array(df_mean[x_col]).reshape(-1, 1)
        y = np.array(df_mean[rasio_col])
        model = LinearRegression()
        model.fit(X, y)

        x_future = np.arange(X.max() + 1, X.max() + tahun_pred + 1).reshape(-1, 1)
        y_future = model.predict(x_future)

        ax.plot(X, y, "o-", label="Data Aktual")
        ax.plot(x_future, y_future, "r--", label=f"Prediksi {tahun_pred} Tahun ke Depan")

    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Prediksi Rata-Rata Rasio Emisi berdasarkan {x_col}")
    ax.legend()
    st.pyplot(fig)
    
    pred_df = pd.DataFrame({x_col: x_future.flatten(), "Prediksi Rasio Emisi": y_future})
    st.dataframe(pred_df)
# === Tabs ===
tab1, tab2, tab3 = st.tabs(["🚲 Kendaraan Roda Dua", "⛽ Bensin", "🚛 Solar"])

# === TAB 1 ===
with tab1:
    st.subheader("🚲 Kendaraan Roda Dua")
    df = data["Kendaraan Roda Dua"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [c for c in df.columns if "umur" in c or "tahun" in c]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x1")
    if st.button("Tampilkan Grafik", key="g1"):
        tampilkan_grafik(df, x_col)
    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p1")
    if st.button("Prediksi", key="pred1"):
        prediksi(df, x_col, tahun_pred)

# === TAB 2 ===
with tab2:
    st.subheader("⛽ Kendaraan Bensin")
    df = data["Bensin"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [c for c in df.columns if "umur" in c or "tahun" in c]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x2")
    kategori = st.selectbox("Pilih kategori:", ["B", "C", "D", "Semua"], key="cat2")
    if st.button("Tampilkan Grafik", key="g2"):
        tampilkan_grafik(df, x_col, kategori)
    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p2")
    if st.button("Prediksi", key="pred2"):
        prediksi(df, x_col, tahun_pred, kategori)

# === TAB 3 ===
with tab3:
    st.subheader("🚛 Kendaraan Solar")
    df = data["Solar"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    x_options = [c for c in df.columns if "umur" in c or "tahun" in c]
    x_col = st.selectbox("Pilih sumbu X:", x_options, key="x3")
    kategori = st.selectbox("Pilih kategori:", ["C", "D", "E", "F", "G", "Semua"], key="cat3")
    if st.button("Tampilkan Grafik", key="g3"):
        tampilkan_grafik(df, x_col, kategori)
    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p3")
    if st.button("Prediksi", key="pred3"):
        prediksi(df, x_col, tahun_pred, kategori)

