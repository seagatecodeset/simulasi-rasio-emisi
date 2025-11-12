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
    if kategori and kategori != "Semua":
        df = df[df["Kategori"] == kategori]

    df_mean = df.groupby(x_col)["Rasio Emisi"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_mean[x_col], df_mean["Rasio Emisi"], marker="o", label="Data Aktual")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Rata-Rata Rasio Emisi berdasarkan {x_col}")
    ax.legend()
    st.pyplot(fig)
    return df_mean

def prediksi(df_mean, x_col, tahun_ke_depan):
    X = np.array(df_mean[x_col]).reshape(-1, 1)
    y = np.array(df_mean["Rasio Emisi"])
    model = LinearRegression().fit(X, y)

    last_x = int(df_mean[x_col].max())
    future_x = np.arange(last_x + 1, last_x + tahun_ke_depan + 1)
    pred_y = model.predict(future_x.reshape(-1, 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_mean[x_col], y, "o-", label="Data Aktual")
    ax.plot(future_x, pred_y, "r--", label="Prediksi")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Rata-Rata Rasio Emisi")
    ax.set_title(f"Prediksi Rasio Emisi {tahun_ke_depan} tahun mendatang")
    ax.legend()
    st.pyplot(fig)

# === Tab Navigasi ===
tab1, tab2, tab3 = st.tabs(["🚲 Kendaraan Roda Dua", "⛽ Bensin", "🚛 Solar"])

# === TAB 1: Kendaraan Roda Dua ===
with tab1:
    st.subheader("🚲 Kendaraan Roda Dua")
    df = data["Kendaraan Roda Dua"]
    x_col = st.selectbox("Pilih sumbu X:", ["Umur Kendaraan", "Tahun Pembuatan"], key="x1")
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
    x_col = st.selectbox("Pilih sumbu X:", ["Umur Kendaraan", "Tahun Pembuatan"], key="x2")
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
    x_col = st.selectbox("Pilih sumbu X:", ["Umur Kendaraan", "Tahun Pembuatan"], key="x3")
    kategori = st.selectbox("Pilih kategori:", ["C", "D", "E", "F", "G", "Semua"], key="cat3")
    if st.button("Tampilkan Grafik", key="g3"):
        df_mean = tampilkan_grafik(df, x_col, kategori)
    tahun_pred = st.number_input("Prediksi berapa tahun mendatang:", 1, 10, 3, key="p3")
    if st.button("Prediksi", key="pred3"):
        df_mean = tampilkan_grafik(df, x_col, kategori)
        prediksi(df_mean, x_col, tahun_pred)
