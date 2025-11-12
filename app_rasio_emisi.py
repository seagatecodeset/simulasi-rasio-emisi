import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulasi Rasio Emisi", layout="wide")

# === Fungsi menampilkan grafik ===
def tampilkan_grafik(df, x_col, kategori, judul):
    st.subheader(f"📊 Grafik Rasio Emisi - {judul}")

    if kategori != "Semua":
        df_filtered = df[df["Klasifikasi"] == kategori]
    else:
        df_filtered = df

    # Hitung rata-rata per tahun dan klasifikasi
    df_mean = df_filtered.groupby([x_col, "Klasifikasi"])["Rasio Emisi"].mean().reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for klas in df_mean["Klasifikasi"].unique():
        subset = df_mean[df_mean["Klasifikasi"] == klas]
        ax.plot(subset[x_col], subset["Rasio Emisi"], marker="o", label=f"Klasifikasi {klas}")

    ax.set_title(f"Rasio Emisi per {x_col}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Rasio Emisi")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    return df_mean

# === Fungsi prediksi sederhana (linear regression) ===
def prediksi(df_mean, x_col, tahun_pred):
    st.subheader("🔮 Hasil Prediksi Rasio Emisi")

    x = np.array(df_mean[x_col]).reshape(-1, 1)
    y = np.array(df_mean["Rasio Emisi"])

    # Regresi linier manual
    coef = np.polyfit(x.flatten(), y, 1)
    poly1d_fn = np.poly1d(coef)

    y_pred = poly1d_fn(tahun_pred)
    hasil_pred = pd.DataFrame({x_col: tahun_pred, "Prediksi Rasio Emisi": y_pred})

    # Plot hasil prediksi
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, "bo-", label="Data Aktual")
    ax.plot(tahun_pred, y_pred, "r--", label="Prediksi")
    ax.set_title(f"Prediksi Rasio Emisi hingga {max(tahun_pred)}")
    ax.set_xlabel(x_col)
    ax.set_ylabel("Rasio Emisi")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Tampilkan tabel hasil prediksi
    st.dataframe(hasil_pred.style.format({x_col: "{:.0f}", "Prediksi Rasio Emisi": "{:.4f}"}))

# === Main App ===
st.title("🌱 Simulasi Rasio Emisi Kendaraan")

# Upload file Excel
uploaded_file = st.file_uploader("📂 Unggah file Excel Data Rasio Emisi", type=["xlsx"])
if uploaded_file:
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    tab_list = st.tabs(sheet_names)

    for i, sheet in enumerate(sheet_names):
        with tab_list[i]:
            df = pd.read_excel(xls, sheet_name=sheet)
            st.markdown(f"### Data: {sheet}")
            st.dataframe(df.head())

            # Pilih kolom tahun dan klasifikasi
            x_col = st.selectbox(f"Pilih kolom X untuk {sheet}", df.columns, index=0, key=f"xcol_{sheet}")
            kategori = st.selectbox(
                f"Pilih Klasifikasi untuk {sheet}",
                ["Semua"] + sorted(df["Klasifikasi"].unique().tolist()),
                key=f"kat_{sheet}"
            )

            # Tampilkan grafik aktual
            df_mean = tampilkan_grafik(df, x_col, kategori, sheet)

            # Input prediksi
            tahun_pred = st.slider(f"Pilih rentang {x_col} untuk prediksi {sheet}", 
                                   int(df_mean[x_col].min()), 
                                   int(df_mean[x_col].max()) + 5, 
                                   (int(df_mean[x_col].max())-2, int(df_mean[x_col].max())+3),
                                   key=f"slider_{sheet}")
            tahun_pred = np.arange(tahun_pred[0], tahun_pred[1]+1)

            # Filter prediksi per klasifikasi
            if kategori == "Semua":
                for klas in df_mean["Klasifikasi"].unique():
                    st.markdown(f"#### Prediksi untuk Klasifikasi {klas}")
                    df_k = df_mean[df_mean["Klasifikasi"] == klas]
                    prediksi(df_k, x_col, tahun_pred)
            else:
                df_k = df_mean[df_mean["Klasifikasi"] == kategori]
                prediksi(df_k, x_col, tahun_pred)
