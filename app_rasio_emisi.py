# app_rasio_emisi.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
import plotly.express as px

st.set_page_config(layout="wide", page_title="Visual & Prediksi Rasio Emisi")

st.title("📈 Visualisasi & Prediksi Rata-Rata Rasio Emisi")
st.caption("Baca file Excel (Data_Rasio_Emisi.xlsx). Setiap sheet: 'Kendaraan Roda Dua', 'Bensin', 'Solar'")

# -----------------------------
# Upload atau baca file lokal
# -----------------------------
uploaded = st.file_uploader("Upload file Excel (.xlsx) — atau biarkan kosong jika file sudah di direktori working (Data_Rasio_Emisi.xlsx)", type=["xlsx"])

if uploaded is None:
    try:
        # coba baca dari working dir
        xls = pd.ExcelFile("Data_Rasio_Emisi.xlsx")
        st.info("Membaca Data_Rasio_Emisi.xlsx dari working directory.")
    except Exception:
        st.warning("Belum ada file. Upload file Excel Anda dulu.")
        st.stop()
else:
    xls = pd.ExcelFile(uploaded)

sheet_names = xls.sheet_names
st.sidebar.header("Pilihan Sheet / Kategori")
sheet_choice = st.sidebar.selectbox("Pilih kategori (sheet):", sheet_names)

# Baca sheet terpilih
df_raw = pd.read_excel(xls, sheet_choice)
st.sidebar.markdown(f"**Sheet:** `{sheet_choice}` — {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")

# Tampilkan preview & kolom pemetaan
st.subheader("Preview data (sheet terpilih)")
st.dataframe(df_raw.head(200))

# Pemetaaan kolom (adaptif bila nama berbeda)
st.sidebar.header("Pemetaan kolom (jika nama kolom berbeda)")
cols = df_raw.columns.tolist()

default_x = None
default_kat = None
default_y = None
for c in cols:
    cl = c.lower()
    if default_x is None and ("umur" in cl or "umur kendaraan" in cl):
        default_x = c
    if default_x is None and ("tahun" in cl):
        default_x = c
    if default_kat is None and ("kategori" in cl or cl.strip().lower() == "kategori"):
        default_kat = c
    if default_y is None and ("rasio" in cl or "rasio emisi" in cl or "rata" in cl):
        default_y = c

# fallback if not found
if default_x is None and len(cols) >= 1:
    default_x = cols[0]
if default_kat is None and len(cols) >= 2:
    default_kat = cols[1]
if default_y is None and len(cols) >= 3:
    default_y = cols[2]

x_col = st.sidebar.selectbox("Kolom sumbu X (Umur/Tahun):", cols, index=cols.index(default_x) if default_x in cols else 0)
kategori_col = st.sidebar.selectbox("Kolom kategori (B/C/D/...):", cols, index=cols.index(default_kat) if default_kat in cols else 1)
y_col = st.sidebar.selectbox("Kolom Y (Rata-Rata Rasio Emisi):", cols, index=cols.index(default_y) if default_y in cols else 2)

st.sidebar.markdown("---")

# Pilihan sumbu X yang disyaratkan oleh UI (Umur atau Tahun)
st.sidebar.header("Pilihan sumbu X untuk grafik")
x_axis_option = st.sidebar.radio("Gunakan sebagai sumbu X:", ("Umur Kendaraan", "Tahun Pembuatan"))
# NOTE: we will map user-chosen x_col to interpretaion

# Filter kategori tambahan (untuk Bensin & Solar)
additional_filter = None
if sheet_choice.lower().strip() == "bensin":
    additional_filter = st.sidebar.multiselect("Filter kategori (B/C/D) — pilih salah satu atau Semua:", options=sorted(df_raw[kategori_col].dropna().unique()), default=["Semua"] if "Semua" in df_raw[kategori_col].unique() else [])
elif sheet_choice.lower().strip() == "solar":
    additional_filter = st.sidebar.multiselect("Filter kategori (C/D/E/F/G) — pilih atau Semua:", options=sorted(df_raw[kategori_col].dropna().unique()), default=["Semua"] if "Semua" in df_raw[kategori_col].unique() else [])
else:
    # Kendaraan roda dua biasanya tidak perlu filter kategori; allow optional selection
    additional_filter = st.sidebar.multiselect("Opsional: filter kategori (jika ada):", options=sorted(df_raw[kategori_col].dropna().unique()), default=[])

# Clean dataframe: ensure numeric X and Y
df = df_raw.copy()
# Try to coerce numeric columns
for col in [x_col, y_col]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# if x axis option is Umur Kendaraan but x_col is Tahun Pembuatan, convert if possible
if x_axis_option == "Umur Kendaraan":
    # if x_col is Tahun, try to compute umur from current year or from dataset if 'Umur Kendaraan' exists
    if "umur" in x_col.lower():
        pass  # use directly
    elif "tahun" in x_col.lower():
        # convert tahun -> umur relative to max year in dataset (or current year)
        now_year = pd.Timestamp.now().year
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        # create computed umur column
        df["__UMUR_COMPUTED__"] = now_year - df[x_col]
        x_col_to_use = "__UMUR_COMPUTED__"
    else:
        # fallback: use x_col as is (assume it already is umur)
        x_col_to_use = x_col
else:
    # Tahun Pembuatan
    if "tahun" in x_col.lower():
        x_col_to_use = x_col
    elif "umur" in x_col.lower():
        # convert umur to year using current year
        now_year = pd.Timestamp.now().year
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df["__TAHUN_COMPUTED__"] = now_year - df[x_col]  # this may be ambiguous but provide option
        x_col_to_use = "__TAHUN_COMPUTED__"
    else:
        x_col_to_use = x_col

# Apply category filter if any (additional_filter not empty and doesn't contain "Semua")
if additional_filter:
    # if default "Semua" used, ignore
    if "Semua" in additional_filter and len(additional_filter) == 1:
        df_filtered = df.copy()
    else:
        df_filtered = df[df[kategori_col].isin(additional_filter)]
else:
    df_filtered = df.copy()

# Drop NA for X/Y
df_filtered = df_filtered[[x_col_to_use, y_col]].dropna()

if df_filtered.empty:
    st.warning("Data kosong setelah filter / pemetaan kolom. Periksa pemetaan kolom atau isi sheet.")
    st.stop()

# Aggregate rata-rata rasio emisi per X
df_group = df_filtered.groupby(x_col_to_use)[y_col].mean().reset_index().sort_values(by=x_col_to_use)
df_group.rename(columns={x_col_to_use: "X", y_col: "Y"}, inplace=True)

st.subheader("Data agregat: rata-rata rasio emisi per X")
st.dataframe(df_group.head(200))

# Pilihan grafik & tampilkan
st.subheader("Grafik Rata-Rata Rasio Emisi")
chart_type = st.selectbox("Pilih jenis grafik:", ["Line + Markers", "Scatter", "Bar"])

fig = None
if chart_type == "Line + Markers":
    fig = px.line(df_group, x="X", y="Y", markers=True, title=f"{sheet_choice} — Rata-Rata Rasio Emisi per {x_axis_option}")
elif chart_type == "Scatter":
    fig = px.scatter(df_group, x="X", y="Y", trendline="lowess", title=f"{sheet_choice} — Rata-Rata Rasio Emisi per {x_axis_option}")
else:
    fig = px.bar(df_group, x="X", y="Y", title=f"{sheet_choice} — Rata-Rata Rasio Emisi per {x_axis_option}")

fig.update_layout(xaxis_title=x_axis_option, yaxis_title="Rata-Rata Rasio Emisi")
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Prediksi ML
# -----------------------------
st.subheader("🔮 Prediksi Rata-Rata Rasio Emisi ke Depan (Machine Learning)")
st.markdown("""
Gunakan model sederhana berbasis regresi polynomial (Ridge + PolynomialFeatures).
Model dibuat dari data agregat (X vs mean(Y)). Anda dapat menentukan berapa langkah waktu ke depan ingin diprediksi.
""")

# Input: berapa tahun mendatang / langkah umur
n_steps = st.number_input("Prediksi berapa tahun/umur ke depan (integer >=1):", min_value=1, max_value=20, value=3, step=1)

# Parameter model
degree = st.selectbox("Derajat polynomial untuk model (degree):", [1, 2, 3], index=1)
alpha = st.number_input("Regularisasi Ridge (alpha):", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

# Prepare data for training
X = df_group["X"].values.reshape(-1, 1)
y = df_group["Y"].values.reshape(-1, 1)

# Fit model (if enough points)
can_fit = len(X) >= 3  # require at least 3 points to fit polynomial reliably
if not can_fit:
    st.warning("Data terlalu sedikit untuk pelatihan model ML (butuh minimal 3 titik agregat). Prediksi tidak tersedia.")
else:
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=alpha))
    model.fit(X, y.ravel())
    # evaluate in-sample
    y_pred_train = model.predict(X)
    r2 = r2_score(y, y_pred_train)

    # Determine future X values for prediction
    x_max = int(np.nanmax(X))
    # If X represents Tahun Pembuatan and x increases over time, prediction for next years should be > x_max
    # If X represents Umur Kendaraan, prediction for future ages will be x_max + 1..n
    future_X = np.array([x_max + i for i in range(1, n_steps + 1)]).reshape(-1, 1)

    y_future = model.predict(future_X)

    # Build plotting frame that includes predictions
    df_pred_plot = pd.DataFrame({"X": future_X.ravel(), "Y": y_future.ravel(), "type": "prediksi"})
    df_plot_all = pd.concat([df_group.assign(type="data"), df_pred_plot], ignore_index=True)

    # Plot including predictions
    fig_pred = px.line(df_plot_all, x="X", y="Y", color="type", markers=True,
                       title=f"Data historis + Prediksi {n_steps} langkah ke depan (degree={degree}, alpha={alpha})")
    fig_pred.update_traces(selector=dict(name="prediksi"), line=dict(dash='dash'))
    fig_pred.update_layout(xaxis_title=x_axis_option, yaxis_title="Rata-Rata Rasio Emisi")
    st.plotly_chart(fig_pred, use_container_width=True)

    # Show prediction table
    df_future_display = pd.DataFrame({
        x_axis_option: future_X.ravel(),
        "Prediksi_RataRasio": np.round(y_future.ravel(), 4)
    })
    st.subheader("Tabel Prediksi")
    st.dataframe(df_future_display)

    st.markdown(f"**Model in-sample R²:** {r2:.3f}")

    st.success("Prediksi selesai — interpretasikan dengan hati-hati. Model sederhana; untuk hasil lebih akurat gunakan dataset lebih besar, fitur tambahan, atau model time-series yang sesuai.")

# -----------------------------
# Catatan & saran
# -----------------------------
st.markdown("""
### Catatan penting
- Kode ini menggunakan model regresi polynomial sederhana (Ridge). Ini cepat dan mudah, tetapi **bukan** model time-series kompleks.
- Jika Anda ingin prediksi berdasarkan waktu nyata (mis. memprediksi untuk *tahun kalender* ke depan), pastikan kolom X adalah `Tahun Pembuatan` yang representatif.
- Untuk perbaikan: gunakan model ARIMA/SARIMA/Prophet/LSTM bila data berurutan per waktu tersedia.
""")
