# ==========================================
# Dashboard Analisis & Prediksi Kepatuhan Emisi Kendaraan
# Streamlit
# ==========================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Dashboard Kepatuhan Emisi Kendaraan",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetricValue"] {font-size: 1.8rem;}
        .small-note {font-size: 0.88rem; color: #666;}
        .section-title {font-weight: 700; font-size: 1.15rem; margin-top: 0.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# FILE NAMES
# =========================================================

FILES = {
    "diesel_main": [
        "diesel_df_with_corrections (2).csv",
        "diesel_df_with_corrections.csv",
    ],
    "diesel_tl": [
        "diesel_df_with_corrections_TL (2).csv",
        "diesel_df_with_corrections_TL.csv",
    ],
    "gasoline_main": [
        "gasoline_df_with_corrections.csv",
    ],
    "gasoline_tl": [
        "gasoline_df_with_corrections_TL (2).csv",
        "gasoline_df_with_corrections_TL.csv",
    ],
    "roda2_main": [
        "roda2_df_with_corrections (2).csv",
        "roda2_df_with_corrections.csv",
    ],
    "roda2_tl": [
        "roda2_df_with_corrections_TL (2).csv",
        "roda2_df_with_corrections_TL.csv",
    ],
}


def resolve_file(candidates):
    for name in candidates:
        p = BASE_DIR / name
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def read_csv_clean(path_str):
    df = pd.read_csv(path_str)
    df.columns = df.columns.astype(str).str.strip()
    # Buang kolom kosong hasil ekspor Excel
    df = df.loc[:, ~df.columns.str.startswith("Unnamed:")]
    return df


def parse_date_mixed(series):
    """Parser tanggal campuran YYYY-MM-DD dan DD/MM/YYYY."""
    s = series.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    mask_iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    if mask_iso.any():
        out.loc[mask_iso] = pd.to_datetime(s.loc[mask_iso], errors="coerce")
    if (~mask_iso).any():
        out.loc[~mask_iso] = pd.to_datetime(
            s.loc[~mask_iso], errors="coerce", dayfirst=True
        )
    return out


# =========================================================
# STANDARDISASI DATA UTAMA
# =========================================================


def prepare_diesel(df):
    x = df.copy()
    if "Tanggal" in x.columns:
        tanggal = parse_date_mixed(x["Tanggal"])
    else:
        tanggal = pd.Series(pd.NaT, index=x.index)

    status_col = "LULUS/TIIDAK" if "LULUS/TIIDAK" in x.columns else "LULUS/TIDAK"
    lulus_bool = x[status_col].astype(bool)

    out = pd.DataFrame(index=x.index)
    out["Kelompok"] = "Diesel"
    out["Bahan Bakar"] = "Diesel"
    out["Tanggal"] = tanggal
    out["Tahun Pembuatan"] = pd.to_numeric(x.get("Tahun Pembuatan"), errors="coerce")
    out["Usia Kendaraan"] = pd.to_numeric(x.get("Usia Kendaraan"), errors="coerce")
    out["Klasifikasi"] = x.get("Klasifikasi", pd.Series("-", index=x.index)).astype(str)
    out["Euro"] = x.get("Euro", pd.Series("-", index=x.index)).astype(str)
    out["Segmen Tahun"] = x.get("segmen_tahun", pd.Series("-", index=x.index)).astype(str)
    out["CO"] = np.nan
    out["HC"] = np.nan
    out["Opasitas"] = pd.to_numeric(x.get("Opasitas"), errors="coerce")
    out["Baku Mutu CO"] = np.nan
    out["Baku Mutu HC"] = np.nan
    out["Baku Mutu Opasitas"] = pd.to_numeric(x.get("Baku Mutu Opasitas"), errors="coerce")
    out["CO_ratio"] = np.nan
    out["HC_ratio"] = np.nan
    out["Opasitas_ratio"] = pd.to_numeric(x.get("Opasitas_ratio"), errors="coerce")
    out["Rasio Emisi"] = out["Opasitas_ratio"]
    out["Parameter Dominan"] = "Opasitas"
    out["Alpha"] = pd.to_numeric(x.get("alpha"), errors="coerce")
    out["Status"] = np.where(lulus_bool, "LULUS", "TIDAK LULUS")
    out["Target"] = (~lulus_bool).astype(int)
    return out



def prepare_gasoline(df):
    x = df.copy()
    tanggal = parse_date_mixed(x["Tanggal"]) if "Tanggal" in x.columns else pd.Series(pd.NaT, index=x.index)

    co_ratio = pd.to_numeric(x.get("CO_ratio"), errors="coerce")
    hc_ratio = pd.to_numeric(x.get("HC_ratio"), errors="coerce")
    valid = co_ratio.notna() & hc_ratio.notna()

    # Status diturunkan langsung dari baku mutu: lulus jika CO dan HC <= 1 kali baku mutu
    tidak_lulus = (co_ratio > 1) | (hc_ratio > 1)

    out = pd.DataFrame(index=x.index)
    out["Kelompok"] = "Bensin Roda 4"
    out["Bahan Bakar"] = "Bensin"
    out["Tanggal"] = tanggal
    out["Tahun Pembuatan"] = pd.to_numeric(x.get("Tahun Pembuatan"), errors="coerce")
    out["Usia Kendaraan"] = pd.to_numeric(x.get("Usia Kendaraan"), errors="coerce")
    out["Klasifikasi"] = x.get("Klasifikasi", pd.Series("-", index=x.index)).astype(str)
    out["Euro"] = x.get("Euro", pd.Series("-", index=x.index)).astype(str)
    out["Segmen Tahun"] = x.get("segmen_tahun", pd.Series("-", index=x.index)).astype(str)
    out["CO"] = pd.to_numeric(x.get("CO"), errors="coerce")
    out["HC"] = pd.to_numeric(x.get("HC"), errors="coerce")
    out["Opasitas"] = np.nan
    out["Baku Mutu CO"] = pd.to_numeric(x.get("Baku Mutu CO"), errors="coerce")
    out["Baku Mutu HC"] = pd.to_numeric(x.get("Baku Mutu HC"), errors="coerce")
    out["Baku Mutu Opasitas"] = np.nan
    out["CO_ratio"] = co_ratio
    out["HC_ratio"] = hc_ratio
    out["Opasitas_ratio"] = np.nan
    out["Rasio Emisi"] = pd.concat([co_ratio, hc_ratio], axis=1).max(axis=1)
    out["Parameter Dominan"] = np.where(co_ratio >= hc_ratio, "CO", "HC")
    out["Alpha"] = pd.to_numeric(x.get("alpha"), errors="coerce")
    out["Status"] = np.where(tidak_lulus, "TIDAK LULUS", "LULUS")
    out["Target"] = tidak_lulus.astype(int)

    # Buang baris yang statusnya tidak dapat dihitung karena ratio hilang
    return out.loc[valid].copy()



def prepare_roda2(df):
    x = df.copy()
    tanggal_col = "Tanggal" if "Tanggal" in x.columns else None
    tanggal = parse_date_mixed(x[tanggal_col]) if tanggal_col else pd.Series(pd.NaT, index=x.index)

    status_col = "LULUS/TIDAK"
    lulus_bool = x[status_col].astype(bool)

    co_ratio = pd.to_numeric(x.get("CO_ratio"), errors="coerce")
    hc_ratio = pd.to_numeric(x.get("HC_ratio"), errors="coerce")

    out = pd.DataFrame(index=x.index)
    out["Kelompok"] = "Roda 2"
    out["Bahan Bakar"] = "Bensin"
    out["Tanggal"] = tanggal
    out["Tahun Pembuatan"] = pd.to_numeric(x.get("Tahun Pembuatan"), errors="coerce")
    out["Usia Kendaraan"] = pd.to_numeric(x.get("Usia Kendaraan"), errors="coerce")
    out["Klasifikasi"] = "A"
    out["Euro"] = x.get("Euro", pd.Series("-", index=x.index)).astype(str)
    out["Segmen Tahun"] = x.get("segmen_tahun", pd.Series("-", index=x.index)).astype(str)
    out["CO"] = pd.to_numeric(x.get("CO"), errors="coerce")
    out["HC"] = pd.to_numeric(x.get("HC"), errors="coerce")
    out["Opasitas"] = np.nan
    out["Baku Mutu CO"] = pd.to_numeric(x.get("Baku Mutu CO"), errors="coerce")
    out["Baku Mutu HC"] = pd.to_numeric(x.get("Baku Mutu HC"), errors="coerce")
    out["Baku Mutu Opasitas"] = np.nan
    out["CO_ratio"] = co_ratio
    out["HC_ratio"] = hc_ratio
    out["Opasitas_ratio"] = np.nan
    out["Rasio Emisi"] = pd.concat([co_ratio, hc_ratio], axis=1).max(axis=1)
    out["Parameter Dominan"] = np.where(co_ratio >= hc_ratio, "CO", "HC")
    out["Alpha"] = pd.to_numeric(x.get("alpha"), errors="coerce")
    out["Status"] = np.where(lulus_bool, "LULUS", "TIDAK LULUS")
    out["Target"] = (~lulus_bool).astype(int)
    return out


@st.cache_data(show_spinner="Memuat dan menyiapkan dataset...")
def load_all_data():
    resolved = {k: resolve_file(v) for k, v in FILES.items()}
    missing = [k for k, p in resolved.items() if p is None]
    if missing:
        raise FileNotFoundError(
            "File berikut tidak ditemukan di folder aplikasi: " + ", ".join(missing)
        )

    diesel_raw = read_csv_clean(str(resolved["diesel_main"]))
    gasoline_raw = read_csv_clean(str(resolved["gasoline_main"]))
    roda2_raw = read_csv_clean(str(resolved["roda2_main"]))

    diesel_tl = read_csv_clean(str(resolved["diesel_tl"]))
    gasoline_tl = read_csv_clean(str(resolved["gasoline_tl"]))
    roda2_tl = read_csv_clean(str(resolved["roda2_tl"]))

    diesel = prepare_diesel(diesel_raw)
    gasoline = prepare_gasoline(gasoline_raw)
    roda2 = prepare_roda2(roda2_raw)

    combined = pd.concat([diesel, gasoline, roda2], ignore_index=True)
    combined["Tahun Uji"] = combined["Tanggal"].dt.year
    combined["Bulan Uji"] = combined["Tanggal"].dt.to_period("M").astype(str)

    return combined, diesel_tl, gasoline_tl, roda2_tl


try:
    data, diesel_tl, gasoline_tl, roda2_tl = load_all_data()
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

# =========================================================
# UTILITIES
# =========================================================


def pct(x):
    return f"{x * 100:.2f}%"


def wilson_interval(k, n, z=1.96):
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return max(0, centre - margin), min(1, centre + margin)


def segment_from_year(group, year):
    if group == "Diesel":
        if year < 2010:
            return "<2010"
        elif year <= 2021:
            return "2010-2021"
        return ">2021"
    elif group == "Bensin Roda 4":
        if year < 2007:
            return "<2007"
        elif year <= 2018:
            return "2007-2018"
        return ">2018"
    else:
        if year < 2010:
            return "<2010"
        elif year <= 2016:
            return "2010-2016"
        return ">2016"


# =========================================================
# MODEL TRAINING (TANPA DATA LEAKAGE)
# =========================================================
# Fitur model sengaja tidak menggunakan CO, HC, Opasitas, maupun rasio emisi,
# karena variabel tersebut secara langsung menentukan status lulus/tidak lulus.

MODEL_FEATURES = [
    "Usia Kendaraan",
    "Kelompok",
    "Klasifikasi",
    "Euro",
    "Segmen Tahun",
]
NUM_FEATURES = ["Usia Kendaraan"]
CAT_FEATURES = ["Kelompok", "Klasifikasi", "Euro", "Segmen Tahun"]


@st.cache_resource(show_spinner="Melatih Logistic Regression dan Random Forest...")
def train_models():
    model_df = data[MODEL_FEATURES + ["Target"]].dropna().copy()

    X = model_df[MODEL_FEATURES]
    y = model_df["Target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_FEATURES,
            ),
        ]
    )

    logistic = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("prep", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=12,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    logistic.fit(X_train, y_train)
    random_forest.fit(X_train, y_train)

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
        "X_test": X_test,
        "y_test": y_test,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }


models = train_models()

# =========================================================
# SIDEBAR FILTER
# =========================================================

st.sidebar.header("🔎 Filter Data")

kelompok_options = ["Semua"] + sorted(data["Kelompok"].dropna().unique().tolist())
kelompok_filter = st.sidebar.selectbox("Kelompok Kendaraan", kelompok_options)

filter_base = data.copy()
if kelompok_filter != "Semua":
    filter_base = filter_base[filter_base["Kelompok"] == kelompok_filter]

kelas_options = ["Semua"] + sorted(filter_base["Klasifikasi"].dropna().astype(str).unique().tolist())
kelas_filter = st.sidebar.selectbox("Klasifikasi", kelas_options)

if kelas_filter != "Semua":
    filter_base = filter_base[filter_base["Klasifikasi"].astype(str) == kelas_filter]

segmen_options = ["Semua"] + sorted(filter_base["Segmen Tahun"].dropna().astype(str).unique().tolist())
segmen_filter = st.sidebar.selectbox("Segmen Tahun", segmen_options)

if segmen_filter != "Semua":
    filter_base = filter_base[filter_base["Segmen Tahun"].astype(str) == segmen_filter]

euro_options = ["Semua"] + sorted(filter_base["Euro"].dropna().astype(str).unique().tolist())
euro_filter = st.sidebar.selectbox("Standar Euro", euro_options)

if euro_filter != "Semua":
    filter_base = filter_base[filter_base["Euro"].astype(str) == euro_filter]

status_filter = st.sidebar.multiselect(
    "Status Kepatuhan",
    ["LULUS", "TIDAK LULUS"],
    default=["LULUS", "TIDAK LULUS"],
)

filtered = filter_base[filter_base["Status"].isin(status_filter)].copy()

# Filter tanggal opsional
valid_dates = filtered["Tanggal"].dropna()
use_date_filter = st.sidebar.checkbox("Gunakan filter tanggal", value=False)
if use_date_filter and not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    date_range = st.sidebar.date_input(
        "Rentang Tanggal Uji",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["Tanggal"].isna()
            | (
                (filtered["Tanggal"].dt.date >= start_date)
                & (filtered["Tanggal"].dt.date <= end_date)
            )
        ].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Model prediksi menggunakan karakteristik kendaraan dan tidak menggunakan CO, HC, opasitas, atau rasio emisi sebagai fitur, untuk menghindari data leakage."
)

# =========================================================
# HEADER
# =========================================================

st.title("📊 Dashboard Analisis & Prediksi Kepatuhan Emisi Kendaraan")
st.caption(
    "Analisis persentase kendaraan yang tidak memenuhi baku mutu, estimasi jumlah kendaraan yang berpotensi terkena tambahan pajak, visualisasi data, dan model statistik."
)

if filtered.empty:
    st.warning("Tidak ada data yang sesuai dengan kombinasi filter yang dipilih.")
    st.stop()

# =========================================================
# KPI GLOBAL / FILTERED
# =========================================================

n_total = len(filtered)
n_tl = int(filtered["Target"].sum())
n_lulus = n_total - n_tl
rate_tl = n_tl / n_total if n_total else 0
ci_low, ci_high = wilson_interval(n_tl, n_total)

# Prediksi probabilistik pada data terfilter menggunakan Logistic Regression
pred_input = filtered[MODEL_FEATURES].dropna().copy()
if len(pred_input) > 0:
    pred_probs = models["Logistic Regression"].predict_proba(pred_input)[:, 1]
    pred_rate = float(np.mean(pred_probs))
    pred_count = float(np.sum(pred_probs))
else:
    pred_rate = np.nan
    pred_count = np.nan

# =========================================================
# TABS
# =========================================================

tab_summary, tab_viz, tab_stats, tab_model, tab_data = st.tabs(
    [
        "📌 Ringkasan",
        "📈 Visualisasi Data",
        "🔬 Analisis Statistik",
        "🤖 Model & Prediksi",
        "📋 Data",
    ]
)

# =========================================================
# TAB 1 - RINGKASAN
# =========================================================

with tab_summary:
    st.subheader("Ringkasan Kepatuhan Emisi")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Kendaraan", f"{n_total:,}")
    c2.metric("Lulus", f"{n_lulus:,}")
    c3.metric("Tidak Lulus", f"{n_tl:,}")
    c4.metric("% Tidak Lulus", f"{rate_tl * 100:.2f}%")
    c5.metric(
        "Prediksi % Tidak Lulus",
        "-" if np.isnan(pred_rate) else f"{pred_rate * 100:.2f}%",
    )

    st.caption(
        f"Interval kepercayaan 95% untuk proporsi aktual tidak lulus: "
        f"{ci_low * 100:.2f}% – {ci_high * 100:.2f}%. "
        "Prediksi menggunakan rerata probabilitas Logistic Regression pada data terfilter."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        status_counts = (
            filtered["Status"].value_counts().rename_axis("Status").reset_index(name="Jumlah")
        )
        fig_status = px.bar(
            status_counts,
            x="Status",
            y="Jumlah",
            text="Jumlah",
            title="Jumlah Kendaraan Lulus dan Tidak Lulus",
        )
        fig_status.update_traces(textposition="outside")
        fig_status.update_layout(height=390)
        st.plotly_chart(fig_status, use_container_width=True)

    with col_b:
        grp = (
            filtered.groupby("Kelompok", dropna=False)["Target"]
            .agg(["count", "sum"])
            .reset_index()
        )
        grp["Persentase Tidak Lulus"] = grp["sum"] / grp["count"] * 100
        fig_group = px.bar(
            grp,
            x="Kelompok",
            y="Persentase Tidak Lulus",
            text=grp["Persentase Tidak Lulus"].map(lambda x: f"{x:.2f}%"),
            title="Persentase Tidak Lulus menurut Kelompok Kendaraan",
        )
        fig_group.update_traces(textposition="outside")
        fig_group.update_layout(height=390, yaxis_title="Tidak Lulus (%)")
        st.plotly_chart(fig_group, use_container_width=True)

    st.markdown("### Estimasi Jumlah Kendaraan yang Berpotensi Terkena Tambahan Pajak")
    col1, col2 = st.columns([1, 2])
    with col1:
        population = st.number_input(
            "Jumlah populasi kendaraan yang ingin diestimasi",
            min_value=1,
            value=max(10000, n_total),
            step=1000,
            key="summary_population",
        )
    with col2:
        if np.isnan(pred_rate):
            st.info("Prediksi belum tersedia karena fitur model pada data terfilter tidak lengkap.")
        else:
            expected = int(round(population * pred_rate))
            st.metric(
                "Estimasi Tidak Lulus / Berpotensi Terkena Tambahan Pajak",
                f"{expected:,} kendaraan",
                f"{pred_rate * 100:.2f}% dari populasi",
            )
            st.caption(
                "Angka ini adalah estimasi jumlah kendaraan yang tidak memenuhi baku mutu. "
                "Nilai tambahan pajak dalam Rupiah belum dihitung karena dataset ini tidak memuat NJKB dan tarif pajak setiap kendaraan."
            )

    monthly = filtered.dropna(subset=["Tanggal"]).copy()
    if not monthly.empty:
        monthly["Bulan"] = monthly["Tanggal"].dt.to_period("M").dt.to_timestamp()
        trend = (
            monthly.groupby("Bulan")["Target"]
            .agg(["count", "sum"])
            .reset_index()
        )
        trend["Tidak Lulus (%)"] = trend["sum"] / trend["count"] * 100
        fig_trend = px.line(
            trend,
            x="Bulan",
            y="Tidak Lulus (%)",
            markers=True,
            title="Tren Persentase Kendaraan Tidak Lulus per Bulan",
        )
        fig_trend.update_layout(height=420)
        st.plotly_chart(fig_trend, use_container_width=True)

# =========================================================
# TAB 2 - VISUALISASI DATA
# =========================================================

with tab_viz:
    st.subheader("Visualisasi Data Interaktif")

    v1, v2 = st.columns(2)

    with v1:
        cls = (
            filtered.groupby(["Kelompok", "Klasifikasi"])["Target"]
            .agg(["count", "sum"])
            .reset_index()
        )
        cls["Tidak Lulus (%)"] = cls["sum"] / cls["count"] * 100
        fig_cls = px.bar(
            cls,
            x="Klasifikasi",
            y="Tidak Lulus (%)",
            color="Kelompok",
            barmode="group",
            hover_data={"count": True, "sum": True},
            title="Persentase Tidak Lulus berdasarkan Klasifikasi",
        )
        fig_cls.update_layout(height=420)
        st.plotly_chart(fig_cls, use_container_width=True)

    with v2:
        seg = (
            filtered.groupby(["Kelompok", "Segmen Tahun"])["Target"]
            .agg(["count", "sum"])
            .reset_index()
        )
        seg["Tidak Lulus (%)"] = seg["sum"] / seg["count"] * 100
        fig_seg = px.bar(
            seg,
            x="Segmen Tahun",
            y="Tidak Lulus (%)",
            color="Kelompok",
            barmode="group",
            hover_data={"count": True, "sum": True},
            title="Persentase Tidak Lulus berdasarkan Segmen Tahun",
        )
        fig_seg.update_layout(height=420)
        st.plotly_chart(fig_seg, use_container_width=True)

    v3, v4 = st.columns(2)

    with v3:
        age_plot = filtered.dropna(subset=["Usia Kendaraan"]).copy()
        fig_age = px.histogram(
            age_plot,
            x="Usia Kendaraan",
            color="Status",
            nbins=35,
            barmode="overlay",
            opacity=0.65,
            title="Distribusi Usia Kendaraan menurut Status",
        )
        fig_age.update_layout(height=420)
        st.plotly_chart(fig_age, use_container_width=True)

    with v4:
        ratio_plot = filtered.dropna(subset=["Rasio Emisi"]).copy()
        # Hindari outlier ekstrem membuat grafik sulit dibaca
        if len(ratio_plot) > 20:
            cap = ratio_plot["Rasio Emisi"].quantile(0.99)
            ratio_plot = ratio_plot[ratio_plot["Rasio Emisi"] <= cap]
        fig_ratio = px.box(
            ratio_plot,
            x="Kelompok",
            y="Rasio Emisi",
            color="Status",
            points=False,
            title="Distribusi Rasio Emisi (dibatasi visual sampai P99)",
        )
        fig_ratio.add_hline(y=1, line_dash="dash", annotation_text="Baku Mutu = 1")
        fig_ratio.update_layout(height=420)
        st.plotly_chart(fig_ratio, use_container_width=True)

    st.markdown("### Heatmap Risiko Tidak Lulus")
    heat = (
        filtered.groupby(["Klasifikasi", "Segmen Tahun"])["Target"]
        .agg(["count", "sum"])
        .reset_index()
    )
    heat["rate"] = np.where(heat["count"] > 0, heat["sum"] / heat["count"] * 100, np.nan)
    pivot_heat = heat.pivot(index="Klasifikasi", columns="Segmen Tahun", values="rate")
    fig_heat = px.imshow(
        pivot_heat,
        text_auto=".1f",
        aspect="auto",
        labels=dict(x="Segmen Tahun", y="Klasifikasi", color="Tidak Lulus (%)"),
        title="Heatmap Persentase Tidak Lulus: Klasifikasi × Segmen Tahun",
    )
    fig_heat.update_layout(height=450)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Eksplorasi Variabel Emisi")
    available_vars = [
        c for c in ["CO", "HC", "Opasitas", "Rasio Emisi", "Usia Kendaraan"]
        if filtered[c].notna().any()
    ]
    x_var = st.selectbox("Variabel untuk histogram", available_vars, key="hist_var")
    tmp = filtered.dropna(subset=[x_var]).copy()
    if len(tmp) > 20:
        q99 = tmp[x_var].quantile(0.99)
        tmp = tmp[tmp[x_var] <= q99]
    fig_hist = px.histogram(
        tmp,
        x=x_var,
        color="Status",
        nbins=50,
        barmode="overlay",
        opacity=0.65,
        title=f"Distribusi {x_var} menurut Status (visual dibatasi sampai P99)",
    )
    fig_hist.update_layout(height=420)
    st.plotly_chart(fig_hist, use_container_width=True)

# =========================================================
# TAB 3 - ANALISIS STATISTIK
# =========================================================

with tab_stats:
    st.subheader("Analisis Statistik")

    st.markdown("### Statistik Deskriptif")
    numeric_cols = [
        c
        for c in [
            "Tahun Pembuatan",
            "Usia Kendaraan",
            "CO",
            "HC",
            "Opasitas",
            "Rasio Emisi",
            "Alpha",
        ]
        if filtered[c].notna().any()
    ]

    desc = filtered[numeric_cols].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    desc = desc.rename(
        columns={
            "count": "N",
            "mean": "Mean",
            "std": "Std",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "95%": "P95",
            "max": "Max",
        }
    )
    st.dataframe(desc.round(4), use_container_width=True)

    st.markdown("### Uji Hubungan Kategorik dengan Status")
    cat_test = st.selectbox(
        "Pilih variabel kategorik",
        ["Kelompok", "Klasifikasi", "Segmen Tahun", "Euro"],
        key="chi_var",
    )
    ct = pd.crosstab(filtered[cat_test], filtered["Status"])
    st.dataframe(ct, use_container_width=True)
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, pval, dof, _ = chi2_contingency(ct)
        s1, s2, s3 = st.columns(3)
        s1.metric("Chi-square", f"{chi2:.3f}")
        s2.metric("df", f"{dof}")
        s3.metric("p-value", f"{pval:.3e}")
        if pval < 0.05:
            st.success(
                f"Terdapat hubungan yang signifikan secara statistik antara {cat_test} dan status kepatuhan (p < 0,05)."
            )
        else:
            st.info(
                f"Belum terdapat bukti hubungan yang signifikan antara {cat_test} dan status kepatuhan pada data terfilter (p ≥ 0,05)."
            )
    else:
        st.info("Uji Chi-square memerlukan minimal dua kategori dan dua status.")

    st.markdown("### Perbandingan Usia Kendaraan: Lulus vs Tidak Lulus")
    age_l = filtered.loc[filtered["Target"] == 0, "Usia Kendaraan"].dropna()
    age_tl = filtered.loc[filtered["Target"] == 1, "Usia Kendaraan"].dropna()
    if len(age_l) > 1 and len(age_tl) > 1:
        u_stat, p_u = mannwhitneyu(age_l, age_tl, alternative="two-sided")
        a1, a2, a3 = st.columns(3)
        a1.metric("Median Usia - Lulus", f"{age_l.median():.1f} tahun")
        a2.metric("Median Usia - Tidak Lulus", f"{age_tl.median():.1f} tahun")
        a3.metric("Mann-Whitney p-value", f"{p_u:.3e}")
    else:
        st.info("Data usia untuk kedua status belum cukup untuk uji Mann-Whitney.")

    st.markdown("### Korelasi Usia Kendaraan dengan Rasio Emisi")
    corr_df = filtered[["Usia Kendaraan", "Rasio Emisi"]].dropna()
    if len(corr_df) >= 3:
        rho, p_s = spearmanr(corr_df["Usia Kendaraan"], corr_df["Rasio Emisi"])
        k1, k2 = st.columns(2)
        k1.metric("Spearman ρ", f"{rho:.3f}")
        k2.metric("p-value", f"{p_s:.3e}")
    else:
        st.info("Data belum cukup untuk menghitung korelasi Spearman.")

    st.markdown("### Odds Ratio dari Logistic Regression")
    logistic = models["Logistic Regression"]
    prep = logistic.named_steps["prep"]
    cat_names = prep.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
    feature_names = NUM_FEATURES + list(cat_names)
    coef = logistic.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "Fitur": feature_names,
            "Koefisien": coef,
            "Odds Ratio": np.exp(coef),
        }
    )
    coef_df["|Koefisien|"] = coef_df["Koefisien"].abs()
    coef_df = coef_df.sort_values("|Koefisien|", ascending=False).drop(columns="|Koefisien|")
    st.dataframe(coef_df.round(4), use_container_width=True, height=430)
    st.caption(
        "Untuk Usia Kendaraan, odds ratio merepresentasikan perubahan odds tidak lulus untuk setiap kenaikan 1 tahun usia. "
        "Untuk variabel kategorik, interpretasi harus dibandingkan dengan kategori referensi yang di-encode oleh model."
    )

    st.markdown("### Referensi Dataset Tidak Lulus (TL)")
    ref_choice = st.selectbox(
        "Pilih dataset TL",
        ["Diesel - ringkasan Q1/Median/Q3/P95/Max", "Bensin Roda 4 - data TL", "Roda 2 - data TL"],
    )
    if ref_choice.startswith("Diesel"):
        st.dataframe(diesel_tl, use_container_width=True)
    elif ref_choice.startswith("Bensin"):
        st.write(f"Jumlah data TL: **{len(gasoline_tl):,}**")
        cols = [c for c in ["CO_ratio", "HC_ratio", "Usia Kendaraan", "Klasifikasi", "segmen_tahun"] if c in gasoline_tl.columns]
        st.dataframe(gasoline_tl[cols].head(1000), use_container_width=True, height=400)
    else:
        st.write(f"Jumlah data TL: **{len(roda2_tl):,}**")
        cols = [c for c in ["CO_ratio", "HC_ratio", "Usia Kendaraan", "Euro", "segmen_tahun"] if c in roda2_tl.columns]
        st.dataframe(roda2_tl[cols].head(1000), use_container_width=True, height=400)

# =========================================================
# TAB 4 - MODEL & PREDIKSI
# =========================================================

with tab_model:
    st.subheader("Model Prediksi Tidak Memenuhi Baku Mutu")

    st.info(
        "Model menggunakan Usia Kendaraan, Kelompok, Klasifikasi, Euro, dan Segmen Tahun. "
        "CO, HC, Opasitas, dan rasio emisi sengaja tidak digunakan sebagai input agar tidak terjadi data leakage."
    )

    model_name = st.selectbox(
        "Pilih model",
        ["Logistic Regression", "Random Forest"],
        index=0,
    )
    model = models[model_name]
    X_test = models["X_test"]
    y_test = models["y_test"]
    prob_test = model.predict_proba(X_test)[:, 1]

    threshold = st.slider(
        "Threshold klasifikasi untuk evaluasi",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        help="Karena kelas Tidak Lulus relatif jarang, threshold 0,50 dapat terlalu konservatif. Persentase prediksi utama tetap dihitung dari probabilitas, bukan jumlah hasil threshold.",
    )
    pred_test = (prob_test >= threshold).astype(int)

    try:
        auc = roc_auc_score(y_test, prob_test)
    except Exception:
        auc = np.nan

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("ROC-AUC", "-" if np.isnan(auc) else f"{auc:.3f}")
    m2.metric("Accuracy", f"{accuracy_score(y_test, pred_test):.3f}")
    m3.metric("Precision", f"{precision_score(y_test, pred_test, zero_division=0):.3f}")
    m4.metric("Recall", f"{recall_score(y_test, pred_test, zero_division=0):.3f}")
    m5.metric("F1-score", f"{f1_score(y_test, pred_test, zero_division=0):.3f}")
    m6.metric("Brier Score", f"{brier_score_loss(y_test, prob_test):.4f}")

    cm = confusion_matrix(y_test, pred_test)
    cm_df = pd.DataFrame(
        cm,
        index=["Aktual Lulus", "Aktual Tidak Lulus"],
        columns=["Prediksi Lulus", "Prediksi Tidak Lulus"],
    )

    mc1, mc2 = st.columns(2)
    with mc1:
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            aspect="auto",
            title=f"Confusion Matrix ({model_name}, threshold={threshold:.2f})",
        )
        fig_cm.update_layout(height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

    with mc2:
        fpr, tpr, _ = roc_curve(y_test, prob_test)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{model_name} AUC={auc:.3f}"))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Acak", line=dict(dash="dash")))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=420,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("### Prediksi Probabilitas menurut Kelompok pada Test Set")
    group_eval = X_test.copy()
    group_eval["Aktual"] = y_test.values
    group_eval["Probabilitas"] = prob_test
    group_summary = (
        group_eval.groupby("Kelompok")
        .agg(
            Jumlah=("Aktual", "size"),
            Aktual_Tidak_Lulus=("Aktual", "mean"),
            Prediksi_Tidak_Lulus=("Probabilitas", "mean"),
        )
        .reset_index()
    )
    group_summary["Aktual Tidak Lulus (%)"] = group_summary["Aktual_Tidak_Lulus"] * 100
    group_summary["Prediksi Tidak Lulus (%)"] = group_summary["Prediksi_Tidak_Lulus"] * 100
    st.dataframe(
        group_summary[["Kelompok", "Jumlah", "Aktual Tidak Lulus (%)", "Prediksi Tidak Lulus (%)"]].round(3),
        use_container_width=True,
    )

    if model_name == "Random Forest":
        st.markdown("### Feature Importance")
        prep_rf = model.named_steps["prep"]
        cat_names_rf = prep_rf.named_transformers_["cat"].get_feature_names_out(CAT_FEATURES)
        names_rf = NUM_FEATURES + list(cat_names_rf)
        imp = model.named_steps["model"].feature_importances_
        imp_df = pd.DataFrame({"Fitur": names_rf, "Importance": imp}).sort_values("Importance", ascending=False).head(20)
        fig_imp = px.bar(
            imp_df.sort_values("Importance"),
            x="Importance",
            y="Fitur",
            orientation="h",
            title="20 Feature Importance Terbesar",
        )
        fig_imp.update_layout(height=520)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("### Simulasi Prediksi Populasi Kendaraan")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_group = st.selectbox(
            "Kelompok Kendaraan",
            ["Diesel", "Bensin Roda 4", "Roda 2"],
            key="sim_group",
        )

    if sim_group == "Diesel":
        class_choices = ["C", "D", "E", "F", "G"]
        euro_choices = ["NON_EURO", "EURO2", "EURO4"]
    elif sim_group == "Bensin Roda 4":
        class_choices = ["B", "C", "D"]
        euro_choices = ["NON_EURO", "EURO2", "EURO4"]
    else:
        class_choices = ["A"]
        euro_choices = ["NON_EURO", "EURO2", "EURO3"]

    with s2:
        sim_class = st.selectbox("Klasifikasi", class_choices, key="sim_class")
    with s3:
        sim_euro = st.selectbox("Euro", euro_choices, key="sim_euro")

    s4, s5, s6 = st.columns(3)
    with s4:
        current_year = pd.Timestamp.now().year
        sim_year = st.number_input(
            "Tahun Pembuatan",
            min_value=1980,
            max_value=current_year,
            value=2015,
            step=1,
            key="sim_year",
        )
    with s5:
        sim_population = st.number_input(
            "Jumlah Populasi",
            min_value=1,
            value=10000,
            step=1000,
            key="sim_pop",
        )
    with s6:
        sim_model_name = st.selectbox(
            "Model untuk simulasi",
            ["Logistic Regression", "Random Forest"],
            index=0,
            key="sim_model",
        )

    sim_age = max(0, current_year - int(sim_year))
    sim_segment = segment_from_year(sim_group, int(sim_year))

    sim_df = pd.DataFrame(
        [
            {
                "Usia Kendaraan": sim_age,
                "Kelompok": sim_group,
                "Klasifikasi": sim_class,
                "Euro": sim_euro,
                "Segmen Tahun": sim_segment,
            }
        ]
    )

    sim_prob = float(models[sim_model_name].predict_proba(sim_df)[0, 1])
    sim_expected = int(round(sim_population * sim_prob))

    p1, p2, p3 = st.columns(3)
    p1.metric("Usia Kendaraan", f"{sim_age} tahun")
    p2.metric("Probabilitas Tidak Lulus", f"{sim_prob * 100:.2f}%")
    p3.metric("Estimasi Jumlah Tidak Lulus", f"{sim_expected:,} kendaraan")

    st.caption(
        f"Segmen tahun yang digunakan model: {sim_segment}. Estimasi jumlah = populasi × probabilitas tidak lulus. "
        "Prediksi ini merupakan estimasi statistik berdasarkan pola dataset historis dan bukan keputusan pajak individual."
    )

# =========================================================
# TAB 5 - DATA
# =========================================================

with tab_data:
    st.subheader("Data Hasil Filter")

    display_cols = [
        "Kelompok",
        "Tanggal",
        "Tahun Pembuatan",
        "Usia Kendaraan",
        "Klasifikasi",
        "Euro",
        "Segmen Tahun",
        "CO",
        "HC",
        "Opasitas",
        "Rasio Emisi",
        "Parameter Dominan",
        "Status",
    ]

    st.write(f"Jumlah baris ditampilkan: **{len(filtered):,}**")
    st.dataframe(filtered[display_cols], use_container_width=True, height=560)

    csv_bytes = filtered[display_cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "⬇️ Download data hasil filter (CSV)",
        data=csv_bytes,
        file_name="hasil_filter_dashboard_emisi.csv",
        mime="text/csv",
    )

    st.markdown("### Ringkasan Sumber Dataset")
    source_summary = pd.DataFrame(
        {
            "Dataset": [
                "Diesel utama",
                "Diesel TL",
                "Bensin roda 4 utama (valid)",
                "Bensin roda 4 TL",
                "Roda 2 utama",
                "Roda 2 TL",
            ],
            "Jumlah Baris": [
                int((data["Kelompok"] == "Diesel").sum()),
                len(diesel_tl),
                int((data["Kelompok"] == "Bensin Roda 4").sum()),
                len(gasoline_tl),
                int((data["Kelompok"] == "Roda 2").sum()),
                len(roda2_tl),
            ],
        }
    )
    st.dataframe(source_summary, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "Catatan metodologis: status aktual ditentukan dari kolom status yang tersedia pada dataset Diesel dan Roda 2. "
    "Untuk dataset Bensin Roda 4, status dihitung dari CO_ratio dan HC_ratio: kendaraan dianggap tidak lulus apabila salah satu rasio > 1. "
    "Model prediksi menggunakan karakteristik kendaraan saja agar tidak terjadi kebocoran target (data leakage)."
)
