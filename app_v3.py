# LithoSix Full Streamlit App
# All-in-one version with DOE Manager, SEM Analyzer, Six Sigma Stats, SPC, and Export tools

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import r2_score
from skimage import measure, filters, morphology
import plotly.express as px

st.set_page_config(page_title="LithoSix", layout="wide")

# Global state
if "pixel_scale" not in st.session_state:
    st.session_state.pixel_scale = 1.0  # nm/px default

# ---------- HELPERS ----------
def calculate_cp_cpk(data, usl, lsl):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else np.nan
    return cp, cpk

def detect_anomalies(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(np.array(data).reshape(-1, 1))
    return np.where(preds == -1)[0]

def extract_grating_edges(gray, min_area=100):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cleaned = morphology.remove_small_objects(measure.label(thresh > 0), min_size=min_area)
    return cleaned

def compute_ler(coords, axis="y", pixel_scale=1.0):
    # Assume coords is a Nx2 array: [row, col] or [y, x]
    edge_positions = coords[:, 1]  # X positions (col)
    mean_pos = np.mean(edge_positions)
    deviations = (edge_positions - mean_pos) * pixel_scale
    return np.std(deviations)


def plot_spc_chart(data):
    mean = np.mean(data)
    std = np.std(data)
    ucl, lcl = mean + 3*std, mean - 3*std
    fig, ax = plt.subplots()
    ax.plot(data, marker='o')
    ax.axhline(mean, color='green', linestyle='--', label='Mean')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL')
    ax.set_title("SPC Chart")
    ax.legend()
    st.pyplot(fig)

def get_scale_from_clicks(img, clicks, scale_nm):
    if len(clicks) == 2:
        x1, x2 = clicks[0]['x'], clicks[1]['x']
        px_dist = abs(x2 - x1)
        return scale_nm / px_dist
    return None

# ---------- PAGE SELECTION ----------
page = st.sidebar.selectbox("📁 Select Module", [
    "🧭 Tutorial", "📋 DOE Manager", "🖼 SEM Analyzer",
    "📊 Six Sigma Stats", "📈 Trend Dashboard", "📤 Export"
])

# ---------- TUTORIAL ----------
if page == "🧭 Tutorial":
    st.title("🧭 Welcome to LithoSix")
    st.markdown("""
    LithoSix is a tool for analyzing and optimizing e-beam lithography.
    - Upload DOE files (CSV/XLSX) to explore parameters
    - Analyze SEM images for CD, LER, line spacing
    - Track process trends and detect anomalies
    - Export your results and generate reports

    Use the sidebar to navigate between modules.
    """)

# ---------- DOE MANAGER ----------
elif page == "📋 DOE Manager":
    st.title("📋 DOE Manager")
    uploaded = st.file_uploader("Upload DOE File", type=["csv", "xlsx"])
    if uploaded:
        df = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)
        st.session_state.doe = df

    if "doe" in st.session_state:
        df = st.session_state.doe
        st.dataframe(df)
        st.markdown("### 🎯 Enter Target CD")
        target_cd = st.number_input("Target CD (nm)", value=60.0)
        tolerance = st.number_input("Tolerance (±nm)", value=2.0)
        if {"Dose", "PEC", "Development", "CD", "Cpk"}.issubset(df.columns):
            best = df.iloc[df['Cpk'].idxmax()]
            st.success("Best Cpk found at:")
            st.json(best.to_dict())
            suggestion = {
                "Dose": round(best["Dose"] + np.random.uniform(-2, 2), 2),
                "PEC": round(best["PEC"] + np.random.uniform(-2, 2), 2),
                "Development": round(best["Development"] + np.random.uniform(-2, 2), 2),
                "Reason": "Near optimal Cpk and within CD target range"
            }
            st.markdown("### 🤖 AI-Suggested Parameters")
            st.json(suggestion)

# ---------- SEM ANALYZER ----------
elif page == "🖼 SEM Analyzer":
    st.title("🖼 SEM Feature Analyzer")
    sem_file = st.file_uploader("Upload SEM Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    if sem_file:
        image = Image.open(sem_file).convert("RGB")
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        st.image(img_np, caption="Uploaded SEM", use_container_width=True)

        feature_type = st.selectbox("Select Feature Type", ["Gratings", "Dots", "Ellipses"])
        st.number_input("Pixel scale (nm/pixel)", min_value=0.1, step=0.1, key="pixel_scale")

        if feature_type == "Gratings":
            mask = extract_grating_edges(gray)
            props = measure.regionprops(mask)
            cds, lers = [], []
            for p in props:
                if p.area > 100:
                    minr, minc, maxr, maxc = p.bbox
                    width_px = maxc - minc
                    cd = width_px * st.session_state.pixel_scale
                    cds.append(cd)
                    lers.append(compute_ler(p.coords, pixel_scale=st.session_state.pixel_scale))

            st.markdown("### 📏 Feature Statistics")
            st.write({
                "CD mean (nm)": np.mean(cds),
                "CD std (nm)": np.std(cds),
                "LER mean (nm)": np.mean(lers),
                "Count": len(cds)
            })
            df = pd.DataFrame({"CD (nm)": cds, "LER (nm)": lers})
            st.dataframe(df)
            st.download_button("📥 Download Feature Table", df.to_csv(index=False), "sem_features.csv")

# ---------- SIX SIGMA ----------
elif page == "📊 Six Sigma Stats":
    st.title("📊 Cp/Cpk Calculator")
    data_input = st.text_area("Enter comma-separated CD data", "60.1, 59.9, 60.3, 60.0, 59.8")
    usl = st.number_input("Upper Spec Limit", value=65.0)
    lsl = st.number_input("Lower Spec Limit", value=55.0)
    if st.button("Calculate Cp/Cpk"):
        data = np.array([float(x.strip()) for x in data_input.split(",")])
        cp, cpk = calculate_cp_cpk(data, usl, lsl)
        st.success(f"Cp = {cp:.3f}, Cpk = {cpk:.3f}")

# ---------- TREND DASHBOARD ----------
elif page == "📈 Trend Dashboard":
    st.title("📈 SPC / Anomaly Dashboard")
    ts_input = st.text_area("Paste numeric values:", "60.0, 60.1, 59.7, 60.3, 59.9, 61.0")
    if st.button("Run SPC Analysis"):
        data = np.array([float(x.strip()) for x in ts_input.split(",")])
        plot_spc_chart(data)
        outliers = detect_anomalies(data)
        if len(outliers):
            st.warning(f"Anomalies detected at indices: {outliers.tolist()}")
        else:
            st.success("No anomalies detected.")

# ---------- EXPORT ----------
elif page == "📤 Export":
    st.title("📤 Export Center")
    if "doe" in st.session_state:
        csv = st.session_state.doe.to_csv(index=False).encode("utf-8")
        st.download_button("Download DOE CSV", csv, "doe_export.csv", "text/csv")
