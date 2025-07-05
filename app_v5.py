# --- Imports ---
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage import filters, measure, morphology
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from scipy import stats
from io import BytesIO
import math

# --- Helper Functions Shared ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()

# -------------------------------
# SEM Analysis Functions
# -------------------------------
def preprocess_image(image, blur_ksize=5, threshold_value=80, contrast_factor=1.2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    adjusted = cv2.convertScaleAbs(binary, alpha=contrast_factor, beta=0)
    return adjusted

def extract_grating_features(edges, scale):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    features = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        x, y, w, h = cv2.boundingRect(cnt)
        width = w * scale
        if len(cnt) >= 2:
            dx = cnt[-1][0][0] - cnt[0][0][0]
            dy = cnt[-1][0][1] - cnt[0][0][1]
            slope_angle = math.degrees(math.atan2(dy, dx))
        else: slope_angle = 0
        features.append({'CD (nm)': width, 'Slope (°)': slope_angle})
    df = pd.DataFrame(features)
    df['LER (nm)'] = df['CD (nm)'].diff().abs()
    df['LWR (nm)'] = df['CD (nm)'].rolling(3).std()
    return df

def extract_dot_ellipse_features(edges, scale):
    labeled = measure.label(edges)
    props = measure.regionprops(labeled)
    data = []
    for prop in props:
        if prop.area < 30: continue
        ecc = prop.eccentricity
        circ = (4 * math.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
        cd = np.mean(prop.major_axis_length + prop.minor_axis_length) / 2 * scale
        data.append({'CD (nm)': cd, 'Circularity': circ, 'Eccentricity': ecc})
    return pd.DataFrame(data)

def overlay_contours(img, edges):
    overlay = img.copy()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return overlay

def get_ai_score(df):
    if df.empty: return 0.0
    weights = {'CD (nm)': -0.2, 'LER (nm)': -0.3, 'Circularity': 0.4, 'Eccentricity': -0.1}
    score = 0
    for col in df.columns:
        if col in weights:
            norm = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-5)
            score += norm.mean() * weights[col]
    return np.clip(score + 0.5, 0, 1)

# -------------------------------
# DOE & Six Sigma Functions
# -------------------------------
def calculate_cp_cpk(data, usl, lsl):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else np.nan
    return cp, cpk

def anova_test(groups):
    f_val, p_val = stats.f_oneway(*groups)
    return f_val, p_val

def generate_synthetic_doe():
    dose = np.random.uniform(80, 120, 50)
    pec = np.random.uniform(5, 15, 50)
    dev = np.random.uniform(10, 30, 50)
    cd = 50 + 0.3 * dose - 0.5 * pec + 0.2 * dev + np.random.normal(0, 2, 50)
    df = pd.DataFrame({'Dose': dose, 'PEC': pec, 'Development': dev, 'CD': cd})
    return df

# -------------------------------
# Trend Analysis Functions
# -------------------------------
def plot_spc_chart(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    ucl, lcl = mean + 3*std, mean - 3*std
    fig, ax = plt.subplots()
    ax.plot(data, marker='o')
    ax.axhline(mean, color='green', linestyle='--', label='Mean')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL')
    ax.legend()
    ax.set_title("SPC Chart")
    st.pyplot(fig)

def detect_anomalies(data):
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(data.reshape(-1, 1))
    return np.where(preds == -1)[0]

def cusum(data, k=0.5, h=5):
    s_pos = np.zeros(len(data))
    s_neg = np.zeros(len(data))
    for i in range(1, len(data)):
        s_pos[i] = max(0, s_pos[i-1] + data[i] - k)
        s_neg[i] = min(0, s_neg[i-1] + data[i] + k)
        if s_pos[i] > h or abs(s_neg[i]) > h:
            return i
    return -1

# -------------------------------
# STREAMLIT APP UI
# -------------------------------
st.set_page_config(page_title="LithoSix", layout="wide")
st.title("🧪 LithoSix: Complete Lithography Assistant")

page = st.sidebar.radio("Select Module", ["SEM Analyzer", "DOE Manager", "Six Sigma Stats", "Trend Dashboard"])

# === SEM ANALYZER ===
if page == "SEM Analyzer":
    st.header("SEM Analyzer")
    st.sidebar.markdown("### SEM Preprocessing")
    scale = st.sidebar.number_input("nm per pixel", value=2.5)
    blur = st.sidebar.slider("Blur", 1, 11, 5, 2)
    threshold = st.sidebar.slider("Threshold", 30, 200, 80)
    contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.2)
    mode = st.sidebar.selectbox("Feature Type", ["Grating", "Dot", "Ellipse"])

    uploaded = st.file_uploader("Upload SEM image", type=['png','jpg','jpeg','tif','tiff'])
    if uploaded:
        img = Image.open(uploaded).convert('RGB')
        img = np.array(img)
        st.image(img, caption="Original Image", use_container_width=True)
        processed = preprocess_image(img, blur, threshold, contrast)
        st.image(processed, caption="Processed", use_column_widthh=True, clamp=True)
        overlay = overlay_contours(img, processed)
        st.image(overlay, caption="Edge Overlay", use_column_widthh=True)

        if mode == "Grating":
            df = extract_grating_features(processed, scale)
        else:
            df = extract_dot_ellipse_features(processed, scale)

        if not df.empty:
            st.subheader("🔬 Feature-Level Metrics")
            st.dataframe(df)
            st.subheader("📊 Summary")
            st.dataframe(df.describe().T.round(3))
            st.download_button("📤 Download as CSV", df.to_csv(index=False).encode(), "sem_features.csv", "text/csv")
            score = get_ai_score(df)
            st.markdown(f"### 🤖 AI Quality Score: `{score:.3f}`")
        else:
            st.warning("No valid features detected.")

# === DOE MANAGER ===
elif page == "DOE Manager":
    st.header("📋 DOE Manager")
    if "doe_history" not in st.session_state:
        st.session_state.doe_history = pd.DataFrame(columns=['Dose', 'PEC', 'Development', 'Cpk'])

    with st.form("add_doe_form"):
        dose = st.number_input("Exposure Dose (mJ/cm²)", 50.0, 200.0, 100.0)
        pec = st.number_input("Post Exposure Bake Temp (°C)", 50.0, 200.0, 90.0)
        dev = st.number_input("Development Time (sec)", 10.0, 120.0, 30.0)
        cpk = st.number_input("Cpk Value", 0.0, 2.0, 1.0)
        submitted = st.form_submit_button("Add DOE Entry")

    if submitted:
        new_row = {'Dose': dose, 'PEC': pec, 'Development': dev, 'Cpk': cpk}
        st.session_state.doe_history = pd.concat([st.session_state.doe_history, pd.DataFrame([new_row])], ignore_index=True)
        st.success("✅ DOE entry added!")

    if not st.session_state.doe_history.empty:
        st.subheader("🧾 DOE History")
        st.dataframe(st.session_state.doe_history)

        st.subheader("🤖 AI-Suggested Next DOE")

        top_row = st.session_state.doe_history.loc[st.session_state.doe_history['Cpk'].idxmax()]
        st.markdown(f"- Highest Cpk so far: **{top_row['Cpk']:.3f}**")

        # Suggest next DOE by perturbing top performer
        next_dose = np.clip(top_row['Dose'] + np.random.uniform(-3, 3), 50, 200)
        next_pec = np.clip(top_row['PEC'] + np.random.uniform(-3, 3), 50, 200)
        next_dev = np.clip(top_row['Development'] + np.random.uniform(-5, 5), 10, 120)

        suggestion = pd.DataFrame([{
            'Dose': round(next_dose, 2),
            'PEC': round(next_pec, 2),
            'Development': round(next_dev, 2)
        }])
        st.dataframe(suggestion)

        with st.expander("📘 Why was this suggested?"):
            st.write("""
            This suggestion is based on the best historical Cpk result, slightly perturbing the parameters using a narrow range
            to explore better process corners. The idea is to balance exploration (try nearby parameters) and exploitation
            (stay close to high-Cpk setups). You could optimize this using gradient-based methods or surrogate models in the future.
            """)
        
        st.download_button(
            "📤 Download DOE History (CSV)",
            st.session_state.doe_history.to_csv(index=False).encode(),
            "doe_history.csv",
            "text/csv"
        )
