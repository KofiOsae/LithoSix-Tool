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
from skimage.filters import sobel, threshold_otsu
from skimage.measure import regionprops, label
import plotly.express as px
import plotly.graph_objects as go
import io


# --- Helper Functions Shared ---
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    return output.getvalue()


# -------------------------------
# SEM Analysis Functions
# -------------------------------
def summarize_metrics(df, fields):
    summary = {}
    for name, key in fields.items():
        if key in df.columns and not df[key].dropna().empty:
            mean = df[key].mean()
            std = df[key].std()
            summary[name] = f"{mean:.2f} ± {std:.2f}"
        else:
            summary[name] = "N/A"
    return pd.DataFrame.from_dict(summary, orient="index", columns=["μ ± σ"])

def preprocess_image(image, blur_ksize=5, threshold_value=100, contrast_factor=1.2):
    """
    Convert image to grayscale, blur, threshold, and apply contrast adjustment.
    Returns a processed binary image suitable for feature extraction.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, binary = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
    adjusted = cv2.convertScaleAbs(binary, alpha=contrast_factor, beta=0)
    return adjusted


def extract_grating_geometry(gray, contours, scale, gray_to_nm=2.0):
    results = []
    centers = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = gray[y:y + h, x:x + w]

        profile_vert = np.mean(roi, axis=1)
        top_idx = np.argmax(np.gradient(profile_vert))
        bot_idx = np.argmin(np.gradient(profile_vert))
        height_px = max(1, bot_idx - top_idx)
        height_nm = height_px * scale

        horiz_proj = np.mean(roi, axis=0)
        edge_thresh = np.max(horiz_proj) * 0.5
        left = np.argmax(horiz_proj > edge_thresh)
        right = len(horiz_proj) - np.argmax(np.flip(horiz_proj) > edge_thresh)
        top_width_px = right - left
        bottom_width_px = w
        cd_px = (top_width_px + bottom_width_px) / 2

        cd_nm = cd_px * scale
        top_width_nm = top_width_px * scale
        bottom_width_nm = bottom_width_px * scale

        slope_rad = math.atan(abs(top_width_px - bottom_width_px) / height_px) if height_px else 0
        slope_deg = math.degrees(slope_rad)

        intensity_top = np.mean(roi[top_idx:top_idx + 3])
        intensity_bottom = np.mean(roi[bot_idx - 3:bot_idx])
        delta_intensity = abs(intensity_bottom - intensity_top)
        est_height_nm = delta_intensity * gray_to_nm

        cx = x + w // 2
        centers.append(cx * scale)

        results.append({
            "Top Width (nm)": top_width_nm,
            "Bottom Width (nm)": bottom_width_nm,
            "CD (nm)": cd_nm,
            "Height (nm)": est_height_nm,
            "Sidewall Angle (°)": slope_deg
        })

    df = pd.DataFrame(results)
    if len(df) > 1:
        df["LER (nm)"] = df["CD (nm)"].diff().abs()
        df["LWR (nm)"] = df["CD (nm)"].rolling(3).std()
    else:
        df["LER (nm)"] = np.nan
        df["LWR (nm)"] = np.nan

    if len(centers) > 1:
        centers = np.sort(centers)
        pitches = np.diff(centers)
        df["Pitch (nm)"] = [*pitches, np.nan]

    summary_fields = {
        "CD (nm)": "CD (nm)",
        "Top Width (nm)": "Top Width (nm)",
        "Bottom Width (nm)": "Bottom Width (nm)",
        "Sidewall Angle (°)": "Sidewall Angle (°)",
        "Height (nm)": "Height (nm)",
        "LER (nm)": "LER (nm)",
        "LWR (nm)": "LWR (nm)",
        "Pitch (nm)": "Pitch (nm)"
    }

    return df.dropna(), summarize_metrics(df, summary_fields)

def extract_dot_features(binary, scale):
    from skimage.measure import regionprops, label
    props = regionprops(label(binary))
    entries = []
    for p in props:
        if p.area < 50:
            continue
        cd = (p.major_axis_length + p.minor_axis_length) / 2 * scale
        circ = (4 * math.pi * p.area) / (p.perimeter ** 2) if p.perimeter > 0 else 0
        ecc = p.eccentricity
        entries.append({"CD (nm)": cd, "Circularity": circ, "Eccentricity": ecc})

    df = pd.DataFrame(entries)
    summary_fields = {
        "CD (nm)": "CD (nm)",
        "Circularity": "Circularity",
        "Eccentricity": "Eccentricity"
    }
    return df, summarize_metrics(df, summary_fields)

def extract_dot_ellipse_features(binary, scale):
    from skimage.measure import regionprops, label
    props = regionprops(label(binary))
    data = []
    for prop in props:
        if prop.area < 50:
            continue
        cd = ((prop.major_axis_length + prop.minor_axis_length) / 2) * scale
        ecc = prop.eccentricity
        circ = (4 * math.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter > 0 else 0
        data.append({'CD (nm)': cd, 'Circularity': circ, 'Eccentricity': ecc})

    df = pd.DataFrame(data)
    summary_fields = {
        "CD (nm)": "CD (nm)",
        "Circularity": "Circularity",
        "Eccentricity": "Eccentricity"
    }
    return df, summarize_metrics(df, summary_fields)

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


def run_sem_analyzer():
    st.header("🖼 SEM Analyzer — Advanced Feature Analysis")

    uploaded = st.file_uploader("Upload SEM Image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'])

    if uploaded:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        st.sidebar.subheader("📐 Scale Calibration")
        coords_str = st.sidebar.text_input("Enter scale bar points (x1,y1,x2,y2)", "100,500,300,500")
        try:
            x1, y1, x2, y2 = map(float, coords_str.strip().split(","))
            pixel_dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            known_len_nm = st.sidebar.number_input("Known distance (nm)", value=200.0)
            scale_nm_per_pixel = known_len_nm / pixel_dist
            st.session_state.scale_nm_per_pixel = scale_nm_per_pixel
            st.sidebar.success(f"Scale = {scale_nm_per_pixel:.3f} nm/pixel")
        except:
            st.sidebar.warning("Invalid coordinates")

        st.sidebar.subheader("🧭 Feature Type & Preprocessing")
        feature_type = st.sidebar.selectbox("Feature Type", ["Grating", "Dot", "Ellipse"])
        blur = st.sidebar.slider("Blur", 1, 15, 5, step=2)
        threshold = st.sidebar.slider("Threshold", 10, 255, 100)
        contrast = st.sidebar.slider("Contrast", 1.0, 3.0, 1.2)

        st.sidebar.subheader("🌄 Grayscale Height Estimation")
        gray_to_nm = st.sidebar.slider("Gray→nm (Height Estimation)", 0.0, 10.0, 2.0)

        if "scale_nm_per_pixel" not in st.session_state:
            st.info("Please calibrate scale to proceed.")
            return

        binary = preprocess_image(image, blur, threshold, contrast)
        edges = cv2.Canny(binary, 50, 150)
        overlay = overlay_contours(img_rgb, edges)

        view = st.radio("🖼 View Mode", ["Overlay", "Processed", "Original"])
        if view == "Overlay":
            st.image(overlay, caption="Contour Overlay", use_column_width=True)
        elif view == "Processed":
            st.image(binary, caption="Processed Binary Image", use_column_width=True)
        else:
            st.image(img_rgb, caption="Original SEM Image", use_column_width=True)

        scale = st.session_state.scale_nm_per_pixel
        df, summary = pd.DataFrame(), pd.DataFrame()

        if feature_type == "Grating":
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            df, summary = extract_grating_geometry(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), contours, scale, gray_to_nm)
        elif feature_type == "Dot":
            df, summary = extract_dot_features(binary, scale)
        elif feature_type == "Ellipse":
            df, summary = extract_dot_ellipse_features(binary, scale)

        if not df.empty:
            st.subheader("📋 Feature Table")
            st.dataframe(df)

            st.subheader("📊 Summary (μ ± σ)")
            st.dataframe(summary)

            st.subheader("🎚 Filter Features (Optional)")
            if st.checkbox("Enable filters"):
                selected = st.multiselect("Filter columns", df.select_dtypes(float).columns.tolist())
                for col in selected:
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    f_min = st.number_input(f"Min {col}", value=min_val)
                    f_max = st.number_input(f"Max {col}", value=max_val)
                    df = df[(df[col] >= f_min) & (df[col] <= f_max)]

            st.subheader("📤 Export Data")
            st.download_button("⬇️ Download Features", df.to_csv(index=False), file_name="sem_features.csv")
            st.download_button("⬇️ Download Summary", summary.to_csv(), file_name="sem_summary.csv")

            st.session_state.sem_data = df
            st.session_state.sem_summary = summary

            st.success(f"{len(df)} features extracted.")
        else:
            st.warning("No valid features found.")
          
# -------------------------------
# STREAMLIT APP UI
# -------------------------------
st.set_page_config(page_title="LithoSix", layout="wide")
st.title("🧪 LithoSix: Complete Lithography Assistant")

page = st.sidebar.radio("Select Module", ["SEM Analyzer", "DOE Manager", "Six Sigma Stats", "Trend Dashboard"])

# === SEM ANALYZER ADVANCE===

if page == "SEM Analyzer":
    run_sem_analyzer()



# === DOE MANAGER ===
elif page == "DOE Manager":
    st.title("🧪 DOE Manager — Recipe Tracking & Optimization")
    st.markdown("""
    Upload and manage experimental process conditions (Dose, PEB, Develop Time).  
    Compute Cpk, find top performers, and get AI-suggested next recipe.
    """)

    uploaded_doe = st.file_uploader("Upload DOE Matrix (CSV)", type=["csv"])
    if uploaded_doe:
        df = pd.read_csv(uploaded_doe)
        st.session_state.doe_data = df  # Persist globally

        st.subheader("📋 Uploaded DOE Matrix")
        st.dataframe(df)

        if "CD" in df.columns:
            mean = df["CD"].mean()
            std = df["CD"].std()
            usl = st.number_input("Upper Spec Limit (nm)", value=mean + 3 * std)
            lsl = st.number_input("Lower Spec Limit (nm)", value=mean - 3 * std)

            df["Cp"] = (usl - lsl) / (6 * std)
            df["Cpk"] = np.minimum((usl - mean), (mean - lsl)) / (3 * std)

            best = df.sort_values("Cpk", ascending=False).iloc[0]
            st.success(f"📌 Best Cpk: {best['Cpk']:.3f} at Recipe: {best.to_dict()}")

            st.subheader("🤖 AI-Suggested Next Experiment")
            step = st.slider("Perturbation Step %", 1, 20, 5)
            suggestion = best.copy()
            for col in ["Dose", "PEB", "Develop Time"]:
                if col in df.columns:
                    suggestion[col] *= (1 + (step / 100.0))
            st.write("Try next:")
            st.write(suggestion.to_frame())
        else:
            st.warning("Column 'CD' not found — cannot compute Cpk.")

        st.download_button("📥 Export DOE with Cpk", df.to_csv(index=False).encode(), "doe_with_cpk.csv")

# === SIX SIGMA STATS ===
elif page == "Six Sigma Stats":
    st.title("📐 Six Sigma & Statistical Analysis")
    st.markdown("Analyze measurement data using process capability and regression tools.")

    use_sem = st.checkbox("Use CD data from SEM Analyzer")
    uploaded_stats = None

    if use_sem and st.session_state.sem_data is not None:
        df = st.session_state.sem_data
        st.success("Loaded CD data from SEM module.")
    else:
        uploaded_stats = st.file_uploader("Upload CSV with measurement data", type=["csv"])
        if uploaded_stats:
            df = pd.read_csv(uploaded_stats)

    if uploaded_stats or use_sem:
        st.dataframe(df)

        col = st.selectbox("Select Measurement Column", df.columns)
        mean = df[col].mean()
        std = df[col].std()
        usl = st.number_input("Upper Spec Limit", value=mean + 3 * std)
        lsl = st.number_input("Lower Spec Limit", value=mean - 3 * std)

        cp = (usl - lsl) / (6 * std)
        cpk = np.minimum((usl - mean), (mean - lsl)) / (3 * std)
        st.metric("Cp", f"{cp:.3f}")
        st.metric("Cpk", f"{cpk:.3f}")

        st.subheader("📊 One-Way ANOVA (optional)")
        group_col = st.selectbox("Grouping Variable", df.columns)
        import scipy.stats as stats
        groups = [grp[col].values for name, grp in df.groupby(group_col)]
        fval, pval = stats.f_oneway(*groups)
        st.write(f"F = {fval:.3f}, p = {pval:.4f}")
        if pval < 0.05:
            st.success("Significant differences between groups (p < 0.05)")

        st.subheader("🤖 Regression: Predict CD")
        x_cols = st.multiselect("Select Features", df.columns)
        if col in x_cols:
            x_cols.remove(col)
        if x_cols:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor()
            X = df[x_cols]
            y = df[col]
            model.fit(X, y)
            r2 = model.score(X, y)
            st.metric("Regression R²", f"{r2:.3f}")


# === TREND DASHBOARD ===
elif page == "Trend Dashboard":
    st.title("📈 Trend Dashboard — SPC + Drift Detection")

    use_sem = st.checkbox("Use SEM CD data")
    if use_sem and st.session_state.sem_data is not None:
        df = st.session_state.sem_data.reset_index(drop=True)
        df["Index"] = df.index
    else:
        uploaded_trend = st.file_uploader("Upload CSV with CD Time Series", type=["csv"])
        if uploaded_trend:
            df = pd.read_csv(uploaded_trend)
            df["Index"] = range(len(df))

    if "CD (nm)" in df.columns:
        st.line_chart(df[["Index", "CD (nm)"]].set_index("Index"))

        mean_cd = df["CD (nm)"].mean()
        std_cd = df["CD (nm)"].std()
        ucl = mean_cd + 3 * std_cd
        lcl = mean_cd - 3 * std_cd

        st.subheader("📏 SPC Limits")
        st.markdown(f"UCL = {ucl:.2f} nm, LCL = {lcl:.2f} nm")

        df["Out of Control"] = (df["CD (nm)"] > ucl) | (df["CD (nm)"] < lcl)
        outliers = df[df["Out of Control"]]
        st.write("🔴 Out-of-control points:", outliers.index.tolist())

        st.subheader("🧪 CUSUM Drift Detection")
        df["cusum_pos"] = (df["CD (nm)"] - mean_cd).clip(lower=0).cumsum()
        df["cusum_neg"] = (mean_cd - df["CD (nm)"]).clip(lower=0).cumsum()
        st.line_chart(df[["Index", "cusum_pos", "cusum_neg"]].set_index("Index"))

        st.subheader("🧠 Isolation Forest Anomaly Detection")
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest()
        df["Anomaly Score"] = iso.fit_predict(df[["CD (nm)"]])
        outliers = df[df["Anomaly Score"] == -1]
        st.write("🔍 Detected Anomalies:", outliers.index.tolist())
    else:
        st.warning("No 'CD (nm)' column found in uploaded or selected data.")
