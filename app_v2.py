import streamlit as st
import numpy as np
import pandas as pd
import cv2
from skimage import filters, color
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
import base64

# === Helper functions ===

def calculate_cp_cpk(data, usl, lsl):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    cp = (usl - lsl) / (6 * std) if std > 0 else np.nan
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std)) if std > 0 else np.nan
    return cp, cpk

def anova_test(groups):
    f_val, p_val = stats.f_oneway(*groups)
    return f_val, p_val

def detect_anomalies_isolation_forest(data):
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(data.reshape(-1,1))
    anomalies = np.where(preds == -1)[0]
    return anomalies

def plot_spc_chart(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    ucl = mean + 3*std
    lcl = mean - 3*std

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data, marker='o', linestyle='-', label='Data')
    ax.axhline(mean, color='green', linestyle='--', label='Mean')
    ax.axhline(ucl, color='red', linestyle='--', label='UCL (Mean + 3σ)')
    ax.axhline(lcl, color='red', linestyle='--', label='LCL (Mean - 3σ)')
    ax.set_title('SPC Chart')
    ax.legend()
    st.pyplot(fig)

def cusum(data, k=0.5, h=5):
    s_pos = np.zeros(len(data))
    s_neg = np.zeros(len(data))
    for i in range(1, len(data)):
        s_pos[i] = max(0, s_pos[i-1] + data[i] - k)
        s_neg[i] = min(0, s_neg[i-1] + data[i] + k)
        if s_pos[i] > h or abs(s_neg[i]) > h:
            return i  # index where anomaly detected
    return -1

def sem_edge_detection(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use Sobel filter for edges
    edges = filters.sobel(gray)
    edges = (edges * 255).astype(np.uint8)
    return edges

def dummy_cnn_quality_score(image):
    # Dummy model: return random score simulating AI
    # Replace with your real model integration
    score = np.clip(np.random.normal(0.75, 0.1), 0, 1)
    return score

def generate_synthetic_regression_data(n=50):
    np.random.seed(42)
    dose = np.random.uniform(80, 120, n)
    pec = np.random.uniform(5, 15, n)
    dev = np.random.uniform(10, 30, n)
    cd = 50 + 0.3 * dose - 0.5 * pec + 0.2 * dev + np.random.normal(0, 2, n)
    df = pd.DataFrame({'Dose': dose, 'PEC': pec, 'Development': dev, 'CD': cd})
    return df

def regression_modeling(df):
    X = df[['Dose', 'PEC', 'Development']]
    y = df['CD']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    return model, y_pred, r2

def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    processed_data = output.getvalue()
    return processed_data

# === Initialize session state for DOE ===
if 'doe_history' not in st.session_state:
    st.session_state.doe_history = pd.DataFrame(columns=['Dose', 'PEC', 'Development', 'Cpk'])

# === Streamlit UI ===

st.set_page_config(page_title="Enhanced AI-Based LithoSix Tool", layout="wide")
st.title("Enhanced AI-Based LithoSix Tool")

page = st.sidebar.selectbox("Select Tool Component", 
                            ["DOE Manager", "SEM Analyzer", "Six Sigma Stats Module", "Trend Dashboard", "Data Export"])

# ------------------ DOE Manager ------------------
if page == "DOE Manager":
    st.header("DOE Manager")
    st.markdown("Define or upload lithography experiments (Dose, PEC, Development, Cpk)")

    uploaded_doe = st.file_uploader("📤 Upload DOE CSV or Excel", type=["csv", "xlsx"])
    if uploaded_doe:
        try:
            if uploaded_doe.name.endswith(".csv"):
                df = pd.read_csv(uploaded_doe)
            else:
                df = pd.read_excel(uploaded_doe)
            st.session_state.doe_history = df
            st.success("DOE data uploaded successfully.")
        except Exception as e:
            st.error(f"Failed to load data: {e}")


    with st.form("add_doe_form"):
        dose = st.number_input("Dose (mJ/cm²)", min_value=0.0, max_value=200.0, value=100.0)
        pec = st.number_input("Post-Exposure Bake (°C)", min_value=0.0, max_value=300.0, value=90.0)
        dev = st.number_input("Development Time (s)", min_value=0.0, max_value=120.0, value=30.0)
        cpk = st.number_input("Measured Cpk", min_value=0.0, max_value=2.0, value=1.0, format="%.3f")
        submitted = st.form_submit_button("Add DOE")

    if submitted:
        new_row = {'Dose': dose, 'PEC': pec, 'Development': dev, 'Cpk': cpk}
        st.session_state.doe_history = pd.concat([st.session_state.doe_history, pd.DataFrame([new_row])], ignore_index=True)
        st.success("DOE added.")

    if not st.session_state.doe_history.empty:
        st.subheader("DOE History")
        st.dataframe(st.session_state.doe_history)

        # AI Suggestion (heuristic): suggest next DOE near highest Cpk
        best_cpk_row = st.session_state.doe_history.loc[st.session_state.doe_history['Cpk'].idxmax()]
        st.markdown("### AI-Suggested Next DOE (heuristic):")
        st.write(f"Based on highest Cpk = {best_cpk_row['Cpk']:.3f}, suggest DOE close to:")
        suggested_dose = np.clip(best_cpk_row['Dose'] + np.random.uniform(-5, 5), 0, 200)
        suggested_pec = np.clip(best_cpk_row['PEC'] + np.random.uniform(-5, 5), 0, 300)
        suggested_dev = np.clip(best_cpk_row['Development'] + np.random.uniform(-5, 5), 0, 120)
        st.write(f"- Dose: {suggested_dose:.1f} mJ/cm²")
        st.write(f"- PEC: {suggested_pec:.1f} °C")
        st.write(f"- Development: {suggested_dev:.1f} s")

# ------------------ SEM Analyzer ------------------
elif page == "SEM Analyzer":
    st.header("SEM Analyzer")
    st.markdown("""
    Upload SEM images, apply edge detection, and get AI-based quality score.
    """)

    uploaded_file = st.file_uploader("Upload SEM Image", type=["png", "jpg", "jpeg", "tif"])


    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(img, caption="Original SEM Image", use_column_width=True)

        edges = sem_edge_detection(img)
        st.image(edges, caption="Edge Detection Result", use_column_width=True)

        score = dummy_cnn_quality_score(edges)
        st.markdown(f"**AI-based Quality Score (0=poor, 1=excellent):** `{score:.3f}`")
    else:
        st.info("Please upload a SEM image.")

# ------------------ Six Sigma Stats Module ------------------
elif page == "Six Sigma Stats Module":
    st.header("Six Sigma Stats Module")
    st.markdown("""
    Calculate Cp, Cpk, perform ANOVA, and regression modeling of CD vs parameters.
    """)
    
    st.markdown("📥 Upload a DOE dataset to auto-populate:")
    uploaded_stats_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_stats_file:
        try:
            if uploaded_stats_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_stats_file)
            else:
                df = pd.read_excel(uploaded_stats_file)
            st.session_state.uploaded_data = df
            st.success("Data uploaded.")
            st.write(df.head())
        except Exception as e:
            st.error(f"Upload error: {e}")


    # Cp/Cpk Calculation
    if 'uploaded_data' in st.session_state:
        data_input = ', '.join([f"{v:.2f}" for v in st.session_state.uploaded_data['CD']])

    st.subheader("Cp and Cpk Calculation")
    data_input = st.text_area("Enter measurement data (comma separated)", value="10.2, 9.8, 10.5, 10.1, 10.3")
    usl = st.number_input("Upper Spec Limit (USL)", value=11.0)
    lsl = st.number_input("Lower Spec Limit (LSL)", value=9.0)
    if st.button("Calculate Cp and Cpk"):
        try:
            data = np.array([float(x.strip()) for x in data_input.split(',') if x.strip() != ''])
            cp, cpk = calculate_cp_cpk(data, usl, lsl)
            st.write(f"Cp = {cp:.3f}")
            st.write(f"Cpk = {cpk:.3f}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

    # ANOVA
    st.subheader("ANOVA Test")
    st.markdown("Enter groups of data, each group in a separate line, values comma separated:")
    anova_input = st.text_area("Example:\n10, 11, 9\n12, 13, 14\n9, 8, 10", height=100)
    if st.button("Perform ANOVA"):
        try:
            groups = []
            for line in anova_input.strip().split('\n'):
                group_vals = [float(x.strip()) for x in line.split(',') if x.strip() != '']
                groups.append(group_vals)
            f_val, p_val = anova_test(groups)
            st.write(f"F-value = {f_val:.3f}")
            st.write(f"P-value = {p_val:.3f}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")

    # Regression Modeling
    st.subheader("Regression Modeling of CD vs Parameters (Synthetic Data)")
    df = generate_synthetic_regression_data()
    st.write(df.head())

    model, y_pred, r2 = regression_modeling(df)
    st.write(f"R² of Random Forest regression model: {r2:.3f}")

    fig, ax = plt.subplots()
    ax.scatter(df['CD'], y_pred, alpha=0.7)
    ax.plot([df['CD'].min(), df['CD'].max()], [df['CD'].min(), df['CD'].max()], 'r--')
    ax.set_xlabel("Actual CD")
    ax.set_ylabel("Predicted CD")
    ax.set_title("Regression Model Performance")
    st.pyplot(fig)

# ------------------ Trend Dashboard ------------------
elif page == "Trend Dashboard":
    st.header("Trend Dashboard")
    st.markdown("""
    Input time-series data to visualize SPC charts and detect anomalies using Isolation Forest and CUSUM.
    """)

    ts_input = st.text_area("Enter time-series data (comma separated)", value="100, 101, 99, 98, 102, 97, 103, 105, 110, 108, 107, 95")
    if st.button("Analyze Trend"):
        try:
            data = np.array([float(x.strip()) for x in ts_input.split(',') if x.strip() != ''])
            st.subheader("SPC Chart")
            plot_spc_chart(data)

            anomalies = detect_anomalies_isolation_forest(data)
            if len(anomalies) > 0:
                st.warning(f"Isolation Forest detected anomalies at indices: {list(anomalies)}")
            else:
                st.success("No anomalies detected by Isolation Forest.")

            cusum_index = cusum(data - np.mean(data))
            if cusum_index >= 0:
                st.warning(f"CUSUM detected anomaly at index {cusum_index}.")
            else:
                st.success("No anomalies detected by CUSUM.")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Data Export ------------------
elif page == "Data Export":
    st.header("Data Export")
    st.markdown("""
    Download DOE history data as CSV or Excel.
    """)

    if st.session_state.doe_history.empty:
        st.info("No DOE data available to export. Please add data from DOE Manager first.")
    else:
        df = st.session_state.doe_history
        st.dataframe(df)

        csv_data = df.to_csv(index=False).encode('utf-8')
        excel_data = to_excel(df)

        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="doe_history.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name="doe_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
