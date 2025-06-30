import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from scipy.stats import f_oneway
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# --- DOE Manager ---
class DOEManager:
    def __init__(self):
        if "doe_history" not in st.session_state:
            st.session_state.doe_history = pd.DataFrame(columns=['dose', 'PEC', 'development', 'Cp', 'Cpk'])
    
    def add_experiment(self, dose, PEC, development, Cp, Cpk):
        new_row = {'dose': dose, 'PEC': PEC, 'development': development, 'Cp': Cp, 'Cpk': Cpk}
        st.session_state.doe_history = st.session_state.doe_history.append(new_row, ignore_index=True)
    
    def suggest_next_DOE(self):
        history = st.session_state.doe_history
        if history.empty:
            return {'dose': 100, 'PEC': 50, 'development': 30}  # default start
        best = history.loc[history['Cpk'].idxmax()]
        new_DOE = {
            'dose': best['dose'] * np.random.uniform(0.95, 1.05),
            'PEC': best['PEC'] * np.random.uniform(0.95, 1.05),
            'development': best['development'] * np.random.uniform(0.95, 1.05),
        }
        return new_DOE

# --- SEM Analyzer ---
def preprocess_sem_image(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def build_cnn_classifier(input_shape=(128,128,1)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # dummy binary output
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@st.cache(allow_output_mutation=True)
def load_cnn_model():
    # For demo, build untrained model (replace with loading your trained weights)
    model = build_cnn_classifier()
    return model

def analyze_sem_image(image, model):
    edges = preprocess_sem_image(image)
    resized = cv2.resize(edges, (128,128))
    input_data = resized.reshape(1,128,128,1) / 255.0
    pred = model.predict(input_data)[0][0]
    return pred, edges

# --- Six Sigma Stats ---
def calculate_cp_cpk(data, USL, LSL):
    mean_val = np.mean(data)
    std_val = np.std(data, ddof=1)
    Cp = (USL - LSL) / (6 * std_val) if std_val != 0 else np.nan
    Cpu = (USL - mean_val) / (3 * std_val) if std_val != 0 else np.nan
    Cpl = (mean_val - LSL) / (3 * std_val) if std_val != 0 else np.nan
    Cpk = min(Cpu, Cpl) if std_val != 0 else np.nan
    return Cp, Cpk

def anova_test(groups):
    f_val, p_val = f_oneway(*groups)
    return f_val, p_val

# --- Trend Dashboard ---
def plot_spc_chart(data):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    UCL = mean + 3*std
    LCL = mean - 3*std
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Measurements'))
    fig.add_hline(y=mean, line_dash='dash', line_color='green', annotation_text="Mean", annotation_position="top left")
    fig.add_hline(y=UCL, line_dash='dot', line_color='red', annotation_text="UCL", annotation_position="top left")
    fig.add_hline(y=LCL, line_dash='dot', line_color='red', annotation_text="LCL", annotation_position="top left")
    fig.update_layout(height=400, title="SPC Chart")
    st.plotly_chart(fig, use_container_width=True)

def detect_anomalies(data):
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(data.reshape(-1,1))
    anomalies = np.where(preds == -1)[0]
    return anomalies

# --- Data Export ---
def export_data(df):
    csv = df.to_csv(index=False).encode()
    xlsx = df.to_excel(index=False, engine='openpyxl')
    return csv

# --- Streamlit App UI ---
st.title("Enhanced AI-Based LithoSix Tool")

# Sidebar navigation
page = st.sidebar.selectbox("Select Tool", ["DOE Manager", "SEM Analyzer", "Six Sigma Stats", "Trend Dashboard", "Data Export"])

# Initialize DOE Manager
doe_manager = DOEManager()

if page == "DOE Manager":
    st.header("DOE Manager")
    with st.form("doe_form"):
        dose = st.number_input("Dose", value=100.0, step=0.1)
        pec = st.number_input("PEC", value=50.0, step=0.1)
        dev = st.number_input("Development Time", value=30.0, step=0.1)
        cp = st.number_input("Cp", value=1.33, step=0.01)
        cpk = st.number_input("Cpk", value=1.25, step=0.01)
        submitted = st.form_submit_button("Add Experiment")
    
    if submitted:
        doe_manager.add_experiment(dose, pec, dev, cp, cpk)
        st.success("Experiment added to DOE history!")

    st.subheader("DOE History")
    st.dataframe(st.session_state.doe_history)
    
    st.subheader("AI Suggested Next DOE")
    suggestion = doe_manager.suggest_next_DOE()
    st.write(suggestion)

elif page == "SEM Analyzer":
    st.header("SEM Analyzer")
    uploaded_file = st.file_uploader("Upload SEM Image", type=["png", "jpg", "jpeg", "tif"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded SEM Image", use_column_width=True)
        
        model = load_cnn_model()
        pred_score, edges = analyze_sem_image(image, model)
        
        st.subheader("SEM Image Edges")
        st.image(edges, caption="Edge Detection Result", use_column_width=True, clamp=True)
        
        st.subheader("CNN Quality Score")
        st.write(f"Predicted quality/confidence score: {pred_score:.3f}")

elif page == "Six Sigma Stats":
    st.header("Six Sigma Statistics")
    st.write("Input measurement data (comma separated):")
    data_input = st.text_area("Data", value="100, 101, 99, 98, 102, 97, 103")
    USL = st.number_input("Upper Spec Limit (USL)", value=105.0)
    LSL = st.number_input("Lower Spec Limit (LSL)", value=95.0)

    if st.button("Calculate Cp, Cpk"):
        try:
            data = np.array([float(x.strip()) for x in data_input.split(',') if x.strip() != ''])
            Cp, Cpk = calculate_cp_cpk(data, USL, LSL)
            st.write(f"Cp = {Cp:.3f}")
            st.write(f"Cpk = {Cpk:.3f}")
        except Exception as e:
            st.error(f"Error processing data: {e}")

    st.write("---")
    st.write("ANOVA Test (enter groups separated by new lines, values comma separated)")
    group_text = st.text_area("Groups", value="100, 101, 99\n98, 102, 97\n103, 104, 105")
    if st.button("Run ANOVA"):
        try:
            groups = []
            for line in group_text.strip().split('\n'):
                group_vals = [float(x.strip()) for x in line.split(',') if x.strip() != '']
                groups.append(group_vals)
            f_val, p_val = anova_test(groups)
            st.write(f"F-value = {f_val:.3f}")
            st.write(f"P-value = {p_val:.3f}")
        except Exception as e:
            st.error(f"Error processing groups: {e}")

elif page == "Trend Dashboard":
    st.header("Trend Dashboard")
    st.write("Input time-series data (comma separated):")
    ts_input = st.text_area("Data", value="100, 101, 99, 98, 102, 97, 103, 105, 110, 108, 107, 95")
    if st.button("Show SPC Chart & Anomalies"):
        try:
            data = np.array([float(x.strip()) for x in ts_input.split(',') if x.strip() != ''])
            plot_spc_chart(data)
            anomalies = detect_anomalies(data)
            if len(anomalies) > 0:
                st.warning(f"Anomalies detected at indices: {list(anomalies)}")
            else:
                st.success("No anomalies detected.")
        except Exception as e:
            st.error(f"Error processing time-series data: {e}")

elif page == "Data Export":
    st.header("Export DOE History")
    if not st.session_state.doe_history.empty:
        st.dataframe(st.session_state.doe_history)
        csv = st.session_state.doe_history.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "doe_history.csv", "text/csv")
        xlsx = st.session_state.doe_history.to_excel(index=False, engine='openpyxl')
        # Note: Streamlit download button does not support excel directly without saving file.
        # We provide CSV for demo.
        st.info("Excel export coming soon (requires saving locally).")
    else:
        st.info("No DOE data available to export.")

