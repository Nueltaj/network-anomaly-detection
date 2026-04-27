import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
import os

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

# ── Load model and scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_model()

# ── Expected features ─────────────────────────────────────────────────────────
EXPECTED_FEATURES = scaler.feature_names_in_.tolist()

# ── Detection function ────────────────────────────────────────────────────────
def run_detection(df):
    # Drop label if present
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])

    # Drop unwanted columns if present
    for col in ['Destination Port', 'Fwd Header Length.1']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Keep only expected features
    missing = [f for f in EXPECTED_FEATURES if f not in df.columns]
    if missing:
        st.error(f"Input file is missing {len(missing)} required features.")
        return None

    df = df[EXPECTED_FEATURES]

    # Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Scale and predict
    scaled = scaler.transform(df)
    predictions = model.predict(scaled)
    df['Prediction'] = ['DDoS' if p == 1 else 'BENIGN' for p in predictions]

    return df

# ── Dashboard UI ──────────────────────────────────────────────────────────────
st.title("🛡️ Network Anomaly Detection System")
st.markdown("Upload a CSV file of network flows to detect DDoS attacks.")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload network traffic CSV", 
    type=["csv"]
)

if uploaded_file is not None:
    with st.spinner("Analysing traffic..."):
        df_input = pd.read_csv(uploaded_file)
        results = run_detection(df_input)

    if results is not None:
        total = len(results)
        ddos = (results['Prediction'] == 'DDoS').sum()
        benign = (results['Prediction'] == 'BENIGN').sum()

        # ── Summary metrics ───────────────────────────────────────────────────
        st.markdown("### Detection Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Flows Analysed", f"{total:,}")
        col2.metric("BENIGN Flows", f"{benign:,}", delta=None)
        col3.metric("DDoS Flows Detected", f"{ddos:,}",
                    delta="ALERT" if ddos > 0 else None,
                    delta_color="inverse")

        # ── Alert banner ──────────────────────────────────────────────────────
        if ddos > 0:
            st.error(f"⚠️ ALERT: {ddos:,} DDoS flows detected in this capture.")
        else:
            st.success("✓ No DDoS anomalies detected. Traffic appears normal.")

        st.markdown("---")

        # ── Bar chart ─────────────────────────────────────────────────────────
        st.markdown("### Traffic Distribution")
        chart_data = pd.DataFrame({
            'Label': ['BENIGN', 'DDoS'],
            'Count': [benign, ddos]
        }).set_index('Label')
        st.bar_chart(chart_data)

        st.markdown("---")

        # ── Results table ─────────────────────────────────────────────────────
        st.markdown("### Detailed Results (first 500 rows)")
        st.dataframe(
            results[['Flow Duration', 'Total Fwd Packets',
                      'Total Backward Packets', 'Flow Bytes/s',
                      'Flow Packets/s', 'Prediction']].head(500),
            use_container_width=True
        )

        # ── Download button ───────────────────────────────────────────────────
        st.markdown("---")
        csv_output = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_output,
            file_name="detection_results.csv",
            mime="text/csv"
        )

else:
    st.info("Awaiting file upload. Please upload a CSV file to begin.")