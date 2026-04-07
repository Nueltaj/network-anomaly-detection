import pandas as pd
import numpy as np
import joblib
import sys

# ── Load saved model and scaler ──────────────────────────────────────────────
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# ── Feature list (same 77 features used during training) ─────────────────────
EXPECTED_FEATURES = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 
    'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
    'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
    'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# ── Core detection function ───────────────────────────────────────────────────
def detect(input_csv: str) -> pd.DataFrame:
    """
    Load a CSV of network flows, run the model, return results with predictions.
    """
    df = pd.read_csv(input_csv)

    # Drop Label column if present (we are predicting, not training)
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])

    # Drop Destination Port and duplicate column if present
    for col in ['Destination Port']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Keep only expected features, in correct order
    df = df[EXPECTED_FEATURES]

    # Handle any infinite or missing values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Scale features
    scaled = scaler.transform(df)

    # Predict
    predictions = model.predict(scaled)
    labels = ['DDoS' if p == 1 else 'BENIGN' for p in predictions]

    # Build results dataframe
    results = df.copy()
    results['Prediction'] = labels

    return results

# ── Main: run from command line ───────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python detection_engine.py <path_to_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    print(f"\nRunning detection on: {input_file}")

    results = detect(input_file)

    total = len(results)
    ddos_count = (results['Prediction'] == 'DDoS').sum()
    benign_count = (results['Prediction'] == 'BENIGN').sum()

    print(f"\n{'='*40}")
    print(f"  Total flows analysed : {total}")
    print(f"  BENIGN               : {benign_count}")
    print(f"  DDoS detected        : {ddos_count}")
    print(f"{'='*40}")

    if ddos_count > 0:
        print(f"\n⚠ ALERT: {ddos_count} DDoS flows detected.")
    else:
        print("\n✓ No anomalies detected.")

    # Save results
    output_path = 'results_output.csv'
    results.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")