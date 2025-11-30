import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter
from scipy.stats import mode
from sklearn.preprocessing import StandardScaler

# Configuration
COLUMNS = [
    'ED_COUNTER', 'ED_INTERPOLATED', 'ED_RAW_CQ', 'ED_AF3', 'ED_F7',
    'ED_F3', 'ED_FC5', 'ED_T7', 'ED_P7', 'ED_O1',
    'ED_O2', 'ED_P8', 'ED_T8', 'ED_FC6', 'ED_F4',
    'ED_F8', 'ED_AF4', 'ED_GYROX', 'ED_GYROY', 'ED_TIMESTAMP',
    'ED_ES_TIMESTAMP', 'ED_FUNC_ID', 'ED_FUNC_VALUE', 'ED_MARKER', 'ED_SYNC_SIGNAL'
]

FOLDER_PATH = r'C:\Users\reddy\Desktop\Mini_project\data'

# Label IDs
FOCUSED_ID = 0
UNFOCUSED_ID = 1
DROWSY_ID = 2

# EEG signal parameters
SAMPLE_LENGTH_SECOND = 4
FREQUENCY_HZ = 128
SAMPLE_LENGTH_HZ = FREQUENCY_HZ * SAMPLE_LENGTH_SECOND

# Brainwave frequency ranges (Hz)
BRAINWAVE_RANGES = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30)
}

# Helper Functions
def get_state(timestamp: int) -> int:
    if timestamp <= 10 * 128 * 60:
        return FOCUSED_ID
    elif timestamp > 20 * 128 * 60:
        return UNFOCUSED_ID
    else:
        return DROWSY_ID


def bandpass_filter(data, low_freq, high_freq, fs, order=4):
    """Apply Butterworth bandpass filter to the EEG data."""
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data)


# =========================================================
# Preprocessing Logic
# =========================================================
def preprocess_data():
    """Main preprocessing pipeline for all EEG .mat files."""

    scaler = StandardScaler(with_mean=True, with_std=True)

    # Lists to store features and labels
    features = []
    delta_features, theta_features, alpha_features, beta_features = [], [], [], []
    labels = []

    # Process each .mat file
    for i in range(1, 35):
        file_name = f"eeg_record{i}.mat"
        file_path = os.path.join(FOLDER_PATH, file_name)
        print(f" Extracting file {i}: {file_path}")

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        # Load .mat file
        mat_data = loadmat(file_path)
        data = mat_data['o'][0][0]['data']  # Adjust 'o' if your structure differs

        # Create DataFrame
        eeg_df = pd.DataFrame(data, columns=COLUMNS)
        eeg_df.reset_index(inplace=True)
        eeg_df.rename(columns={'index': 'timestamp'}, inplace=True)

        # Assign mental states
        eeg_df['state'] = eeg_df['timestamp'].apply(get_state)

        # Extract EEG channels
        feature = eeg_df.iloc[:, 4:18].values  # EEG channels (AF3–AF4)
        label = eeg_df['state'].values

        # Standardize data
        feature = scaler.fit_transform(feature)

        # Apply bandpass filters for brainwave bands
        brainwave_features = {}
        for wave, (low, high) in BRAINWAVE_RANGES.items():
            filtered = np.apply_along_axis(bandpass_filter, 0, feature, low, high, FREQUENCY_HZ)
            brainwave_features[wave] = filtered

        # Segment into 4-second samples
        num_samples = len(feature) // SAMPLE_LENGTH_HZ
        feature = feature[:num_samples * SAMPLE_LENGTH_HZ]
        label = label[:num_samples * SAMPLE_LENGTH_HZ]

        feature = feature.reshape(num_samples, SAMPLE_LENGTH_HZ, 14, 1)
        label = label.reshape(num_samples, SAMPLE_LENGTH_HZ)
        consensus_labels = mode(label, axis=1)[0].flatten()

        # Append data
        features.append(feature)
        labels.append(consensus_labels)

        for wave, bw_feature in brainwave_features.items():
            bw_feature = bw_feature[:num_samples * SAMPLE_LENGTH_HZ]
            bw_feature = bw_feature.reshape(num_samples, SAMPLE_LENGTH_HZ, 14, 1)
            if wave == "Delta":
                delta_features.append(bw_feature)
            elif wave == "Theta":
                theta_features.append(bw_feature)
            elif wave == "Alpha":
                alpha_features.append(bw_feature)
            elif wave == "Beta":
                beta_features.append(bw_feature)

    # Combine all data
    features = np.vstack(features)
    delta_features = np.vstack(delta_features)
    theta_features = np.vstack(theta_features)
    alpha_features = np.vstack(alpha_features)
    beta_features = np.vstack(beta_features)
    labels = np.concatenate(labels)

    # Print final shapes
    print(f"\n Delta Features Shape: {delta_features.shape}")
    print(f" Theta Features Shape: {theta_features.shape}")
    print(f" Alpha Features Shape: {alpha_features.shape}")
    print(f" Beta Features Shape: {beta_features.shape}")
    print(f" Labels Shape: {labels.shape}\n")

    return {
        "features": features,
        "delta": delta_features,
        "theta": theta_features,
        "alpha": alpha_features,
        "beta": beta_features,
        "labels": labels
    }


# =========================================================
# Script Entry Point
# =========================================================
if __name__ == "__main__":
    data = preprocess_data()
    print("Preprocessing completed successfully!")
