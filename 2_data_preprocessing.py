"""
Data Preprocessing for CAUEEG Dataset
This script extracts features from EDF files and prepares the dataset for model training
"""

import os
import json
import pandas as pd
import numpy as np
import mne
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CAUEEG DATASET - DATA PREPROCESSING & FEATURE EXTRACTION")
print("="*80)

# Paths
base_path = r"c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\caueeg-dataset"
edf_dir = os.path.join(base_path, "signal", "edf")
annotation_path = os.path.join(base_path, "annotation.json")
output_dir = r"c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\csvcaueeg"

# Load annotations
with open(annotation_path, 'r') as f:
    annotation_data = json.load(f)

patients = annotation_data['data']

# Create diagnosis mapping
def classify_alzheimer_stage(symptoms):
    """
    Classify into Alzheimer's progression stages:
    0 - Normal/Control
    1 - MCI (Mild Cognitive Impairment) - Early stage
    2 - Alzheimer's Disease - Dementia stage
    """
    if 'normal' in symptoms:
        return 0  # Normal
    elif 'mci' in symptoms:
        return 1  # MCI - Early Alzheimer's
    elif 'dementia' in symptoms and 'ad' in symptoms:
        return 2  # Alzheimer's Dementia
    else:
        return -1  # Other conditions (will be excluded)

def extract_eeg_features(edf_file_path):
    """
    Extract comprehensive features from EDF file
    Features include: statistical, spectral, and complexity measures
    """
    try:
        # Read EDF file
        raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
        
        # Get data and sampling frequency
        data = raw.get_data()
        sfreq = raw.info['sfreq']
        
        # Get channel names (exclude non-EEG channels like EKG, Photic)
        ch_names = raw.ch_names
        eeg_channels = [ch for ch in ch_names if not any(x in ch.upper() for x in ['EKG', 'PHOTIC', 'ECG'])]
        
        # Select only EEG channels
        eeg_indices = [raw.ch_names.index(ch) for ch in eeg_channels]
        eeg_data = data[eeg_indices, :]
        
        features = {}
        
        # 1. STATISTICAL FEATURES (for each channel)
        for i, ch_name in enumerate(eeg_channels):
            ch_data = eeg_data[i, :]
            
            # Basic statistics
            features[f'{ch_name}_mean'] = np.mean(ch_data)
            features[f'{ch_name}_std'] = np.std(ch_data)
            features[f'{ch_name}_var'] = np.var(ch_data)
            features[f'{ch_name}_skew'] = skew(ch_data)
            features[f'{ch_name}_kurtosis'] = kurtosis(ch_data)
            features[f'{ch_name}_min'] = np.min(ch_data)
            features[f'{ch_name}_max'] = np.max(ch_data)
            features[f'{ch_name}_range'] = np.ptp(ch_data)
            
            # Percentiles
            features[f'{ch_name}_p25'] = np.percentile(ch_data, 25)
            features[f'{ch_name}_p50'] = np.percentile(ch_data, 50)
            features[f'{ch_name}_p75'] = np.percentile(ch_data, 75)
            
            # Zero crossing rate
            features[f'{ch_name}_zcr'] = np.sum(np.diff(np.sign(ch_data)) != 0) / len(ch_data)
        
        # 2. SPECTRAL FEATURES (Power in different frequency bands)
        # Define frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        for i, ch_name in enumerate(eeg_channels):
            ch_data = eeg_data[i, :]
            
            # Compute power spectral density
            freqs, psd = scipy_signal.welch(ch_data, fs=sfreq, nperseg=min(256, len(ch_data)))
            
            # Calculate band powers
            for band_name, (low_freq, high_freq) in bands.items():
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.trapz(psd[band_mask], freqs[band_mask])
                features[f'{ch_name}_{band_name}_power'] = band_power
            
            # Total power
            total_power = np.trapz(psd, freqs)
            features[f'{ch_name}_total_power'] = total_power
            
            # Relative band powers
            for band_name in bands.keys():
                features[f'{ch_name}_{band_name}_rel_power'] = (
                    features[f'{ch_name}_{band_name}_power'] / total_power if total_power > 0 else 0
                )
            
            # Spectral edge frequency (95%)
            cumsum_psd = np.cumsum(psd)
            sef_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
            features[f'{ch_name}_sef95'] = freqs[sef_idx[0]] if len(sef_idx) > 0 else 0
        
        # 3. GLOBAL FEATURES (across all channels)
        # Average power across channels
        for band_name in bands.keys():
            band_powers = [features[f'{ch}_{band_name}_power'] for ch in eeg_channels]
            features[f'global_{band_name}_power'] = np.mean(band_powers)
            features[f'global_{band_name}_std'] = np.std(band_powers)
        
        # Coherence between hemispheres (simplified)
        # Left hemisphere channels vs Right hemisphere channels
        left_channels = [ch for ch in eeg_channels if any(x in ch for x in ['1', '3', '7', 'Z'])]
        right_channels = [ch for ch in eeg_channels if any(x in ch for x in ['2', '4', '8'])]
        
        if left_channels and right_channels:
            left_idx = [eeg_channels.index(ch) for ch in left_channels if ch in eeg_channels]
            right_idx = [eeg_channels.index(ch) for ch in right_channels if ch in eeg_channels]
            
            if left_idx and right_idx:
                left_mean = np.mean(eeg_data[left_idx, :], axis=0)
                right_mean = np.mean(eeg_data[right_idx, :], axis=0)
                
                # Correlation between hemispheres
                features['hemisphere_correlation'] = np.corrcoef(left_mean, right_mean)[0, 1]
        
        return features
    
    except Exception as e:
        print(f"Error processing {edf_file_path}: {str(e)}")
        return None

# Process all EDF files
print("\n1. PROCESSING EDF FILES AND EXTRACTING FEATURES...")
print(f"   Total patients to process: {len(patients)}")

all_features = []
all_labels = []
all_serials = []
all_ages = []

processed_count = 0
skipped_count = 0

for patient in tqdm(patients, desc="Extracting features"):
    serial = patient['serial']
    age = patient['age']
    symptoms = patient['symptom']
    
    # Classify stage
    stage = classify_alzheimer_stage(symptoms)
    
    # Skip "Other conditions" (stage -1)
    if stage == -1:
        skipped_count += 1
        continue
    
    # Check if EDF file exists
    edf_file = os.path.join(edf_dir, f"{serial}.edf")
    
    if not os.path.exists(edf_file):
        skipped_count += 1
        continue
    
    # Extract features
    features = extract_eeg_features(edf_file)
    
    if features is not None:
        all_features.append(features)
        all_labels.append(stage)
        all_serials.append(serial)
        all_ages.append(age)
        processed_count += 1

print(f"\n   Successfully processed: {processed_count} patients")
print(f"   Skipped: {skipped_count} patients")

# Create DataFrame
print("\n2. CREATING FEATURE DATAFRAME...")
features_df = pd.DataFrame(all_features)
features_df['serial'] = all_serials
features_df['age'] = all_ages
features_df['alzheimer_stage'] = all_labels

print(f"   Total features extracted: {len(features_df.columns) - 3}")
print(f"   Total samples: {len(features_df)}")

# Handle missing values
print("\n3. HANDLING MISSING VALUES...")
missing_before = features_df.isnull().sum().sum()
print(f"   Missing values before: {missing_before}")

# Fill missing values with median
for col in features_df.columns:
    if features_df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
        features_df[col].fillna(features_df[col].median(), inplace=True)

missing_after = features_df.isnull().sum().sum()
print(f"   Missing values after: {missing_after}")

# Handle infinite values
print("\n4. HANDLING INFINITE VALUES...")
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in features_df.columns:
    if features_df[col].dtype in [np.float64, np.float32]:
        features_df[col].fillna(features_df[col].median(), inplace=True)

# Save processed data
print("\n5. SAVING PROCESSED DATA...")
output_file = os.path.join(output_dir, 'processed_features.csv')
features_df.to_csv(output_file, index=False)
print(f"   Saved to: {output_file}")

# Display class distribution
print("\n6. CLASS DISTRIBUTION:")
stage_names = {
    0: 'Normal/Control',
    1: 'MCI (Early Stage)',
    2: 'Alzheimer\'s Dementia'
}

for stage in sorted(features_df['alzheimer_stage'].unique()):
    count = (features_df['alzheimer_stage'] == stage).sum()
    percentage = count / len(features_df) * 100
    print(f"   Stage {stage} - {stage_names[stage]}: {count} samples ({percentage:.1f}%)")

# Feature statistics
print("\n7. FEATURE STATISTICS:")
print(f"   Total features: {len(features_df.columns) - 3}")
print(f"   Feature types:")
print(f"   - Statistical features per channel: 11")
print(f"   - Spectral features per channel: 11 (5 bands + total + 5 relative)")
print(f"   - Global features: ~15")

print("\n" + "="*80)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("="*80)
