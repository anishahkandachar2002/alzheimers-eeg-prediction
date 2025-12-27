"""
Prediction Pipeline for Alzheimer's Stage Classification
Upload an EDF file and predict the Alzheimer's stage
"""

import os
import numpy as np
import pandas as pd
import mne
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

class AlzheimerPredictor:
    """
    Alzheimer's Stage Predictor from EDF files
    """
    
    def __init__(self, model_dir):
        """
        Initialize predictor with trained model and preprocessing components
        
        Args:
            model_dir: Directory containing model files
        """
        self.model_dir = model_dir
        
        # Load model
        self.model = joblib.load(os.path.join(model_dir, 'alzheimer_model.pkl'))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        # Load feature selector
        self.selector = joblib.load(os.path.join(model_dir, 'feature_selector.pkl'))
        
        # Load metadata
        with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Load selected features
        with open(os.path.join(model_dir, 'selected_features.txt'), 'r') as f:
            self.selected_features = [line.strip() for line in f.readlines()]
        
        self.stage_names = self.metadata['stage_names']
        
        print("Alzheimer's Stage Predictor Loaded Successfully!")
        print(f"Model: {self.metadata['model_name']}")
        print(f"Accuracy: {self.metadata['accuracy']:.4f}")
        print(f"Number of features: {self.metadata['n_features']}")
    
    def extract_features_from_edf(self, edf_file_path):
        """
        Extract features from an EDF or SET file
        
        Args:
            edf_file_path: Path to EDF or SET file
            
        Returns:
            Dictionary of features
        """
        try:
            # Detect file type and read accordingly
            file_ext = edf_file_path.lower().split('.')[-1]
            
            if file_ext == 'set':
                # Read EEGLAB .set file
                raw = mne.io.read_raw_eeglab(edf_file_path, preload=True, verbose=False)
            elif file_ext == 'edf':
                # Read EDF file
                raw = mne.io.read_raw_edf(edf_file_path, preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported file format: .{file_ext}. Only .edf and .set files are supported.")
            
            # Get data and sampling frequency
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            
            # Get channel names (exclude non-EEG channels)
            ch_names = raw.ch_names
            eeg_channels = [ch for ch in ch_names if not any(x in ch.upper() for x in ['EKG', 'PHOTIC', 'ECG'])]
            
            # Select only EEG channels
            eeg_indices = [raw.ch_names.index(ch) for ch in eeg_channels]
            eeg_data = data[eeg_indices, :]
            
            features = {}
            
            # 1. STATISTICAL FEATURES
            for i, ch_name in enumerate(eeg_channels):
                ch_data = eeg_data[i, :]
                
                features[f'{ch_name}_mean'] = np.mean(ch_data)
                features[f'{ch_name}_std'] = np.std(ch_data)
                features[f'{ch_name}_var'] = np.var(ch_data)
                features[f'{ch_name}_skew'] = skew(ch_data)
                features[f'{ch_name}_kurtosis'] = kurtosis(ch_data)
                features[f'{ch_name}_min'] = np.min(ch_data)
                features[f'{ch_name}_max'] = np.max(ch_data)
                features[f'{ch_name}_range'] = np.ptp(ch_data)
                features[f'{ch_name}_p25'] = np.percentile(ch_data, 25)
                features[f'{ch_name}_p50'] = np.percentile(ch_data, 50)
                features[f'{ch_name}_p75'] = np.percentile(ch_data, 75)
                features[f'{ch_name}_zcr'] = np.sum(np.diff(np.sign(ch_data)) != 0) / len(ch_data)
            
            # 2. SPECTRAL FEATURES
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
                
                # Spectral edge frequency
                cumsum_psd = np.cumsum(psd)
                sef_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
                features[f'{ch_name}_sef95'] = freqs[sef_idx[0]] if len(sef_idx) > 0 else 0
            
            # 3. GLOBAL FEATURES
            for band_name in bands.keys():
                band_powers = [features[f'{ch}_{band_name}_power'] for ch in eeg_channels]
                features[f'global_{band_name}_power'] = np.mean(band_powers)
                features[f'global_{band_name}_std'] = np.std(band_powers)
            
            # Hemisphere correlation
            left_channels = [ch for ch in eeg_channels if any(x in ch for x in ['1', '3', '7', 'Z'])]
            right_channels = [ch for ch in eeg_channels if any(x in ch for x in ['2', '4', '8'])]
            
            if left_channels and right_channels:
                left_idx = [eeg_channels.index(ch) for ch in left_channels if ch in eeg_channels]
                right_idx = [eeg_channels.index(ch) for ch in right_channels if ch in eeg_channels]
                
                if left_idx and right_idx:
                    left_mean = np.mean(eeg_data[left_idx, :], axis=0)
                    right_mean = np.mean(eeg_data[right_idx, :], axis=0)
                    features['hemisphere_correlation'] = np.corrcoef(left_mean, right_mean)[0, 1]
            
            return features
        
        except Exception as e:
            raise Exception(f"Error extracting features from EDF file: {str(e)}")
    
    def predict(self, edf_file_path, age=None):
        """
        Predict Alzheimer's stage from EDF file
        
        Args:
            edf_file_path: Path to EDF file
            age: Patient age (optional, will use median if not provided)
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        print(f"Extracting features from: {edf_file_path}")
        features = self.extract_features_from_edf(edf_file_path)
        
        if features is None:
            raise Exception("Failed to extract features from EDF file")
        
        # Add age
        if age is None:
            age = 75  # Median age from training data
            print(f"Age not provided, using median age: {age}")
        
        features['age'] = age
        
        # Create DataFrame
        features_df = pd.DataFrame([features])
        
        # Handle missing columns (add with 0 if not present)
        # This ensures compatibility with the trained model
        all_feature_cols = self.scaler.feature_names_in_
        for col in all_feature_cols:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training data
        features_df = features_df[all_feature_cols]
        
        # Handle missing values
        features_df.fillna(0, inplace=True)
        features_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        # Select features
        features_selected = self.selector.transform(features_scaled)
        
        # Predict
        prediction = self.model.predict(features_selected)[0]
        probabilities = self.model.predict_proba(features_selected)[0]
        
        # Prepare results
        results = {
            'predicted_stage': int(prediction),
            'stage_name': self.stage_names[str(int(prediction))],
            'confidence': float(probabilities[prediction]),
            'probabilities': {
                self.stage_names[str(i)]: float(probabilities[i]) 
                for i in range(len(probabilities))
            },
            'age': age
        }
        
        return results
    
    def predict_and_display(self, edf_file_path, age=None):
        """
        Predict and display results in a user-friendly format
        
        Args:
            edf_file_path: Path to EDF file
            age: Patient age (optional)
        """
        results = self.predict(edf_file_path, age)
        
        print("\n" + "="*80)
        print("ALZHEIMER'S STAGE PREDICTION RESULTS")
        print("="*80)
        print(f"\nEDF File: {os.path.basename(edf_file_path)}")
        print(f"Patient Age: {results['age']} years")
        print(f"\nPredicted Stage: {results['predicted_stage']} - {results['stage_name']}")
        print(f"Confidence: {results['confidence']*100:.2f}%")
        print(f"\nProbabilities for each stage:")
        for stage_name, prob in results['probabilities'].items():
            print(f"  {stage_name}: {prob*100:.2f}%")
        print("="*80)
        
        return results


def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Alzheimer\'s stage from EDF file')
    parser.add_argument('edf_file', type=str, help='Path to EDF file')
    parser.add_argument('--age', type=int, default=None, help='Patient age (optional)')
    parser.add_argument('--model_dir', type=str, 
                       default=r'c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\csvcaueeg',
                       help='Directory containing model files')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = AlzheimerPredictor(args.model_dir)
    
    # Make prediction
    predictor.predict_and_display(args.edf_file, args.age)


if __name__ == "__main__":
    main()
