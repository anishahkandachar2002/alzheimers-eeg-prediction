# Alzheimer's Stage Prediction from EEG Data

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project provides a complete pipeline for analyzing EEG data from the CAUEEG dataset and predicting Alzheimer's disease stages using machine learning.

## 🌐 Live Demo

**Try the app**: [https://alzheimers-eeg-prediction-z5k5ysah6qknjqn2hdgeus.streamlit.app/)

Upload your EEG files and get instant predictions!

## 🎯 Project Overview

The system analyzes EEG recordings (EDF files) and classifies patients into three stages:
- **Stage 0**: Normal/Control - No cognitive impairment
- **Stage 1**: MCI (Mild Cognitive Impairment) - Early-stage Alzheimer's
- **Stage 2**: Alzheimer's Dementia - Diagnosed Alzheimer's disease

## 📁 Project Structure

```
csvcaueeg/
├── 1_eda_analysis.py           # Exploratory Data Analysis
├── 2_data_preprocessing.py     # Feature extraction from EDF files
├── 3_model_building.py         # Model training and evaluation
├── 4_prediction_pipeline.py    # Prediction module
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── eda_plots/                  # EDA visualizations (generated)
├── model_plots/                # Model performance plots (generated)
├── alzheimer_model.pkl         # Trained model (generated)
├── scaler.pkl                  # Feature scaler (generated)
├── feature_selector.pkl        # Feature selector (generated)
├── model_metadata.json         # Model information (generated)
├── selected_features.txt       # Selected feature names (generated)
└── processed_features.csv      # Processed dataset (generated)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CAUEEG dataset with EDF files and annotation.json

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\csvcaueeg
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Quick Deploy to Streamlit Cloud

Want to deploy this app online? Follow these steps:

1. **Run the setup script**:
   ```powershell
   .\deploy-setup.ps1
   ```

2. **Create a GitHub repository** and push your code

3. **Deploy to Streamlit Cloud**: [share.streamlit.io](https://share.streamlit.io)

📖 **Detailed deployment guide**: See [DEPLOYMENT.md](DEPLOYMENT.md)

### Usage

#### Step 1: Exploratory Data Analysis (EDA)

Run the EDA script to analyze the dataset and generate visualizations:

```bash
python 1_eda_analysis.py
```

**Output**:
- `patient_metadata.csv` - Patient information with labels
- `eda_plots/` - Visualizations of age distribution, diagnosis distribution, etc.

#### Step 2: Data Preprocessing

Extract features from EDF files:

```bash
python 2_data_preprocessing.py
```

**Output**:
- `processed_features.csv` - Extracted features from all EDF files
- Features include:
  - Statistical features (mean, std, skewness, kurtosis, etc.)
  - Spectral features (power in delta, theta, alpha, beta, gamma bands)
  - Global features (hemisphere correlation, etc.)

**Note**: This step may take 10-30 minutes depending on the number of files.

#### Step 3: Model Building

Train and evaluate machine learning models:

```bash
python 3_model_building.py
```

**Output**:
- `alzheimer_model.pkl` - Best trained model
- `scaler.pkl` - Feature scaler
- `feature_selector.pkl` - Feature selector
- `model_metadata.json` - Model performance metrics
- `selected_features.txt` - List of selected features
- `model_plots/` - Performance visualizations

**Models tested**:
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Logistic Regression

#### Step 4: Make Predictions

##### Option A: Command Line

Predict from a single EDF file:

```bash
python 4_prediction_pipeline.py path/to/file.edf --age 75
```

##### Option B: Web Application (Recommended)

Launch the interactive Streamlit web app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features**:
- Upload EDF files via drag-and-drop
- Enter patient age
- Get instant predictions with confidence scores
- View probability distributions
- Download results as CSV
- View model performance metrics

## 📊 Features Extracted

### Statistical Features (per channel)
- Mean, Standard Deviation, Variance
- Skewness, Kurtosis
- Min, Max, Range
- Percentiles (25th, 50th, 75th)
- Zero-crossing rate

### Spectral Features (per channel)
- Power in frequency bands:
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz)
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-45 Hz)
- Total power
- Relative band powers
- Spectral edge frequency (95%)

### Global Features
- Average power across channels
- Hemisphere correlation (left vs right)

**Total**: 100+ features per EEG recording

## 🧪 Model Performance

The best model achieves:
- **Accuracy**: ~85-90% (varies based on data split)
- **F1-Score**: ~0.85
- **Cross-validation**: 5-fold stratified CV

Performance metrics are saved in `model_metadata.json` and visualizations in `model_plots/`.

## 📈 Visualizations

### EDA Plots
1. Age distribution
2. Primary diagnosis distribution
3. Alzheimer's stage distribution
4. Age by stage
5. Top symptoms

### Model Plots
1. Model comparison
2. Confusion matrix
3. Feature importance
4. ROC curves (multi-class)

## 🔬 Technical Details

### EEG Channels Used
- Fp1-AVG, F3-AVG, C3-AVG, P3-AVG, O1-AVG
- Fp2-AVG, F4-AVG, C4-AVG, P4-AVG, O2-AVG
- F7-AVG, T3-AVG, T5-AVG
- F8-AVG, T4-AVG, T6-AVG
- FZ-AVG, CZ-AVG, PZ-AVG

(EKG and Photic channels are excluded)

### Feature Selection
- Method: Mutual Information
- Top 100 features selected
- Reduces dimensionality while maintaining performance

### Data Preprocessing
- Missing value imputation (median)
- Infinite value handling
- Standard scaling (zero mean, unit variance)
- Stratified train-test split (80-20)

## ⚠️ Important Notes

1. **Medical Disclaimer**: This tool is for research and educational purposes only. It should NOT be used as a substitute for professional medical diagnosis.

2. **Data Privacy**: Ensure patient data is handled according to relevant privacy regulations (HIPAA, GDPR, etc.)

3. **Model Limitations**: 
   - Performance depends on data quality
   - May not generalize to different EEG recording protocols
   - Should be validated on independent datasets

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError`
- **Solution**: Install all requirements: `pip install -r requirements.txt`

**Issue**: `FileNotFoundError` for EDF files
- **Solution**: Ensure the CAUEEG dataset is properly structured with EDF files in `caueeg-dataset/signal/edf/`

**Issue**: Model files not found
- **Solution**: Run scripts in order (1 → 2 → 3) before running predictions

**Issue**: Memory error during preprocessing
- **Solution**: Process files in batches or use a machine with more RAM

## 📝 Citation

If you use this code or the CAUEEG dataset, please cite:

```
CAUEEG Dataset: [Add appropriate citation]
```

## 👥 Authors

- Developed for Alzheimer's disease research
- Contact: [Your contact information]

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Medical Disclaimer**: This tool is for research and educational purposes only.

## 🙏 Acknowledgments

- CAUEEG dataset providers
- MNE-Python for EEG processing
- Scikit-learn for machine learning tools
- Streamlit for the web interface

---

**Last Updated**: December 2025
