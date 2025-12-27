# 🚀 Quick Start Guide

## Alzheimer's Stage Prediction from EEG Data

### ⚡ Fastest Way to Get Started

#### Option 1: Run Complete Pipeline (Recommended)

```bash
# Navigate to the project directory
cd c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\csvcaueeg

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline (EDA + Preprocessing + Model Training)
python run_pipeline.py
```

This will:
1. ✅ Analyze the dataset (EDA)
2. ✅ Extract features from EDF files
3. ✅ Train and evaluate models
4. ✅ Save the best model

**Time**: 30-60 minutes

---

#### Option 2: Run Step-by-Step

```bash
# Step 1: EDA
python 1_eda_analysis.py

# Step 2: Preprocessing (may take 15-30 minutes)
python 2_data_preprocessing.py

# Step 3: Model Building
python 3_model_building.py
```

---

### 🔮 Making Predictions

#### Method 1: Web Application (Easiest)

```bash
streamlit run app.py
```

Then:
1. Open browser to `http://localhost:8501`
2. Upload an EDF file
3. Enter patient age (optional)
4. Click "Predict"
5. View results and download

#### Method 2: Command Line

```bash
python 4_prediction_pipeline.py path/to/file.edf --age 75
```

Example:
```bash
python 4_prediction_pipeline.py ../caueeg-dataset/signal/edf/00001.edf --age 78
```

---

### 📊 View Results

After running the pipeline, check:

1. **EDA Visualizations**: `csvcaueeg/eda_plots/`
2. **Model Performance**: `csvcaueeg/model_plots/`
3. **Model Metrics**: `csvcaueeg/model_metadata.json`
4. **Processed Data**: `csvcaueeg/processed_features.csv`

---

### 🎯 Expected Output

**Stage 0**: Normal/Control - No cognitive impairment  
**Stage 1**: MCI (Mild Cognitive Impairment) - Early Alzheimer's  
**Stage 2**: Alzheimer's Dementia - Diagnosed Alzheimer's

---

### ❓ Troubleshooting

**Problem**: Missing packages  
**Solution**: `pip install -r requirements.txt`

**Problem**: EDF files not found  
**Solution**: Ensure dataset is at `c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\caueeg-dataset\`

**Problem**: Out of memory  
**Solution**: Close other applications or process fewer files

---

### 📞 Need Help?

Check the full README.md for detailed documentation.

---

**Ready to start? Run:**
```bash
python run_pipeline.py
```
