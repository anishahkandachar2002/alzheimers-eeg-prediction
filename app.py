"""
Streamlit Web Application for Alzheimer's Stage Prediction
Upload EDF file and get predictions with visualizations
"""

import streamlit as st
import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_pipeline import AlzheimerPredictor

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Stage Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .stage-0 {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .stage-1 {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .stage-2 {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    .metric-card {
        padding: 1.5rem;
        border-radius: 8px;
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    # Use the directory where this script is located (works on all platforms)
    model_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        st.session_state.predictor = AlzheimerPredictor(model_dir)
        st.session_state.model_loaded = True
    except Exception as e:
        st.session_state.model_loaded = False
        st.session_state.error_message = str(e)

# Header
st.markdown('<div class="main-header">🧠 Alzheimer\'s Stage Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered EEG Analysis for Early Detection</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=150)
    st.markdown("### About")
    st.info("""
    This application uses machine learning to predict Alzheimer's disease stages from EEG (EDF) files.
    
    **Stages:**
    - **Stage 0**: Normal/Control
    - **Stage 1**: MCI (Mild Cognitive Impairment)
    - **Stage 2**: Alzheimer's Dementia
    """)
    
    if st.session_state.model_loaded:
        st.success("✅ Model Loaded Successfully")
        st.markdown("### Model Information")
        metadata = st.session_state.predictor.metadata
        st.metric("Model Type", metadata['model_name'])
        st.metric("Accuracy", f"{metadata['accuracy']*100:.2f}%")
        st.metric("F1-Score", f"{metadata['f1_score']*100:.2f}%")
    else:
        st.error("❌ Model Not Loaded")
        st.error(st.session_state.error_message)

# Main content
if st.session_state.model_loaded:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Predict", "📊 Model Performance", "ℹ️ Information"])
    
    with tab1:
        st.markdown("### Upload EDF File for Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an EDF or SET file",
                type=['edf', 'set'],
                help="Upload an EEG recording in EDF or EEGLAB SET format"
            )
        
        with col2:
            age = st.number_input(
                "Patient Age (optional)",
                min_value=18,
                max_value=120,
                value=75,
                help="Enter patient age or leave default"
            )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_dir = os.path.join(st.session_state.predictor.model_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Predict button
            if st.button("🔮 Predict Alzheimer's Stage", type="primary"):
                with st.spinner("Analyzing EEG data..."):
                    try:
                        # Make prediction
                        results = st.session_state.predictor.predict(temp_file_path, age)
                        
                        # Display results
                        st.markdown("---")
                        st.markdown("## 📋 Prediction Results")
                        
                        # Main prediction card
                        stage = results['predicted_stage']
                        stage_name = results['stage_name']
                        confidence = results['confidence']
                        
                        stage_class = f"stage-{stage}"
                        st.markdown(f"""
                        <div class="prediction-box {stage_class}">
                            <h2>Predicted Stage: {stage}</h2>
                            <h3>{stage_name}</h3>
                            <h4>Confidence: {confidence*100:.2f}%</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Stage", f"Stage {stage}")
                        
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.2f}%")
                        
                        with col3:
                            st.metric("Patient Age", f"{age} years")
                        
                        # Probability distribution
                        st.markdown("### 📊 Probability Distribution")
                        
                        prob_df = pd.DataFrame([
                            {'Stage': k, 'Probability': v*100}
                            for k, v in results['probabilities'].items()
                        ])
                        
                        fig = px.bar(
                            prob_df,
                            x='Stage',
                            y='Probability',
                            color='Probability',
                            color_continuous_scale='RdYlGn_r',
                            title='Probability for Each Stage',
                            labels={'Probability': 'Probability (%)'}
                        )
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_column_width=True)
                        
                        # Gauge chart for confidence
                        st.markdown("### 🎯 Confidence Meter")
                        
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=confidence*100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Prediction Confidence"},
                            delta={'reference': 80},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 75], 'color': "gray"},
                                    {'range': [75, 100], 'color': "lightgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_column_width=True)
                        
                        # Interpretation
                        st.markdown("### 💡 Interpretation")
                        
                        if stage == 0:
                            st.success("""
                            **Normal/Control**: The EEG patterns suggest normal cognitive function. 
                            No significant signs of cognitive impairment detected.
                            """)
                        elif stage == 1:
                            st.warning("""
                            **MCI (Mild Cognitive Impairment)**: The EEG patterns suggest early-stage 
                            cognitive changes. This is often considered a precursor to Alzheimer's disease. 
                            Regular monitoring and lifestyle interventions are recommended.
                            """)
                        elif stage == 2:
                            st.error("""
                            **Alzheimer's Dementia**: The EEG patterns are consistent with Alzheimer's disease. 
                            Medical consultation and comprehensive evaluation are strongly recommended.
                            """)
                        
                        # Download results
                        st.markdown("### 💾 Download Results")
                        
                        results_df = pd.DataFrame([{
                            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Filename': uploaded_file.name,
                            'Age': age,
                            'Predicted_Stage': stage,
                            'Stage_Name': stage_name,
                            'Confidence': f"{confidence*100:.2f}%",
                            **{f'Prob_{k}': f"{v*100:.2f}%" for k, v in results['probabilities'].items()}
                        }])
                        
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.remove(temp_file_path)
    
    with tab2:
        st.markdown("### 📊 Model Performance Metrics")
        
        metadata = st.session_state.predictor.metadata
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Type", metadata['model_name'])
        
        with col2:
            st.metric("Test Accuracy", f"{metadata['accuracy']*100:.2f}%")
        
        with col3:
            st.metric("F1-Score", f"{metadata['f1_score']*100:.2f}%")
        
        with col4:
            st.metric("CV Accuracy", f"{metadata['cv_mean']*100:.2f}%")
        
        st.markdown("---")
        
        # Display plots if available
        plots_dir = os.path.join(st.session_state.predictor.model_dir, 'model_plots')
        
        if os.path.exists(plots_dir):
            st.markdown("### 📈 Performance Visualizations")
            
            plot_files = {
                'Model Comparison': '1_model_comparison.png',
                'Confusion Matrix': '2_confusion_matrix.png',
                'Feature Importance': '3_feature_importance.png',
                'ROC Curves': '4_roc_curves.png'
            }
            
            for title, filename in plot_files.items():
                plot_path = os.path.join(plots_dir, filename)
                if os.path.exists(plot_path):
                    st.markdown(f"#### {title}")
                    st.image(plot_path, use_column_width=True)
    
    with tab3:
        st.markdown("### ℹ️ About This Application")
        
        st.markdown("""
        ## Alzheimer's Stage Prediction System
        
        This application uses advanced machine learning techniques to analyze EEG (Electroencephalogram) 
        signals and predict the stage of Alzheimer's disease.
        
        ### How It Works
        
        1. **Feature Extraction**: The system extracts comprehensive features from EDF or SET files including:
           - Statistical features (mean, std, skewness, kurtosis, etc.)
           - Spectral features (power in different frequency bands: delta, theta, alpha, beta, gamma)
           - Global features (hemisphere correlation, etc.)
        
        2. **Machine Learning Model**: A trained classifier analyzes these features to predict the stage
        
        3. **Prediction**: The model outputs a prediction with confidence scores for each stage
        
        ### Supported File Formats
        
        - **EDF (European Data Format)**: Standard format for EEG recordings
        - **SET (EEGLAB)**: MATLAB EEGLAB format for EEG data
        
        ### Stages
        
        - **Stage 0 - Normal/Control**: No cognitive impairment
        - **Stage 1 - MCI (Mild Cognitive Impairment)**: Early-stage changes, potential precursor to Alzheimer's
        - **Stage 2 - Alzheimer's Dementia**: Diagnosed Alzheimer's disease
        
        ### Dataset
        
        The model was trained on the CAUEEG dataset, which contains EEG recordings from patients 
        with various cognitive conditions.
        
        ### Disclaimer
        
        ⚠️ **Important**: This tool is for research and educational purposes only. It should NOT be used 
        as a substitute for professional medical diagnosis. Always consult with qualified healthcare 
        professionals for medical advice and diagnosis.
        
        ### Technical Details
        
        - **Features**: 100+ features extracted from EEG signals
        - **Model**: Random Forest / Gradient Boosting Classifier
        - **Validation**: Cross-validated on independent test set
        
        ### Contact & Support
        
        For questions or issues, please contact the development team.
        """)

else:
    st.error("⚠️ Model could not be loaded. Please ensure all model files are present in the csvcaueeg directory.")
    st.info("""
    Required files:
    - alzheimer_model.pkl
    - scaler.pkl
    - feature_selector.pkl
    - model_metadata.json
    - selected_features.txt
    
    Please run the training pipeline first:
    1. Run 1_eda_analysis.py
    2. Run 2_data_preprocessing.py
    3. Run 3_model_building.py
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p>Alzheimer's Stage Predictor | Powered by Machine Learning</p>
    <p>© 2025 | For Research and Educational Purposes Only</p>
</div>
""", unsafe_allow_html=True)
