"""
Exploratory Data Analysis (EDA) for CAUEEG Dataset
This script analyzes the JSON annotations and provides insights into the dataset
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load annotation data
annotation_path = r"c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\caueeg-dataset\annotation.json"

with open(annotation_path, 'r') as f:
    data = json.load(f)

# Extract patient data
patients = data['data']
signal_headers = data['signal_header']

print("="*80)
print("CAUEEG DATASET - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Create DataFrame
df = pd.DataFrame(patients)
print(f"\n1. DATASET OVERVIEW")
print(f"   Total number of patients: {len(df)}")
print(f"   Number of EEG channels: {len(signal_headers)}")
print(f"\n   EEG Channels: {', '.join(signal_headers)}")

# Age statistics
print(f"\n2. AGE DISTRIBUTION")
print(f"   Mean age: {df['age'].mean():.2f} years")
print(f"   Median age: {df['age'].median():.2f} years")
print(f"   Age range: {df['age'].min()} - {df['age'].max()} years")
print(f"   Std deviation: {df['age'].std():.2f} years")

# Symptom analysis
print(f"\n3. SYMPTOM ANALYSIS")

# Flatten all symptoms
all_symptoms = []
for symptoms in df['symptom']:
    all_symptoms.extend(symptoms)

symptom_counts = Counter(all_symptoms)
print(f"   Total unique symptoms: {len(symptom_counts)}")
print(f"\n   Top 10 Most Common Symptoms:")
for symptom, count in symptom_counts.most_common(10):
    print(f"   - {symptom}: {count} patients ({count/len(df)*100:.1f}%)")

# Create primary diagnosis categories
def categorize_diagnosis(symptoms):
    """Categorize patients into main diagnosis groups"""
    if 'dementia' in symptoms:
        if 'ad' in symptoms:
            return 'Alzheimer\'s Disease (AD)'
        elif 'vd' in symptoms:
            return 'Vascular Dementia (VD)'
        elif 'ad_vd_mixed' in symptoms:
            return 'Mixed Dementia'
        else:
            return 'Other Dementia'
    elif 'mci' in symptoms:
        return 'Mild Cognitive Impairment (MCI)'
    elif 'normal' in symptoms:
        return 'Normal/Control'
    elif 'parkinson_synd' in symptoms or 'parkinson_disease' in symptoms:
        return 'Parkinson\'s'
    elif 'ftd' in symptoms:
        return 'Frontotemporal Dementia (FTD)'
    elif 'nph' in symptoms:
        return 'Normal Pressure Hydrocephalus (NPH)'
    else:
        return 'Other'

df['primary_diagnosis'] = df['symptom'].apply(categorize_diagnosis)

print(f"\n4. PRIMARY DIAGNOSIS DISTRIBUTION")
diagnosis_counts = df['primary_diagnosis'].value_counts()
for diagnosis, count in diagnosis_counts.items():
    print(f"   - {diagnosis}: {count} patients ({count/len(df)*100:.1f}%)")

# Create Alzheimer's stage classification
def classify_alzheimer_stage(symptoms):
    """
    Classify into Alzheimer's progression stages:
    0 - Normal/Control
    1 - MCI (Mild Cognitive Impairment) - Early stage
    2 - Alzheimer's Disease - Dementia stage
    3 - Other conditions (Parkinson's, FTD, etc.)
    """
    if 'normal' in symptoms:
        return 0  # Normal
    elif 'mci' in symptoms:
        return 1  # MCI - Early Alzheimer's
    elif 'dementia' in symptoms and 'ad' in symptoms:
        return 2  # Alzheimer's Dementia
    else:
        return 3  # Other conditions

df['alzheimer_stage'] = df['symptom'].apply(classify_alzheimer_stage)

stage_names = {
    0: 'Normal/Control',
    1: 'MCI (Early Stage)',
    2: 'Alzheimer\'s Dementia',
    3: 'Other Conditions'
}

print(f"\n5. ALZHEIMER'S STAGE CLASSIFICATION (Target Variable)")
stage_counts = df['alzheimer_stage'].value_counts().sort_index()
for stage, count in stage_counts.items():
    print(f"   Stage {stage} - {stage_names[stage]}: {count} patients ({count/len(df)*100:.1f}%)")

# Save processed data
output_dir = r"c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\csvcaueeg"
os.makedirs(output_dir, exist_ok=True)

df.to_csv(os.path.join(output_dir, 'patient_metadata.csv'), index=False)
print(f"\n6. SAVED")
print(f"   Patient metadata saved to: {os.path.join(output_dir, 'patient_metadata.csv')}")

# Visualizations
print(f"\n7. GENERATING VISUALIZATIONS...")

# Create output directory for plots
plots_dir = os.path.join(output_dir, 'eda_plots')
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Age Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Age (years)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Age Distribution of Patients', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(df['age'], vert=True)
plt.ylabel('Age (years)', fontsize=12)
plt.title('Age Distribution (Box Plot)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '1_age_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Primary Diagnosis Distribution
plt.figure(figsize=(12, 6))
diagnosis_counts.plot(kind='barh', color='coral', edgecolor='black')
plt.xlabel('Number of Patients', fontsize=12)
plt.ylabel('Diagnosis', fontsize=12)
plt.title('Primary Diagnosis Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '2_diagnosis_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Alzheimer's Stage Distribution
plt.figure(figsize=(10, 6))
stage_labels = [f"Stage {s}\n{stage_names[s]}" for s in stage_counts.index]
colors = ['#2ecc71', '#f39c12', '#e74c3c', '#95a5a6']
plt.bar(range(len(stage_counts)), stage_counts.values, color=colors, edgecolor='black', alpha=0.8)
plt.xticks(range(len(stage_counts)), stage_labels, fontsize=10)
plt.ylabel('Number of Patients', fontsize=12)
plt.title('Alzheimer\'s Stage Classification Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(stage_counts.values):
    plt.text(i, v + 5, str(v), ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '3_alzheimer_stages.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Age by Alzheimer's Stage
plt.figure(figsize=(12, 6))
stage_labels_short = [stage_names[s] for s in sorted(df['alzheimer_stage'].unique())]
df_sorted = df.sort_values('alzheimer_stage')
sns.violinplot(data=df_sorted, x='alzheimer_stage', y='age', palette='Set2')
plt.xticks(range(len(stage_labels_short)), stage_labels_short, rotation=15, ha='right')
plt.xlabel('Alzheimer\'s Stage', fontsize=12)
plt.ylabel('Age (years)', fontsize=12)
plt.title('Age Distribution by Alzheimer\'s Stage', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '4_age_by_stage.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Top Symptoms
plt.figure(figsize=(12, 8))
top_symptoms = dict(symptom_counts.most_common(15))
plt.barh(list(top_symptoms.keys()), list(top_symptoms.values()), color='steelblue', edgecolor='black')
plt.xlabel('Number of Patients', fontsize=12)
plt.ylabel('Symptom', fontsize=12)
plt.title('Top 15 Most Common Symptoms', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '5_top_symptoms.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"   All plots saved to: {plots_dir}")

# Summary statistics
print(f"\n8. SUMMARY STATISTICS")
print(f"   Age by Alzheimer's Stage:")
print(df.groupby('alzheimer_stage')['age'].describe())

print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY!")
print("="*80)
