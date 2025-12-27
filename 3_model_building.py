"""
Model Building and Training for Alzheimer's Stage Prediction
This script trains multiple models and selects the best one
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, f1_score, roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ALZHEIMER'S STAGE PREDICTION - MODEL BUILDING")
print("="*80)

# Paths
data_dir = r"c:\Users\Anisha\Desktop\isa-2\caueeg-dataset\csvcaueeg"
data_file = os.path.join(data_dir, 'processed_features.csv')

# Load data
print("\n1. LOADING PROCESSED DATA...")
df = pd.read_csv(data_file)
print(f"   Total samples: {len(df)}")
print(f"   Total features: {len(df.columns) - 3}")

# Separate features and target
X = df.drop(['serial', 'age', 'alzheimer_stage'], axis=1)
y = df['alzheimer_stage']

# Add age as a feature
X['age'] = df['age']

print(f"\n   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# Class distribution
print("\n2. CLASS DISTRIBUTION:")
stage_names = {
    0: 'Normal/Control',
    1: 'MCI (Early Stage)',
    2: 'Alzheimer\'s Dementia'
}

for stage in sorted(y.unique()):
    count = (y == stage).sum()
    percentage = count / len(y) * 100
    print(f"   Stage {stage} - {stage_names[stage]}: {count} samples ({percentage:.1f}%)")

# Split data
print("\n3. SPLITTING DATA...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# Feature scaling
print("\n4. FEATURE SCALING...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_file = os.path.join(data_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_file)
print(f"   Scaler saved to: {scaler_file}")

# Feature selection
print("\n5. FEATURE SELECTION...")
print("   Selecting top features using mutual information...")

# Select top 100 features
k_best = min(100, X_train_scaled.shape[1])
selector = SelectKBest(score_func=mutual_info_classif, k=k_best)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# Get selected feature names
selected_features_mask = selector.get_support()
selected_features = X.columns[selected_features_mask].tolist()
print(f"   Selected {len(selected_features)} features")

# Save feature selector
selector_file = os.path.join(data_dir, 'feature_selector.pkl')
joblib.dump(selector, selector_file)
print(f"   Feature selector saved to: {selector_file}")

# Save selected feature names
feature_names_file = os.path.join(data_dir, 'selected_features.txt')
with open(feature_names_file, 'w') as f:
    for feat in selected_features:
        f.write(f"{feat}\n")
print(f"   Selected feature names saved to: {feature_names_file}")

# Model training
print("\n6. TRAINING MODELS...")
print("   Testing multiple algorithms...\n")

models = {
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                           min_samples_split=5, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=5, 
                                                    learning_rate=0.1, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42)
}

results = {}
trained_models = {}

for name, model in models.items():
    print(f"   Training {name}...")
    
    # Train model
    model.fit(X_train_selected, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_selected)
    y_pred_proba = model.predict_proba(X_test_selected)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')
    
    results[name] = {
        'accuracy': accuracy,
        'f1_score': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    trained_models[name] = model
    
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      F1-Score: {f1:.4f}")
    print(f"      CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")

# Select best model
print("\n7. SELECTING BEST MODEL...")
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = trained_models[best_model_name]
best_results = results[best_model_name]

print(f"   Best Model: {best_model_name}")
print(f"   Test Accuracy: {best_results['accuracy']:.4f}")
print(f"   Test F1-Score: {best_results['f1_score']:.4f}")

# Save best model
model_file = os.path.join(data_dir, 'alzheimer_model.pkl')
joblib.dump(best_model, model_file)
print(f"   Model saved to: {model_file}")

# Detailed evaluation
print("\n8. DETAILED EVALUATION OF BEST MODEL:")
print("\n   Classification Report:")
print(classification_report(y_test, best_results['y_pred'], 
                          target_names=[stage_names[i] for i in sorted(y.unique())]))

# Confusion Matrix
cm = confusion_matrix(y_test, best_results['y_pred'])
print("\n   Confusion Matrix:")
print(cm)

# Visualizations
print("\n9. GENERATING VISUALIZATIONS...")
plots_dir = os.path.join(data_dir, 'model_plots')
os.makedirs(plots_dir, exist_ok=True)

# Plot 1: Model Comparison
plt.figure(figsize=(12, 6))
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
f1_scores = [results[m]['f1_score'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

plt.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue', edgecolor='black')
plt.bar(x + width/2, f1_scores, width, label='F1-Score', color='coral', edgecolor='black')

plt.xlabel('Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x, model_names, rotation=15, ha='right')
plt.legend()
plt.ylim([0, 1.1])
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '1_model_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[stage_names[i] for i in sorted(y.unique())],
            yticklabels=[stage_names[i] for i in sorted(y.unique())],
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Stage', fontsize=12)
plt.ylabel('True Stage', fontsize=12)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '2_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Feature Importance (for tree-based models)
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = best_model.feature_importances_
    indices = np.argsort(feature_importance)[-20:]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importance[indices], color='steelblue', edgecolor='black')
    plt.yticks(range(len(indices)), [selected_features[i] for i in indices], fontsize=9)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top 20 Most Important Features - {best_model_name}', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, '3_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Plot 4: ROC Curves (One-vs-Rest)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))
n_classes = y_test_bin.shape[1]

plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']

for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], best_results['y_pred_proba'][:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{stage_names[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, '4_roc_curves.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"   All plots saved to: {plots_dir}")

# Save model metadata
print("\n10. SAVING MODEL METADATA...")
metadata = {
    'model_name': best_model_name,
    'accuracy': float(best_results['accuracy']),
    'f1_score': float(best_results['f1_score']),
    'cv_mean': float(best_results['cv_mean']),
    'cv_std': float(best_results['cv_std']),
    'n_features': len(selected_features),
    'n_samples_train': len(X_train),
    'n_samples_test': len(X_test),
    'stage_names': stage_names
}

import json
metadata_file = os.path.join(data_dir, 'model_metadata.json')
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=4)
print(f"   Metadata saved to: {metadata_file}")

print("\n" + "="*80)
print("MODEL BUILDING COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nFINAL MODEL: {best_model_name}")
print(f"Test Accuracy: {best_results['accuracy']:.4f}")
print(f"Test F1-Score: {best_results['f1_score']:.4f}")
print("="*80)
