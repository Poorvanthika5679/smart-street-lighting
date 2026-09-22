"""
Machine Learning Fault Prediction Module
Implements Random Forest Classifier using Scikit-learn for predicting street lamp faults,
evaluating performance (Accuracy, Precision, Recall, F1, Confusion Matrix),
and providing individual and batch lamp fault risk scoring.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

FEATURE_COLS_NUMERIC = [
    'Operating_Hours',
    'Energy_Consumption',
    'Voltage',
    'Current',
    'Previous_Faults',
    'Lamp_Age'
]

FEATURE_COLS_CATEGORICAL = [
    'Lamp_Type',
    'Zone'
]


def train_fault_prediction_model(df):
    """
    Trains a Random Forest Classifier on the dataset.
    Returns:
    - (pipeline, metrics_dict, confusion_mat_fig, feature_imp_fig) if successful
    - None, None, None, None if Fault_Status is missing or invalid
    """
    if 'Fault_Status' not in df.columns:
        return None, None, None, None, "Fault prediction requires labeled Fault_Status data."
        
    # Check if target has at least 2 distinct classes
    unique_targets = df['Fault_Status'].dropna().unique()
    if len(unique_targets) < 2:
        return None, None, None, None, "Target 'Fault_Status' must contain both 0 (Normal) and 1 (Fault) classes for training."

    # Filter available features
    num_features = [col for col in FEATURE_COLS_NUMERIC if col in df.columns]
    cat_features = [col for col in FEATURE_COLS_CATEGORICAL if col in df.columns]
    
    if len(num_features) == 0:
        return None, None, None, None, "Insufficient numeric telemetry features for model training."

    # Prepare features and target
    X = df[num_features + cat_features].copy()
    y = df['Fault_Status'].astype(int)
    
    # Preprocessor
    transformers = []
    if num_features:
        transformers.append(('num', 'passthrough', num_features))
    if cat_features:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features))
        
    preprocessor = ColumnTransformer(transformers=transformers)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_split=4,
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    # Stratified Train/Test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate performance metrics
    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall': round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        'f1': round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    # Confusion Matrix figure
    cm = confusion_matrix(y_test, y_pred)
    cm_labels = ['Normal (0)', 'Fault (1)']
    
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted Condition", y="Actual Condition", color="Count"),
        x=cm_labels,
        y=cm_labels,
        color_continuous_scale=[[0, '#1e293b'], [0.5, '#3b82f6'], [1, '#38bdf8']]
    )
    cm_fig.update_layout(
        title=dict(text="<b>Confusion Matrix</b>", font=dict(family="Inter, sans-serif", size=14, color="#f8fafc")),
        paper_bgcolor="#1e293b",
        plot_bgcolor="#0f172a",
        font=dict(color="#f8fafc"),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    # Feature Importance
    try:
        classifier = pipeline.named_steps['classifier']
        ohe = pipeline.named_steps['preprocessor'].named_transformers_.get('cat', None)
        
        feature_names = list(num_features)
        if ohe is not None and hasattr(ohe, 'get_feature_names_out'):
            feature_names.extend(list(ohe.get_feature_names_out(cat_features)))
            
        importances = classifier.feature_importances_
        if len(feature_names) == len(importances):
            fi_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=True).tail(8)
            
            fi_fig = px.bar(
                fi_df,
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale=['#818cf8', '#38bdf8'],
                labels={'Importance': 'Relative Weight', 'Feature': 'Telemetry Metric'}
            )
            fi_fig.update_layout(
                title=dict(text="<b>Top Feature Importances</b>", font=dict(family="Inter, sans-serif", size=14, color="#f8fafc")),
                paper_bgcolor="#1e293b",
                plot_bgcolor="#0f172a",
                font=dict(color="#f8fafc"),
                margin=dict(l=40, r=40, t=50, b=40)
            )
        else:
            fi_fig = None
    except Exception:
        fi_fig = None
        
    return pipeline, metrics, cm_fig, fi_fig, None


def predict_single_lamp(pipeline, lamp_row):
    """
    Predicts fault probability and risk level for an individual lamp row.
    Returns:
    - fault_probability (float 0.0 - 100.0)
    - risk_level ('LOW', 'MEDIUM', 'HIGH')
    - risk_color (hex)
    """
    if pipeline is None:
        # Fallback heuristic if ML pipeline unavailable
        v = lamp_row.get('Voltage', 220.0)
        h = lamp_row.get('Operating_Hours', 5000.0)
        pf = lamp_row.get('Previous_Faults', 0)
        prob = 0.1
        if v < 195 or v > 245:
            prob += 0.45
        if pf >= 3:
            prob += 0.30
        if h > 10000:
            prob += 0.15
        prob = min(0.95, max(0.05, prob)) * 100.0
    else:
        df_single = pd.DataFrame([lamp_row])
        prob = float(pipeline.predict_proba(df_single)[:, 1][0]) * 100.0
        
    prob = round(prob, 1)
    
    if prob >= 70.0:
        risk_level = "HIGH"
        risk_color = "#f87171"
    elif prob >= 35.0:
        risk_level = "MEDIUM"
        risk_color = "#fbbf24"
    else:
        risk_level = "LOW"
        risk_color = "#34d399"
        
    return prob, risk_level, risk_color


def score_dataset_fault_risks(pipeline, df):
    """
    Appends 'Fault_Probability', 'Risk_Level', and 'Recommended_Action' to the dataframe.
    Used by Main Dashboard, Fault Alerts, and Maintenance pages.
    """
    df_scored = df.copy()
    
    if pipeline is not None:
        try:
            probs = pipeline.predict_proba(df_scored)[:, 1] * 100.0
        except Exception:
            probs = []
    else:
        probs = []
        
    if len(probs) != len(df_scored):
        # Heuristic scoring fallback
        probs = []
        for _, row in df_scored.iterrows():
            p, _, _ = predict_single_lamp(None, row)
            probs.append(p)
            
    df_scored['Fault_Probability'] = [round(float(p), 1) for p in probs]
    
    def get_level(p):
        if p >= 70.0:
            return 'HIGH'
        elif p >= 35.0:
            return 'MEDIUM'
        return 'LOW'
        
    df_scored['Risk_Level'] = df_scored['Fault_Probability'].apply(get_level)
    
    # Generate maintenance recommendations
    def get_action(row):
        prob = row['Fault_Probability']
        v = row.get('Voltage', 220)
        h = row.get('Operating_Hours', 5000)
        pf = row.get('Previous_Faults', 0)
        
        if prob >= 70.0 or row.get('Lamp_Status') == 'Maintenance Required':
            if v < 195 or v > 245:
                return 'Immediate electrical inspection (Voltage anomaly)'
            elif pf >= 3:
                return 'Immediate overhaul (Repeated fault history)'
            else:
                return 'Immediate inspection & ballast testing'
        elif prob >= 45.0 or h > 11000:
            if h > 11000:
                return 'Preventive maintenance (Near end-of-life hours)'
            else:
                return 'Electrical inspection (Current draw irregularity)'
        elif prob >= 35.0:
            return 'Monitor condition in next maintenance cycle'
        else:
            return 'Normal operation - standard routine check'
            
    df_scored['Recommended_Action'] = df_scored.apply(get_action, axis=1)
    
    return df_scored
