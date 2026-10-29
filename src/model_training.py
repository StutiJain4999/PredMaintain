"""
Machine Learning Training, Evaluation & Live Prediction Module
Preserves the four required algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naïve Bayes
"""

import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

CARD_BG = "#172033"
BORDER_COLOR = "#293548"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
ACCENT_BLUE = "#60A5FA"
ACCENT_PURPLE = "#A78BFA"
ACCENT_GREEN = "#34D399"
ACCENT_AMBER = "#FBBF24"
ACCENT_ROSE = "#FB7185"


def train_all_models(split_data):
    """
    Train all 4 models on X_train and evaluate on X_test.
    Returns:
        models (dict): Fitted model objects
        results (pd.DataFrame): Comparison table with Accuracy, Precision, Recall, F1, Training Time
        reports (dict): Classification report string and confusion matrix per model
    """
    X_train = split_data["X_train"]
    X_test = split_data["X_test"]
    Y_train = split_data["Y_train"]
    Y_test = split_data["Y_test"]
    
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Naive Bayes": GaussianNB()
    }
    
    records = []
    reports = {}
    
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, Y_train)
        train_time = round((time.time() - t0) * 1000, 2)  # milliseconds
        
        preds = clf.predict(X_test)
        
        acc = accuracy_score(Y_test, preds)
        prec = precision_score(Y_test, preds, zero_division=0)
        rec = recall_score(Y_test, preds, zero_division=0)
        f1 = f1_score(Y_test, preds, zero_division=0)
        cm = confusion_matrix(Y_test, preds)
        cr = classification_report(Y_test, preds, target_names=["No Failure", "Failure"], zero_division=0)
        
        records.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "Training Time (ms)": train_time
        })
        
        reports[name] = {
            "predictions": preds,
            "confusion_matrix": cm,
            "classification_report": cr
        }
        
    results_df = pd.DataFrame(records)
    return models, results_df, reports


def get_best_model(results_df, models_dict):
    """Select best-fit model using F1-score as the primary metric."""
    best_idx = results_df["F1 Score"].idxmax()
    best_name = results_df.loc[best_idx, "Model"]
    return {
        "name": best_name,
        "model": models_dict[best_name],
        "accuracy": results_df.loc[best_idx, "Accuracy"],
        "precision": results_df.loc[best_idx, "Precision"],
        "recall": results_df.loc[best_idx, "Recall"],
        "f1": results_df.loc[best_idx, "F1 Score"]
    }


def plot_model_metrics_comparison(results_df):
    """Interactive multi-metric comparison chart."""
    melted = pd.melt(
        results_df,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score"
    )
    
    fig = px.bar(
        melted, x="Model", y="Score", color="Metric",
        barmode="group",
        color_discrete_map={
            "Accuracy": ACCENT_BLUE,
            "Precision": ACCENT_PURPLE,
            "Recall": ACCENT_AMBER,
            "F1 Score": ACCENT_GREEN
        },
        text=melted["Score"].apply(lambda v: f"{v:.3f}")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title={
            "text": "Model Performance Across All Key Metrics",
            "y": 0.95, "x": 0.05,
            "font": {"size": 16, "color": TEXT_COLOR}
        },
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR},
        height=420,
        margin=dict(l=40, r=40, t=60, b=40),
        yaxis=dict(
            range=[0, 1.15],
            gridcolor=BORDER_COLOR,
            tickfont={"color": MUTED_TEXT}
        ),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            tickfont={"color": MUTED_TEXT}
        ),
        legend=dict(
            font={"color": TEXT_COLOR},
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom", y=1.02, xanchor="right", x=1
        )
    )
    return fig


def plot_metric_bar(results_df, metric_col, color=ACCENT_BLUE):
    """Bar chart for a single metric."""
    fig = px.bar(
        results_df, x="Model", y=metric_col,
        color="Model",
        color_discrete_sequence=[ACCENT_BLUE, ACCENT_GREEN, ACCENT_PURPLE, ACCENT_AMBER],
        text=results_df[metric_col].apply(lambda v: f"{v:.4f}")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title={
            "text": f"Model {metric_col} Comparison",
            "y": 0.95, "x": 0.05,
            "font": {"size": 16, "color": TEXT_COLOR}
        },
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR},
        height=380,
        showlegend=False,
        yaxis=dict(range=[0, 1.15], gridcolor=BORDER_COLOR, tickfont={"color": MUTED_TEXT}),
        xaxis=dict(gridcolor=BORDER_COLOR, tickfont={"color": MUTED_TEXT})
    )
    return fig


def plot_confusion_matrix_interactive(cm, model_name="Best Model"):
    """Interactive heatmap representing Confusion Matrix."""
    z = cm
    x = ["Predicted Healthy (0)", "Predicted Failure (1)"]
    y = ["Actual Healthy (0)", "Actual Failure (1)"]
    
    annotations = []
    labels = [
        [f"True Negative (TN)<br><b>{cm[0, 0]}</b>", f"False Positive (FP)<br><b>{cm[0, 1]}</b>"],
        [f"False Negative (FN)<br><b>{cm[1, 0]}</b>", f"True Positive (TP)<br><b>{cm[1, 1]}</b>"]
    ]
    
    for i in range(2):
        for j in range(2):
            annotations.append(dict(
                x=x[j], y=y[i],
                text=labels[i][j],
                font=dict(color=TEXT_COLOR, size=14),
                showarrow=False
            ))
            
    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        colorscale=[[0, "#111827"], [0.5, "#1E3A8A"], [1, "#3B82F6"]],
        showscale=False
    ))
    fig.update_layout(
        title={
            "text": f"Confusion Matrix: {model_name}",
            "y": 0.95, "x": 0.05,
            "font": {"size": 16, "color": TEXT_COLOR}
        },
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR},
        height=400,
        annotations=annotations,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(tickfont={"color": TEXT_COLOR}),
        yaxis=dict(tickfont={"color": TEXT_COLOR})
    )
    return fig


def plot_feature_importance_interactive(model, feature_names):
    """Feature importance plot for Decision Tree."""
    if not hasattr(model, "feature_importances_"):
        return None
        
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=True)
    
    fig = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Tealgrn",
        text=imp_df["Importance"].apply(lambda v: f"{v:.4f}")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        title={
            "text": "Decision Tree Feature Importances",
            "y": 0.95, "x": 0.05,
            "font": {"size": 16, "color": TEXT_COLOR}
        },
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR},
        height=420,
        coloraxis_showscale=False,
        yaxis=dict(tickfont={"color": TEXT_COLOR}),
        xaxis=dict(gridcolor=BORDER_COLOR, tickfont={"color": MUTED_TEXT})
    )
    return fig


def predict_machine_failure(model, scaler, feature_names, product_type, air_temp, process_temp, speed, torque, tool_wear):
    """
    Accept user inputs, preprocess with matching pipeline, and return prediction & risk insights.
    """
    temp_diff = float(process_temp) - float(air_temp)
    type_l = 1 if product_type == "L" else 0
    type_m = 1 if product_type == "M" else 0
    
    raw_dict = {
        "Air temperature [K]": [float(air_temp)],
        "Process temperature [K]": [float(process_temp)],
        "Rotational speed [rpm]": [float(speed)],
        "Torque [Nm]": [float(torque)],
        "Tool wear [min]": [float(tool_wear)],
        "Temperature Difference": [temp_diff],
        "Type_L": [type_l],
        "Type_M": [type_m]
    }
    input_df = pd.DataFrame(raw_dict)[feature_names]
    
    # Scale strictly with the pre-fitted scaler
    input_scaled = scaler.transform(input_df)
    
    prediction = int(model.predict(input_scaled)[0])
    
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(input_scaled)[0][1])
    else:
        prob = 1.0 if prediction == 1 else 0.0
        
    # Determine risk category
    if prob >= 0.50 or prediction == 1:
        risk_level = "CRITICAL / HIGH RISK"
        badge_color = ACCENT_ROSE
    elif prob >= 0.25:
        risk_level = "MODERATE RISK"
        badge_color = ACCENT_AMBER
    else:
        risk_level = "HEALTHY / LOW RISK"
        badge_color = ACCENT_GREEN
        
    # Generate contextual maintenance recommendation
    recommendations = []
    if tool_wear >= 180:
        recommendations.append("Tool wear is dangerously high (>180 min); schedule cutting tool replacement.")
    if torque >= 55:
        recommendations.append("Spindle torque exceeds 55 Nm; inspect workpiece feed rate and spindle resistance.")
    if temp_diff < 9.0:
        recommendations.append("Thermal dissipation delta is constricted (<9 K); check coolant flow and air filters.")
    if speed < 1350:
        recommendations.append("Spindle speed is abnormally depressed; check drive motor load.")
    if not recommendations:
        if prediction == 1:
            recommendations.append("Multi-factor operational stress detected; perform preventive visual inspection.")
        else:
            recommendations.append("Telemetry operates within nominal tolerances. Normal maintenance schedule applies.")
            
    return {
        "prediction": prediction,
        "prediction_label": "FAILURE IMMINENT" if prediction == 1 else "NORMAL OPERATION",
        "failure_probability": prob,
        "risk_level": risk_level,
        "badge_color": badge_color,
        "temp_diff": temp_diff,
        "recommendations": recommendations
    }
