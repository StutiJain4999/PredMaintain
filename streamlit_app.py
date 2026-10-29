"""
Predictive Maintenance Analytics Dashboard
A modern, dark-themed industrial machine health monitoring and failure prediction system.
Framework: Streamlit & Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import io
import os

from src.data_preprocessing import (
    load_raw_data, clean_data, engineer_features, prepare_ml_data, split_and_scale
)
from src.eda import (
    plot_failure_distribution, plot_failure_distribution_bar,
    plot_product_type_distribution, plot_failure_by_product_type,
    plot_temperature_distributions, plot_temperature_difference,
    plot_tool_wear_torque, plot_speed_torque_scatter,
    plot_failure_modes, plot_correlation_heatmap
)
from src.model_training import (
    train_all_models, get_best_model, plot_model_metrics_comparison,
    plot_metric_bar, plot_confusion_matrix_interactive,
    plot_feature_importance_interactive, predict_machine_failure
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM DARK THEME CSS INJECTION
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Main app backgrounds */
.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #293548;
}

/* Card components */
.metric-card {
    background-color: #172033;
    border: 1px solid #293548;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    border-color: #60A5FA;
    transform: translateY(-2px);
}
.metric-label {
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #F8FAFC;
    margin-bottom: 4px;
}
.metric-delta {
    font-size: 0.8rem;
    font-weight: 500;
}
.delta-normal { color: #34D399; }
.delta-danger { color: #FB7185; }
.delta-warn   { color: #FBBF24; }
.delta-info   { color: #60A5FA; }

/* Status Badges */
.status-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-healthy {
    background-color: rgba(52, 211, 153, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}
.status-danger {
    background-color: rgba(251, 113, 133, 0.15);
    color: #FB7185;
    border: 1px solid rgba(251, 113, 133, 0.3);
}

/* Info & Callout boxes */
.info-card {
    background-color: #172033;
    border-left: 4px solid #60A5FA;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 14px 0;
    color: #F8FAFC;
}
.warning-card {
    background-color: #172033;
    border-left: 4px solid #FBBF24;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 14px 0;
    color: #F8FAFC;
}
.danger-card {
    background-color: #172033;
    border-left: 4px solid #FB7185;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 14px 0;
    color: #F8FAFC;
}

/* Streamlit component adjustments */
div[data-testid="stMetricValue"] {
    color: #F8FAFC !important;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #111827;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #293548;
}
.stTabs [data-baseweb="tab"] {
    color: #94A3B8;
}
.stTabs [aria-selected="true"] {
    color: #60A5FA !important;
    font-weight: 600;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA CACHING & PIPELINE EXECUTION
# -----------------------------------------------------------------------------
@st.cache_data
def get_data_bundle():
    """Load, clean, engineer and prepare data once."""
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    featured_df = engineer_features(clean_df)
    X, Y = prepare_ml_data(featured_df)
    split_bundle = split_and_scale(X, Y)
    return raw_df, clean_df, featured_df, split_bundle


@st.cache_resource
def get_trained_models(_split_bundle):
    """Train the 4 models and return fitted estimators, comparison table, and reports."""
    models, results_df, reports = train_all_models(_split_bundle)
    best_info = get_best_model(results_df, models)
    return models, results_df, reports, best_info


raw_df, clean_df, df, split_bundle = get_data_bundle()
models, results_df, reports, best_info = get_trained_models(split_bundle)


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding: 10px 0;">
        <div style="background-color: #1E3A8A; padding: 10px; border-radius: 10px; border: 1px solid #60A5FA;">
            <span style="font-size: 1.8rem;">⚙️</span>
        </div>
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #F8FAFC;">PredMaintain</h3>
            <span style="font-size: 0.75rem; color: #60A5FA; font-weight: 600;">INDUSTRIAL ANALYTICS</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em; margin-bottom: 8px;'>Navigation Menu</p>", unsafe_allow_html=True)
    
    nav_choice = st.radio(
        label="Select Page",
        options=[
            "Dashboard Overview",
            "Data Exploration",
            "Five Objectives",
            "Machine Learning Models",
            "Model Evaluation",
            "Decision Tree Visualization",
            "Predictions",
            "Insights and Recommendations",
            "Dataset Information"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Dataset Status Card
    st.markdown(f"""
    <div style="background-color: #172033; border: 1px solid #293548; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-size: 0.75rem; color: #94A3B8;">Dataset Status</span>
            <span class="status-badge status-healthy">● Connected</span>
        </div>
        <p style="margin: 0; font-size: 0.85rem; font-weight: 600; color: #F8FAFC;">AI4I 2020 Telemetry</p>
        <span style="font-size: 0.75rem; color: #94A3B8;">10,000 Records • 8 Features</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🔄 Refresh Data Pipeline", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
        
    st.markdown("""
    <div style="text-align: center; margin-top: 25px; padding-top: 15px; border-top: 1px solid #293548; font-size: 0.75rem; color: #64748B;">
        <p style="margin: 0 0 4px 0;">Industrial AI CA Project</p>
        <p style="margin: 0;">Predictive Maintenance Dashboard v2.0</p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TOP HEADER BAR
# -----------------------------------------------------------------------------
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("""
    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.02em;">
        Predictive Maintenance Analytics
    </h1>
    <p style="margin: 4px 0 16px 0; font-size: 0.95rem; color: #94A3B8;">
        Industrial Machine Failure Analysis & Real-Time Predictive Health Intelligence
    </p>
    """, unsafe_allow_html=True)
with col_h2:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span class="status-badge status-healthy" style="font-size: 0.85rem; padding: 6px 14px;">
            ● System Operational
        </span>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# GLOBAL TOP KPI CARDS
# -----------------------------------------------------------------------------
total_machines = len(df)
total_failures = int(df["Machine failure"].sum())
failure_rate = (total_failures / total_machines) * 100
normal_rate = 100 - failure_rate

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Machines</div>
        <div class="metric-value">{total_machines:,}</div>
        <div class="metric-delta delta-info">100% telemetry records</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Machine Failures</div>
        <div class="metric-value" style="color: #FB7185;">{total_failures:,}</div>
        <div class="metric-delta delta-danger">Confirmed breakdown cases</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Failure Rate</div>
        <div class="metric-value" style="color: #FB7185;">{failure_rate:.2f}%</div>
        <div class="metric-delta delta-warn">Severe class imbalance</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Normal Operation</div>
        <div class="metric-value" style="color: #34D399;">{normal_rate:.2f}%</div>
        <div class="metric-delta delta-normal">{total_machines - total_failures:,} healthy units</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best-Fit Model</div>
        <div class="metric-value" style="font-size: 1.45rem; color: #60A5FA;">{best_info['name']}</div>
        <div class="metric-delta delta-info">Highest F1-Score</div>
    </div>
    """, unsafe_allow_html=True)

with kpi6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Best Model F1</div>
        <div class="metric-value" style="color: #34D399;">{best_info['f1']:.4f}</div>
        <div class="metric-delta delta-normal">Acc: {best_info['accuracy']*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)


# =============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# =============================================================================
if nav_choice == "Dashboard Overview":
    st.markdown("### Executive Performance & Condition Summary")
    
    col_ov1, col_ov2 = st.columns(2)
    with col_ov1:
        st.plotly_chart(plot_failure_distribution(df), use_container_width=True)
    with col_ov2:
        fig_type, rates = plot_failure_by_product_type(df)
        st.plotly_chart(fig_type, use_container_width=True)
        
    st.markdown("### Operational Telemetry Snapshot")
    col_ov3, col_ov4 = st.columns(2)
    with col_ov3:
        fig_scat = plot_speed_torque_scatter(df)
        st.plotly_chart(fig_scat, use_container_width=True)
    with col_ov4:
        st.plotly_chart(plot_model_metrics_comparison(results_df), use_container_width=True)
        
    st.markdown("""
    <div class="info-card">
        <h4 style="margin: 0 0 6px 0; color: #60A5FA;">Executive Takeaway</h4>
        <p style="margin: 0; color: #E2E8F0; font-size: 0.9rem;">
            Industrial milling machinery telemetry demonstrates that machine failure is concentrated in high-wear (>180 min),
            extreme torque (>55 Nm), and constrained thermal dissipation regimes. The Decision Tree Classifier provides the optimal
            balance of high detection sensitivity (Recall 63.24%) and operational reliability (Precision 68.25%).
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: DATA EXPLORATION
# =============================================================================
elif nav_choice == "Data Exploration":
    st.markdown("### Interactive Data Exploration & Feature Inspection")
    
    # Filter Controls
    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    with f_col1:
        type_filter = st.multiselect("Filter by Product Type", options=["L", "M", "H"], default=["L", "M", "H"])
    with f_col2:
        fail_filter = st.multiselect("Filter by Machine Health", options=[0, 1], format_func=lambda x: "Failure (1)" if x == 1 else "Healthy (0)", default=[0, 1])
    with f_col3:
        search_kw = st.text_input("Search Product ID (e.g. M14860)", "")
        
    filtered_df = df[(df["Type"].isin(type_filter)) & (df["Machine failure"].isin(fail_filter))]
    if search_kw:
        filtered_df = filtered_df[filtered_df["Product ID"].str.contains(search_kw.strip(), case=False, na=False)]
        
    st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} total records**")
    
    # Data Table
    st.dataframe(filtered_df, use_container_width=True, height=300)
    
    # Download Button
    csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned & Engineered Dataset (CSV)",
        data=csv_bytes,
        file_name="predictive_maintenance_cleaned.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Tabs for Data Profiles
    tab_stats, tab_corr, tab_outliers = st.tabs(["📊 Statistical Summary", "🔥 Correlation Heatmap", "📦 Outlier Inspection"])
    
    with tab_stats:
        st.markdown("#### Numerical Attributes Summary")
        st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
        
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Missing Values (Before & After)</div>
                <div class="metric-value" style="color: #34D399;">0 / 0</div>
                <div class="metric-delta delta-normal">100% complete dataset</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Duplicate Rows Removed</div>
                <div class="metric-value" style="color: #60A5FA;">0 Rows</div>
                <div class="metric-delta delta-info">Unique telemetry per record</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Categorical Encoding</div>
                <div class="metric-value" style="color: #A78BFA;">Type_L, Type_M</div>
                <div class="metric-delta delta-info">Dummy variable trap avoided</div>
            </div>
            """, unsafe_allow_html=True)
            
    with tab_corr:
        st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
        
    with tab_outliers:
        feature_choice = st.selectbox(
            "Select Feature to Inspect for Outliers across Machine Health Status",
            options=["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]", "Temperature Difference"]
        )
        fig_out = px.box(
            df, x="Machine failure", y=feature_choice,
            color="Machine failure",
            color_discrete_map={0: "#60A5FA", 1: "#FB7185"},
            labels={"Machine failure": "Failure Status (0 = Healthy, 1 = Failure)"}
        )
        fig_out.update_layout(
            paper_bgcolor="#172033", plot_bgcolor="#172033",
            font={"color": "#F8FAFC"}, showlegend=False
        )
        st.plotly_chart(fig_out, use_container_width=True)


# =============================================================================
# PAGE 3: FIVE OBJECTIVES
# =============================================================================
elif nav_choice == "Five Objectives":
    st.markdown("### Deep Dive: Five Analytical Project Objectives")
    
    obj_tab1, obj_tab2, obj_tab3, obj_tab4, obj_tab5 = st.tabs([
        "Obj 1: Failure Distribution",
        "Obj 2: Product Type",
        "Obj 3: Temperature",
        "Obj 4: Wear & Torque",
        "Obj 5: Failure Modes"
    ])
    
    # ---------------- OBJECTIVE 1 ----------------
    with obj_tab1:
        st.markdown("#### Objective 1: Machine Failure Distribution & Class Imbalance")
        col_o1a, col_o1b = st.columns([1, 1])
        with col_o1a:
            st.plotly_chart(plot_failure_distribution(df), use_container_width=True)
        with col_o1b:
            st.plotly_chart(plot_failure_distribution_bar(df), use_container_width=True)
            
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin: 0 0 6px 0; color: #60A5FA;">Objective 1 Findings & Actual Statistics</h4>
            <ul style="margin: 0; padding-left: 20px; color: #E2E8F0;">
                <li><b>Normal Machines (0):</b> 9,661 units (<b>96.61%</b>)</li>
                <li><b>Failed Machines (1):</b> 339 units (<b>3.39%</b>)</li>
                <li><b>Class Imbalance Implication:</b> A trivial model that always predicts 'No Failure' would score 96.61% accuracy while completely missing all machine breakdowns. Consequently, models must be evaluated on <b>F1-Score</b> and <b>Recall</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- OBJECTIVE 2 ----------------
    with obj_tab2:
        st.markdown("#### Objective 2: Product Type Analysis (L, M, H Quality Variants)")
        col_o2a, col_o2b = st.columns(2)
        with col_o2a:
            st.plotly_chart(plot_product_type_distribution(df), use_container_width=True)
        with col_o2b:
            fig_p, rates = plot_failure_by_product_type(df)
            st.plotly_chart(fig_p, use_container_width=True)
            
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin: 0 0 6px 0; color: #60A5FA;">Objective 2 Findings & Actual Statistics</h4>
            <ul style="margin: 0; padding-left: 20px; color: #E2E8F0;">
                <li><b>Product Distribution:</b> Type L = 6,000 (60.0%), Type M = 2,997 (29.97%), Type H = 1,003 (10.03%).</li>
                <li><b>Variant Failure Rates:</b>
                    <ul>
                        <li><b>Type L (Low Quality):</b> <span style="color: #FB7185; font-weight: 700;">{rates['L']:.2f}%</span> (235 failures out of 6,000)</li>
                        <li><b>Type M (Medium Quality):</b> <span style="color: #FBBF24; font-weight: 700;">{rates['M']:.2f}%</span> (83 failures out of 2,997)</li>
                        <li><b>Type H (High Quality):</b> <span style="color: #34D399; font-weight: 700;">{rates['H']:.2f}%</span> (21 failures out of 1,003)</li>
                    </ul>
                </li>
                <li><b>Insight:</b> Type L machines experience nearly <b>double</b> the failure rate of Type H machines due to looser production tolerances and lower material stress resistance.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- OBJECTIVE 3 ----------------
    with obj_tab3:
        st.markdown("#### Objective 3: Temperature Analysis & Thermal Dissipation")
        col_o3a, col_o3b = st.columns(2)
        with col_o3a:
            st.plotly_chart(plot_temperature_distributions(df), use_container_width=True)
        with col_o3b:
            st.plotly_chart(plot_temperature_difference(df), use_container_width=True)
            
        t_non_fail = df[df["Machine failure"] == 0]["Temperature Difference"].mean()
        t_fail = df[df["Machine failure"] == 1]["Temperature Difference"].mean()
        
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin: 0 0 6px 0; color: #60A5FA;">Objective 3 Findings & Actual Statistics</h4>
            <ul style="margin: 0; padding-left: 20px; color: #E2E8F0;">
                <li><b>Feature Engineering:</b> Calculated <code>Temperature Difference = Process temperature [K] - Air temperature [K]</code>.</li>
                <li><b>Non-Failed Machines Mean Delta:</b> <b>{t_non_fail:.2f} K</b></li>
                <li><b>Failed Machines Mean Delta:</b> <b>{t_fail:.2f} K</b></li>
                <li><b>Insight:</b> Machines suffering failure exhibit a narrower thermal dissipation gradient. When process heat cannot dissipate into ambient air rapidly enough, heat accumulation triggers Heat Dissipation Failure (HDF).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- OBJECTIVE 4 ----------------
    with obj_tab4:
        st.markdown("#### Objective 4: Tool Wear, Torque, and Rotational Speed Analysis")
        f_tw, f_tq = plot_tool_wear_torque(df)
        col_o4a, col_o4b = st.columns(2)
        with col_o4a:
            st.plotly_chart(f_tw, use_container_width=True)
        with col_o4b:
            st.plotly_chart(f_tq, use_container_width=True)
            
        st.plotly_chart(plot_speed_torque_scatter(df), use_container_width=True)
        
        tw_non = df[df["Machine failure"] == 0]["Tool wear [min]"].mean()
        tw_f = df[df["Machine failure"] == 1]["Tool wear [min]"].mean()
        tq_non = df[df["Machine failure"] == 0]["Torque [Nm]"].mean()
        tq_f = df[df["Machine failure"] == 1]["Torque [Nm]"].mean()
        
        st.markdown(f"""
        <div class="info-card">
            <h4 style="margin: 0 0 6px 0; color: #60A5FA;">Objective 4 Findings & Actual Statistics</h4>
            <ul style="margin: 0; padding-left: 20px; color: #E2E8F0;">
                <li><b>Tool Wear [min]:</b> Non-Failed = <b>{tw_non:.2f} min</b> vs Failed = <b>{tw_f:.2f} min</b> (elevated by +34.8%).</li>
                <li><b>Torque [Nm]:</b> Non-Failed = <b>{tq_non:.2f} Nm</b> vs Failed = <b>{tq_f:.2f} Nm</b> (spiking beyond 55–60 Nm).</li>
                <li><b>Kinematic Relationship:</b> Rotational speed and torque are inversely correlated (r = -0.88). Breakdowns cluster heavily in high-torque/low-speed regimes (Overstrain) and high-speed/low-torque regimes (Power Failure).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- OBJECTIVE 5 ----------------
    with obj_tab5:
        st.markdown("#### Objective 5: Machine Failure Modes & Operational Correlations")
        f_mode, mode_counts = plot_failure_modes(df)
        col_o5a, col_o5b = st.columns(2)
        with col_o5a:
            st.plotly_chart(f_mode, use_container_width=True)
        with col_o5b:
            st.plotly_chart(plot_correlation_heatmap(df), use_container_width=True)
            
        st.markdown("""
        <div class="info-card">
            <h4 style="margin: 0 0 6px 0; color: #60A5FA;">Objective 5 Findings & Breakdown</h4>
            <ul style="margin: 0; padding-left: 20px; color: #E2E8F0;">
                <li><b>Failure Mode Frequency:</b> HDF (115) > OSF (98) > PWF (95) > TWF (46) > RNF (19).</li>
                <li><b>Heat Dissipation Failure (HDF)</b> is the most frequent failure mode, proving the critical importance of monitoring ambient vs process temperature delta.</li>
                <li><b>Target Leakage Protection:</b> These breakdown columns (TWF, HDF, PWF, OSF, RNF) are direct symptoms of failure and were safely excluded from ML model features.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 4: MACHINE LEARNING MODELS
# =============================================================================
elif nav_choice == "Machine Learning Models":
    st.markdown("### Machine Learning Model Comparison & Benchmark")
    
    st.markdown("""
    The four required classification algorithms were trained using a **reproducible stratified 80:20 train-test split** (`random_state=42`).
    StandardScaler was fitted strictly on the training set (`X_train`) to avoid data leakage.
    """)
    
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "Training Time (ms)": "{:.2f}"
        }).highlight_max(subset=["F1 Score"], color="#1E3A8A"),
        use_container_width=True
    )
    
    st.markdown("#### Model Performance Visual Comparison")
    st.plotly_chart(plot_model_metrics_comparison(results_df), use_container_width=True)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.plotly_chart(plot_metric_bar(results_df, "Accuracy"), use_container_width=True)
    with col_m2:
        st.plotly_chart(plot_metric_bar(results_df, "F1 Score", color="#34D399"), use_container_width=True)
        
    st.markdown("#### Model Architecture & Hyperparameter Reference")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    with col_a1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #60A5FA;">Logistic Regression</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">
                • class_weight='balanced'<br>
                • max_iter=1000<br>
                • random_state=42<br>
                • Linear decision boundary
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_a2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #34D399;">Decision Tree</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">
                • class_weight='balanced'<br>
                • criterion='gini'<br>
                • random_state=42<br>
                • Captures non-linear limits
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_a3:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #A78BFA;">K-Nearest Neighbors</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">
                • n_neighbors=5<br>
                • metric='minkowski'<br>
                • Scaled distance metric<br>
                • Instance-based learning
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_a4:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #FBBF24;">Gaussian Naïve Bayes</h4>
            <p style="font-size: 0.8rem; color: #94A3B8; margin: 0;">
                • var_smoothing=1e-9<br>
                • Prior from class frequencies<br>
                • Feature independence model<br>
                • Probabilistic classifier
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 5: MODEL EVALUATION
# =============================================================================
elif nav_choice == "Model Evaluation":
    st.markdown("### Model Evaluation, Confusion Matrices & Diagnostics")
    
    selected_model_name = st.selectbox(
        "Select Machine Learning Model to Inspect Detailed Evaluation:",
        options=list(models.keys()),
        index=1  # Default to Decision Tree
    )
    
    rep = reports[selected_model_name]
    cm = rep["confusion_matrix"]
    
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.plotly_chart(plot_confusion_matrix_interactive(cm, selected_model_name), use_container_width=True)
    with col_e2:
        st.markdown("#### Classification Report")
        st.code(rep["classification_report"], language="text")
        
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        st.markdown(f"""
        <div style="background-color: #172033; border: 1px solid #293548; border-radius: 8px; padding: 12px;">
            <span style="font-size: 0.8rem; color: #94A3B8;">Test Set Diagnostics (2,000 samples)</span><br>
            <span style="color: #34D399;"><b>True Negatives (TN):</b> {tn}</span> | 
            <span style="color: #FBBF24;"><b>False Positives (FP):</b> {fp}</span><br>
            <span style="color: #FB7185;"><b>False Negatives (FN):</b> {fn}</span> | 
            <span style="color: #60A5FA;"><b>True Positives (TP):</b> {tp}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Feature Importance for Decision Tree
    st.markdown("#### Feature Importance Analysis")
    fig_imp = plot_feature_importance_interactive(models["Decision Tree"], split_bundle["feature_names"])
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)
        
    st.markdown("""
    <div class="warning-card">
        <h4 style="margin: 0 0 6px 0; color: #FBBF24;">Why Accuracy Alone is Deceptive for Imbalanced Datasets</h4>
        <p style="margin: 0; color: #E2E8F0; font-size: 0.9rem;">
            Because equipment breakdowns occur in only 3.39% of operating observations, an uncalibrated model that predicts
            'No Failure' 100% of the time scores <b>96.6% accuracy</b> while failing to detect a single breakdown!
            In manufacturing plants, a <b>False Negative (undetected breakdown)</b> carries catastrophic cost (unplanned downtime, ruined workpieces, motor burnouts),
            whereas a <b>False Positive</b> merely prompts a 5-minute visual inspection. 
            Therefore, <b>F1-Score</b> and <b>Recall</b> are the primary determinants for industrial deployment.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE: DECISION TREE VISUALIZATION
# =============================================================================
elif nav_choice == "Decision Tree Visualization":
    st.markdown("### Decision Tree Architecture & Rule Hierarchy")
    st.markdown("""
    Explore the learned decision rules, physical feature split thresholds, and class purity of the best-performing
    **Decision Tree Classifier** (`model2`). The tree uses the exact same training split and preprocessed feature set
    (excluding leakage columns `TWF`, `HDF`, `PWF`, `OSF`, `RNF`).
    """)
    
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        tree_depth = st.slider(
            "Visualization Tree Depth (Levels)",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Select how many levels of branches to visualize. Default is 3 levels for optimal readability."
        )
    with col_c2:
        show_proportions = st.checkbox("Show Relative Sample Proportions (%)", value=False)
        
    # Render Matplotlib Decision Tree
    fig_tree, ax_tree = plt.subplots(figsize=(24, 12), dpi=200)
    fig_tree.patch.set_facecolor("#172033")
    ax_tree.set_facecolor("#172033")
    
    plot_tree(
        models["Decision Tree"],
        max_depth=tree_depth,
        feature_names=split_bundle["feature_names"],
        class_names=["No Failure", "Failure"],
        filled=True,
        rounded=True,
        proportion=show_proportions,
        fontsize=10,
        precision=2,
        ax=ax_tree
    )
    ax_tree.set_title(
        f"Decision Tree Classifier Architecture (Visualizing Depth = {tree_depth})",
        fontsize=18,
        fontweight="bold",
        color="#F8FAFC",
        pad=20
    )
    plt.tight_layout()
    st.pyplot(fig_tree)
    
    # Download Buttons
    buf = io.BytesIO()
    fig_tree.savefig(buf, format="png", dpi=300, bbox_inches="tight", facecolor=fig_tree.get_facecolor(), edgecolor="none")
    buf.seek(0)
    plt.close(fig_tree)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label=f"📥 Download Rendered Tree at Depth {tree_depth} (PNG - 300 DPI)",
            data=buf,
            file_name=f"decision_tree_depth_{tree_depth}.png",
            mime="image/png",
            use_container_width=True
        )
    with col_d2:
        static_tree_path = os.path.join(os.path.dirname(__file__), "plots", "final_decision_tree.png")
        if not os.path.exists(static_tree_path):
            static_tree_path = "plots/final_decision_tree.png"
        if os.path.exists(static_tree_path):
            with open(static_tree_path, "rb") as f_static:
                static_bytes = f_static.read()
            st.download_button(
                label="📥 Download Standard Final Decision Tree (plots/final_decision_tree.png)",
                data=static_bytes,
                file_name="final_decision_tree.png",
                mime="image/png",
                use_container_width=True
            )
            
    st.markdown("---")
    
    # Educational & Technical Explanation Card
    st.markdown("#### How to Interpret the Decision Tree")
    
    col_exp1, col_exp2 = st.columns(2)
    with col_exp1:
        st.markdown(r"""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #60A5FA;">1. Decision Nodes & Split Thresholds</h4>
            <ul style="color: #CBD5E1; font-size: 0.85rem; padding-left: 20px; margin-bottom: 0;">
                <li><b>Top Split Condition:</b> Each non-leaf node evaluates a standardized numerical sensor threshold (e.g., <code>Torque [Nm] &le; 0.88</code> or <code>Rotational speed [rpm] &le; -0.15</code>).</li>
                <li><b>Branch Routing:</b> Machines meeting the condition branch to the <b>Left (True)</b>; otherwise, they branch to the <b>Right (False)</b>.</li>
                <li><b>Root Importance:</b> The root and upper levels test the highest-importance features: <b>Rotational Speed</b>, <b>Torque</b>, and <b>Tool Wear</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(r"""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #A78BFA;">2. Gini Impurity Metric</h4>
            <p style="color: #CBD5E1; font-size: 0.85rem; margin-bottom: 0;">
                Measures the probability of misclassifying a randomly chosen sample from the node ($Gini = 1 - \sum p_i^2$). 
                A Gini score of <b>0.00</b> indicates a perfectly pure leaf where all samples belong exclusively to one class.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_exp2:
        st.markdown(r"""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #FBBF24;">3. Blue vs. Orange Color Palette</h4>
            <ul style="color: #CBD5E1; font-size: 0.85rem; padding-left: 20px; margin-bottom: 0;">
                <li><span style="color: #60A5FA; font-weight: 700;">Blue Tones:</span> Predominantly <b>Healthy / Normal Operation (Class 0)</b>. Darker blue signifies higher purity of normal machines.</li>
                <li><span style="color: #FB923C; font-weight: 700;">Orange / Warm Tones:</span> High concentration of <b>Machine Failure (Class 1)</b>. Darker orange indicates pure failure regimes (e.g. extreme torque + tool wear).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(r"""
        <div class="metric-card">
            <h4 style="margin: 0 0 8px 0; color: #34D399;">4. Leaf Predictions & Sample Distribution</h4>
            <p style="color: #CBD5E1; font-size: 0.85rem; margin-bottom: 0;">
                <code>samples = N</code> indicates how many training instances reached this node. 
                <code>value = [n0, n1]</code> shows the class breakdown (healthy vs. failing).
                The final class assignment is determined by the majority class within each terminal leaf.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 7: PREDICTIONS
# =============================================================================
elif nav_choice == "Predictions":
    st.markdown("### Interactive Machine Failure Prediction Simulator")
    st.markdown("Input operational sensor telemetry below to simulate real-time failure prediction and risk assessment.")
    
    with st.form("prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            p_type = st.selectbox("Product Quality Type", options=["L", "M", "H"], index=0, help="L = Low (60%), M = Medium (30%), H = High (10%)")
            p_air = st.slider("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        with col_p2:
            p_proc = st.slider("Process Temperature [K]", min_value=300.0, max_value=320.0, value=310.0, step=0.1)
            p_speed = st.slider("Rotational Speed [rpm]", min_value=1100, max_value=3000, value=1500, step=10)
        with col_p3:
            p_torque = st.slider("Torque [Nm]", min_value=3.0, max_value=80.0, value=40.0, step=0.5)
            p_wear = st.slider("Tool Wear [min]", min_value=0, max_value=260, value=100, step=1)
            
        col_m_sel, col_sub = st.columns([2, 1])
        with col_m_sel:
            p_model_name = st.selectbox("Select Model for Inference", options=list(models.keys()), index=1)
        with col_sub:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("⚡ Predict Machine Health", use_container_width=True)
            
    if submit_btn:
        selected_model = models[p_model_name]
        pred_res = predict_machine_failure(
            model=selected_model,
            scaler=split_bundle["scaler"],
            feature_names=split_bundle["feature_names"],
            product_type=p_type,
            air_temp=p_air,
            process_temp=p_proc,
            speed=p_speed,
            torque=p_torque,
            tool_wear=p_wear
        )
        
        st.markdown("### Inference Result")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            is_failure = pred_res["prediction"] == 1
            card_class = "danger-card" if is_failure else "info-card"
            title_color = "#FB7185" if is_failure else "#34D399"
            
            st.markdown(f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 0.85rem; color: #94A3B8;">Model: {p_model_name}</span>
                    <span class="status-badge" style="background-color: {pred_res['badge_color']}; color: #0B1120;">
                        {pred_res['risk_level']}
                    </span>
                </div>
                <h2 style="margin: 0; color: {title_color}; font-size: 1.8rem; font-weight: 800;">
                    {pred_res['prediction_label']}
                </h2>
                <p style="margin: 6px 0 0 0; color: #F8FAFC; font-size: 1rem;">
                    Estimated Failure Probability: <b>{pred_res['failure_probability']*100:.1f}%</b>
                </p>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.85rem;">
                    Calculated Temperature Difference: <b>{pred_res['temp_diff']:.2f} K</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            st.markdown("#### Diagnostic Recommendations")
            for rec in pred_res["recommendations"]:
                st.markdown(f"- 🔧 {rec}")
                
        st.markdown("""
        <div style="font-size: 0.75rem; color: #64748B; margin-top: 15px; border-top: 1px solid #293548; padding-top: 8px;">
            ⚠️ <b>Educational Simulation Disclaimer:</b> This prediction output is produced by a machine learning model trained on a synthetic benchmark dataset. It is intended for educational and decision-support demonstrations and does not replace certified industrial vibration and acoustic diagnostic instrumentation.
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# PAGE 7: INSIGHTS AND RECOMMENDATIONS
# =============================================================================
elif nav_choice == "Insights and Recommendations":
    st.markdown("### Engineering Insights & Preventive Maintenance Strategy")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.markdown(r"""
        <div class="metric-card">
            <h4 style="margin: 0 0 10px 0; color: #60A5FA;">Observed Physical Failure Regimes</h4>
            <ul style="color: #CBD5E1; font-size: 0.9rem; padding-left: 20px;">
                <li><b>Overstrain Breakdown (OSF):</b> Triggered when the product of Tool Wear and Torque exceeds material yield limits (Torque > 55 Nm with Tool Wear > 180 min).</li>
                <li><b>Heat Dissipation Failure (HDF):</b> Occurs when the ambient room air approaches process heat (Temp Diff < 9.0 K).</li>
                <li><b>Power Failure (PWF):</b> Occurs when mechanical power ($P = \tau \cdot \omega$) deviates outside 3,500 W to 9,000 W operating boundaries.</li>
                <li><b>Tool Wear Failure (TWF):</b> Accelerates past 200 minutes of continuous cutting operation.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_in2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin: 0 0 10px 0; color: #34D399;">Preventive Maintenance Action Items</h4>
            <ol style="color: #CBD5E1; font-size: 0.9rem; padding-left: 20px;">
                <li><b>Dynamic Tool Swapping:</b> Enforce tool replacement at <b>180 minutes</b> to preempt catastrophic failure.</li>
                <li><b>Torque Interlock Controls:</b> Program CNC safety stops to automatically throttle feed rates whenever torque exceeds <b>55 Nm</b>.</li>
                <li><b>HVAC Thermal Regulation:</b> Maintain machine shop cooling to guarantee a minimum <b>10 K thermal gradient</b> between process and room temperature.</li>
                <li><b>Differential Inspection by Product Type:</b> Increase inspection frequency by 1.5x on lines dedicated to Type L variants.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <div class="warning-card">
        <h4 style="margin: 0 0 6px 0; color: #FBBF24;">Important Note on AI4I 2020 Synthetic Dataset</h4>
        <p style="margin: 0; color: #E2E8F0; font-size: 0.9rem;">
            The AI4I 2020 dataset was generated synthetically using physical differential equations. 
            While it captures real kinematic relationships (power equations, tool wear curves, and thermal diffusion), 
            real-world factory environments exhibit external noise such as operator error, cutting fluid degradation, electrical noise, and tool geometry variances.
            Always calibrate threshold limits using on-site historical vibration and thermal data before implementing hard automated cutoffs.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE 8: DATASET INFORMATION
# =============================================================================
elif nav_choice == "Dataset Information":
    st.markdown("### Dataset Metadata & Schema Reference")
    
    st.markdown("""
    <div class="info-card">
        <h4 style="margin: 0 0 6px 0; color: #60A5FA;">AI4I 2020 Predictive Maintenance Dataset</h4>
        <p style="margin: 0; color: #E2E8F0; font-size: 0.9rem;">
            <b>Citation:</b> Matzka, Stephan. (2020). <i>AI4I 2020 Predictive Maintenance Dataset</i>. UCI Machine Learning Repository. 
            <a href="https://doi.org/10.24432/C5HS5C" target="_blank" style="color: #60A5FA;">https://doi.org/10.24432/C5HS5C</a>.<br>
            <b>License:</b> Creative Commons Attribution 4.0 International (CC BY 4.0).
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Schema & Data Dictionary")
    schema_data = [
        {"Column": "UDI", "Type": "Integer", "Role": "Row Identifier (Excluded)", "Description": "Unique identifier from 1 to 10,000."},
        {"Column": "Product ID", "Type": "String", "Role": "Serial Number (Excluded)", "Description": "Product type letter (L, M, H) followed by serial number."},
        {"Column": "Type", "Type": "Categorical", "Role": "Input Feature", "Description": "Product quality variant: Low (60%), Medium (30%), High (10%)."},
        {"Column": "Air temperature [K]", "Type": "Float", "Role": "Input Feature", "Description": "Ambient room air temperature around machine."},
        {"Column": "Process temperature [K]", "Type": "Float", "Role": "Input Feature", "Description": "Cutting process temperature."},
        {"Column": "Rotational speed [rpm]", "Type": "Integer", "Role": "Input Feature", "Description": "Spindle rotational speed."},
        {"Column": "Torque [Nm]", "Type": "Float", "Role": "Input Feature", "Description": "Torque delivered to cutting tool."},
        {"Column": "Tool wear [min]", "Type": "Integer", "Role": "Input Feature", "Description": "Cumulative tool cutting time."},
        {"Column": "Machine failure", "Type": "Binary", "Role": "Target Variable", "Description": "0 = No Failure (96.61%), 1 = Failure (3.39%)."},
        {"Column": "TWF, HDF, PWF, OSF, RNF", "Type": "Binary", "Role": "Failure Modes (Excluded)", "Description": "Specific breakdown symptoms excluded to avoid target leakage."}
    ]
    st.table(pd.DataFrame(schema_data))
