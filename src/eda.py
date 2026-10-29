"""
Exploratory Data Analysis (EDA) & Interactive Visualizations Module
Creates dark-themed Plotly charts for the 5 analytical objectives.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Dark Theme Color Palette
CARD_BG = "#172033"
BORDER_COLOR = "#293548"
TEXT_COLOR = "#F8FAFC"
MUTED_TEXT = "#94A3B8"
ACCENT_BLUE = "#60A5FA"
ACCENT_PURPLE = "#A78BFA"
ACCENT_GREEN = "#34D399"
ACCENT_AMBER = "#FBBF24"
ACCENT_ROSE = "#FB7185"

# Coordinated Pastel Accents
PASTEL_AQUA = "#A8DADC"
PASTEL_PINK = "#F4ACB7"
PASTEL_LAVENDER = "#BDB2FF"
PASTEL_PEACH = "#FFD6A5"
PASTEL_GREEN = "#CAFFBF"
PASTEL_BLUE = "#BDE0FE"


def apply_dark_layout(fig, title="", height=400):
    """Apply consistent premium dark styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": TEXT_COLOR, "family": "Inter, sans-serif"}
        },
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font={"color": TEXT_COLOR, "family": "Inter, sans-serif"},
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            font={"color": TEXT_COLOR},
            bgcolor="rgba(0,0,0,0)",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont={"color": MUTED_TEXT},
            title_font={"color": TEXT_COLOR}
        ),
        yaxis=dict(
            gridcolor=BORDER_COLOR,
            zerolinecolor=BORDER_COLOR,
            tickfont={"color": MUTED_TEXT},
            title_font={"color": TEXT_COLOR}
        )
    )
    return fig


def plot_failure_distribution(df):
    """Objective 1: Donut and Bar representation of Machine Failure."""
    counts = df["Machine failure"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Status_Label"] = counts["Status"].map({0: "No Failure (Healthy)", 1: "Machine Failure"})
    counts["Percentage"] = (counts["Count"] / len(df) * 100).round(2)
    
    fig = go.Figure(data=[
        go.Pie(
            labels=counts["Status_Label"],
            values=counts["Count"],
            hole=0.6,
            marker=dict(colors=[ACCENT_BLUE, ACCENT_ROSE], line=dict(color=CARD_BG, width=3)),
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            textfont=dict(size=14, color=TEXT_COLOR)
        )
    ])
    fig = apply_dark_layout(fig, "Objective 1: Overall Machine Failure Breakdown", height=380)
    return fig


def plot_failure_distribution_bar(df):
    """Objective 1: Bar chart with exact counts."""
    counts = df["Machine failure"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    counts["Status_Label"] = counts["Status"].map({0: "Healthy (0)", 1: "Failure (1)"})
    
    fig = px.bar(
        counts, x="Status_Label", y="Count",
        color="Status_Label",
        color_discrete_map={"Healthy (0)": ACCENT_BLUE, "Failure (1)": ACCENT_ROSE},
        text="Count"
    )
    fig.update_traces(textposition="outside", textfont=dict(color=TEXT_COLOR))
    fig = apply_dark_layout(fig, "Objective 1: Machine Health Class Distribution", height=380)
    fig.update_layout(showlegend=False)
    return fig


def plot_product_type_distribution(df):
    """Objective 2: Distribution of Product Types L, M, and H."""
    type_counts = df["Type"].value_counts().reset_index()
    type_counts.columns = ["Product Type", "Count"]
    
    fig = px.bar(
        type_counts, x="Product Type", y="Count",
        color="Product Type",
        color_discrete_map={"L": ACCENT_PURPLE, "M": ACCENT_AMBER, "H": ACCENT_GREEN},
        text="Count",
        category_orders={"Product Type": ["L", "M", "H"]}
    )
    fig.update_traces(textposition="outside")
    fig = apply_dark_layout(fig, "Objective 2: Distribution of Product Types (L, M, H)", height=380)
    fig.update_layout(showlegend=False)
    return fig


def plot_failure_by_product_type(df):
    """Objective 2: Failure counts and rates across Product Types."""
    grouped = df.groupby(["Type", "Machine failure"]).size().reset_index(name="Count")
    grouped["Status"] = grouped["Machine failure"].map({0: "Healthy", 1: "Failure"})
    
    # Calculate failure rates
    rates = df.groupby("Type")["Machine failure"].mean() * 100
    
    fig = px.bar(
        grouped, x="Type", y="Count", color="Status",
        barmode="group",
        color_discrete_map={"Healthy": ACCENT_BLUE, "Failure": ACCENT_ROSE},
        category_orders={"Type": ["L", "M", "H"]}
    )
    fig = apply_dark_layout(fig, "Objective 2: Machine Failure by Product Type", height=380)
    return fig, rates


def plot_temperature_distributions(df):
    """Objective 3: Air and Process Temperature Distributions."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["Air temperature [K]"],
        name="Air Temperature [K]",
        marker_color=ACCENT_BLUE,
        opacity=0.75,
        nbinsx=35
    ))
    fig.add_trace(go.Histogram(
        x=df["Process temperature [K]"],
        name="Process Temperature [K]",
        marker_color=ACCENT_PURPLE,
        opacity=0.75,
        nbinsx=35
    ))
    fig.update_layout(barmode="overlay")
    fig = apply_dark_layout(fig, "Objective 3: Air vs Process Temperature Distributions", height=380)
    return fig


def plot_temperature_difference(df):
    """Objective 3: Boxplot of Temperature Difference vs Machine Failure."""
    temp_df = df.copy()
    if "Temperature Difference" not in temp_df.columns:
        temp_df["Temperature Difference"] = (
            temp_df["Process temperature [K]"] - temp_df["Air temperature [K]"]
        )
    temp_df["Status"] = temp_df["Machine failure"].map({0: "No Failure", 1: "Failure"})
    
    fig = px.box(
        temp_df, x="Status", y="Temperature Difference",
        color="Status",
        color_discrete_map={"No Failure": ACCENT_GREEN, "Failure": ACCENT_AMBER},
        points="outliers"
    )
    fig = apply_dark_layout(fig, "Objective 3: Temperature Difference (Process - Air) vs Failure", height=380)
    fig.update_layout(showlegend=False)
    return fig


def plot_tool_wear_torque(df):
    """Objective 4: Tool wear & Torque comparisons."""
    tool_df = df.copy()
    tool_df["Status"] = tool_df["Machine failure"].map({0: "No Failure", 1: "Failure"})
    
    fig1 = px.box(
        tool_df, x="Status", y="Tool wear [min]",
        color="Status",
        color_discrete_map={"No Failure": ACCENT_BLUE, "Failure": ACCENT_ROSE}
    )
    fig1 = apply_dark_layout(fig1, "Objective 4: Tool Wear vs Machine Failure", height=380)
    fig1.update_layout(showlegend=False)
    
    fig2 = px.box(
        tool_df, x="Status", y="Torque [Nm]",
        color="Status",
        color_discrete_map={"No Failure": ACCENT_AMBER, "Failure": ACCENT_PURPLE}
    )
    fig2 = apply_dark_layout(fig2, "Objective 4: Torque vs Machine Failure", height=380)
    fig2.update_layout(showlegend=False)
    
    return fig1, fig2


def plot_speed_torque_scatter(df):
    """Objective 4: Scatter plot of Rotational Speed vs Torque."""
    scat_df = df.copy()
    scat_df["Status"] = scat_df["Machine failure"].map({0: "Healthy", 1: "Failure"})
    
    fig = px.scatter(
        scat_df, x="Rotational speed [rpm]", y="Torque [Nm]",
        color="Status",
        color_discrete_map={"Healthy": ACCENT_BLUE, "Failure": ACCENT_ROSE},
        opacity=0.7,
        hover_data=["Tool wear [min]", "Type"]
    )
    fig = apply_dark_layout(fig, "Objective 4: Rotational Speed vs Torque (Failure Regimes)", height=420)
    return fig


def plot_failure_modes(df):
    """Objective 5: Bar chart of specific breakdown types."""
    failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    present_cols = [c for c in failure_cols if c in df.columns]
    counts = df[present_cols].sum().reset_index()
    counts.columns = ["Failure Mode", "Count"]
    
    full_names = {
        "TWF": "Tool Wear Failure (TWF)",
        "HDF": "Heat Dissipation Failure (HDF)",
        "PWF": "Power Failure (PWF)",
        "OSF": "Overstrain Failure (OSF)",
        "RNF": "Random Failure (RNF)"
    }
    counts["Full Name"] = counts["Failure Mode"].map(full_names)
    
    fig = px.bar(
        counts, x="Failure Mode", y="Count",
        color="Failure Mode",
        color_discrete_sequence=[ACCENT_BLUE, ACCENT_ROSE, ACCENT_PURPLE, ACCENT_AMBER, ACCENT_GREEN],
        text="Count",
        hover_data=["Full Name"]
    )
    fig.update_traces(textposition="outside")
    fig = apply_dark_layout(fig, "Objective 5: Specific Machine Failure Mode Occurrences", height=380)
    fig.update_layout(showlegend=False)
    return fig, counts


def plot_correlation_heatmap(df):
    """Objective 5 & Exploration: Correlation Heatmap."""
    numeric_df = df.select_dtypes(include=np.number).drop(columns=["UDI"], errors="ignore")
    corr = numeric_df.corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1
    )
    fig = apply_dark_layout(fig, "Correlation Heatmap of Operational Features", height=500)
    fig.update_layout(
        coloraxis_colorbar=dict(
            title=dict(text="Correlation", font=dict(color=TEXT_COLOR)),
            tickfont=dict(color=TEXT_COLOR)
        )
    )
    return fig
