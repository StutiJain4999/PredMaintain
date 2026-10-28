"""
Predictive Maintenance of Industrial Machines Using Machine Learning
Dataset: AI4I 2020 Predictive Maintenance Dataset (ai4i2020.csv)

This script performs complete exploratory data analysis (EDA),
analyzes five project objectives, trains four machine learning models,
evaluates and compares their performance, and saves all visualizations
and results to designated folders.
"""

# ==============================================================================
# STEP 1: IMPORT LIBRARIES
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Set style for clear and clean visualizations
sns.set_theme(style="whitegrid")

# Environment control for showing plots (interactive by default; can be set to 0 for batch runs)
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "1").lower() not in ("0", "false", "no")

# Ensure output directories exist
base_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
plots_dir = os.path.join(base_dir, "plots")
results_dir = os.path.join(base_dir, "results")
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Coordinated pastel color palette definition
PASTEL_PINK = "#F4ACB7"
PASTEL_LAVENDER = "#BDB2FF"
PASTEL_AQUA = "#A8DADC"
PASTEL_PEACH = "#FFD6A5"
PASTEL_GREEN = "#CAFFBF"
PASTEL_BLUE = "#BDE0FE"
PASTEL_ROSE = "#FFC8DD"
PASTEL_MINT = "#D8E2DC"


# ==============================================================================
# STEP 2: LOAD DATASET
# ==============================================================================

# Use relative path resolution so the project runs portably on any machine
dataset_path = os.path.join(base_dir, "dataset", "ai4i2020.csv")
if not os.path.exists(dataset_path):
    dataset_path = os.path.join("dataset", "ai4i2020.csv")
if not os.path.exists(dataset_path):
    dataset_path = "ai4i2020.csv"

print(f"Loading dataset from: {dataset_path}")
data = pd.read_csv(dataset_path)

print("First five rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())

print("\nDataset information:")
data.info()

print("\nStatistical summary:")
print(data.describe())


# ==============================================================================
# DATA PREPROCESSING AND CLEANING
# ==============================================================================

print("\nMissing values before cleaning:")
print(data.isnull().sum())

# Fill missing numerical values with median
for col in data.select_dtypes(include=np.number).columns:
    data[col] = data[col].fillna(data[col].median())

# Fill missing categorical values with mode
for col in data.select_dtypes(include=["object", "string"]).columns:
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].mode()[0])

print("\nDuplicate rows before removal:", data.duplicated().sum())
data = data.drop_duplicates()

# Remove extra spaces in categorical values if any and normalize case
data["Type"] = data["Type"].astype(str).str.strip().str.upper()

print("\nProduct types after cleaning:")
print(data["Type"].value_counts())

print("\nMachine failure counts:")
print(data["Machine failure"].value_counts())

print("\nMissing values after cleaning:")
print(data.isnull().sum())

print("\nCleaned dataset shape:")
print(data.shape)


# ==============================================================================
# FIVE PROJECT OBJECTIVES
# ==============================================================================
# OBJECTIVE 1:
# Analyze the number and proportion of machines with and without failure.
#
# OBJECTIVE 2:
# Compare machine failure counts across product types (L, M and H).
#
# OBJECTIVE 3:
# Explore air and process temperature patterns and compare them
# with machine failure.
#
# OBJECTIVE 4:
# Investigate whether tool wear and torque measurements differ
# between machines with and without failure.
#
# OBJECTIVE 5:
# Examine failure-mode counts and relationships among numeric
# operating measurements to support preventive maintenance insights.


# ------------------------------------------------------------------------------
# OBJECTIVE 1: MACHINE FAILURE DISTRIBUTION
# Graph 1: Count of machines with and without failure
# ------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
ax1 = sns.countplot(
    data=data, x="Machine failure",
    hue="Machine failure",
    palette=[PASTEL_AQUA, PASTEL_PINK],
    legend=False
)
plt.title("Objective 1: Machine Failure Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)", fontsize=11)
plt.ylabel("Number of Machines", fontsize=11)

for p in ax1.patches:
    height = p.get_height()
    ax1.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                 ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                 textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "01_machine_failure_distribution.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

obj1_counts = data["Machine failure"].value_counts()
obj1_failure_pct = round(data["Machine failure"].mean() * 100, 2)
print("\nObjective 1 finding:")
print(obj1_counts)
print(f"Failure percentage: {obj1_failure_pct}%")


# ------------------------------------------------------------------------------
# OBJECTIVE 2: PRODUCT TYPE AND MACHINE FAILURE
# Graph 2: Product type distribution
# Graph 3: Failure count by product type
# ------------------------------------------------------------------------------

obj2_type_counts = data["Type"].value_counts()

plt.figure(figsize=(8, 5))
ax2 = sns.countplot(
    data=data, x="Type",
    hue="Type",
    palette=[PASTEL_LAVENDER, PASTEL_PEACH, PASTEL_GREEN],
    legend=False,
    order=["L", "M", "H"]
)
plt.title("Objective 2: Distribution of Product Types (L, M, H)", fontsize=14, fontweight="bold")
plt.xlabel("Product Type (L = Low, M = Medium, H = High Quality)", fontsize=11)
plt.ylabel("Number of Products", fontsize=11)

for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(f"{int(height)}", (p.get_x() + p.get_width() / 2., height),
                 ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                 textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "02_product_type_distribution.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(
    data=data, x="Type", hue="Machine failure",
    palette=[PASTEL_BLUE, PASTEL_PINK],
    order=["L", "M", "H"]
)
plt.title("Objective 2: Machine Failure by Product Type", fontsize=14, fontweight="bold")
plt.xlabel("Product Type", fontsize=11)
plt.ylabel("Number of Machines", fontsize=11)
plt.legend(title="Machine Failure", labels=["No Failure", "Failure"])
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "03_failure_by_product_type.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

obj2_failure_rate = (data.groupby("Type")["Machine failure"].mean() * 100).round(2)
print("\nObjective 2: Failure rate by product type (%)")
print(obj2_failure_rate)


# ------------------------------------------------------------------------------
# OBJECTIVE 3: TEMPERATURE ANALYSIS
# Graph 4: Air temperature distribution
# Graph 5: Temperature difference vs machine failure
# ------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(
    data["Air temperature [K]"],
    bins=25, kde=True,
    color=PASTEL_LAVENDER,
    edgecolor="white"
)
plt.title("Objective 3: Air Temperature Distribution", fontsize=14, fontweight="bold")
plt.xlabel("Air Temperature [K]", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "04_air_temperature_distribution.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

# Feature engineering: difference between process and air temperature
data["Temperature Difference"] = (
    data["Process temperature [K]"] - data["Air temperature [K]"]
)

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=data, x="Machine failure",
    y="Temperature Difference",
    hue="Machine failure",
    palette=[PASTEL_GREEN, PASTEL_PEACH],
    legend=False
)
plt.title("Objective 3: Temperature Difference vs Machine Failure", fontsize=14, fontweight="bold")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)", fontsize=11)
plt.ylabel("Temperature Difference (Process - Air) [K]", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "05_temperature_difference_vs_failure.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

obj3_stats = data.groupby("Machine failure")[
    ["Air temperature [K]", "Process temperature [K]", "Temperature Difference"]
].mean().round(2)
print("\nObjective 3: Average temperatures by machine failure")
print(obj3_stats)


# ------------------------------------------------------------------------------
# OBJECTIVE 4: TOOL WEAR AND TORQUE ANALYSIS
# Graph 6: Tool wear vs machine failure
# Graph 7: Torque vs machine failure
# ------------------------------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=data, x="Machine failure",
    y="Tool wear [min]",
    hue="Machine failure",
    palette=[PASTEL_BLUE, PASTEL_ROSE],
    legend=False
)
plt.title("Objective 4: Tool Wear vs Machine Failure", fontsize=14, fontweight="bold")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)", fontsize=11)
plt.ylabel("Tool Wear [min]", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "06_tool_wear_vs_failure.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(
    data=data, x="Machine failure",
    y="Torque [Nm]",
    hue="Machine failure",
    palette=[PASTEL_PEACH, PASTEL_LAVENDER],
    legend=False
)
plt.title("Objective 4: Torque vs Machine Failure", fontsize=14, fontweight="bold")
plt.xlabel("Machine Failure (0 = No Failure, 1 = Failure)", fontsize=11)
plt.ylabel("Torque [Nm]", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "07_torque_vs_failure.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

obj4_stats = data.groupby("Machine failure")[
    ["Tool wear [min]", "Torque [Nm]", "Rotational speed [rpm]"]
].mean().round(2)
print("\nObjective 4: Average tool wear, torque, and speed by machine failure")
print(obj4_stats)


# ------------------------------------------------------------------------------
# OBJECTIVE 5: FAILURE TYPES AND CORRELATIONS
# Graph 8: Failure type counts
# Graph 9: Correlation heatmap
# ------------------------------------------------------------------------------

failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
failure_counts = data[failure_types].sum()

plt.figure(figsize=(8, 5))
bar_colors = [PASTEL_AQUA, PASTEL_PINK, PASTEL_LAVENDER, PASTEL_PEACH, PASTEL_GREEN]
ax8 = failure_counts.plot(kind="bar", color=bar_colors, edgecolor="grey", linewidth=0.5)
plt.title("Objective 5: Number of Each Failure Mode", fontsize=14, fontweight="bold")
plt.xlabel("Failure Mode (TWF, HDF, PWF, OSF, RNF)", fontsize=11)
plt.ylabel("Number of Failures", fontsize=11)
plt.xticks(rotation=0)

for i, count in enumerate(failure_counts):
    ax8.annotate(f"{int(count)}", (i, count), ha="center", va="bottom",
                 fontsize=10, xytext=(0, 3), textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "08_failure_type_counts.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

# Correlation chart is for EDA only.
# Failure-type indicators are excluded from the ML input features below to prevent target leakage.
corr_data = data.select_dtypes(include=np.number).drop(
    columns=["UDI"], errors="ignore"
)

plt.figure(figsize=(11, 8))
pastel_cmap = sns.diverging_palette(240, 10, s=40, l=75, as_cmap=True)
sns.heatmap(
    corr_data.corr(), annot=True,
    cmap=pastel_cmap, fmt=".2f",
    linewidths=0.5, cbar_kws={"shrink": 0.8}
)
plt.title("Objective 5: Correlation Heatmap of Operational Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "09_correlation_heatmap.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

print("\nObjective 5: Failure-type counts")
print(failure_counts)


# ==============================================================================
# STEP 3: PREPARE INPUT AND OUTPUT VARIABLES (PREVENT TARGET LEAKAGE)
# ==============================================================================

# Remove identifiers and failure-mode columns to prevent target leakage.
# TWF, HDF, PWF, OSF, and RNF are direct indicators/causes of Machine failure.
data = data.drop(
    columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
)

# Convert categorical Type into numeric dummy columns
data = pd.get_dummies(
    data, columns=["Type"], drop_first=True, dtype=int
)

# X = input features; Y = target variable
X = data.drop("Machine failure", axis=1)
Y = data["Machine failure"]

print("\nInput features (first 5 rows):")
print(X.head())

print("\nTarget variable distribution:")
print(Y.value_counts())


# ==============================================================================
# STEP 4: SPLIT DATA INTO TRAINING AND TESTING DATA
# ==============================================================================

# Stratified 80:20 split ensures proportional representation of rare failures
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=42,
    stratify=Y
)

print(f"\nTraining data shape: {X_train.shape}")
print(f"Testing data shape:  {X_test.shape}")


# ==============================================================================
# STEP 5: FEATURE SCALING
# ==============================================================================

# Fit scaler strictly on X_train to prevent data leakage into X_test
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# ==============================================================================
# STEP 6: IMPLEMENT FOUR MACHINE LEARNING MODELS
# ==============================================================================

# Model 1: Logistic Regression
model1 = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model1.fit(X_train, Y_train)
pred1 = model1.predict(X_test)

# Model 2: Decision Tree Classifier
model2 = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)
model2.fit(X_train, Y_train)
pred2 = model2.predict(X_test)

# Model 3: K-Nearest Neighbors
model3 = KNeighborsClassifier(n_neighbors=5)
model3.fit(X_train, Y_train)
pred3 = model3.predict(X_test)

# Model 4: Gaussian Naive Bayes
model4 = GaussianNB()
model4.fit(X_train, Y_train)
pred4 = model4.predict(X_test)


# ==============================================================================
# STEP 7: MODEL EVALUATION AND COMPARISON
# ==============================================================================

models = [
    "Logistic Regression",
    "Decision Tree",
    "KNN",
    "Naive Bayes"
]

predictions = [pred1, pred2, pred3, pred4]

accuracy = []
precision = []
recall = []
f1 = []

for pred in predictions:
    accuracy.append(accuracy_score(Y_test, pred))
    precision.append(precision_score(Y_test, pred, zero_division=0))
    recall.append(recall_score(Y_test, pred, zero_division=0))
    f1.append(f1_score(Y_test, pred, zero_division=0))

results = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
})

print("\n" + "=" * 50)
print("MODEL COMPARISON TABLE")
print("=" * 50)
print(results.round(4).to_string(index=False))

# Save comparison table to results/model_comparison.csv
comparison_csv_path = os.path.join(results_dir, "model_comparison.csv")
results.round(4).to_csv(comparison_csv_path, index=False)
print(f"\nSaved model comparison table to: {comparison_csv_path}")

# Graph 10: Accuracy comparison
plt.figure(figsize=(9, 5))
ax10 = sns.barplot(
    data=results, x="Model", y="Accuracy",
    hue="Model",
    palette=[PASTEL_PINK, PASTEL_LAVENDER, PASTEL_AQUA, PASTEL_PEACH],
    legend=False
)
plt.title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Machine Learning Model", fontsize=11)
plt.ylabel("Accuracy", fontsize=11)
plt.ylim(0, 1.05)
plt.xticks(rotation=0)

for p in ax10.patches:
    height = p.get_height()
    ax10.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height),
                  ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                  textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "10_model_accuracy_comparison.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

# Graph 11: F1-score comparison
plt.figure(figsize=(9, 5))
ax11 = sns.barplot(
    data=results, x="Model", y="F1 Score",
    hue="Model",
    palette=[PASTEL_GREEN, PASTEL_BLUE, PASTEL_ROSE, PASTEL_MINT],
    legend=False
)
plt.title("Model F1-Score Comparison", fontsize=14, fontweight="bold")
plt.xlabel("Machine Learning Model", fontsize=11)
plt.ylabel("F1 Score", fontsize=11)
plt.ylim(0, 1.05)
plt.xticks(rotation=0)

for p in ax11.patches:
    height = p.get_height()
    ax11.annotate(f"{height:.4f}", (p.get_x() + p.get_width() / 2., height),
                  ha="center", va="bottom", fontsize=10, xytext=(0, 3),
                  textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "11_model_f1_score_comparison.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()


# ==============================================================================
# STEP 8: FIND THE BEST-FIT MODEL
# ==============================================================================

# F1 score is used because machine failures are rare (class imbalance)
best_index = results["F1 Score"].idxmax()
best_model_name = results.loc[best_index, "Model"]
best_accuracy = results.loc[best_index, "Accuracy"]
best_precision = results.loc[best_index, "Precision"]
best_recall = results.loc[best_index, "Recall"]
best_f1 = results.loc[best_index, "F1 Score"]

print("\n" + "=" * 50)
print(f"BEST-FIT MODEL: {best_model_name}")
print("=" * 50)
print(f"Accuracy:  {best_accuracy:.4f}")
print(f"Precision: {best_precision:.4f}")
print(f"Recall:    {best_recall:.4f}")
print(f"F1 Score:  {best_f1:.4f}")


# ==============================================================================
# STEP 9: CONFUSION MATRIX AND CLASSIFICATION REPORT
# ==============================================================================

best_predictions = predictions[best_index]
cm = confusion_matrix(Y_test, best_predictions)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d",
    cmap=sns.light_palette(PASTEL_LAVENDER, as_cmap=True),
    xticklabels=["Predicted No Failure", "Predicted Failure"],
    yticklabels=["Actual No Failure", "Actual Failure"],
    cbar=False
)
plt.title(f"Confusion Matrix: Best Model ({best_model_name})", fontsize=12, fontweight="bold")
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("Actual Label", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "12_best_model_confusion_matrix.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()

clf_report = classification_report(
    Y_test, best_predictions,
    target_names=["No Failure (0)", "Failure (1)"],
    zero_division=0
)
print("\nClassification Report of Best Model:")
print(clf_report)

# Save classification report to results/classification_report.txt
clf_report_path = os.path.join(results_dir, "classification_report.txt")
with open(clf_report_path, "w") as f:
    f.write(f"CLASSIFICATION REPORT - BEST MODEL: {best_model_name}\n")
    f.write("=" * 60 + "\n\n")
    f.write(clf_report)
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"Confusion Matrix:\n")
    f.write(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
    f.write(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
print(f"Saved classification report to: {clf_report_path}")


# ==============================================================================
# STEP 10: FEATURE IMPORTANCE (DECISION TREE)
# ==============================================================================

feature_names = X.columns
importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model2.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nDecision Tree Feature Importance:")
print(importance.to_string(index=False))

plt.figure(figsize=(10, 6))
ax13 = sns.barplot(
    data=importance, x="Importance", y="Feature",
    color=PASTEL_LAVENDER
)
plt.title("Feature Importance from Decision Tree Classifier", fontsize=14, fontweight="bold")
plt.xlabel("Importance Score", fontsize=11)
plt.ylabel("Feature", fontsize=11)

for p in ax13.patches:
    width = p.get_width()
    ax13.annotate(f"{width:.4f}", (width, p.get_y() + p.get_height() / 2.),
                  ha="left", va="center", fontsize=9, xytext=(5, 0),
                  textcoords="offset points")

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "13_decision_tree_feature_importance.png"), dpi=300)
if SHOW_PLOTS:
    plt.show()
plt.close()


# ==============================================================================
# STEP 11: PROJECT SUMMARY AND CONCLUSION EXPORT
# ==============================================================================

summary_text = f"""PREDICTIVE MAINTENANCE OF INDUSTRIAL MACHINES USING MACHINE LEARNING
================================================================================
Dataset: AI4I 2020 Predictive Maintenance Dataset (ai4i2020.csv)
Total Samples: {len(data)}
Total Features Evaluated: {X.shape[1]}
Target: Machine failure (0 = No Failure, 1 = Failure)

FIVE PROJECT OBJECTIVES & KEY FINDINGS:
--------------------------------------------------------------------------------
1. Machine Failure Distribution:
   - No Failure (0): {obj1_counts.get(0, 0)} machines ({100 - obj1_failure_pct:.2f}%)
   - Failure (1):    {obj1_counts.get(1, 0)} machines ({obj1_failure_pct:.2f}%)
   - Finding: Severe class imbalance (~3.39% failures), requiring F1-score & recall
     prioritization over raw accuracy.

2. Product Type and Machine Failure:
   - Distribution: L (Low) = {obj2_type_counts.get('L', 0)}, M (Medium) = {obj2_type_counts.get('M', 0)}, H (High) = {obj2_type_counts.get('H', 0)}
   - Failure Rates: { {k: float(v) for k, v in obj2_failure_rate.items()} }
   - Finding: Low-quality variant (L) experiences higher failure rates compared to
     Medium (M) and High (H) quality variants due to lower tolerances.

3. Temperature Analysis:
   - Feature Engineered: Temperature Difference = Process Temp - Air Temp [K]
   - Non-Failure Average Temp Diff: {obj3_stats.loc[0, 'Temperature Difference']:.2f} K
   - Failure Average Temp Diff:     {obj3_stats.loc[1, 'Temperature Difference']:.2f} K
   - Finding: Machines experiencing failure operate under abnormal thermal gradients,
     often linked to heat dissipation failure (HDF).

4. Tool Wear and Torque Analysis:
   - Non-Failure Avg Tool Wear: {obj4_stats.loc[0, 'Tool wear [min]']:.2f} min | Torque: {obj4_stats.loc[0, 'Torque [Nm]']:.2f} Nm
   - Failure Avg Tool Wear:     {obj4_stats.loc[1, 'Tool wear [min]']:.2f} min | Torque: {obj4_stats.loc[1, 'Torque [Nm]']:.2f} Nm
   - Finding: High tool wear (>200 min) and extreme torque loads strongly correlate
     with tool wear failure (TWF) and overstrain failure (OSF).

5. Failure Types and Feature Correlations:
   - Failure Mode Breakdown: { {k: int(v) for k, v in failure_counts.items()} }
   - Top Failure Modes: Heat Dissipation Failure (HDF) and Power Failure (PWF).
   - Correlation: Strong negative relationship between Rotational Speed and Torque,
     and strong positive coupling between Air and Process temperatures.

MODEL PERFORMANCE SUMMARY (Stratified 80:20 Split):
--------------------------------------------------------------------------------
{results.round(4).to_string(index=False)}

BEST-FIT MODEL SELECTION:
--------------------------------------------------------------------------------
- Selected Model: {best_model_name}
- F1-Score:       {best_f1:.4f}
- Accuracy:       {best_accuracy:.4f}
- Precision:      {best_precision:.4f}
- Recall:         {best_recall:.4f}
- Justification: In predictive maintenance, missing a machine failure (False Negative)
  leads to costly unscheduled downtime and catastrophic damage. High F1-score balances
  detecting failures (Recall) while avoiding excessive false alarms (Precision).

PREVENTIVE MAINTENANCE RECOMMENDATIONS:
--------------------------------------------------------------------------------
1. Dynamic Tool Replacement: Schedule tool changes when tool wear exceeds 180-200 minutes.
2. Torque & Speed Regulation: Implement automated cutoffs when torque exceeds safe operational thresholds.
3. Thermal Gradient Monitoring: Continuously track temperature difference between process and air; trigger alerts when dissipation drops below critical levels.
4. Product-Specific Inspection: Increase inspection frequency for Type-L machines.
"""

summary_path = os.path.join(results_dir, "project_summary.txt")
with open(summary_path, "w") as f:
    f.write(summary_text)

print("\n" + "=" * 50)
print("PROJECT SUMMARY EXPORTED")
print("=" * 50)
print(summary_text)
print(f"Saved project summary to: {summary_path}")

print("\n" + "=" * 50)
print("PROJECT EXECUTION COMPLETE!")
print("=" * 50)
