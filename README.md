# Predictive Maintenance of Industrial Machines Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

An end-to-end Machine Learning and Exploratory Data Analysis (EDA) project built for Continuous Assessment (CA) and portfolio presentation. This project predicts equipment failures in industrial manufacturing machinery using multivariate sensor data, preventing costly unplanned downtime and enabling data-driven condition-based maintenance.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Motivation and Real-World Application](#motivation-and-real-world-application)
4. [Dataset Description & Citation](#dataset-description--citation)
5. [Repository Structure](#repository-structure)
6. [Technologies & Python Libraries](#technologies--python-libraries)
7. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
8. [Data Leakage Prevention](#data-leakage-prevention)
9. [Five Project Objectives & Exploratory Data Analysis (EDA)](#five-project-objectives--exploratory-data-analysis-eda)
   - [Objective 1: Machine Failure Distribution](#objective-1-machine-failure-distribution)
   - [Objective 2: Product Type & Failure Rates](#objective-2-product-type--failure-rates)
   - [Objective 3: Temperature Analysis & Thermal Dissipation](#objective-3-temperature-analysis--thermal-dissipation)
   - [Objective 4: Tool Wear & Torque Stress Analysis](#objective-4-tool-wear--torque-stress-analysis)
   - [Objective 5: Failure Mode Breakdown & Feature Correlation](#objective-5-failure-mode-breakdown--feature-correlation)
10. [Machine Learning Models & Evaluation](#machine-learning-models--evaluation)
    - [Model Comparison Table](#actual-model-comparison-table)
    - [Model Comparison Visualizations](#model-comparison-visualizations)
    - [Best-Fit Model Selection & Justification](#best-fit-model-selection--justification)
    - [Confusion Matrix & Classification Report](#confusion-matrix--classification-report)
    - [Feature Importance Analysis](#feature-importance-analysis)
11. [Findings, Conclusions & Recommendations](#findings-conclusions--recommendations)
12. [Dataset Limitations](#dataset-limitations)
13. [Installation & Setup](#installation--setup)
14. [How to Run the Project](#how-to-run-the-project)
15. [Output Files Generated](#output-files-generated)
16. [Future Improvements](#future-improvements)
17. [License & Acknowledgements](#license--acknowledgements)

---

## Project Overview

In contemporary Industry 4.0 environments, modern manufacturing facilities rely on automated milling, turning, and computer numerical control (CNC) machines operating under continuous stress. Traditional maintenance policies typically fall into two sub-optimal extremes:
* **Reactive Maintenance (Run-to-Failure):** Machines are operated until catastrophic breakdown occurs, incurring extensive equipment damage, safety risks, and expensive unscheduled downtime.
* **Preventive Maintenance (Time-Based):** Components are serviced or replaced at rigid, fixed calendar intervals, frequently discarding serviceable parts prematurely and driving up operating expenditures.

**Predictive Maintenance (PdM)** bridges this divide by continuously evaluating real-time operational sensor telemetry—including rotational speed, torque, air and process temperatures, and cumulative tool wear—to forecast component failure *before* it occurs. This project implements an interpretable, reliable machine learning pipeline that classifies machine health state, identifies high-risk operational regimes, and provides actionable recommendations.

---

## Problem Statement

Given 10,000 operational records capturing real-time physical sensor telemetry, our goal is to construct a binary classification model that accurately predicts whether an industrial machine will suffer a **Machine failure** (`0 = No Failure`, `1 = Failure`). 

Key challenges addressed in this project include:
1. **Severe Class Imbalance:** Failures represent only **3.39%** of operational observations (339 failure cases vs. 9,661 normal operating cases). Standard models optimizing purely for overall accuracy can achieve 96.6% accuracy simply by predicting non-failure every time while missing every breakdown.
2. **Prevention of Target Leakage:** The dataset includes specific breakdown category indicators (Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, Random Failure) that are consequential symptoms of failure rather than predictive signals. These must be rigorously isolated during feature preparation.
3. **Model Interpretability:** Factory maintenance engineers require transparent insight into *which* physical variables (torque, speed, heat differential, tool life) trigger failure to schedule timely physical interventions.

---

## Motivation and Real-World Application

Unplanned downtime in heavy manufacturing facilities can cost upward of \$20,000 to \$500,000 per hour depending on plant throughput. Implementing this machine learning solution enables:
* **Dynamic Tool Replacement:** Rather than swapping tooling after arbitrary hours, cutting tools are replaced based on predicted wear limits and torque thresholds.
* **Real-Time Thermal Monitoring:** Automated alerts trigger when the dissipation delta between process temperature and ambient air narrows.
* **Process Parameter Regulation:** Detection of dangerous speed-torque operational combinations to prevent motor stalling and shaft overstrain.

---

## Dataset Description & Citation

This project utilizes the **AI4I 2020 Predictive Maintenance Dataset**, an industry-standard benchmark reflecting real-world milling machine telemetry.

* **Total Samples:** 10,000 data points
* **Features:** 14 original columns
* **Target Variable:** `Machine failure` (binary flag indicating equipment failure)
* **Dataset Location in Repository:** `dataset/ai4i2020.csv`
* **Licence:** Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))

### Raw Data Attributes

| Column Name | Data Type | Units / Range | Description |
|:---|:---:|:---:|:---|
| `UDI` | Integer | 1 to 10000 | Unique Row Identifier |
| `Product ID` | Object / String | L/M/H + serial | Product variant prefix followed by serial number |
| `Type` | Categorical | L (60%), M (30%), H (10%) | Quality variant: Low (L), Medium (M), High (H) |
| `Air temperature [K]` | Float | ~295 K – 304 K | Ambient air temperature generated using a random walk |
| `Process temperature [K]` | Float | ~305 K – 313 K | Temperature of the machine process during operation |
| `Rotational speed [rpm]` | Integer | ~1168 – 2886 rpm | Spindle rotational speed calculated from power equation |
| `Torque [Nm]` | Float | ~3.8 – 76.6 Nm | Torque delivered to the workpiece |
| `Tool wear [min]` | Integer | 0 – 253 min | Cumulative operating time on cutting tool |
| `Machine failure` | Binary (0 / 1) | 0 (No), 1 (Yes) | Overall machine failure indicator (**Target**) |
| `TWF` | Binary (0 / 1) | 0 or 1 | Tool Wear Failure mode indicator |
| `HDF` | Binary (0 / 1) | 0 or 1 | Heat Dissipation Failure mode indicator |
| `PWF` | Binary (0 / 1) | 0 or 1 | Power Failure mode indicator |
| `OSF` | Binary (0 / 1) | 0 or 1 | Overstrain Failure mode indicator |
| `RNF` | Binary (0 / 1) | 0 or 1 | Random Failure mode indicator |

> **Citation:**
> Matzka, Stephan. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).

---

## Repository Structure

```text
Predictive-Maintenance-of-Industrial-Machines/
│
├── README.md                                    # Comprehensive project documentation
├── Predictive Maintenance of Industrial Machines.py  # End-to-end Python script
├── requirements.txt                             # Pinned project dependencies
├── .gitignore                                   # Ignore cache, bytecode, env files
│
├── dataset/
│   └── ai4i2020.csv                             # AI4I 2020 dataset (CC BY 4.0)
│
├── plots/                                       # 13 high-resolution pastel visualizations
│   ├── 01_machine_failure_distribution.png
│   ├── 02_product_type_distribution.png
│   ├── 03_failure_by_product_type.png
│   ├── 04_air_temperature_distribution.png
│   ├── 05_temperature_difference_vs_failure.png
│   ├── 06_tool_wear_vs_failure.png
│   ├── 07_torque_vs_failure.png
│   ├── 08_failure_type_counts.png
│   ├── 09_correlation_heatmap.png
│   ├── 10_model_accuracy_comparison.png
│   ├── 11_model_f1_score_comparison.png
│   ├── 12_best_model_confusion_matrix.png
│   └── 13_decision_tree_feature_importance.png
│
└── results/                                     # Machine-generated reports & metrics
    ├── model_comparison.csv                     # Metrics summary table (Accuracy, Precision, Recall, F1)
    ├── classification_report.txt                # Precision, recall, f1, and confusion matrix
    └── project_summary.txt                      # Detailed analysis summary of all 5 objectives
```

---

## Technologies & Python Libraries

The implementation avoids black-box frameworks and utilizes straightforward, reliable Python scientific stack libraries:

* **Python 3.12+** – Core programming language runtime.
* **Pandas** – Data ingestion, exploratory manipulation, cleaning, and tabular summary exports.
* **NumPy** – High-performance vectorized numerical operations and array transformations.
* **Matplotlib & Seaborn** – Publication-quality visual analytics styled with coordinated pastel palettes.
* **Scikit-Learn** – Data splitting, feature standardization, model training, cross-model metric evaluations, and decision tree feature importances.

---

## Data Preprocessing & Feature Engineering

1. **Handling Missing Values:**
   * Numerical features were checked using `.isnull().sum()` and imputed with column medians (0 missing values present in raw dataset).
   * Categorical features were verified and filled with column modes.
2. **Duplicate Detection:**
   * Examined duplicate records with `data.duplicated().sum()`. Exactly 0 duplicate records were detected; no rows were dropped.
3. **Categorical Normalization & Encoding:**
   * String whitespace stripped and capitalized: `data["Type"] = data["Type"].astype(str).str.strip().str.upper()`.
   * Encoded using dummy one-hot indicators (`Type_L`, `Type_M`) with `drop_first=True` to prevent multicollinearity (dummy variable trap).
4. **Feature Engineering (`Temperature Difference`):**
   * Created domain-specific feature: 
     $$\text{Temperature Difference} = \text{Process temperature [K]} - \text{Air temperature [K]}$$
   * Physical Rationale: A milling machine dissipates heat produced during cutting into ambient air. When the delta between workpiece heat and ambient air drops below a critical point, heat dissipation fails (HDF), triggering emergency stops.

---

## Data Leakage Prevention

> [!IMPORTANT]
> **Strict Target Leakage Isolation:**
> The dataset includes five failure-type flags: `TWF` (Tool Wear Failure), `HDF` (Heat Dissipation Failure), `PWF` (Power Failure), `OSF` (Overstrain Failure), and `RNF` (Random Failure). 
> In a production factory setting, maintenance sensors cannot observe these breakdown flags *before* a failure occurs because they define the failure itself. 
> To prevent target leakage, all five failure columns along with arbitrary identifiers (`UDI`, `Product ID`) were **strictly excluded** from model training inputs.
> 
> Furthermore, the `StandardScaler` was fitted **exclusively on the training dataset** (`X_train`) and merely applied to transform the test set (`X_test`), preventing statistical data leakage from evaluation samples.

---

## Five Project Objectives & Exploratory Data Analysis (EDA)

All visualizations were generated using coordinated, accessible pastel color codes:
* **Pastel Aqua (`#A8DADC`)** | **Pastel Pink (`#F4ACB7`)** | **Pastel Lavender (`#BDB2FF`)**
* **Pastel Peach (`#FFD6A5`)** | **Pastel Green (`#CAFFBF`)** | **Pastel Blue (`#BDE0FE`)** | **Pastel Rose (`#FFC8DD`)**

---

### Objective 1: Machine Failure Distribution
* **Purpose:** Determine the baseline frequency and proportion of machines operating normally versus those experiencing failure.
* **Actual Findings:**
  * **No Failure (0):** 9,661 machines (**96.61%**)
  * **Failure (1):** 339 machines (**3.39%**)
  * **Insight:** The operational dataset exhibits extreme class imbalance. Because failures represent only ~3.4% of the dataset, naive accuracy is a deceptive metric. Models must be judged on F1-score and Recall.

![Machine Failure Distribution](plots/01_machine_failure_distribution.png)

---

### Objective 2: Product Type & Failure Rates
* **Purpose:** Compare product variants across quality tiers—Low (`L`), Medium (`M`), and High (`H`)—to evaluate if product grade impacts operational reliability.
* **Actual Findings:**
  * **Variant Counts:** `L` = 6,000 (60.0%), `M` = 2,997 (29.97%), `H` = 1,003 (10.03%)
  * **Failure Rates by Variant:**
    * **Type L (Low Quality):** **3.92%** failure rate (235 failures out of 6,000)
    * **Type M (Medium Quality):** **2.77%** failure rate (83 failures out of 2,997)
    * **Type H (High Quality):** **2.09%** failure rate (21 failures out of 1,003)
  * **Insight:** Low-quality product variants exhibit nearly **double** the failure rate of high-quality variants. Variant L machines operate closer to material stress tolerances and require closer inspection intervals.

| Product Type Distribution | Machine Failure by Product Type |
|:---:|:---:|
| ![Product Type Distribution](plots/02_product_type_distribution.png) | ![Failure by Product Type](plots/03_failure_by_product_type.png) |

---

### Objective 3: Temperature Analysis & Thermal Dissipation
* **Purpose:** Examine air temperature distribution, construct the `Temperature Difference` feature, and compare thermal profiles for failed versus non-failed machines.
* **Actual Findings:**
  * **Air Temperature:** Normal Gaussian distribution centered around ~300.0 K (min 295.3 K, max 304.5 K).
  * **Process Temperature:** Centered around ~310.0 K (min 305.7 K, max 313.8 K).
  * **Mean Temperature Difference:**
    * **Non-Failed Machines:** **10.02 K**
    * **Failed Machines:** **9.40 K**
  * **Insight:** Machines that experienced failure operated with a significantly smaller temperature delta ($9.40\text{ K}$ vs $10.02\text{ K}$). Insufficient heat dissipation between the spindle and the room environment causes thermal accumulation, precipitating heat dissipation breakdowns.

| Air Temperature Distribution | Temperature Difference vs Failure |
|:---:|:---:|
| ![Air Temperature Distribution](plots/04_air_temperature_distribution.png) | ![Temperature Difference vs Failure](plots/05_temperature_difference_vs_failure.png) |

---

### Objective 4: Tool Wear & Torque Stress Analysis
* **Purpose:** Evaluate whether mechanical loading (torque in Nm) and cumulative wear time (tool wear in minutes) separate healthy machines from failing machines.
* **Actual Findings:**
  * **Tool Wear:**
    * **Non-Failed Machines:** Mean **106.69 minutes**
    * **Failed Machines:** Mean **143.78 minutes** (frequently peaking above 200–240 minutes)
  * **Torque:**
    * **Non-Failed Machines:** Mean **39.63 Nm** (standard operating range: 30–50 Nm)
    * **Failed Machines:** Mean **50.17 Nm** (frequently spiking beyond 60 Nm)
  * **Rotational Speed:**
    * Non-Failed: Mean **1540.26 rpm** | Failed: Mean **1496.49 rpm**
  * **Insight:** Machines with high cumulative tool wear combined with torque overload exhibit extreme failure risk, triggering tool wear failure (TWF) and overstrain failure (OSF).

| Tool Wear vs Machine Failure | Torque vs Machine Failure |
|:---:|:---:|
| ![Tool Wear vs Machine Failure](plots/06_tool_wear_vs_failure.png) | ![Torque vs Machine Failure](plots/07_torque_vs_failure.png) |

---

### Objective 5: Failure Mode Breakdown & Feature Correlation
* **Purpose:** Quantify individual failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and examine cross-feature correlations to uncover operational interactions.
* **Actual Findings:**
  * **Failure Mode Counts:**
    * **HDF (Heat Dissipation Failure):** **115** occurrences (most frequent)
    * **OSF (Overstrain Failure):** **98** occurrences
    * **PWF (Power Failure):** **95** occurrences
    * **TWF (Tool Wear Failure):** **46** occurrences
    * **RNF (Random Failure):** **19** occurrences
  * **Key Correlations:**
    * **Torque vs. Rotational Speed:** Strong negative correlation ($r = -0.88$). As torque increases, spindle speed decelerates, reflecting the mechanical power formula $P = \tau \cdot \omega$.
    * **Air vs. Process Temperature:** Strong positive correlation ($r = +0.88$). Process heat tracks ambient temperature conditions.
    * **Machine Failure Correlations:** Positively correlated with Torque ($r = +0.19$) and Tool Wear ($r = +0.11$).

| Failure Mode Counts | Correlation Heatmap |
|:---:|:---:|
| ![Failure Mode Counts](plots/08_failure_type_counts.png) | ![Correlation Heatmap](plots/09_correlation_heatmap.png) |

---

## Machine Learning Models & Evaluation

### Train-Test Split & Scaling Methodology
* **Split Ratio:** 80% Training (8,000 samples) : 20% Testing (2,000 samples).
* **Stratification:** `stratify=Y` guarantees the 3.39% failure incidence is identically preserved in both training (271 failures) and testing (68 failures) partitions.
* **Random Seed:** `random_state=42` ensures complete experimental reproducibility.
* **Feature Scaling:** `StandardScaler` fitted strictly on `X_train` and applied to `X_test`.

### Algorithms Evaluated
1. **Logistic Regression (`model1`):** Linear decision boundary with `class_weight="balanced"`.
2. **Decision Tree Classifier (`model2`):** Non-linear rule-based tree with `class_weight="balanced"`.
3. **K-Nearest Neighbors (`model3`):** Distance-based instance learning ($k=5$).
4. **Gaussian Naïve Bayes (`model4`):** Probabilistic generative model assuming feature independence.

---

### Actual Model Comparison Table

All performance values below were measured by running `Predictive Maintenance of Industrial Machines.py` on the real test dataset:

| Model | Accuracy | Precision | Recall | F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 82.60% (0.8260) | 14.29% (0.1429) | **82.35% (0.8235)** | 0.2435 |
| **Decision Tree Classifier** | **97.75% (0.9775)** | **68.25% (0.6825)** | **63.24% (0.6324)** | **0.6565** |
| **K-Nearest Neighbors (KNN)** | 97.50% (0.9750) | 87.50% (0.8750) | 30.88% (0.3088) | 0.4565 |
| **Gaussian Naïve Bayes** | 95.90% (0.9590) | 29.41% (0.2941) | 14.71% (0.1471) | 0.1961 |

*Data exported to `results/model_comparison.csv`.*

---

### Model Comparison Visualizations

| Accuracy Comparison Across Models | F1-Score Comparison Across Models |
|:---:|:---:|
| ![Model Accuracy Comparison](plots/10_model_accuracy_comparison.png) | ![Model F1 Score Comparison](plots/11_model_f1_score_comparison.png) |

---

### Best-Fit Model Selection & Justification

> [!NOTE]
> ### Winner: **Decision Tree Classifier**
> * **Overall Accuracy:** **97.75%**
> * **Precision:** **68.25%**
> * **Recall:** **63.24%**
> * **F1-Score:** **0.6565** (Highest across all evaluated algorithms)

#### Why F1-Score Over Accuracy?
In predictive maintenance, **class imbalance makes accuracy deceptive**. For example, a "dumb" baseline model that predicts non-failure 100% of the time achieves **96.6% accuracy**, yet its operational utility is zero because every machine failure goes undetected.

#### Navigating the Precision-Recall Trade-off:
* **The Cost of a False Negative (Missed Breakdown):** In an industrial plant, a missed failure causes sudden tool breakage, damaged workpieces, fire hazards, and millions of dollars in secondary conveyor damage. High **Recall** is crucial.
* **The Cost of a False Positive (False Alarm):** A false alarm prompts a maintenance technician to visually inspect the tool during a standard shift change, causing negligible disruption.
* **Why Decision Tree Won:** 
  * Logistic Regression achieved high recall (82.35%) but suffered from terrible precision (14.29%), creating 494 false alarms per 100 actual issues.
  * KNN achieved high precision (87.50%) but failed to detect over 69% of machine failures (recall 30.88%).
  * Gaussian Naïve Bayes struggled due to feature dependencies (speed vs torque).
  * **Decision Tree** provided the optimal harmonic balance between detecting true failures ($63.24\%$ recall) while maintaining strong operational precision ($68.25\%$), yielding the highest **F1-Score (0.6565)**.

---

### Confusion Matrix & Classification Report

For the 2,000 machines in the unseen test set (1,932 healthy, 68 failures):

| Metric | Non-Failure (0) | Failure (1) | Macro Avg | Weighted Avg |
|:---|:---:|:---:|:---:|:---:|
| **Precision** | 0.99 | 0.68 | 0.83 | 0.98 |
| **Recall** | 0.99 | 0.63 | 0.81 | 0.98 |
| **F1-Score** | 0.99 | 0.66 | 0.82 | 0.98 |
| **Support** | 1,932 | 68 | 2,000 | 2,000 |

* **True Negatives (TN):** **1,912** healthy machines correctly identified.
* **False Positives (FP):** **20** healthy machines flagged for unnecessary inspection.
* **False Negatives (FN):** **25** failures missed.
* **True Positives (TP):** **43** impending failures caught in advance.

![Best Model Confusion Matrix](plots/12_best_model_confusion_matrix.png)

---

### Feature Importance Analysis

From the trained Decision Tree Classifier, feature importances reveal the physical variables governing equipment failure:

| Feature Name | Importance Score | Percentage Contribution |
|:---|:---:|:---:|
| **Rotational speed [rpm]** | 0.3355 | **33.55%** |
| **Torque [Nm]** | 0.3085 | **30.85%** |
| **Tool wear [min]** | 0.2547 | **25.47%** |
| **Temperature Difference** | 0.0546 | **5.46%** |
| **Air temperature [K]** | 0.0210 | **2.10%** |
| **Process temperature [K]** | 0.0163 | **1.63%** |
| **Type_L (Low Quality)** | 0.0049 | **0.49%** |
| **Type_M (Medium Quality)** | 0.0045 | **0.45%** |

![Decision Tree Feature Importance](plots/13_decision_tree_feature_importance.png)

*Key Takeaway:* Over **89.8%** of the decision tree's predictive power stems from just three core kinematic features: **Rotational Speed (33.55%)**, **Torque (30.85%)**, and **Tool Wear (25.47%)**.

---

## Findings, Conclusions & Recommendations

### Summary of Findings
1. **Severe Imbalance:** Only 3.39% of operating hours witness failure.
2. **Kinematic Drivers:** High torque coupled with abnormal rotational speed is the primary driver of equipment overstrain and power failure.
3. **Tool Lifecycle:** Tool wear beyond 180–200 minutes drastically accelerates failure probability.
4. **Thermal Gradient Warning:** A narrowing temperature delta between workpiece process and room air precedes heat dissipation failures.
5. **Quality Disparity:** Type-L products experience higher failure rates (3.92%) compared to Type-H products (2.09%).

### Actionable Preventive Maintenance Recommendations
1. **Automated Tool Wear Retraction:** Implement hard control limits that trigger automated tool changes once cumulative tool wear surpasses **180 minutes**, well before catastrophic wear at 240+ minutes.
2. **Torque Limiter Interlocks:** Integrate digital limit switches on CNC motors that throttle feed rates when continuous torque exceeds **55 Nm**.
3. **Ambient Climate Regulation:** Maintain shop floor air temperatures with HVAC cooling to sustain at least a **10 K dissipation buffer** between cutting process heat and ambient air.
4. **Targeted Inspections for Variant L:** Increase quality inspection frequencies for machining lines dedicated to Type L workpieces.

---

## Dataset Limitations

> [!WARNING]
> ### Synthetic Data Disclaimer
> The AI4I 2020 Predictive Maintenance Dataset is a **synthetic benchmark dataset** generated by mathematical models mimicking milling machine kinematics.
> While the dataset accurately reflects theoretical relationships (such as power formulas and tool wear equations), its results should **not be interpreted as guaranteed performance metrics** in commercial industrial plants. Real manufacturing environments introduce external factors such as vibration, cutting fluid degradation, electrical harmonics, and sensor drift that require on-site validation and fine-tuning.

---

## Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.12.5)
* Git

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/StutiJain4999/Predictive-Maintenance-of-Industrial-Machines.git
   cd Predictive-Maintenance-of-Industrial-Machines
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   # On Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run the Project

Run the main Python script from the root of the repository:

```bash
python "Predictive Maintenance of Industrial Machines.py"
```

### Running in Headless / Automated Mode (Skip GUI Windows)
If running on a remote server, continuous integration (CI) pipeline, or if you prefer saving plots without interactive pop-ups:
```powershell
# Windows PowerShell
$env:SHOW_PLOTS="0"; python "Predictive Maintenance of Industrial Machines.py"

# Linux / macOS Bash
SHOW_PLOTS=0 python "Predictive Maintenance of Industrial Machines.py"
```

---

## Output Files Generated

Executing the script automatically creates and populates:

* **`plots/` Directory:**
  * `01_machine_failure_distribution.png`
  * `02_product_type_distribution.png`
  * `03_failure_by_product_type.png`
  * `04_air_temperature_distribution.png`
  * `05_temperature_difference_vs_failure.png`
  * `06_tool_wear_vs_failure.png`
  * `07_torque_vs_failure.png`
  * `08_failure_type_counts.png`
  * `09_correlation_heatmap.png`
  * `10_model_accuracy_comparison.png`
  * `11_model_f1_score_comparison.png`
  * `12_best_model_confusion_matrix.png`
  * `13_decision_tree_feature_importance.png`
* **`results/` Directory:**
  * `model_comparison.csv` – Model performance metrics table.
  * `classification_report.txt` – Full classification report and confusion matrix.
  * `project_summary.txt` – Detailed summary of objectives, findings, and maintenance recommendations.

---

## Future Improvements

1. **Cost-Sensitive Threshold Tuning:** Adjust probability classification thresholds to optimize for custom business cost matrices (balancing false negative downtime costs against false positive inspection costs).
2. **Advanced Ensemble Methods:** Evaluate Random Forests, XGBoost, and LightGBM for non-linear boundary refinement.
3. **Resampling Techniques:** Experiment with SMOTE (Synthetic Minority Over-sampling Technique) or class weighting adjustments to enhance minority class recall.
4. **Multi-Class Failure Classification:** Train secondary multi-output models to predict the exact failure mode (`TWF`, `HDF`, `PWF`, `OSF`) once machine failure is flagged.
5. **Interactive Web Dashboard:** Build a Streamlit or Gradio interface enabling shop floor operators to input live sensor telemetry and receive immediate health assessments.

---

## License & Acknowledgements

* **Dataset:** Stephan Matzka, *AI4I 2020 Predictive Maintenance Dataset*, School of Engineering - Technology and Life, Hochschule für Technik und Wirtschaft Berlin, 2020. Released under Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
* **Code & Documentation:** Developed by **Stuti Jain** for college Continuous Assessment (CA) project and portfolio demonstration.
