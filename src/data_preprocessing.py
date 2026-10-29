"""
Data Preprocessing & Feature Engineering Module
Dataset: AI4I 2020 Predictive Maintenance Dataset (ai4i2020.csv)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_dataset_path():
    """Locate dataset file reliably relative to repository root or script."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    candidates = [
        os.path.join(repo_root, "dataset", "ai4i2020.csv"),
        os.path.join(repo_root, "ai4i2020.csv"),
        os.path.join("dataset", "ai4i2020.csv"),
        "ai4i2020.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not locate ai4i2020.csv dataset file.")


def load_raw_data():
    """Load the original unmodified dataset."""
    path = get_dataset_path()
    return pd.read_csv(path)


def clean_data(df):
    """
    Perform cleaning operations:
    - Impute missing numerical values with median
    - Impute missing categorical values with mode
    - Remove duplicate rows
    - Normalize 'Type' column
    """
    cleaned = df.copy()
    
    # Impute numerical missing values
    for col in cleaned.select_dtypes(include=np.number).columns:
        cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        
    # Impute categorical missing values
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        if cleaned[col].isnull().sum() > 0:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
            
    # Drop duplicate records
    cleaned = cleaned.drop_duplicates()
    
    # Normalize product type
    if "Type" in cleaned.columns:
        cleaned["Type"] = cleaned["Type"].astype(str).str.strip().str.upper()
        
    return cleaned


def engineer_features(df):
    """
    Feature Engineering:
    Calculate Temperature Difference = Process temperature [K] - Air temperature [K]
    """
    featured = df.copy()
    if "Process temperature [K]" in featured.columns and "Air temperature [K]" in featured.columns:
        featured["Temperature Difference"] = (
            featured["Process temperature [K]"] - featured["Air temperature [K]"]
        )
    return featured


def prepare_ml_data(df):
    """
    Strict Target Leakage Prevention:
    - Exclude identifiers (UDI, Product ID)
    - Exclude breakdown-specific mode flags (TWF, HDF, PWF, OSF, RNF)
    - Encode categorical 'Type' with drop_first=True
    - Separate X (features) and Y (target: Machine failure)
    """
    ml_df = df.copy()
    
    # Ensure Temperature Difference is present
    if "Temperature Difference" not in ml_df.columns:
        ml_df = engineer_features(ml_df)
        
    # Drop leakage columns and identifiers
    drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    ml_df = ml_df.drop(columns=[c for c in drop_cols if c in ml_df.columns], errors="ignore")
    
    # One-hot encode categorical Type
    if "Type" in ml_df.columns:
        ml_df = pd.get_dummies(ml_df, columns=["Type"], drop_first=True, dtype=int)
        
    # Separate input features and target
    X = ml_df.drop("Machine failure", axis=1)
    Y = ml_df["Machine failure"]
    
    return X, Y


def split_and_scale(X, Y, test_size=0.20, random_state=42):
    """
    Split data reproducibly (80:20 stratified) and fit scaler strictly on X_train.
    """
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size,
        random_state=random_state,
        stratify=Y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "Y_train": Y_train,
        "Y_test": Y_test,
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "scaler": scaler,
        "feature_names": X.columns.tolist()
    }
