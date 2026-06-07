"""
Data preprocessing module for fraud detection
Author: [Your Name]
Date: June 7, 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with error handling
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
        
    Raises:
        FileNotFoundError: If file doesn't exist
        pd.errors.EmptyDataError: If file is empty
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        raise

def check_missing_values(df: pd.DataFrame, name: str = "Dataset") -> dict:
    """
    Check for missing values in DataFrame
    
    Args:
        df: Input DataFrame
        name: Name of dataset for logging
        
    Returns:
        Dictionary with missing value counts per column
    """
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    
    if len(missing_cols) > 0:
        logger.warning(f"{name}: Found missing values in {len(missing_cols)} columns")
        for col, count in missing_cols.items():
            logger.warning(f"  - {col}: {count} missing ({count/len(df)*100:.2f}%)")
    else:
        logger.info(f"{name}: No missing values found")
    
    return missing_counts.to_dict()

def convert_to_datetime(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Convert specified columns to datetime format
    
    Args:
        df: Input DataFrame
        columns: List of column names to convert
        
    Returns:
        DataFrame with converted columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col])
                logger.info(f"Converted {col} to datetime")
            except Exception as e:
                logger.error(f"Failed to convert {col}: {e}")
                raise
    
    return df_copy

def create_time_features(df: pd.DataFrame, signup_col: str, purchase_col: str) -> pd.DataFrame:
    """
    Create time-based features for fraud detection
    
    Args:
        df: Input DataFrame with datetime columns
        signup_col: Name of signup time column
        purchase_col: Name of purchase time column
        
    Returns:
        DataFrame with new time features
    """
    df_copy = df.copy()
    
    # Time since signup (hours)
    df_copy['time_since_signup_hours'] = (
        df_copy[purchase_col] - df_copy[signup_col]
    ).dt.total_seconds() / 3600
    
    # Hour of day (0-23)
    df_copy['purchase_hour'] = df_copy[purchase_col].dt.hour
    
    # Day of week (0=Monday, 6=Sunday)
    df_copy['purchase_dayofweek'] = df_copy[purchase_col].dt.dayofweek
    
    logger.info(f"Created time features: time_since_signup_hours, purchase_hour, purchase_dayofweek")
    logger.info(f"  - Fraud within 1 hour: {(df_copy['time_since_signup_hours'] < 1).sum()} transactions")
    
    return df_copy

def calculate_transaction_velocity(df: pd.DataFrame, user_col: str, time_col: str) -> pd.DataFrame:
    """
    Calculate transaction velocity features per user
    
    Args:
        df: Input DataFrame
        user_col: User ID column name
        time_col: Transaction time column name
        
    Returns:
        DataFrame with velocity features
    """
    df_copy = df.sort_values([user_col, time_col]).copy()
    
    # Count transactions per user
    df_copy['user_transaction_count'] = df_copy.groupby(user_col)[user_col].transform('count')
    
    # Hours since last purchase
    df_copy['hours_since_last_purchase'] = (
        df_copy.groupby(user_col)[time_col].diff().dt.total_seconds() / 3600
    )
    df_copy['hours_since_last_purchase'] = df_copy['hours_since_last_purchase'].fillna(0)
    
    logger.info(f"Created velocity features for {df_copy[user_col].nunique()} unique users")
    
    return df_copy

def encode_categorical(df: pd.DataFrame, categorical_cols: list, drop_first: bool = True) -> pd.DataFrame:
    """
    One-hot encode categorical columns
    
    Args:
        df: Input DataFrame
        categorical_cols: List of column names to encode
        drop_first: Whether to drop first category to avoid multicollinearity
        
    Returns:
        DataFrame with encoded columns
    """
    df_copy = df.copy()
    
    for col in categorical_cols:
        if col in df_copy.columns:
            dummies = pd.get_dummies(df_copy[col], prefix=col, drop_first=drop_first)
            df_copy = pd.concat([df_copy, dummies], axis=1)
            df_copy = df_copy.drop(columns=[col])
            logger.info(f"Encoded {col} into {dummies.shape[1]} features")
    
    return df_copy

def scale_numerical(df: pd.DataFrame, numerical_cols: list, scaler_type: str = 'standard') -> tuple:
    """
    Scale numerical features
    
    Args:
        df: Input DataFrame
        numerical_cols: List of column names to scale
        scaler_type: 'standard' or 'minmax'
        
    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    df_copy = df.copy()
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df_copy[numerical_cols] = scaler.fit_transform(df_copy[numerical_cols])
    logger.info(f"Scaled {len(numerical_cols)} numerical features using {scaler_type}")
    
    return df_copy, scaler