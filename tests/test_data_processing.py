import pytest
import pandas as pd
import numpy as np
import os

def test_raw_data_exists():
    """Test that raw data files exist"""
    raw_files = [
        'data/raw/Fraud_Data.csv',
        'data/raw/IpAddress_to_Country.csv',
        'data/raw/creditcard.csv'
    ]
    
    for file in raw_files:
        assert os.path.exists(file), f"{file} not found!"

def test_processed_data_exists():
    """Test that processed files exist"""
    processed_files = [
        'data/processed/fraud_data_final.csv',
        'data/processed/creditcard_final.csv'
    ]
    
    for file in processed_files:
        assert os.path.exists(file), f"{file} not found!"

def test_fraud_data_columns():
    """Test that fraud data has required columns"""
    df = pd.read_csv('data/processed/fraud_data_final.csv')
    required_columns = ['class', 'time_since_signup_hours', 'purchase_hour', 'country']
    
    for col in required_columns:
        assert col in df.columns, f"Column '{col}' missing from fraud data!"

def test_credit_data_columns():
    """Test that credit data has required columns"""
    df = pd.read_csv('data/processed/creditcard_final.csv')
    required_columns = ['Class', 'Amount_scaled', 'Time']
    
    for col in required_columns:
        assert col in df.columns, f"Column '{col}' missing from credit data!"

def test_no_missing_values():
    """Test that processed data has no missing values"""
    df_fraud = pd.read_csv('data/processed/fraud_data_final.csv')
    df_credit = pd.read_csv('data/processed/creditcard_final.csv')
    
    assert df_fraud.isnull().sum().sum() == 0, "Fraud data has missing values!"
    assert df_credit.isnull().sum().sum() == 0, "Credit data has missing values!"

def test_fraud_class_balance():
    """Test that fraud data has expected class distribution"""
    df = pd.read_csv('data/processed/fraud_data_final.csv')
    fraud_pct = df['class'].mean() * 100
    
    # Should be around 9.36% as in original
    assert 8 < fraud_pct < 11, f"Fraud percentage is {fraud_pct}%, expected ~9.36%"

def test_credit_class_balance():
    """Test that credit data has expected class distribution"""
    df = pd.read_csv('data/processed/creditcard_final.csv')
    fraud_pct = df['Class'].mean() * 100
    
    # Should be around 0.17% as in original
    assert 0.1 < fraud_pct < 0.3, f"Fraud percentage is {fraud_pct}%, expected ~0.17%"

def test_time_since_signup_range():
    """Test that time_since_signup_hours has reasonable values"""
    df = pd.read_csv('data/processed/fraud_data_final.csv')
    time_values = df['time_since_signup_hours']
    
    # Should be non-negative and not too extreme
    assert (time_values >= 0).all(), "Negative time since signup found!"
    assert time_values.max() < 3000, f"Max time {time_values.max()} hours is too high!"

def test_smote_resampled_files_exist():
    """Test that SMOTE resampled files exist"""
    smote_files = [
        'data/processed/X_fraud_train_resampled.npy',
        'data/processed/y_fraud_train_resampled.npy',
        'data/processed/X_fraud_test.npy',
        'data/processed/y_fraud_test.npy',
        'data/processed/X_credit_train_resampled.npy',
        'data/processed/y_credit_train_resampled.npy',
        'data/processed/X_credit_test.npy',
        'data/processed/y_credit_test.npy'
    ]
    
    for file in smote_files:
        assert os.path.exists(file), f"{file} not found!"

def test_smote_balanced():
    """Test that SMOTE created balanced classes"""
    y_fraud_train = np.load('data/processed/y_fraud_train_resampled.npy')
    y_credit_train = np.load('data/processed/y_credit_train_resampled.npy')
    
    # Should be exactly 50% fraud
    assert abs(y_fraud_train.mean() - 0.5) < 0.01, "Fraud data not balanced after SMOTE!"
    assert abs(y_credit_train.mean() - 0.5) < 0.01, "Credit data not balanced after SMOTE!"