"""
Script to test the preprocessing pipeline
Run with: python scripts/run_preprocessing.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import (
    load_data, check_missing_values, convert_to_datetime,
    create_time_features, calculate_transaction_velocity,
    encode_categorical, scale_numerical
)
from src.imbalance import stratified_split, apply_smote, get_imbalance_report

def test_ecommerce_pipeline():
    """Test the complete e-commerce preprocessing pipeline"""
    print("\n" + "="*60)
    print("TESTING E-COMMERCE PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_data('data/raw/Fraud_Data.csv')
    print(f"✓ Loaded data: {df.shape}")
    
    # Check missing values
    missing = check_missing_values(df, "E-commerce")
    print(f"✓ Missing values checked")
    
    # Convert to datetime
    df = convert_to_datetime(df, ['signup_time', 'purchase_time'])
    print(f"✓ Converted to datetime")
    
    # Create time features
    df = create_time_features(df, 'signup_time', 'purchase_time')
    print(f"✓ Created time features")
    
    # Calculate velocity
    df = calculate_transaction_velocity(df, 'user_id', 'purchase_time')
    print(f"✓ Calculated velocity features")
    
    # Encode categorical
    categorical_cols = ['source', 'browser', 'sex']
    df = encode_categorical(df, categorical_cols)
    print(f"✓ Encoded categorical features")
    
    # Scale numerical
    numerical_cols = ['purchase_value', 'age', 'time_since_signup_hours']
    df, scaler = scale_numerical(df, numerical_cols)
    print(f"✓ Scaled numerical features")
    
    print("\n✅ E-commerce pipeline test PASSED!")
    return True

def test_creditcard_pipeline():
    """Test the credit card preprocessing pipeline"""
    print("\n" + "="*60)
    print("TESTING CREDIT CARD PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load data
    df = load_data('data/raw/creditcard.csv')
    print(f"✓ Loaded data: {df.shape}")
    
    # Check missing values
    missing = check_missing_values(df, "Credit Card")
    print(f"✓ Missing values checked")
    
    # Scale Amount
    df, scaler = scale_numerical(df, ['Amount'])
    print(f"✓ Scaled Amount feature")
    
    print("\n Credit card pipeline test PASSED!")
    return True

if __name__ == "__main__":
    try:
        test_ecommerce_pipeline()
        test_creditcard_pipeline()
        print("\n" + "="*60)
        print(" ALL TESTS PASSED! ")
        print("="*60)
    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        sys.exit(1)