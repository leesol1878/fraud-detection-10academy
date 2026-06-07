"""
Class imbalance handling module for fraud detection
Author: [Your Name]
Date: June 7, 2026
"""

import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def stratified_split(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Perform stratified train-test split to preserve class distribution
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples, Fraud %: {y_train.mean()*100:.4f}%")
        logger.info(f"Test set: {X_test.shape[0]} samples, Fraud %: {y_test.mean()*100:.4f}%")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise

def apply_smote(X_train, y_train, random_state: int = 42):
    """
    Apply SMOTE to balance training data
    
    Args:
        X_train: Training features
        y_train: Training targets
        random_state: Random seed for reproducibility
        
    Returns:
        X_resampled, y_resampled
    """
    try:
        # Log before SMOTE
        before_counter = Counter(y_train)
        before_fraud_pct = y_train.mean() * 100
        
        logger.info(f"Before SMOTE - Class distribution: {before_counter}")
        logger.info(f"Before SMOTE - Fraud %: {before_fraud_pct:.4f}%")
        
        # Apply SMOTE
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Log after SMOTE
        after_counter = Counter(y_resampled)
        after_fraud_pct = y_resampled.mean() * 100
        
        logger.info(f"After SMOTE - Class distribution: {after_counter}")
        logger.info(f"After SMOTE - Fraud %: {after_fraud_pct:.2f}%")
        logger.info(f"SMOTE created {len(X_resampled) - len(X_train)} synthetic samples")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"SMOTE application failed: {e}")
        raise

def get_imbalance_report(y_train, y_test):
    """
    Generate a report on class imbalance
    
    Args:
        y_train: Training targets
        y_test: Test targets
        
    Returns:
        Dictionary with imbalance metrics
    """
    report = {
        'train_fraud_pct': y_train.mean() * 100,
        'test_fraud_pct': y_test.mean() * 100,
        'train_ratio': Counter(y_train),
        'test_ratio': Counter(y_test),
        'imbalance_ratio': (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else float('inf')
    }
    
    logger.info(f"Imbalance Report:")
    logger.info(f"  Train Fraud %: {report['train_fraud_pct']:.4f}%")
    logger.info(f"  Test Fraud %: {report['test_fraud_pct']:.4f}%")
    logger.info(f"  Imbalance Ratio: {report['imbalance_ratio']:.2f}:1")
    
    return report