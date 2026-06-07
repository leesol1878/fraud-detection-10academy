"""
Model evaluation module for imbalanced fraud detection
Author: Tsion Solomon
Date: June 7, 2026
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    roc_auc_score, average_precision_score
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model with metrics suitable for imbalanced data
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC metrics)
        
    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {
        'f1_score': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
    
    # Log results
    tn, fp, fn, tp = metrics['confusion_matrix'][0]
    logger.info(f"Evaluation Results:")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    if y_pred_proba is not None:
        logger.info(f"  AUC-PR: {metrics['auc_pr']:.4f} (Primary metric)")
    logger.info(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    return metrics