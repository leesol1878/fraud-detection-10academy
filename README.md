# \# Fraud Detection for E-commerce and Bank Transactions

# 

# 10 Academy Week 5-6 Challenge (June 4-16, 2026)

# 

# \## Overview

# This project builds fraud detection models for two transaction streams:

# \- \*\*E-commerce transactions\*\* (with user, device, and behavioral context)

# \- \*\*Bank credit card transactions\*\* (anonymized PCA features for privacy)

# 

# \## Business Problem

# Adey Innovations Inc. needs to detect fraud across both platforms while minimizing:

# \- \*\*False positives\*\* (legitimate transactions flagged as fraud) → Customer frustration

# \- \*\*False negatives\*\* (missed fraud) → Direct financial loss

# 

\## Project Structure

fraud-detection-10academy/
===

# ├── data/

# │ ├── raw/ # Original datasets (not committed to git)

# │ └── processed/ # Cleaned and feature-engineered data

# ├── notebooks/

# │ ├── eda-fraud-data.ipynb # E-commerce EDA

# │ ├── eda-creditcard.ipynb # Credit card EDA

# │ ├── feature-engineering.ipynb # Feature creation

# │ ├── modeling.ipynb # Model training \& evaluation

# │ └── shap-explainability.ipynb # SHAP analysis

# ├── src/ # Reusable Python modules

# ├── tests/ # Unit tests

├── models/ # Saved model artifacts
└── scripts/ # Utility scripts
===







\## Setup Instructions



\### 1. Clone the repository

```bash

git clone https://github.com/lees01878/fraud-detection-10academy.git

cd fraud-detection-10academy

