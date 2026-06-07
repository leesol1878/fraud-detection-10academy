import pytest
import os

def test_gitignore_exists():
    """Test that .gitignore exists"""    
    assert os.path.exists('.gitignore'), ".gitignore file missing!"


def test_requirements_exists():
    """Test that requirements.txt exists"""
    assert os.path.exists('requirements.txt'), "requirements.txt missing!"

def test_readme_exists():
    """Test that README.md exists"""
    assert os.path.exists('README.md'), "README.md missing!"

def test_notebooks_exist():
    """Test that notebook files exist"""
    notebooks = [
        'notebooks/eda-fraud-data.ipynb',
        'notebooks/eda-creditcard.ipynb',
        'notebooks/feature-engineering.ipynb',
    ]
    
    for notebook in notebooks:
        assert os.path.exists(notebook), f"{notebook} missing!"

def test_imports():
    """Test that all required libraries can be imported"""
    try:
        import pandas
        import numpy
        import sklearn
        import matplotlib
        import seaborn
        result = True
    except ImportError as e:
        result = False
        print(f"Import error: {e}")
    
    assert result, "Some imports failed"

def test_src_exists():
    """Test that src directory exists"""
    assert os.path.exists('src'), "src directory missing!"
    assert os.path.exists('src/__init__.py'), "src/__init__.py missing!"

def test_tests_exists():
    """Test that tests directory exists"""
    assert os.path.exists('tests'), "tests directory missing!"
    assert os.path.exists('tests/__init__.py'), "tests/__init__.py missing!"

def test_sanity():
    """Sanity check - always passes"""
    assert True