# import pytest

# Placeholder test to ensure CI passes

def test__placeholder():
    assert True# import pytest

# Placeholder test to ensure CI passes

def test__placeholder():
    assert True

# Example template for future tests
# from src import data_processing
# def test_some_data_processing_function():
#     result = data_processing.some_function(args)
#     assert result == expected

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processing import load_data, calculate_rfm


def test_load_data():
    """Test load_data function loads CSV file correctly."""
    # Create a temporary test CSV file
    test_data = pd.DataFrame({
        'CustomerId': [1, 2, 3],
        'Amount': [100, 200, 300],
        'TransactionStartTime': ['2023-01-01', '2023-01-02', '2023-01-03']
    })
    
    # Save test file
    test_file = 'test_data.csv'
    test_data.to_csv(test_file, index=False)
    
    try:
        # Test loading
        loaded_data = load_data(test_file)
        
        # Assertions
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 3
        assert list(loaded_data.columns) == ['CustomerId', 'Amount', 'TransactionStartTime']
        assert loaded_data['Amount'].sum() == 600
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)


def test_calculate_rfm():
    """Test calculate_rfm function computes RFM metrics correctly."""
    # Create test data
    test_data = pd.DataFrame({
        'CustomerId': [1, 1, 2, 2, 3],
        'TransactionStartTime': [
            '2023-01-01', '2023-01-02',  # Customer 1: 2 transactions
            '2023-01-01', '2023-01-03',  # Customer 2: 2 transactions  
            '2023-01-01'                 # Customer 3: 1 transaction
        ],
        'Amount': [100, 150, 200, 250, 300]
    })
    
    # Calculate RFM
    rfm_result = calculate_rfm(test_data)
    
    # Assertions
    assert isinstance(rfm_result, pd.DataFrame)
    assert len(rfm_result) == 3  # 3 unique customers
    assert 'recency' in rfm_result.columns
    assert 'frequency' in rfm_result.columns
    assert 'monetary' in rfm_result.columns
    
    # Check specific values
    customer_1_rfm = rfm_result[rfm_result['CustomerId'] == 1].iloc[0]
    assert customer_1_rfm['frequency'] == 2  # 2 transactions
    assert customer_1_rfm['monetary'] == 250  # 100 + 150
    
    customer_3_rfm = rfm_result[rfm_result['CustomerId'] == 3].iloc[0]
    assert customer_3_rfm['frequency'] == 1  # 1 transaction
    assert customer_3_rfm['monetary'] == 300  # 300


def test_calculate_rfm_with_snapshot_date():
    """Test calculate_rfm function with custom snapshot date."""
    # Create test data
    test_data = pd.DataFrame({
        'CustomerId': [1, 2],
        'TransactionStartTime': ['2023-01-01', '2023-01-02'],
        'Amount': [100, 200]
    })
    
    # Use custom snapshot date
    snapshot_date = pd.Timestamp('2023-01-05')
    rfm_result = calculate_rfm(test_data, snapshot_date=snapshot_date)
    
    # Assertions
    assert isinstance(rfm_result, pd.DataFrame)
    assert len(rfm_result) == 2
    
    # Check recency calculations
    customer_1_recency = rfm_result[rfm_result['CustomerId'] == 1]['recency'].iloc[0]
    customer_2_recency = rfm_result[rfm_result['CustomerId'] == 2]['recency'].iloc[0]
    
    # Customer 1: 2023-01-01 to 2023-01-05 = 4 days
    # Customer 2: 2023-01-02 to 2023-01-05 = 3 days
    assert customer_1_recency == 4
    assert customer_2_recency == 3


# Example template for future tests
# from src import data_processing
# def test_some_data_processing_function():
#     result = data_processing.some_function(args)
#     assert result == expected
