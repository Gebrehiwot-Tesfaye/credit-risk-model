#!/usr/bin/env python3
"""
Simple test script to verify data loading and basic functionality.
"""

import os
import sys

# Add src to path
sys.path.append('src')

try:
    from data_processing import load_data, add_is_high_risk
    print("✅ Successfully imported data_processing")
    
    # Check if data file exists
    data_path = os.path.join('data', 'raw', 'data.csv')
    print(f"Looking for data at: {os.path.abspath(data_path)}")
    
    if os.path.exists(data_path):
        print("✅ Data file found")
        
        # Try to load data
        df = load_data(data_path)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        
        # Try to add target variable
        df_with_target = add_is_high_risk(df)
        print(f"✅ Target variable added. Shape: {df_with_target.shape}")
        
        # Check target distribution
        target_counts = df_with_target['is_high_risk'].value_counts()
        print(f"✅ Target distribution: {target_counts.to_dict()}")
        
        print("\n🎉 All tests passed! You can now run the training script.")
        
    else:
        print(f"❌ Data file not found at: {data_path}")
        print("Please make sure the data.csv file is in the data/raw/ directory")
        
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc() 