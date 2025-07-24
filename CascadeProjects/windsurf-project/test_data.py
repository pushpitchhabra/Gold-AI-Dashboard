#!/usr/bin/env python3
"""
Simple test script to verify data loading works
"""

import pandas as pd
import numpy as np
from datetime import datetime

def test_data_loading():
    """Test basic data loading functionality"""
    print("Testing data loading...")
    
    # Path to the historical data
    data_path = "/Users/pushpitchhabra/Desktop/Projects/Windsurf/Gold Windsurf/CascadeProjects/30 Min - Gold Histdata.com summary.csv"
    
    try:
        # Load the CSV file
        data = pd.read_csv(data_path)
        print(f"✓ Data loaded successfully: {data.shape}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(data.head())
        
        # Convert datetime column
        data['Datetime'] = pd.to_datetime(data['Datetime'], format='%d/%m/%y %H:%M')
        print(f"✓ Datetime conversion successful")
        
        # Set datetime as index
        data.set_index('Datetime', inplace=True)
        data.sort_index(inplace=True)
        
        # Add Volume column
        data['Volume'] = 0
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✓ Data preprocessing complete")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Total records: {len(data)}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    data = test_data_loading()
    if data is not None:
        print("\n✅ Data loading test PASSED")
    else:
        print("\n❌ Data loading test FAILED")
