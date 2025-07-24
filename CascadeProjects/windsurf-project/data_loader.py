import pandas as pd
import numpy as np
from datetime import datetime

class GoldDataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """Load and preprocess the gold data"""
        try:
            # Load the CSV file
            self.data = pd.read_csv(self.data_path)
            
            # Convert datetime column
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'], format='%d/%m/%y %H:%M')
            
            # Set datetime as index
            self.data.set_index('Datetime', inplace=True)
            
            # Sort by datetime
            self.data.sort_index(inplace=True)
            
            # Add Volume column with zeros (required for backtesting library)
            self.data['Volume'] = 0
            
            # Ensure proper column order for backtesting
            self.data = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            print(f"Data loaded successfully!")
            print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
            print(f"Total records: {len(self.data)}")
            print(f"Data shape: {self.data.shape}")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_data_info(self):
        """Get basic information about the dataset"""
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return
            
        print("\n=== DATA INFORMATION ===")
        print(f"Date range: {self.data.index[0]} to {self.data.index[-1]}")
        print(f"Total records: {len(self.data)}")
        print(f"Columns: {list(self.data.columns)}")
        print("\nFirst 5 rows:")
        print(self.data.head())
        print("\nLast 5 rows:")
        print(self.data.tail())
        print("\nBasic statistics:")
        print(self.data.describe())
        
    def check_data_quality(self):
        """Check for missing values and data quality issues"""
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return
            
        print("\n=== DATA QUALITY CHECK ===")
        print("Missing values:")
        print(self.data.isnull().sum())
        
        print("\nDuplicate timestamps:")
        duplicates = self.data.index.duplicated().sum()
        print(f"Found {duplicates} duplicate timestamps")
        
        print("\nPrice consistency check:")
        # Check if High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        inconsistent = (
            (self.data['High'] < self.data['Low']) |
            (self.data['High'] < self.data['Open']) |
            (self.data['High'] < self.data['Close']) |
            (self.data['Low'] > self.data['Open']) |
            (self.data['Low'] > self.data['Close'])
        ).sum()
        print(f"Found {inconsistent} inconsistent OHLC records")
