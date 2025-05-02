#!/usr/bin/env python3

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing NDVI and futures data."""
    
    def __init__(self, config):
        """Initialize the data processor with configuration."""
        self.config = config
        self.ndvi_scaler = MinMaxScaler()
        self.futures_scaler = MinMaxScaler()
        
    def load_futures_data(self, file_path):
        """Load and preprocess futures data."""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Scale the data
            scaled_data = self.futures_scaler.fit_transform(df)
            
            return scaled_data, df.index
            
        except Exception as e:
            logger.error(f"Error loading futures data: {str(e)}")
            raise
            
    def load_ndvi_data(self, directory):
        """Load and preprocess NDVI data."""
        try:
            ndvi_files = sorted([f for f in os.listdir(directory) if f.endswith('.csv')])
            ndvi_data = []
            dates = []
            
            for file in ndvi_files:
                df = pd.read_csv(os.path.join(directory, file))
                ndvi_data.append(df['ndvi'].values)
                dates.append(pd.to_datetime(file.split('.')[0]))
                
            # Scale the data
            scaled_data = self.ndvi_scaler.fit_transform(np.array(ndvi_data).reshape(-1, 1))
            
            return scaled_data, dates
            
        except Exception as e:
            logger.error(f"Error loading NDVI data: {str(e)}")
            raise
            
    def create_sequences(self, data, sequence_length):
        """Create sequences for model input."""
        try:
            sequences = []
            for i in range(len(data) - sequence_length):
                sequences.append(data[i:(i + sequence_length)])
            return np.array(sequences)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise
            
    def align_data(self, ndvi_data, futures_data, ndvi_dates, futures_dates):
        """Align NDVI and futures data by date."""
        try:
            # Create date range
            start_date = max(ndvi_dates[0], futures_dates[0])
            end_date = min(ndvi_dates[-1], futures_dates[-1])
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Align data
            aligned_ndvi = []
            aligned_futures = []
            
            for date in date_range:
                if date in ndvi_dates and date in futures_dates:
                    ndvi_idx = ndvi_dates.index(date)
                    futures_idx = futures_dates.index(date)
                    aligned_ndvi.append(ndvi_data[ndvi_idx])
                    aligned_futures.append(futures_data[futures_idx])
                    
            return np.array(aligned_ndvi), np.array(aligned_futures), date_range
            
        except Exception as e:
            logger.error(f"Error aligning data: {str(e)}")
            raise
            
    def save_scalers(self, directory):
        """Save scalers for later use."""
        try:
            import joblib
            os.makedirs(directory, exist_ok=True)
            joblib.dump(self.ndvi_scaler, os.path.join(directory, 'ndvi_scaler.joblib'))
            joblib.dump(self.futures_scaler, os.path.join(directory, 'futures_scaler.joblib'))
            
        except Exception as e:
            logger.error(f"Error saving scalers: {str(e)}")
            raise
            
    def load_scalers(self, directory):
        """Load saved scalers."""
        try:
            import joblib
            self.ndvi_scaler = joblib.load(os.path.join(directory, 'ndvi_scaler.joblib'))
            self.futures_scaler = joblib.load(os.path.join(directory, 'futures_scaler.joblib'))
            
        except Exception as e:
            logger.error(f"Error loading scalers: {str(e)}")
            raise 
