import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import rasterio
from rasterio.transform import from_origin
import tensorflow as tf
from datetime import datetime, timedelta
import glob
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_ndvi_data(data_dir, start_date=None, end_date=None):
    """
    Load and preprocess NDVI data from GeoTIFF files.
    
    Args:
        data_dir: Directory containing NDVI GeoTIFF files
        start_date: Start date for data (optional)
        end_date: End date for data (optional)
    
    Returns:
        tuple: (ndvi_data, dates)
    """
    logger.info("Loading NDVI data...")
    
    # Get all GeoTIFF files
    ndvi_files = sorted(glob.glob(os.path.join(data_dir, '*.tif')))
    
    if not ndvi_files:
        raise FileNotFoundError(f"No NDVI files found in {data_dir}")
    
    # Extract dates from filenames
    dates = []
    for file in ndvi_files:
        date_str = os.path.basename(file).split('_')[0]
        date = datetime.strptime(date_str, '%Y%m%d')
        dates.append(date)
    
    # Filter by date range if specified
    if start_date and end_date:
        mask = (np.array(dates) >= start_date) & (np.array(dates) <= end_date)
        ndvi_files = np.array(ndvi_files)[mask]
        dates = np.array(dates)[mask]
    
    # Load and preprocess NDVI data
    ndvi_data = []
    for file in ndvi_files:
        with rasterio.open(file) as src:
            data = src.read(1)  # Read first band
            # Replace no-data values with NaN
            data[data == src.nodata] = np.nan
            # Normalize to [-1, 1] range
            data = (data - 0.5) * 2
            ndvi_data.append(data)
    
    ndvi_data = np.array(ndvi_data)
    logger.info(f"Loaded {len(ndvi_data)} NDVI images with shape {ndvi_data.shape}")
    
    return ndvi_data, dates

def load_futures_data(data_dir, start_date=None, end_date=None):
    """
    Load and preprocess futures data from CSV files.
    
    Args:
        data_dir: Directory containing futures CSV files
        start_date: Start date for data (optional)
        end_date: End date for data (optional)
    
    Returns:
        tuple: (futures_data, dates)
    """
    logger.info("Loading futures data...")
    
    # Load futures data
    futures_file = os.path.join(data_dir, 'corn_futures_weekly.csv')
    if not os.path.exists(futures_file):
        raise FileNotFoundError(f"Futures data file not found: {futures_file}")
    
    df = pd.read_csv(futures_file, parse_dates=['Date'])
    
    # Filter by date range if specified
    if start_date and end_date:
        df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Extract features
    features = ['log_return', 'future_return']
    futures_data = df[features].values
    
    # Normalize features
    scaler = StandardScaler()
    futures_data = scaler.fit_transform(futures_data)
    
    dates = df['Date'].values
    logger.info(f"Loaded {len(futures_data)} futures data points with shape {futures_data.shape}")
    
    return futures_data, dates, scaler

def create_sequences(ndvi_data, futures_data, dates, sequence_length=8):
    """
    Create sequences for model training.
    
    Args:
        ndvi_data: NDVI data array
        futures_data: Futures data array
        dates: Array of dates
        sequence_length: Length of input sequences
    
    Returns:
        tuple: (X_ndvi, X_futures, y)
    """
    logger.info("Creating sequences...")
    
    # Find common dates
    ndvi_dates = pd.Series(dates[0])
    futures_dates = pd.Series(dates[1])
    common_dates = pd.Series(np.intersect1d(ndvi_dates, futures_dates))
    
    # Create sequences
    X_ndvi = []
    X_futures = []
    y = []
    
    for i in range(len(common_dates) - sequence_length):
        # Get sequence dates
        seq_dates = common_dates[i:i + sequence_length]
        target_date = common_dates[i + sequence_length]
        
        # Get NDVI data for sequence
        ndvi_indices = [np.where(ndvi_dates == date)[0][0] for date in seq_dates]
        ndvi_seq = ndvi_data[ndvi_indices]
        
        # Get futures data for sequence
        futures_indices = [np.where(futures_dates == date)[0][0] for date in seq_dates]
        futures_seq = futures_data[futures_indices, 0]  # Use log_return
        
        # Get target (future return)
        target_idx = np.where(futures_dates == target_date)[0][0]
        target = futures_data[target_idx, 1]  # Use future_return
        
        X_ndvi.append(ndvi_seq)
        X_futures.append(futures_seq)
        y.append(target)
    
    X_ndvi = np.array(X_ndvi)
    X_futures = np.array(X_futures)
    y = np.array(y)
    
    logger.info(f"Created {len(X_ndvi)} sequences")
    logger.info(f"X_ndvi shape: {X_ndvi.shape}")
    logger.info(f"X_futures shape: {X_futures.shape}")
    logger.info(f"y shape: {y.shape}")
    
    return X_ndvi, X_futures, y

def prepare_tuning_data(
    ndvi_dir,
    futures_dir,
    start_date=None,
    end_date=None,
    sequence_length=8
):
    """
    Prepare data for hyperparameter tuning.
    
    Args:
        ndvi_dir: Directory containing NDVI data
        futures_dir: Directory containing futures data
        start_date: Start date for data (optional)
        end_date: End date for data (optional)
        sequence_length: Length of input sequences
    
    Returns:
        tuple: (X_ndvi, X_futures, y, scaler)
    """
    # Load data
    ndvi_data, ndvi_dates = load_ndvi_data(ndvi_dir, start_date, end_date)
    futures_data, futures_dates, scaler = load_futures_data(futures_dir, start_date, end_date)
    
    # Create sequences
    X_ndvi, X_futures, y = create_sequences(
        ndvi_data,
        futures_data,
        [ndvi_dates, futures_dates],
        sequence_length
    )
    
    return X_ndvi, X_futures, y, scaler

if __name__ == "__main__":
    # Example usage
    ndvi_dir = "data/ndvi"
    futures_dir = "data/futures"
    
    # Set date range for last 5 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    try:
        X_ndvi, X_futures, y, scaler = prepare_tuning_data(
            ndvi_dir,
            futures_dir,
            start_date,
            end_date
        )
        
        # Save preprocessed data
        os.makedirs("data/processed", exist_ok=True)
        np.save("data/processed/X_ndvi.npy", X_ndvi)
        np.save("data/processed/X_futures.npy", X_futures)
        np.save("data/processed/y.npy", y)
        joblib.dump(scaler, "data/processed/scaler.joblib")
        
        logger.info("Data preparation completed successfully")
        logger.info(f"X_ndvi shape: {X_ndvi.shape}")
        logger.info(f"X_futures shape: {X_futures.shape}")
        logger.info(f"y shape: {y.shape}")
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise 
