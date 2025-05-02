import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_directory_structure():
    """Create necessary directories for the project."""
    directories = [
        'data/ndvi',
        'data/futures',
        'data/processed',
        'experiments'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def generate_sample_ndvi_data():
    """Generate sample NDVI data for testing."""
    # Create a 100x100 grid for each month
    grid_size = (100, 100)
    months = 12
    
    for month in range(1, months + 1):
        # Generate random NDVI values between 0 and 1
        ndvi_data = np.random.uniform(0, 1, grid_size)
        
        # Save as .npy file
        filename = f"data/ndvi/ndvi_2023_{month:02d}.npy"
        np.save(filename, ndvi_data)
        logger.info(f"Generated sample NDVI data for month {month}")

def generate_sample_futures_data():
    """Generate sample futures data for testing."""
    # Generate dates for 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(weeks=i) for i in range(52)]
    
    # Generate random futures prices
    base_price = 500  # Base price in USD
    prices = [base_price + np.random.normal(0, 10) for _ in range(52)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    
    # Save to CSV
    df.to_csv('data/futures/corn_futures_weekly.csv', index=False)
    logger.info("Generated sample futures data")

def modify_prepare_tuning_data():
    """Modify prepare_tuning_data.py to handle .npy files."""
    # This function would modify the prepare_tuning_data.py file
    # to handle .npy files instead of .tif files
    pass

def main():
    """Main function to set up the test environment."""
    logger.info("Starting environment setup...")
    
    # Create directory structure
    create_directory_structure()
    
    # Generate sample data
    generate_sample_ndvi_data()
    generate_sample_futures_data()
    
    # Modify prepare_tuning_data.py
    modify_prepare_tuning_data()
    
    logger.info("Environment setup completed successfully!")

if __name__ == "__main__":
    main() 
