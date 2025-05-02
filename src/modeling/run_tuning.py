import os
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json

from .hyperparameter_tuning import optimize_hyperparameters
from .prepare_tuning_data import prepare_tuning_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_prepared_data(data_dir):
    """
    Load preprocessed data for tuning.
    
    Args:
        data_dir: Directory containing preprocessed data
    
    Returns:
        tuple: (X_ndvi, X_futures, y, scaler)
    """
    logger.info("Loading preprocessed data...")
    
    X_ndvi = np.load(os.path.join(data_dir, 'X_ndvi.npy'))
    X_futures = np.load(os.path.join(data_dir, 'X_futures.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    scaler = joblib.load(os.path.join(data_dir, 'scaler.joblib'))
    
    logger.info(f"Loaded data shapes:")
    logger.info(f"X_ndvi: {X_ndvi.shape}")
    logger.info(f"X_futures: {X_futures.shape}")
    logger.info(f"y: {y.shape}")
    
    return X_ndvi, X_futures, y, scaler

def analyze_data(X_ndvi, X_futures, y, save_dir):
    """
    Analyze and visualize the prepared data.
    
    Args:
        X_ndvi: NDVI data
        X_futures: Futures data
        y: Target values
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Analyze NDVI data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(X_ndvi.flatten(), bins=50)
    plt.title('NDVI Value Distribution')
    plt.xlabel('NDVI Value')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.imshow(X_ndvi[0, 0], cmap='RdYlGn')
    plt.colorbar()
    plt.title('Sample NDVI Image')
    plt.savefig(os.path.join(save_dir, 'ndvi_analysis.png'))
    plt.close()
    
    # Analyze futures data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(X_futures[:100, 0])
    plt.title('Sample Futures Returns')
    plt.xlabel('Time Step')
    plt.ylabel('Return')
    
    plt.subplot(1, 2, 2)
    plt.hist(y, bins=50)
    plt.title('Target Distribution')
    plt.xlabel('Future Return')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'futures_analysis.png'))
    plt.close()
    
    # Correlation analysis
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(X_futures.T, y)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(os.path.join(save_dir, 'correlation_matrix.png'))
    plt.close()

def run_tuning_experiment(
    data_dir,
    model_type='hybrid',
    n_trials=100,
    n_splits=5,
    study_name=None
):
    """
    Run hyperparameter tuning experiment.
    
    Args:
        data_dir: Directory containing preprocessed data
        model_type: Type of model ('hybrid' or 'futures')
        n_trials: Number of optimization trials
        n_splits: Number of cross-validation splits
        study_name: Name for the Optuna study
    
    Returns:
        dict: Best hyperparameters and optimization results
    """
    # Create experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = os.path.join('experiments', f'{model_type}_tuning_{timestamp}')
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Load data
    X_ndvi, X_futures, y, scaler = load_prepared_data(data_dir)
    
    # Analyze data
    analyze_data(X_ndvi, X_futures, y, os.path.join(experiment_dir, 'data_analysis'))
    
    # Split data into train and test sets
    train_size = int(0.8 * len(X_ndvi))
    X_ndvi_train, X_ndvi_test = X_ndvi[:train_size], X_ndvi[train_size:]
    X_futures_train, X_futures_test = X_futures[:train_size], X_futures[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Run optimization
    logger.info(f"Starting {model_type} model optimization...")
    best_params = optimize_hyperparameters(
        X_ndvi_train, X_futures_train, y_train,
        model_type=model_type,
        n_trials=n_trials,
        n_splits=n_splits,
        study_name=study_name
    )
    
    # Save results
    results = {
        'best_params': best_params,
        'data_shapes': {
            'X_ndvi': X_ndvi.shape,
            'X_futures': X_futures.shape,
            'y': y.shape
        },
        'train_test_split': {
            'train_size': train_size,
            'test_size': len(X_ndvi) - train_size
        },
        'optimization_config': {
            'n_trials': n_trials,
            'n_splits': n_splits,
            'model_type': model_type
        },
        'timestamp': timestamp
    }
    
    with open(os.path.join(experiment_dir, 'experiment_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Optimization completed. Results saved to {experiment_dir}")
    return results

if __name__ == "__main__":
    # Example usage
    data_dir = "data/processed"
    
    try:
        # Run hybrid model optimization
        logger.info("Starting hybrid model optimization...")
        hybrid_results = run_tuning_experiment(
            data_dir,
            model_type='hybrid',
            n_trials=50,
            n_splits=5
        )
        logger.info("Hybrid model optimization completed")
        logger.info("Best parameters:")
        logger.info(hybrid_results['best_params'])
        
        # Run futures-only model optimization
        logger.info("\nStarting futures-only model optimization...")
        futures_results = run_tuning_experiment(
            data_dir,
            model_type='futures',
            n_trials=50,
            n_splits=5
        )
        logger.info("Futures-only model optimization completed")
        logger.info("Best parameters:")
        logger.info(futures_results['best_params'])
        
    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}")
        raise 
