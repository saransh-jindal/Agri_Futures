import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.integration import TFKerasPruningCallback
import joblib
from datetime import datetime
import json

from .model import create_hybrid_model, create_futures_only_model
from .train import create_training_callbacks, evaluate_predictions

class TimeSeriesCrossValidator:
    """Time series cross-validation with proper data handling."""
    def __init__(self, n_splits=5, test_size=0.15):
        self.n_splits = n_splits
        self.test_size = test_size
        self.tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(test_size * 100))
    
    def split(self, X):
        """Generate train-test splits for time series data."""
        for train_idx, test_idx in self.tscv.split(X):
            yield train_idx, test_idx

def create_hybrid_model_trial(trial, ndvi_shape, futures_shape):
    """
    Create hybrid model with hyperparameters from trial.
    
    Args:
        trial: Optuna trial object
        ndvi_shape: Shape of NDVI input
        futures_shape: Shape of futures input
    
    Returns:
        Model: Hybrid model with trial hyperparameters
    """
    # CNN hyperparameters
    cnn_filters = [
        trial.suggest_int('cnn_filters_1', 16, 64),
        trial.suggest_int('cnn_filters_2', 32, 128),
        trial.suggest_int('cnn_filters_3', 64, 256)
    ]
    cnn_kernel_size = trial.suggest_int('cnn_kernel_size', 3, 5)
    cnn_dropout = trial.suggest_float('cnn_dropout', 0.1, 0.5)
    
    # LSTM hyperparameters
    lstm_units = trial.suggest_int('lstm_units', 32, 256)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    
    # Dense layer hyperparameters
    dense_units = [
        trial.suggest_int('dense_units_1', 32, 256),
        trial.suggest_int('dense_units_2', 16, 128)
    ]
    dense_dropout = trial.suggest_float('dense_dropout', 0.1, 0.5)
    
    # Learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Create model with custom hyperparameters
    model = create_hybrid_model(
        ndvi_shape=ndvi_shape,
        futures_shape=futures_shape,
        cnn_filters=cnn_filters,
        cnn_kernel_size=cnn_kernel_size,
        cnn_dropout=cnn_dropout,
        lstm_units=lstm_units,
        lstm_dropout=lstm_dropout,
        dense_units=dense_units,
        dense_dropout=dense_dropout,
        learning_rate=learning_rate
    )
    
    return model

def create_futures_model_trial(trial, input_shape):
    """
    Create futures-only model with hyperparameters from trial.
    
    Args:
        trial: Optuna trial object
        input_shape: Shape of futures input
    
    Returns:
        Model: Futures-only model with trial hyperparameters
    """
    # LSTM hyperparameters
    lstm_units_1 = trial.suggest_int('lstm_units_1', 32, 256)
    lstm_units_2 = trial.suggest_int('lstm_units_2', 16, 128)
    lstm_dropout = trial.suggest_float('lstm_dropout', 0.1, 0.5)
    
    # Dense layer hyperparameters
    dense_units = trial.suggest_int('dense_units', 16, 128)
    dense_dropout = trial.suggest_float('dense_dropout', 0.1, 0.5)
    
    # Learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Create model with custom hyperparameters
    model = create_futures_only_model(
        input_shape=input_shape,
        lstm_units=[lstm_units_1, lstm_units_2],
        lstm_dropout=lstm_dropout,
        dense_units=dense_units,
        dense_dropout=dense_dropout,
        learning_rate=learning_rate
    )
    
    return model

def objective_hybrid(trial, X_ndvi, X_futures, y, cv, model_dir):
    """
    Objective function for hybrid model hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_ndvi: NDVI data
        X_futures: Futures data
        y: Target values
        cv: Cross-validator
        model_dir: Directory to save models
    
    Returns:
        float: Mean validation score
    """
    # Create model with trial hyperparameters
    model = create_hybrid_model_trial(
        trial,
        ndvi_shape=X_ndvi.shape[1:],
        futures_shape=X_futures.shape[1:]
    )
    
    # Cross-validation scores
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_ndvi)):
        # Split data
        X_ndvi_train, X_ndvi_val = X_ndvi[train_idx], X_ndvi[val_idx]
        X_futures_train, X_futures_val = X_futures[train_idx], X_futures[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create callbacks
        callbacks = create_training_callbacks(
            os.path.join(model_dir, f'fold_{fold}'),
            patience=5
        )
        callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))
        
        # Train model
        history = model.fit(
            [X_ndvi_train, X_futures_train],
            y_train,
            validation_data=([X_ndvi_val, X_futures_val], y_val),
            batch_size=trial.suggest_int('batch_size', 16, 128),
            epochs=50,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict([X_ndvi_val, X_futures_val])
        score = -np.mean((y_val - y_pred) ** 2)  # Negative MSE for maximization
        
        cv_scores.append(score)
    
    return np.mean(cv_scores)

def objective_futures(trial, X, y, cv, model_dir):
    """
    Objective function for futures-only model hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X: Futures data
        y: Target values
        cv: Cross-validator
        model_dir: Directory to save models
    
    Returns:
        float: Mean validation score
    """
    # Create model with trial hyperparameters
    model = create_futures_model_trial(trial, input_shape=X.shape[1:])
    
    # Cross-validation scores
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create callbacks
        callbacks = create_training_callbacks(
            os.path.join(model_dir, f'fold_{fold}'),
            patience=5
        )
        callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))
        
        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=trial.suggest_int('batch_size', 16, 128),
            epochs=50,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred = model.predict(X_val)
        score = -np.mean((y_val - y_pred) ** 2)  # Negative MSE for maximization
        
        cv_scores.append(score)
    
    return np.mean(cv_scores)

def optimize_hyperparameters(
    X_ndvi, X_futures, y,
    model_type='hybrid',
    n_trials=100,
    n_splits=5,
    study_name=None
):
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        X_ndvi: NDVI data
        X_futures: Futures data
        y: Target values
        model_type: Type of model ('hybrid' or 'futures')
        n_trials: Number of optimization trials
        n_splits: Number of cross-validation splits
        study_name: Name for the Optuna study
    
    Returns:
        dict: Best hyperparameters
    """
    # Create study
    if study_name is None:
        study_name = f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Create cross-validator
    cv = TimeSeriesCrossValidator(n_splits=n_splits)
    
    # Create model directory
    model_dir = os.path.join('models', f'{model_type}_optimization')
    os.makedirs(model_dir, exist_ok=True)
    
    # Optimize
    if model_type == 'hybrid':
        study.optimize(
            lambda trial: objective_hybrid(trial, X_ndvi, X_futures, y, cv, model_dir),
            n_trials=n_trials
        )
    else:
        study.optimize(
            lambda trial: objective_futures(trial, X_futures, y, cv, model_dir),
            n_trials=n_trials
        )
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'n_trials': len(study.trials),
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(model_dir, 'optimization_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(os.path.join(model_dir, 'optimization_history.png'))
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(os.path.join(model_dir, 'parameter_importance.png'))
    
    return study.best_params

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Generate dummy data
    n_samples = 1000
    ndvi_shape = (64, 64, 1)
    futures_shape = (8, 1)
    
    X_ndvi = np.random.randn(n_samples, *ndvi_shape)
    X_futures = np.random.randn(n_samples, *futures_shape)
    y = np.random.randn(n_samples)
    
    # Optimize hybrid model
    print("\nOptimizing Hybrid Model...")
    hybrid_params = optimize_hyperparameters(
        X_ndvi, X_futures, y,
        model_type='hybrid',
        n_trials=50
    )
    print("\nBest Hybrid Model Parameters:")
    print(hybrid_params)
    
    # Optimize futures-only model
    print("\nOptimizing Futures-Only Model...")
    futures_params = optimize_hyperparameters(
        X_ndvi, X_futures, y,
        model_type='futures',
        n_trials=50
    )
    print("\nBest Futures-Only Model Parameters:")
    print(futures_params) 
