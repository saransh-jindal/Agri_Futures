import os
import sys
import logging
from pathlib import Path
import optuna
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from sklearn.model_selection import TimeSeriesSplit
import json
from datetime import datetime

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from modeling.model import create_hybrid_model, create_futures_model
from modeling.train import DataGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    def __init__(self, config_path=None):
        """Initialize the hyperparameter tuner.
        
        Args:
            config_path: Path to configuration file (default: None)
        """
        self.config = self._load_config(config_path)
        self._setup_directories()
        self._setup_gpu()
        
        # Load processed data
        self.X = np.load('data/processed/X_train.npy')
        self.y = np.load('data/processed/y_train.npy')
        
        # Create time series cross-validation splits
        self.tscv = TimeSeriesSplit(n_splits=5)
    
    def _load_config(self, config_path):
        """Load configuration from file or use defaults."""
        default_config = {
            'tuning': {
                'n_trials': 50,
                'timeout': 3600,  # 1 hour
                'study_name': 'hybrid_model_tuning',
                'metric': 'val_loss',
                'direction': 'minimize'
            },
            'model': {
                'sequence_length': 30,
                'batch_size': 32,
                'epochs': 100,
                'patience': 10
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _setup_directories(self):
        """Set up necessary directories."""
        directories = [
            'tuning_results',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _setup_gpu(self):
        """Configure GPU settings."""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s)")
            else:
                logger.info("No GPU found, using CPU")
        except Exception as e:
            logger.warning(f"Error setting up GPU: {str(e)}")
    
    def _objective(self, trial):
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
        
        Returns:
            float: Validation loss
        """
        # Define hyperparameter search space
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'cnn_filters': [
                trial.suggest_int('cnn_filters_1', 16, 128),
                trial.suggest_int('cnn_filters_2', 32, 256),
                trial.suggest_int('cnn_filters_3', 64, 512)
            ],
            'lstm_units': [
                trial.suggest_int('lstm_units_1', 32, 256),
                trial.suggest_int('lstm_units_2', 16, 128)
            ],
            'dense_units': [
                trial.suggest_int('dense_units_1', 16, 128),
                trial.suggest_int('dense_units_2', 8, 64)
            ],
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'l2_reg': trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
        }
        
        # Initialize cross-validation scores
        cv_scores = []
        
        # Perform time series cross-validation
        for train_idx, val_idx in self.tscv.split(self.X):
            # Split data
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # Create data generators
            train_generator = DataGenerator(X_train, y_train, self.config['model']['batch_size'])
            val_generator = DataGenerator(X_val, y_val, self.config['model']['batch_size'])
            
            # Create model
            model = create_hybrid_model(
                sequence_length=self.config['model']['sequence_length'],
                cnn_filters=params['cnn_filters'],
                lstm_units=params['lstm_units'],
                dense_units=params['dense_units'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg']
            )
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                loss='mse',
                metrics=['mae', 'mape']
            )
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['model']['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Train model
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=self.config['model']['epochs'],
                callbacks=callbacks,
                verbose=0
            )
            
            # Get best validation score
            best_val_loss = min(history.history['val_loss'])
            cv_scores.append(best_val_loss)
        
        # Return mean validation score
        return np.mean(cv_scores)
    
    def tune_hyperparameters(self):
        """Run hyperparameter tuning."""
        try:
            logger.info("Starting hyperparameter tuning...")
            
            # Create study
            study = optuna.create_study(
                study_name=self.config['tuning']['study_name'],
                direction=self.config['tuning']['direction']
            )
            
            # Run optimization
            study.optimize(
                self._objective,
                n_trials=self.config['tuning']['n_trials'],
                timeout=self.config['tuning']['timeout']
            )
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = Path('tuning_results') / timestamp
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best parameters
            best_params = study.best_params
            with open(results_dir / 'best_params.json', 'w') as f:
                json.dump(best_params, f, indent=4)
            
            # Save study
            joblib.dump(study, results_dir / 'study.pkl')
            
            # Create summary report
            self._create_summary_report(study, results_dir)
            
            logger.info("Hyperparameter tuning completed successfully")
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning: {str(e)}")
            raise
    
    def _create_summary_report(self, study, results_dir):
        """Create a summary report of tuning results.
        
        Args:
            study: Optuna study object
            results_dir: Directory to save results
        """
        report = f"""Hyperparameter Tuning Summary Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. Best Parameters
----------------
{json.dumps(study.best_params, indent=4)}

2. Best Score
------------
{study.best_value:.4f}

3. Optimization History
---------------------
Number of trials: {len(study.trials)}
Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}
Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}

4. Parameter Importance
---------------------
{optuna.importance.get_param_importances(study)}
"""
        
        with open(results_dir / 'tuning_summary.txt', 'w') as f:
            f.write(report)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    return parser.parse_args()

def main():
    """Main function to run hyperparameter tuning."""
    args = parse_args()
    
    tuner = HyperparameterTuner(config_path=args.config)
    best_params = tuner.tune_hyperparameters()
    
    logger.info(f"Best parameters: {best_params}")

if __name__ == "__main__":
    main() 
