#!/usr/bin/env python3

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from modeling.model import create_hybrid_model, create_futures_model
from modeling.data import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, config):
        """Initialize the model trainer with configuration."""
        self.config = config
        self.data_processor = DataProcessor(config)
        
    def prepare_data(self):
        """Prepare training data."""
        try:
            # Load data
            futures_data, futures_dates = self.data_processor.load_futures_data(
                self.config['futures_file']
            )
            ndvi_data, ndvi_dates = self.data_processor.load_ndvi_data(
                self.config['ndvi_dir']
            )
            
            # Align data
            ndvi_data, futures_data, dates = self.data_processor.align_data(
                ndvi_data, futures_data, ndvi_dates, futures_dates
            )
            
            # Create sequences
            ndvi_sequences = self.data_processor.create_sequences(
                ndvi_data, self.config['sequence_length']
            )
            futures_sequences = self.data_processor.create_sequences(
                futures_data, self.config['sequence_length']
            )
            
            # Create target variable (next day's price movement)
            target = np.where(
                futures_data[self.config['sequence_length']:, 0] > 
                futures_data[self.config['sequence_length']-1:-1, 0],
                1, 0
            )
            
            # Split data
            train_size = int(len(target) * self.config['train_split'])
            val_size = int(len(target) * self.config['val_split'])
            
            train_data = {
                'ndvi': ndvi_sequences[:train_size],
                'futures': futures_sequences[:train_size],
                'target': target[:train_size]
            }
            
            val_data = {
                'ndvi': ndvi_sequences[train_size:train_size+val_size],
                'futures': futures_sequences[train_size:train_size+val_size],
                'target': target[train_size:train_size+val_size]
            }
            
            test_data = {
                'ndvi': ndvi_sequences[train_size+val_size:],
                'futures': futures_sequences[train_size+val_size:],
                'target': target[train_size+val_size:]
            }
            
            return train_data, val_data, test_data
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def train(self, train_data, val_data):
        """Train the model."""
        try:
            # Create model
            model = create_hybrid_model(self.config)
            
            # Create callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True
                ),
                ModelCheckpoint(
                    filepath=os.path.join(self.config['model_dir'], 'best_model.h5'),
                    monitor='val_loss',
                    save_best_only=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config['patience']//2,
                    min_lr=self.config['min_lr']
                )
            ]
            
            # Train model
            history = model.fit(
                [train_data['ndvi'], train_data['futures']],
                train_data['target'],
                validation_data=(
                    [val_data['ndvi'], val_data['futures']],
                    val_data['target']
                ),
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Save model
            model.save(os.path.join(self.config['model_dir'], 'final_model.h5'))
            
            # Save scalers
            self.data_processor.save_scalers(self.config['model_dir'])
            
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def evaluate(self, model, test_data):
        """Evaluate the model."""
        try:
            # Evaluate model
            results = model.evaluate(
                [test_data['ndvi'], test_data['futures']],
                test_data['target'],
                batch_size=self.config['batch_size'],
                verbose=1
            )
            
            # Get predictions
            predictions = model.predict(
                [test_data['ndvi'], test_data['futures']],
                batch_size=self.config['batch_size']
            )
            
            return {
                'loss': results[0],
                'accuracy': results[1],
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

def train_models(
    X_futures: np.ndarray,
    X_ndvi: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """
    Train hybrid and futures models.
    
    Args:
        X_futures: Futures data array
        X_ndvi: NDVI data array
        y: Target values
        config: Training configuration
        
    Returns:
        Tuple of (hybrid_model, futures_model)
    """
    try:
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join('experiments', f'experiment_{timestamp}')
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Split data
        train_idx = int(len(X_futures) * config['train_split'])
        val_idx = int(len(X_futures) * (config['train_split'] + config['val_split']))
        
        X_futures_train = X_futures[:train_idx]
        X_futures_val = X_futures[train_idx:val_idx]
        X_futures_test = X_futures[val_idx:]
        
        X_ndvi_train = X_ndvi[:train_idx]
        X_ndvi_val = X_ndvi[train_idx:val_idx]
        X_ndvi_test = X_ndvi[val_idx:]
        
        y_train = y[:train_idx]
        y_val = y[train_idx:val_idx]
        y_test = y[val_idx:]
        
        # Create models
        hybrid_model = create_hybrid_model(
            sequence_length=config['sequence_length'],
            cnn_filters=config['cnn_filters'],
            lstm_units=config['lstm_units'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            l2_reg=config['l2_reg']
        )
        
        futures_model = create_futures_model(
            sequence_length=config['sequence_length'],
            lstm_units=config['lstm_units'],
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            l2_reg=config['l2_reg']
        )
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=config['patience'],
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(experiment_dir, 'hybrid_model.h5'),
                monitor='val_loss',
                save_best_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(experiment_dir, 'logs')
            )
        ]
        
        # Train hybrid model
        logger.info("Training hybrid model...")
        hybrid_history = hybrid_model.fit(
            [X_futures_train, X_ndvi_train],
            y_train,
            validation_data=([X_futures_val, X_ndvi_val], y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Train futures model
        logger.info("Training futures model...")
        futures_history = futures_model.fit(
            X_futures_train,
            y_train,
            validation_data=(X_futures_val, y_val),
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save models
        hybrid_model.save(os.path.join(experiment_dir, 'hybrid_model.h5'))
        futures_model.save(os.path.join(experiment_dir, 'futures_model.h5'))
        
        # Save training history
        np.save(os.path.join(experiment_dir, 'hybrid_history.npy'), hybrid_history.history)
        np.save(os.path.join(experiment_dir, 'futures_history.npy'), futures_history.history)
        
        # Save configuration
        with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
            import json
            json.dump(config, f, indent=4)
        
        logger.info(f"Training completed. Results saved in {experiment_dir}")
        
        return hybrid_model, futures_model
        
    except Exception as e:
        logger.error(f"Error training models: {str(e)}")
        raise

def evaluate_models(
    hybrid_model: tf.keras.Model,
    futures_model: tf.keras.Model,
    X_futures_test: np.ndarray,
    X_ndvi_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate trained models.
    
    Args:
        hybrid_model: Trained hybrid model
        futures_model: Trained futures model
        X_futures_test: Test futures data
        X_ndvi_test: Test NDVI data
        y_test: Test target values
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        # Evaluate hybrid model
        hybrid_metrics = hybrid_model.evaluate(
            [X_futures_test, X_ndvi_test],
            y_test,
            verbose=0
        )
        
        # Evaluate futures model
        futures_metrics = futures_model.evaluate(
            X_futures_test,
            y_test,
            verbose=0
        )
        
        # Get predictions
        hybrid_preds = hybrid_model.predict([X_futures_test, X_ndvi_test])
        futures_preds = futures_model.predict(X_futures_test)
        
        # Calculate additional metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'hybrid': {
                'mse': mean_squared_error(y_test, hybrid_preds),
                'mae': mean_absolute_error(y_test, hybrid_preds),
                'r2': r2_score(y_test, hybrid_preds)
            },
            'futures': {
                'mse': mean_squared_error(y_test, futures_preds),
                'mae': mean_absolute_error(y_test, futures_preds),
                'r2': r2_score(y_test, futures_preds)
            }
        }
        
        logger.info("Model evaluation completed")
        return metrics
        
    except Exception as e:
        logger.error(f"Error evaluating models: {str(e)}")
        raise 
