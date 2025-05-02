#!/usr/bin/env python3

import os
import logging
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple
from datetime import datetime, timedelta
from .data import DataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, config):
        """Initialize the predictor with configuration."""
        self.config = config
        self.data_processor = DataProcessor(config)
        self.model = None
        
    def load_model(self):
        """Load the trained model."""
        try:
            model_path = os.path.join(self.config['model_dir'], 'best_model.h5')
            self.model = tf.keras.models.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def prepare_input(self, ndvi_data, futures_data):
        """Prepare input data for prediction."""
        try:
            # Create sequences
            ndvi_sequences = self.data_processor.create_sequences(
                ndvi_data, self.config['sequence_length']
            )
            futures_sequences = self.data_processor.create_sequences(
                futures_data, self.config['sequence_length']
            )
            
            return ndvi_sequences, futures_sequences
            
        except Exception as e:
            logger.error(f"Error preparing input: {str(e)}")
            raise
            
    def predict(self, ndvi_sequences, futures_sequences):
        """Make predictions."""
        try:
            if self.model is None:
                self.load_model()
                
            # Get predictions
            predictions = self.model.predict(
                [ndvi_sequences, futures_sequences],
                batch_size=self.config['batch_size']
            )
            
            # Calculate confidence intervals
            confidence = np.abs(predictions - 0.5) * 2  # Scale to [0, 1]
            
            return {
                'predictions': predictions,
                'confidence': confidence,
                'direction': np.where(predictions > 0.5, 'up', 'down')
            }
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
            
    def predict_latest(self):
        """Make predictions using the latest data."""
        try:
            # Load latest data
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
            
            # Prepare input
            ndvi_sequences, futures_sequences = self.prepare_input(
                ndvi_data, futures_data
            )
            
            # Make predictions
            results = self.predict(ndvi_sequences, futures_sequences)
            
            # Add dates
            results['dates'] = dates[self.config['sequence_length']:]
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting latest: {str(e)}")
            raise

def make_predictions(
    experiment_dir: str,
    horizon: int = 7,
    confidence_interval: float = 0.95
) -> Dict[str, Any]:
    """
    Generate predictions using trained models.
    
    Args:
        experiment_dir: Directory containing trained models
        horizon: Prediction horizon in days
        confidence_interval: Confidence interval for predictions
        
    Returns:
        Dictionary containing predictions and confidence intervals
    """
    try:
        # Load models
        hybrid_model = tf.keras.models.load_model(os.path.join(experiment_dir, 'hybrid_model.h5'))
        futures_model = tf.keras.models.load_model(os.path.join(experiment_dir, 'futures_model.h5'))
        
        # Load configuration
        with open(os.path.join(experiment_dir, 'config.json'), 'r') as f:
            import json
            config = json.load(f)
        
        # Load latest data
        data_processor = DataProcessor(
            ndvi_dir=config['data']['ndvi_dir'],
            futures_file=config['data']['futures_file'],
            sequence_length=config['data']['sequence_length']
        )
        
        # Load and process data
        X_futures, X_ndvi, _ = data_processor.load_data()
        
        # Get latest sequences
        latest_futures = X_futures[-1:]
        latest_ndvi = X_ndvi[-1:]
        
        # Generate predictions
        hybrid_preds = []
        futures_preds = []
        
        for _ in range(horizon):
            # Hybrid model prediction
            hybrid_pred = hybrid_model.predict([latest_futures, latest_ndvi])
            hybrid_preds.append(hybrid_pred[0, 0])
            
            # Futures model prediction
            futures_pred = futures_model.predict(latest_futures)
            futures_preds.append(futures_pred[0, 0])
            
            # Update sequences for next prediction
            latest_futures = np.roll(latest_futures, -1, axis=1)
            latest_futures[0, -1] = hybrid_pred[0, 0]  # Use hybrid model prediction
            
            latest_ndvi = np.roll(latest_ndvi, -1, axis=1)
            latest_ndvi[0, -1] = latest_ndvi[0, -2]  # Use last NDVI value
        
        # Inverse transform predictions
        hybrid_preds = data_processor.inverse_transform_predictions(np.array(hybrid_preds))
        futures_preds = data_processor.inverse_transform_predictions(np.array(futures_preds))
        
        # Calculate confidence intervals
        z_score = 1.96  # 95% confidence interval
        hybrid_std = np.std(hybrid_preds)
        futures_std = np.std(futures_preds)
        
        hybrid_ci = z_score * hybrid_std
        futures_ci = z_score * futures_std
        
        # Generate dates
        last_date = datetime.now()
        dates = [last_date + timedelta(days=i+1) for i in range(horizon)]
        
        # Create results dictionary
        results = {
            'dates': [d.strftime('%Y-%m-%d') for d in dates],
            'hybrid': {
                'predictions': hybrid_preds.tolist(),
                'confidence_intervals': {
                    'lower': (hybrid_preds - hybrid_ci).tolist(),
                    'upper': (hybrid_preds + hybrid_ci).tolist()
                }
            },
            'futures': {
                'predictions': futures_preds.tolist(),
                'confidence_intervals': {
                    'lower': (futures_preds - futures_ci).tolist(),
                    'upper': (futures_preds + futures_ci).tolist()
                }
            }
        }
        
        # Save predictions
        output_dir = os.path.join('predictions', 'latest')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save as CSV
        import pandas as pd
        df = pd.DataFrame({
            'date': results['dates'],
            'hybrid_prediction': results['hybrid']['predictions'],
            'hybrid_lower_ci': results['hybrid']['confidence_intervals']['lower'],
            'hybrid_upper_ci': results['hybrid']['confidence_intervals']['upper'],
            'futures_prediction': results['futures']['predictions'],
            'futures_lower_ci': results['futures']['confidence_intervals']['lower'],
            'futures_upper_ci': results['futures']['confidence_intervals']['upper']
        })
        df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
        
        logger.info(f"Generated predictions for {horizon} days ahead")
        return results
        
    except Exception as e:
        logger.error(f"Error generating predictions: {str(e)}")
        raise

def load_latest_predictions() -> Dict[str, Any]:
    """
    Load latest predictions from disk.
    
    Returns:
        Dictionary containing predictions
    """
    try:
        predictions_file = os.path.join('predictions', 'latest', 'predictions.json')
        with open(predictions_file, 'r') as f:
            import json
            predictions = json.load(f)
        
        logger.info("Loaded latest predictions")
        return predictions
        
    except Exception as e:
        logger.error(f"Error loading predictions: {str(e)}")
        raise 
