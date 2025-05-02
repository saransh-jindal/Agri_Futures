#!/usr/bin/env python3

import os
import sys
import logging
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from modeling.data import DataProcessor
from modeling.model import create_hybrid_model, create_futures_model
from modeling.train import train_models
from modeling.predict import make_predictions
from modeling.evaluate import ModelEvaluator
from modeling.dashboard import Dashboard
from modeling.scheduler import PipelineScheduler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Class to run the complete ML pipeline."""
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline runner.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_directories()
        self._setup_gpu()
        
        # Initialize components
        self.data_processor = DataProcessor(
            ndvi_dir=self.config['data']['ndvi_dir'],
            futures_file=self.config['data']['futures_file'],
            sequence_length=self.config['data']['sequence_length']
        )
        
        self.evaluator = ModelEvaluator(
            predictions_dir='predictions/latest',
            output_dir='evaluation_results'
        )
        
        self.dashboard = Dashboard(
            predictions_dir='predictions/latest',
            port=self.config.get('dashboard', {}).get('port', 8050)
        )
        
        self.scheduler = PipelineScheduler(config_path)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def _setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            'data/ndvi',
            'data/futures',
            'experiments',
            'predictions/latest',
            'predictions/latest/ndvi',
            'evaluation_results',
            'logs'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def _setup_gpu(self) -> None:
        """Configure GPU settings."""
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Found {len(gpus)} GPU(s) and configured memory growth")
            else:
                logger.warning("No GPU found, using CPU")
        except Exception as e:
            logger.warning(f"Error configuring GPU: {str(e)}")
    
    def run_pipeline(self) -> None:
        """Run the complete ML pipeline."""
        try:
            # 1. Data Processing
            logger.info("Starting data processing...")
            X_futures, X_ndvi, y = self.data_processor.load_data()
            logger.info("Data processing completed")
            
            # 2. Model Training
            logger.info("Starting model training...")
            train_models(
                X_futures=X_futures,
                X_ndvi=X_ndvi,
                y=y,
                config=self.config['model']
            )
            logger.info("Model training completed")
            
            # 3. Generate Predictions
            logger.info("Generating predictions...")
            predictions = make_predictions(
                experiment_dir='experiments/latest',
                horizon=self.config['prediction']['horizon'],
                confidence_interval=self.config['prediction']['confidence_interval']
            )
            logger.info("Predictions generated")
            
            # 4. Model Evaluation
            logger.info("Starting model evaluation...")
            metrics = self.evaluator.evaluate_predictions()
            self.evaluator.create_visualizations()
            logger.info("Model evaluation completed")
            
            # 5. Start Dashboard
            logger.info("Starting dashboard...")
            self.dashboard.run_server()
            
            # 6. Start Scheduler
            logger.info("Starting scheduler...")
            self.scheduler.start()
            
            logger.info("Pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.scheduler.stop()
            logger.info("Scheduler stopped")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description='Run ML Pipeline')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['run', 'schedule'],
                      default='run', help='Pipeline mode')
    
    args = parser.parse_args()
    
    try:
        runner = PipelineRunner(args.config)
        
        if args.mode == 'run':
            runner.run_pipeline()
        else:
            runner.scheduler.start()
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        runner.cleanup()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 
