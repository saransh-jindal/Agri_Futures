#!/usr/bin/env python3

import os
import logging
import json
import time
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import signal
import sys
from .train import ModelTrainer
from .predict import Predictor
from .dashboard import Dashboard

logger = logging.getLogger(__name__)

class Scheduler:
    def __init__(self, config):
        """Initialize the scheduler with configuration."""
        self.config = config
        self.trainer = ModelTrainer(config)
        self.predictor = Predictor(config)
        self.dashboard = Dashboard(config)
        
    def load_config(self):
        """Load scheduler configuration."""
        try:
            with open(self.config['scheduler_config'], 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading scheduler config: {str(e)}")
            raise
            
    def train_job(self):
        """Execute training job."""
        try:
            logger.info("Starting training job")
            
            # Prepare data
            train_data, val_data, test_data = self.trainer.prepare_data()
            
            # Train model
            history = self.trainer.train(train_data, val_data)
            
            # Evaluate model
            results = self.trainer.evaluate(self.trainer.model, test_data)
            
            logger.info(f"Training job completed. Accuracy: {results['accuracy']:.2%}")
            
        except Exception as e:
            logger.error(f"Error in training job: {str(e)}")
            raise
            
    def predict_job(self):
        """Execute prediction job."""
        try:
            logger.info("Starting prediction job")
            
            # Make predictions
            results = self.predictor.predict_latest()
            
            # Save predictions
            os.makedirs(self.config['predictions_dir'], exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            with open(
                os.path.join(self.config['predictions_dir'], f'predictions_{timestamp}.json'),
                'w'
            ) as f:
                json.dump(results, f)
                
            logger.info("Prediction job completed")
            
        except Exception as e:
            logger.error(f"Error in prediction job: {str(e)}")
            raise
            
    def setup_schedule(self):
        """Set up the schedule based on configuration."""
        try:
            scheduler_config = self.load_config()
            
            # Daily execution
            if scheduler_config['daily_execution']['enabled']:
                schedule.every().day.at(
                    scheduler_config['daily_execution']['time']
                ).do(self.train_job)
                
            # Weekly execution
            if scheduler_config['weekly_execution']['enabled']:
                schedule.every().week.at(
                    scheduler_config['weekly_execution']['time']
                ).do(self.train_job)
                
            # Monthly execution
            if scheduler_config['monthly_execution']['enabled']:
                schedule.every().month.at(
                    scheduler_config['monthly_execution']['time']
                ).do(self.train_job)
                
            # Regular prediction updates
            schedule.every(
                scheduler_config['check_interval']
            ).minutes.do(self.predict_job)
            
        except Exception as e:
            logger.error(f"Error setting up schedule: {str(e)}")
            raise
            
    def run(self):
        """Run the scheduler."""
        try:
            # Set up schedule
            self.setup_schedule()
            
            # Start dashboard in a separate thread
            dashboard_thread = threading.Thread(
                target=self.dashboard.run,
                daemon=True
            )
            dashboard_thread.start()
            
            # Run scheduler
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error running scheduler: {str(e)}")
            raise

def main():
    """Main function to run the scheduler."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create scheduler
    scheduler = Scheduler(
        config={
            'scheduler_config': 'config/scheduler.json',
            'predictions_dir': 'predictions'
        }
    )
    
    try:
        # Start scheduler
        scheduler.run()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        raise
    except Exception as e:
        logger.error(f"Error in scheduler: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
