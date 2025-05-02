import os
import json
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading
from queue import Queue
import schedule

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DashboardUpdater:
    def __init__(self, predictions_dir, experiments_dir, update_interval=3600):
        """Initialize the dashboard updater.
        
        Args:
            predictions_dir: Directory containing prediction results
            experiments_dir: Directory containing experiment results
            update_interval: Time interval between updates in seconds (default: 1 hour)
        """
        self.predictions_dir = Path(predictions_dir)
        self.experiments_dir = Path(experiments_dir)
        self.update_interval = update_interval
        self.update_queue = Queue()
        self.latest_predictions = None
        self.latest_metrics = None
        
        # Create necessary directories
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
    
    def start(self):
        """Start the dashboard updater."""
        logger.info("Starting dashboard updater...")
        
        # Start file system observer
        self.observer = Observer()
        self.observer.schedule(
            PredictionFileHandler(self.update_queue),
            str(self.predictions_dir),
            recursive=False
        )
        self.observer.start()
        
        # Start update scheduler
        schedule.every(self.update_interval).seconds.do(self.check_for_updates)
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_updates)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the dashboard updater."""
        logger.info("Stopping dashboard updater...")
        self.observer.stop()
        self.observer.join()
        self.processing_thread.join()
    
    def _process_updates(self):
        """Process updates from the queue."""
        while True:
            try:
                event = self.update_queue.get()
                if event is None:
                    break
                
                self._handle_update(event)
                self.update_queue.task_done()
            except Exception as e:
                logger.error(f"Error processing update: {str(e)}")
    
    def _handle_update(self, event):
        """Handle a file system update event.
        
        Args:
            event: FileSystemEvent object
        """
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix not in ['.csv', '.json', '.npy']:
            return
        
        logger.info(f"Processing update: {file_path}")
        
        try:
            if file_path.suffix == '.csv':
                self._update_predictions(file_path)
            elif file_path.suffix == '.json':
                self._update_metrics(file_path)
            elif file_path.suffix == '.npy':
                self._update_ndvi_data(file_path)
        except Exception as e:
            logger.error(f"Error handling update {file_path}: {str(e)}")
    
    def _update_predictions(self, file_path):
        """Update predictions data.
        
        Args:
            file_path: Path to predictions CSV file
        """
        try:
            # Read new predictions
            new_predictions = pd.read_csv(file_path)
            new_predictions['date'] = pd.to_datetime(new_predictions['date'])
            
            # Update or create predictions file
            predictions_file = self.predictions_dir / 'futures_data.csv'
            if predictions_file.exists():
                current_predictions = pd.read_csv(predictions_file)
                current_predictions['date'] = pd.to_datetime(current_predictions['date'])
                
                # Merge new predictions with existing data
                updated_predictions = pd.concat([
                    current_predictions,
                    new_predictions[~new_predictions['date'].isin(current_predictions['date'])]
                ]).sort_values('date')
            else:
                updated_predictions = new_predictions
            
            # Save updated predictions
            updated_predictions.to_csv(predictions_file, index=False)
            logger.info(f"Updated predictions file with {len(new_predictions)} new records")
            
            # Update latest predictions
            self.latest_predictions = updated_predictions
            
        except Exception as e:
            logger.error(f"Error updating predictions: {str(e)}")
    
    def _update_metrics(self, file_path):
        """Update model metrics.
        
        Args:
            file_path: Path to metrics JSON file
        """
        try:
            # Read new metrics
            with open(file_path, 'r') as f:
                new_metrics = json.load(f)
            
            # Update metrics file
            metrics_file = self.predictions_dir / 'model_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    current_metrics = json.load(f)
                
                # Update metrics
                current_metrics.update(new_metrics)
                updated_metrics = current_metrics
            else:
                updated_metrics = new_metrics
            
            # Save updated metrics
            with open(metrics_file, 'w') as f:
                json.dump(updated_metrics, f, indent=4)
            logger.info("Updated model metrics")
            
            # Update latest metrics
            self.latest_metrics = updated_metrics
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def _update_ndvi_data(self, file_path):
        """Update NDVI data.
        
        Args:
            file_path: Path to NDVI data file
        """
        try:
            # Create NDVI directory if it doesn't exist
            ndvi_dir = self.predictions_dir / 'ndvi'
            ndvi_dir.mkdir(exist_ok=True)
            
            # Copy new NDVI file
            shutil.copy2(file_path, ndvi_dir / file_path.name)
            logger.info(f"Updated NDVI data: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error updating NDVI data: {str(e)}")
    
    def check_for_updates(self):
        """Check for new predictions and update dashboard."""
        logger.info("Checking for updates...")
        
        # Check for new predictions in experiments directory
        for exp_dir in self.experiments_dir.glob('*_tuning_*'):
            try:
                # Check for new predictions
                pred_file = exp_dir / 'predictions.csv'
                if pred_file.exists():
                    self._update_predictions(pred_file)
                
                # Check for new metrics
                metrics_file = exp_dir / 'experiment_results.json'
                if metrics_file.exists():
                    self._update_metrics(metrics_file)
                
                # Check for new NDVI data
                ndvi_file = exp_dir / 'ndvi_data.npy'
                if ndvi_file.exists():
                    self._update_ndvi_data(ndvi_file)
                
            except Exception as e:
                logger.error(f"Error checking experiment directory {exp_dir}: {str(e)}")

class PredictionFileHandler(FileSystemEventHandler):
    def __init__(self, update_queue):
        """Initialize the file handler.
        
        Args:
            update_queue: Queue for update events
        """
        self.update_queue = update_queue
    
    def on_created(self, event):
        """Handle file creation event."""
        self.update_queue.put(event)
    
    def on_modified(self, event):
        """Handle file modification event."""
        self.update_queue.put(event)

def main():
    """Main function to run the dashboard updater."""
    # Set up directories
    predictions_dir = 'predictions/latest'
    experiments_dir = 'experiments'
    
    # Create and start dashboard updater
    updater = DashboardUpdater(predictions_dir, experiments_dir)
    updater.start()

if __name__ == "__main__":
    main() 
