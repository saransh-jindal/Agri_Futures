import os
import sys
import signal
import logging
import subprocess
import time
from pathlib import Path
import psutil
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DashboardManager:
    def __init__(self, dashboard_port=8050, update_interval=3600):
        """Initialize the dashboard manager.
        
        Args:
            dashboard_port: Port for the dashboard server
            update_interval: Time interval between updates in seconds
        """
        self.dashboard_port = dashboard_port
        self.update_interval = update_interval
        self.dashboard_process = None
        self.updater_process = None
        self.is_running = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def start(self):
        """Start both the dashboard and updater."""
        try:
            logger.info("Starting dashboard system...")
            
            # Create necessary directories
            self._setup_directories()
            
            # Start the updater
            self._start_updater()
            
            # Wait for updater to initialize
            time.sleep(2)
            
            # Start the dashboard
            self._start_dashboard()
            
            self.is_running = True
            logger.info("Dashboard system started successfully")
            
            # Monitor processes
            self._monitor_processes()
            
        except Exception as e:
            logger.error(f"Error starting dashboard system: {str(e)}")
            self.stop()
            sys.exit(1)
    
    def stop(self):
        """Stop both the dashboard and updater."""
        logger.info("Stopping dashboard system...")
        
        try:
            # Stop dashboard
            if self.dashboard_process:
                self._stop_process(self.dashboard_process)
                self.dashboard_process = None
            
            # Stop updater
            if self.updater_process:
                self._stop_process(self.updater_process)
                self.updater_process = None
            
            self.is_running = False
            logger.info("Dashboard system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping dashboard system: {str(e)}")
    
    def _setup_directories(self):
        """Set up necessary directories."""
        directories = [
            'predictions/latest',
            'predictions/latest/ndvi',
            'experiments',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _start_dashboard(self):
        """Start the dashboard server."""
        try:
            logger.info("Starting dashboard server...")
            
            # Start dashboard process
            self.dashboard_process = subprocess.Popen(
                [sys.executable, 'src/modeling/dashboard.py'],
                env={**os.environ, 'DASHBOARD_PORT': str(self.dashboard_port)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for dashboard to start
            time.sleep(5)
            
            if self.dashboard_process.poll() is not None:
                raise Exception("Dashboard failed to start")
            
            logger.info(f"Dashboard server started on port {self.dashboard_port}")
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {str(e)}")
            raise
    
    def _start_updater(self):
        """Start the dashboard updater."""
        try:
            logger.info("Starting dashboard updater...")
            
            # Start updater process
            self.updater_process = subprocess.Popen(
                [sys.executable, 'src/modeling/update_dashboard.py'],
                env={**os.environ, 'UPDATE_INTERVAL': str(self.update_interval)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for updater to start
            time.sleep(2)
            
            if self.updater_process.poll() is not None:
                raise Exception("Updater failed to start")
            
            logger.info("Dashboard updater started successfully")
            
        except Exception as e:
            logger.error(f"Error starting updater: {str(e)}")
            raise
    
    def _stop_process(self, process):
        """Stop a subprocess gracefully.
        
        Args:
            process: Subprocess to stop
        """
        try:
            # Get process and all its children
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)
            
            # Stop children first
            for child in children:
                child.terminate()
            
            # Stop parent
            parent.terminate()
            
            # Wait for processes to terminate
            gone, alive = psutil.wait_procs([parent] + children, timeout=3)
            
            # Force kill if still alive
            for p in alive:
                p.kill()
            
        except Exception as e:
            logger.error(f"Error stopping process: {str(e)}")
    
    def _monitor_processes(self):
        """Monitor dashboard and updater processes."""
        while self.is_running:
            try:
                # Check dashboard
                if self.dashboard_process and self.dashboard_process.poll() is not None:
                    logger.error("Dashboard process died unexpectedly")
                    self._restart_dashboard()
                
                # Check updater
                if self.updater_process and self.updater_process.poll() is not None:
                    logger.error("Updater process died unexpectedly")
                    self._restart_updater()
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error monitoring processes: {str(e)}")
    
    def _restart_dashboard(self):
        """Restart the dashboard server."""
        logger.info("Restarting dashboard server...")
        self._stop_process(self.dashboard_process)
        self._start_dashboard()
    
    def _restart_updater(self):
        """Restart the dashboard updater."""
        logger.info("Restarting dashboard updater...")
        self._stop_process(self.updater_process)
        self._start_updater()
    
    def _handle_signal(self, signum, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the dashboard system')
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port for the dashboard server (default: 8050)'
    )
    parser.add_argument(
        '--update-interval',
        type=int,
        default=3600,
        help='Update interval in seconds (default: 3600)'
    )
    return parser.parse_args()

def main():
    """Main function to run the dashboard system."""
    # Parse arguments
    args = parse_args()
    
    # Create and start dashboard manager
    manager = DashboardManager(
        dashboard_port=args.port,
        update_interval=args.update_interval
    )
    
    try:
        manager.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        manager.stop()
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        manager.stop()
        sys.exit(1)

if __name__ == "__main__":
    main() 
