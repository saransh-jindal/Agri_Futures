import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_directory_creation():
    """Test creating directories."""
    logger.info("Testing directory creation...")
    
    # Test directory
    test_dir = "test_dir"
    
    try:
        # Create directory
        Path(test_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created test directory: {test_dir}")
        
        # Verify directory exists
        if os.path.exists(test_dir):
            logger.info("Directory creation successful!")
        else:
            logger.error("Directory creation failed!")
            
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")
        raise

if __name__ == "__main__":
    test_directory_creation() 
