
import logging
from datetime import datetime
from pathlib import Path

# def setup_logging():
#     """Configure logging settings"""
#     # Create logs directory if it doesn't exist
#     Path("logs").mkdir(exist_ok=True)
    
#     # Set up file handler with timestamp
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_file = f"logs/experiment_{timestamp}.log"
    
#     # Configure logging format and settings
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         handlers=[
#             logging.FileHandler(log_file),
#             logging.StreamHandler()  # Also print to console
#         ]
#     )
    
#     return logging.getLogger(__name__)



def setup_logging(run_name):
    """Configure logging settings with run-specific directory"""
    # Create logs directory for this run
    log_dir = Path(f"logs/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file handler with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"experiment_{timestamp}.log"
    
    # Configure logging format and settings
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)