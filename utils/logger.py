import logging
import os
from datetime import datetime
from pathlib import Path

class IDSLogger:
    """Centralized logging system for the IDS"""
    
    def __init__(self, name="IDS", log_level=logging.INFO):
        self.name = name
        self.log_level = log_level
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with file and console handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = log_dir / f"ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)

# Global logger instance
ids_logger = IDSLogger()