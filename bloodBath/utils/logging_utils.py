"""
Logging utilities for the bloodBath sync client
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str = 'bloodBath',
                level: int = logging.INFO,
                log_file: Optional[Path] = None,
                console_output: bool = True) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'bloodBath') -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


def set_log_level(level: int):
    """Set the log level for all bloodBath loggers"""
    logging.getLogger('bloodBath').setLevel(level)
    for handler in logging.getLogger('bloodBath').handlers:
        handler.setLevel(level)
