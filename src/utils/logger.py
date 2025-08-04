"""
ArcheryAI Pro - Logging Utility
Centralized logging configuration for the application
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import os

def setup_logger(name: str = "ArcheryAI_Pro", level: str = "INFO") -> logging.Logger:
    """
    Setup and configure the application logger.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler for detailed logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"archery_ai_pro_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = logging.FileHandler(
        log_dir / f"archery_ai_pro_errors_{timestamp}.log"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'ArcheryAI_Pro')
    
    return logging.getLogger(name)

def log_performance_metrics(logger: logging.Logger, metrics: dict, video_name: str = ""):
    """
    Log performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Performance metrics dictionary
        video_name: Name of the analyzed video
    """
    logger.info(f"=== Performance Metrics for {video_name} ===")
    
    # Log overall scores
    if 'overall_performance' in metrics:
        logger.info(f"Overall Performance: {metrics['overall_performance']:.3f}")
    
    if 'accuracy_score' in metrics:
        logger.info(f"Accuracy Score: {metrics['accuracy_score']:.3f}")
    
    if 'consistency_score' in metrics:
        logger.info(f"Consistency Score: {metrics['consistency_score']:.3f}")
    
    if 'efficiency_score' in metrics:
        logger.info(f"Efficiency Score: {metrics['efficiency_score']:.3f}")
    
    if 'stability_score' in metrics:
        logger.info(f"Stability Score: {metrics['stability_score']:.3f}")
    
    # Log improvement areas
    if 'improvement_areas' in metrics and metrics['improvement_areas']:
        logger.info("Areas for Improvement:")
        for area in metrics['improvement_areas']:
            logger.info(f"  - {area}")
    
    logger.info("=" * 50)

def log_analysis_summary(logger: logging.Logger, summary: dict):
    """
    Log analysis summary in a structured format.
    
    Args:
        logger: Logger instance
        summary: Analysis summary dictionary
    """
    logger.info("=== Analysis Summary ===")
    
    for key, value in summary.items():
        if isinstance(value, float):
            logger.info(f"{key.replace('_', ' ').title()}: {value:.3f}")
        else:
            logger.info(f"{key.replace('_', ' ').title()}: {value}")
    
    logger.info("=" * 30) 