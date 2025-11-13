from loguru import logger
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    output_dir: str = "./logs",
    level: str = "INFO",
    experiment_name: str = "experiment"
) -> None:
    """Setup loguru logger with file and console output"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<level>{time:HH:mm:ss}</level> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
    )
    
    # File handler
    log_file = Path(output_dir) / f"{experiment_name}.log"
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation="500 MB",
    )
    
    logger.info(f"Logger initialized. Output: {log_file}")
    return logger

def get_logger():
    """Get configured logger"""
    return logger
