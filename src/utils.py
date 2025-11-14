import os
import random
import numpy as np
import torch
from pathlib import Path
from loguru import logger
import torch.distributed as dist

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Random seed set to {seed}")

def setup_directories(config):
    """Create necessary directories"""
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.output_dir) / "checkpoints"
    logger.info(f"Output directory: {config.output_dir}")

def get_device():
    """Get device (cuda if available)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    return device

def get_world_size_and_rank():
    """Получает размер мира (кол-во GPU) и ранг (индекс текущего GPU)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0 # Если DDP не запущен, считаем, что работаем на 1 GPU

def setup_distributed():
    """Инициализирует Distributed Process Group, если это необходимо."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Только инициализируем DDP, если он еще не инициализирован
        if not dist.is_initialized():
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            logger.info(f"DDP initialized: Rank {rank}/{world_size}. Backend: NCCL.")
        
        # Выбираем устройство (GPU) по рангу
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(rank)
        return device, rank, world_size
    else:
        logger.info("DDP not detected. Running on single device.")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        return device, 0, 1 # device, rank, world_size

def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
