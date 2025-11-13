"""
Main training script for COCONUT code LLM
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import load_config
from logger import setup_logger
from model import load_model_and_tokenizer
from data import get_dataloader
from optimizer import OptimizerManager
from trainer import CoconutTrainer
from utils import set_seed, setup_directories, get_device

def main():
    # Load config
    config = load_config("config/default.yaml")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Setup
    setup_logger(config.output_dir, experiment_name=config.experiment_name)
    setup_directories(config)
    set_seed(config.seed)
    device = get_device()
    
    logger = __import__('loguru').logger
    logger.info(f"Project: {config.project_name}")
    logger.info(f"Experiment: {config.experiment_name}")
    # logger.info(f"Config: {config.from_dict()}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    model = model.to(device)
    
    # Create optimizer
    optimizer_manager = OptimizerManager(model, config)
    
    # Create trainer
    trainer = CoconutTrainer(model, tokenizer, config, optimizer_manager)
    
    # COCONUT multi-stage training
    logger.info(f"Starting COCONUT training with {config.training.num_stages + 1} stages")
    
    for stage in range(config.training.num_stages + 1):
        logger.info(f"Loading data for stage {stage}")
        dataloader = get_dataloader(config, tokenizer, stage)
        
        trainer.train_stage(stage, dataloader)
    
    logger.info("Training completed!")
    
    # Save final model
    final_dir = Path(config.output_dir) / "final_model"
    model.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Final model saved to {final_dir}")

if __name__ == "__main__":
    main()
