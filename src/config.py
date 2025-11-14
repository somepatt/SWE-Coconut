# src/config.py

from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from pathlib import Path
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    
    use_quantization: bool = True
    quantization: dict = Field(default_factory=dict) 
    
    use_lora: bool = True
    lora: dict = Field(default_factory=dict)

class DataConfig(BaseModel):
    dataset_name: str
    split: str = "train"
    max_seq_length: int = 1024
    max_prompt_length: int = 512
    batch_size: int = 2
    num_workers: int = 4
    gradient_accumulation_steps: int = 1
    cache_dir: str = "./data_cache"

class CoconutTrainingConfig(BaseModel):
    num_stages: int = 3
    epoch_per_stage: int = 1
    
    continuous_thought_steps: int = 4
    warmup_steps: int = 100
    num_training_steps: int = 5000
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    
    uniform_prob: float = 0.0
    pad_latent_to_max: bool = False

class OptimizerConfig(BaseModel):
    name: str = "adamw"
    lr: float = 5e-5
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 100

class MonitoringConfig(BaseModel):
    use_wandb: bool = True
    wandb_project: str = "coconut-code"
    wandb_entity: Optional[str] = None
    log_model: bool = False

class TrainingConfig(BaseModel):
    project_name: str
    experiment_name: str
    output_dir: str = "./outputs"
    seed: int = 42
    
    model: ModelConfig
    data: DataConfig
    training: CoconutTrainingConfig
    optimizer: OptimizerConfig
    monitoring: MonitoringConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingConfig":
        """Load config from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Load config from dictionary"""
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str):
        """Save config to YAML file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """Load configuration from YAML or use defaults"""
    if config_path is None:
        config_path = "config/default.yaml" 
    
    return TrainingConfig.from_yaml(config_path)