import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional
from loguru import logger

class CoconutModel(nn.Module):
    """
    COCONUT model wrapper for code generation with latent reasoning.
    
    Extends base LLM with:
    - Latent thought embeddings between language tokens
    - Multi-stage training (replacing language CoT with latent)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logger
        
        self.model = self._load_model()
        
        if config.training.latent_dim > 0:
            self._add_latent_modules()
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load base model with quantization and LoRA"""
        self.logger.info(f"Loading model: {self.config.model.name}")
        
        # Quantization config
        bnb_config = None
        if self.config.model.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=self.config.model.quantization['load_in_4bit'],
                bnb_4bit_quant_type=self.config.model.quantization['bnb_4bit_quant_type'],
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=self.config.model.quantization['bnb_4bit_use_double_quant'],
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        self.logger.info(f"Model loaded: {model.config.model_type}")
        
        # Prepare for LoRA
        if self.config.model.use_quantization:
            model = prepare_model_for_kbit_training(model)
        
        # Add LoRA
        if self.config.model.use_lora:
            lora_config = LoraConfig(
                r=self.config.model.lora['r'],
                lora_alpha=self.config.model.lora['lora_alpha'],
                target_modules=self.config.model.lora['target_modules'],
                lora_dropout=self.config.model.lora['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            self.logger.info("LoRA adapters added")
        
        return model
    
    def _add_latent_modules(self):
        """Add latent reasoning modules"""
        hidden_size = self.model.config.hidden_size
        latent_dim = self.config.training.latent_dim
        
        # Latent embedding projection
        self.latent_proj = nn.Linear(hidden_size, latent_dim)
        self.latent_unproj = nn.Linear(latent_dim, hidden_size)
        
        self.logger.info(f"Added latent modules: {hidden_size} -> {latent_dim} -> {hidden_size}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: int = 0,
    ):
        """Forward pass with optional latent reasoning"""
        
        # Standard forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        
        return outputs
    
    def get_trainable_params(self):
        """Get trainable parameters"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total params: {total:,}, Trainable: {trainable:,}")
        return trainable, total


def load_model_and_tokenizer(config):
    """Load model and tokenizer"""
    logger.info("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = CoconutModel(config)
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer
