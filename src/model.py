# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Tuple, Dict
from loguru import logger

class LatentReasoningModule(nn.Module):
    """
    Latent reasoning module for COCONUT.
    Converts language tokens to continuous latent thoughts and back.
    """
    
    def __init__(self, hidden_size: int, latent_dim: int, num_steps: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_steps = num_steps
        
        # Encode language hidden states to latent space
        self.to_latent = nn.Sequential(
            nn.Linear(hidden_size, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )
        
        # Decode latent thoughts back to hidden space
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, hidden_size),
        )
        
        # Latent reasoning RNN (iterate over latent space)
        self.latent_rnn = nn.GRUCell(latent_dim, latent_dim)
    
    def encode_to_latent(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Convert hidden states to latent thoughts"""
        return self.to_latent(hidden_states)
    
    def decode_from_latent(self, latent_states: torch.Tensor) -> torch.Tensor:
        """Convert latent thoughts back to hidden states"""
        return self.from_latent(latent_states)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_reasoning_steps: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform latent reasoning.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            num_reasoning_steps: Number of latent reasoning iterations
            
        Returns:
            decoded_states: [batch, seq_len, hidden_size]
            latent_trajectory: [batch, seq_len, num_steps, latent_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Encode to latent space
        latent = self.encode_to_latent(hidden_states)  # [batch, seq_len, latent_dim]
        
        # Store trajectory for logging/analysis
        latent_trajectory = latent.unsqueeze(2).repeat(1, 1, num_reasoning_steps, 1)
        
        # Reasoning iterations in latent space
        for step in range(num_reasoning_steps):
            # Apply RNN to update latent thoughts
            latent_updated = self.latent_rnn(
                latent.view(-1, self.latent_dim),
                latent.view(-1, self.latent_dim)
            )
            latent = latent_updated.view(batch_size, seq_len, self.latent_dim)
            
            # Store step
            latent_trajectory[:, :, step, :] = latent
        
        # Decode back to hidden space
        decoded_states = self.decode_from_latent(latent)  # [batch, seq_len, hidden_size]
        
        return decoded_states, latent_trajectory


class CoconutModel(nn.Module):
    """
    COCONUT model with proper stage-dependent latent reasoning.
    
    Stage 0: Pure language tokens (no latent)
    Stage 1: 33% latent replacement
    Stage 2: 66% latent replacement
    Stage 3: 100% latent replacement (full reasoning in latent space)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logger
        
        # Load base model
        self.model = self._load_model()
        
        # Add latent reasoning if configured
        if config.training.latent_dim > 0:
            self.latent_module = LatentReasoningModule(
                hidden_size=self.model.config.hidden_size,
                latent_dim=config.training.latent_dim,
                num_steps=config.training.continuous_thought_steps,
            )
    
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
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            device_map=self.config.model.device_map,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
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
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: int = 0,
    ) -> Dict:
        """
        Forward pass with stage-dependent latent reasoning.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            labels: [batch, seq_len]
            stage: Training stage (0-3)
                - 0: Language only
                - 1: 33% latent
                - 2: 66% latent
                - 3: 100% latent
        """
        
        # Get hidden states from base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        logits = outputs.logits
        
        if stage > 0 and self.config.training.latent_dim > 0:
            # Коэффициент замены: на какую долю токенов применить latent reasoning
            latent_ratio = stage / self.config.training.num_stages
            
            self.logger.debug(
                f"Stage {stage}: Applying latent reasoning to {latent_ratio:.1%} of tokens"
            )
            
            # Применить латентное рассуждение
            latent_hidden_states, latent_trajectory = self.latent_module(
                hidden_states,
                num_reasoning_steps=self.config.training.continuous_thought_steps,
            )
            
            # Blend между языковыми токенами и латентными мыслями
            hidden_states = (
                (1 - latent_ratio) * hidden_states +  # Language tokens
                latent_ratio * latent_hidden_states     # Latent reasoning
            )
            
            # Пересчитать logits с обновленными hidden states
            logits = self.model.lm_head(hidden_states)
        else:
            latent_trajectory = None
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view(shift_logits.shape[0], -1)
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
            'latent_trajectory': latent_trajectory,
            'stage': stage,
        }
    
    def get_trainable_params(self):
        """Get trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
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
