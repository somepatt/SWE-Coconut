import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from typing import Optional, Tuple, Dict, List
from loguru import logger
import torch.nn.parallel as ddp
from src.utils import setup_distributed, get_world_size_and_rank

class CoconutModel(nn.Module):
    """
    COCONUT: Chain of Continuous Thought

    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = logger
        
        # Load base model
        self.model = self._load_model()
        
        # Get embedding layer
        self.embedding = self.model.get_input_embeddings()
        
        # Special token IDs for latent reasoning
        self.latent_token_id = None  # Will be set after tokenizer is loaded
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load base model with quantization and LoRA"""
        self.logger.info(f"Loading model: {self.config.model.name}")
        
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
            device_map=None,
            trust_remote_code=self.config.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        
        self.logger.info(f"Model loaded: {model.config.model_type}")
        
        if self.config.model.use_quantization:
            base_model = prepare_model_for_kbit_training(model)

        model.gradient_checkpointing_enable()
        self.logger.info("Gradient checkpointing enabled (required for COCONUT loop)")
        
        adapters_path = getattr(self.config.model, "resume_from_checkpoint", None)
        
        if self.config.model.use_lora:
            # создаём такую же LoRA-конфигурацию, как при первом запуске
            lora_cfg = self.config.model.lora
            lora_config = LoraConfig(
                r=lora_cfg["r"],
                lora_alpha=lora_cfg["lora_alpha"],
                target_modules=lora_cfg["target_modules"],
                lora_dropout=lora_cfg["lora_dropout"],
                bias="none",
                task_type="CAUSAL_LM",
            )

            model = get_peft_model(base_model, lora_config)
            self.logger.info("LoRA adapters initialized.")

            # 4) если указан чекпоинт — подгружаем веса адаптера вручную
            ckpt_dir = getattr(self.config.model, "resume_from_checkpoint", None)
            if ckpt_dir:
                adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
                self.logger.info(f"Loading LoRA weights from {adapter_path}")

                if not os.path.isfile(adapter_path):
                    self.logger.error(f"adapter_model.safetensors not found at {adapter_path}")
                else:
                    state_dict = load_file(adapter_path)

                    # 🔑 выкидываем эмбеддинги и lm_head, которые конфликтуют по размеру
                    keys_to_drop = [
                        k for k in state_dict.keys()
                        if "embed_tokens" in k or "lm_head" in k
                    ]
                    if keys_to_drop:
                        self.logger.warning(
                            f"Dropping {len(keys_to_drop)} keys from adapter "
                            f"state_dict (embed_tokens / lm_head) due to size mismatch."
                        )
                        for k in keys_to_drop:
                            state_dict.pop(k)

                    missing, unexpected = model.load_state_dict(state_dict, strict=False)
                    self.logger.info(
                        f"LoRA weights loaded. "
                        f"Missing keys: {len(missing)}, unexpected: {len(unexpected)}"
                    )

        return model

    
    def set_latent_token_id(self, token_id: int):
        """Set the special token ID for latent reasoning"""
        self.latent_token_id = token_id
        self.logger.info(f"Latent token ID set to: {token_id}")
    
    def _find_latent_positions(
        self, 
        input_ids: torch.Tensor
    ) -> List[List[int]]:
        """
        Find positions of <latent> tokens in each sequence.
        
        Returns:
            List of lists: [[pos1, pos2, ...], [pos1, ...], ...]
            One list per batch item
        """
        if self.latent_token_id is None:
            return [[] for _ in range(input_ids.shape[0])]
        
        # Find all latent token positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        
        # Group by batch
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]
        
        return latent_lists
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        ✅ COCONUT forward pass with continuous thought.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [batch, seq_len]
            labels: [batch, seq_len]
            stage: Training stage (determines how many latent tokens to use)
        
        Returns:
            Dict with logits, loss, and reasoning trajectory
        """
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(
                0, seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # ✅ Step 1: Get initial embeddings
        inputs_embeds = self.embedding(input_ids)  # [batch, seq_len, hidden_size]
        
        # ✅ Step 2: Find latent token positions
        latent_lists = self._find_latent_positions(input_ids)
        max_n_latents = max([len(l) for l in latent_lists])
        
        if max_n_latents == 0:
            # No latent reasoning, standard forward pass
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            
            logits = outputs.logits
        else:
            
            all_logits = []
            kv_cache = None
            next_compute_range = (0, seq_len)
            
            # Find first latent token position
            if max_n_latents > 0:
                first_latent_pos = min([l[0] for l in latent_lists if len(l) > 0])
                next_compute_range = (0, first_latent_pos)
            
            # ✅ Step 3: Iterative latent reasoning
            for pass_idx in range(max_n_latents):
                # Forward pass for this chunk
                if kv_cache is None:
                    # First pass - no cache
                    outputs = self.model(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0]:next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[
                            :, next_compute_range[0]:next_compute_range[1]
                        ] if attention_mask is not None else None,
                        position_ids=position_ids[
                            :, next_compute_range[0]:next_compute_range[1]
                        ],
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=True,
                    )
                    hidden_states_offset = 0
                else:
                    # Subsequent passes - use KV cache
                    past_key_values = kv_cache

                    outputs = self.model(
                        inputs_embeds=inputs_embeds[
                            :, next_compute_range[0]:next_compute_range[1], :
                        ],
                        attention_mask=attention_mask[:, :next_compute_range[1]]
                            if attention_mask is not None else None,
                        position_ids=position_ids[
                            :, next_compute_range[0]:next_compute_range[1]
                        ],
                        past_key_values=past_key_values,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=True,
                    )
                    hidden_states_offset = next_compute_range[0]
                
                # Collect logits
                all_logits.append(outputs.logits)
                
                # Get hidden states
                hidden_states = outputs.hidden_states[-1]
                kv_cache = outputs.past_key_values
                
                # ✅ Step 4: REPLACE latent token embeddings with hidden states
                
                # Find which latent tokens to replace in this pass
                filling_indices = [
                    (batch_idx, token_pos)
                    for batch_idx, pos_list in enumerate(latent_lists)
                    if len(pos_list) > pass_idx
                    for token_pos in [pos_list[pass_idx]]
                ]
                
                if filling_indices:
                    # Convert inputs_embeds to list for modification
                    # (to avoid in-place operations)
                    inputs_embeds_list = [
                        [inputs_embeds[b, t, :] for t in range(seq_len)]
                        for b in range(batch_size)
                    ]
                    
                    # Replace latent tokens with hidden states
                    for batch_idx, token_idx in filling_indices:
                        # ✅ KEY: Use hidden state from PREVIOUS position
                        inputs_embeds_list[batch_idx][token_idx] = hidden_states[
                            batch_idx, 
                            token_idx - 1 - hidden_states_offset, 
                            :
                        ]
                    
                    # Reassemble inputs_embeds
                    inputs_embeds = torch.stack([
                        torch.stack(inputs_embeds_list[b])
                        for b in range(batch_size)
                    ])
                
                # Update compute range for next iteration
                next_compute_range = (
                    next_compute_range[1],
                    seq_len if pass_idx + 1 >= max_n_latents 
                    else next_compute_range[1] + 1
                )
            
            # ✅ Step 5: Final pass with all latent tokens replaced
            past_key_values = kv_cache
            
            outputs = self.model(
                inputs_embeds=inputs_embeds[
                    :, next_compute_range[0]:next_compute_range[1], :
                ],
                attention_mask=attention_mask[:, :next_compute_range[1]]
                    if attention_mask is not None else None,
                position_ids=position_ids[
                    :, next_compute_range[0]:next_compute_range[1]
                ],
                past_key_values=past_key_values,
                output_hidden_states=True,
                return_dict=True,
            )
            
            all_logits.append(outputs.logits)
            
            # Concatenate all logits
            logits = torch.cat(all_logits, dim=1)  # [batch, seq_len, vocab_size]
        
        
        return {
            'logits': logits,
            # 'inputs_embeds': inputs_embeds,
            # 'hidden_states': outputs.hidden_states[-1] if outputs.hidden_states else None,
        }
    
    def get_trainable_params(self):
        """Get trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        self.logger.info(f"Total params: {total:,}, Trainable: {trainable:,}")
        return trainable, total


def load_model_and_tokenizer(config):
    """Load model and tokenizer with latent token setup"""
    logger.info("Loading model and tokenizer...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    
    # Add special tokens for latent reasoning
    special_tokens = {
        'additional_special_tokens': ['<bot>', '<eot>', '<thought>']
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    logger.info(f"Added {num_added} special tokens: {special_tokens.keys()}")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = CoconutModel(config)
    
    # Resize embeddings if new tokens were added
    if num_added > 0:
        model.model.resize_token_embeddings(len(tokenizer))
        model.embedding = model.model.get_input_embeddings()

    device, rank, world_size = setup_distributed()
    model = model.to(device)

    if world_size > 1:
        # NOTE: find_unused_parameters=True может быть полезен для COCONUT, 
        # но может замедлить работу. Начни без него.
        model = ddp.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True 
        )
        model.logger.info(f"Model wrapped in DDP on GPU {rank}.")
    
    # Set latent token ID
    latent_token_id = tokenizer.convert_tokens_to_ids('<thought>')
    model_to_configure = model.module if world_size > 1 else model
    model_to_configure.set_latent_token_id(latent_token_id)
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer
