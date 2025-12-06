# src/trainer.py
import os
import torch
import time
from itertools import islice
from pathlib import Path
from typing import Optional, Dict
from loguru import logger
import torch.nn.functional as F
from tqdm import tqdm

class CoconutTrainer:
    """COCONUT trainer with multi-stage training for latent reasoning"""
    
    def __init__(self, model, tokenizer, config, optimizer_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer_manager = optimizer_manager
        self.logger = logger
        
        self.global_step = 0
        self.best_loss = float('inf')
        
    def train_stage(self, stage: int, dataloader):
        """Train single COCONUT stage"""
        self.logger.info(f"=" * 80)
        self.logger.info(f"Starting Stage {stage}/{self.config.training.num_stages}")
        self.logger.info(f"Latent replacement ratio: {stage / self.config.training.num_stages:.1%}")
        self.logger.info(f"=" * 80)
        
        stage_loss = 0.0
        stage_examples = 0
        
        device = next(self.model.parameters()).device

        for epoch in range(self.config.training.epoch_per_stage):
            self.logger.info(f"Epoch {epoch}/{self.config.training.epoch_per_stage} at stage {stage}")
            
            epoch_loss = 0.0
            epoch_examples = 0
            
            # dataloader = islice(dataloader, 9000)
            pbar = tqdm(dataloader, desc=f"Stage {stage} Epoch {epoch}")
            
            for batch_idx, batch in enumerate(pbar):
                self.global_step += 1
                step_start_time = time.time()
                
                # Move batch to device
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                try:
                    loss = self._compute_loss(batch, stage)
                    
                    # Check for NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.warning(f"NaN/Inf loss detected: {loss.item()}, skipping")
                        self.optimizer_manager.optimizer.zero_grad()
                        torch.cuda.empty_cache()
                        continue
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % self.config.data.gradient_accumulation_steps == 0:
                        grad_norm = self.optimizer_manager.step()
                        if grad_norm > 500:
                            logger.warning(f"Dropping step due to huge grad_norm: {grad_norm}")
                            self.optimizer_manager.optimizer.zero_grad()
                            continue
                        
                        batch_loss_value = float(loss.detach().item())
                        epoch_loss += batch_loss_value
                        epoch_examples += 1
                        stage_loss += batch_loss_value
                        stage_examples += 1
                        
                        # Logging
                        if self.global_step % self.config.training.logging_steps == 0:
                            lr = self.optimizer_manager.get_lr()
                            step_time = time.time() - step_start_time
                            throughput = self.config.data.batch_size / step_time
                            
                            self.logger.info(
                                f"Step {self.global_step} | "
                                f"Loss: {batch_loss_value:.4f} | "
                                f"Grad Norm: {grad_norm:.4f} | "
                                f"LR: {lr:.2e} | "
                                f"Throughput: {throughput:.1f} ex/s"
                            )
                            
                            # W&B logging
                            if self.config.monitoring.use_wandb:
                                self._log_to_wandb({
                                    "loss": batch_loss_value,
                                    "grad_norm": grad_norm,
                                    "learning_rate": lr,
                                    "throughput": throughput,
                                    "stage": stage,
                                })
                        
                        # Checkpoint
                        if self.global_step % self.config.training.save_steps == 0:
                            self._save_checkpoint(stage, self.global_step)
                        
                        # Memory cleanup
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    self.logger.error(f"Error in training step {self.global_step}: {e}")
                    self.optimizer_manager.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                
                # Update progress bar
                if epoch_examples > 0:
                    pbar.set_postfix({'loss': epoch_loss / epoch_examples})
        
        # Stage summary
        avg_loss = stage_loss / max(stage_examples, 1)
        self.logger.info(
            f"Stage {stage} completed. "
            f"Average loss: {avg_loss:.4f}, "
            f"Total examples: {stage_examples}"
        )
    
    def _compute_loss(self, batch: Dict, stage: int) -> torch.Tensor:
        """Compute COCONUT loss with stage-aware latent reasoning"""
        
        # 1. Получаем 'labels' из батча, а не 'loss_mask'
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels'] 
        position_ids = batch.get('position_ids', None) # .get для безопасности

        # ✅ 2. Делаем forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids, 
        )
        
        logits = outputs['logits']
        
        # ✅ 3. Сдвигаем logits и labels для "next token prediction"
        # Logits: (batch, seq_len, vocab_size) -> (batch, seq_len-1, vocab_size)
        # Labels: (batch, seq_len) -> (batch, seq_len-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous() # Используем 'labels', а не 'input_ids'
        
        # ✅ 4. Считаем loss
        # F.cross_entropy *автоматически* проигнорирует все токены, 
        # где shift_labels == -100 (это стандартное поведение).
        # Вся логика с 'shift_mask' больше не нужна.
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100 # Явно указываем (хотя это и так default)
        )
        
        # Проверяем, что loss не NaN (на всякий случай)
        if torch.isnan(loss) or torch.isinf(loss):
            self.logger.warning("NaN/Inf loss detected after F.cross_entropy")
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        # ✅ 5. Нормализуем loss
        loss = loss / self.config.data.gradient_accumulation_steps
        
        return loss
        
    def _save_checkpoint(self, stage: int, step: int):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / f"stage_{stage}" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 🔹 1. Разворачиваем DDP, если он есть
            model_to_save = self.model.module if hasattr(self.model, "module") else self.model

            # 🔹 2. Сохраняем HF-модель (внутри CoconutModel)
            model_to_save.model.save_pretrained(str(checkpoint_dir))
            
            # 🔹 3. Токенайзер
            self.tokenizer.save_pretrained(str(checkpoint_dir))
            
            # 🔹 4. Конфиг
            self.config.to_yaml(str(checkpoint_dir / "config.yaml"))
            
            self.logger.info(f"Checkpoint saved: {checkpoint_dir}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
    
    def _log_to_wandb(self, metrics: Dict):
        """Log metrics to Weights & Biases"""
        try:
            import wandb
            wandb.log(metrics)
        except ImportError:
            pass
        except Exception as e:
            self.logger.warning(f"W&B logging error: {e}")
