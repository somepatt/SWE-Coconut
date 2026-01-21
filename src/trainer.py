# src/trainer.py
import os
import torch
import time
from itertools import islice
from pathlib import Path
from typing import Optional, Dict, List
from loguru import logger
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

class CoconutTrainer:
    """COCONUT trainer with multi-stage training for latent reasoning"""
    
    def __init__(self, model, tokenizer, config, optimizer_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.optimizer_manager = optimizer_manager
        self.logger = logger
        self.simcot_enabled = getattr(self.config, "simcot", None) and self.config.simcot.enabled
        
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

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        position_ids = batch.get('position_ids', None)

        # Prepare SIM-CoT step IDs for forward pass (DDP-compatible)
        simcot_step_ids = None
        if self.simcot_enabled and batch.get("steps_tokenized"):
            simcot_step_ids = self._prepare_simcot_step_ids(
                batch["steps_tokenized"],
                num_latents=self._count_latents(input_ids),
            )

        # Forward pass with SIM-CoT computed inside model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_latent_embeds=bool(self.simcot_enabled),
            simcot_step_ids=simcot_step_ids,
        )

        logits = outputs['logits']

        # Next token prediction loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        if torch.isnan(lm_loss) or torch.isinf(lm_loss):
            self.logger.warning("NaN/Inf loss detected after F.cross_entropy")
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        total_loss = lm_loss * getattr(self.config.simcot, "lambda_lm", 1.0)

        # Add SIM-CoT loss if computed in forward
        if 'simcot_loss' in outputs and outputs['simcot_loss'] is not None:
            simcot_loss = outputs['simcot_loss']
            total_loss = total_loss + simcot_loss * getattr(
                self.config.simcot, "lambda_step", 1.0
            )

        total_loss = total_loss / self.config.data.gradient_accumulation_steps

        return total_loss

    def _count_latents(self, input_ids: torch.Tensor) -> int:
        """Count max number of latent tokens in batch"""
        model_inner = self.model.module if hasattr(self.model, "module") else self.model
        latent_token_id = model_inner.latent_token_id
        if latent_token_id is None:
            return 0
        counts = (input_ids == latent_token_id).sum(dim=1)
        return int(counts.max().item())

    def _prepare_simcot_step_ids(
        self,
        steps_tokenized,
        num_latents: int
    ) -> list:
        """
        Prepare step IDs for SIM-CoT loss, bucketed by latent tokens.
        Returns: List[batch][num_latents] of token id lists
        """
        batch_size = len(steps_tokenized)
        result = []
        for batch_idx in range(batch_size):
            buckets = self._bucket_steps(steps_tokenized[batch_idx], num_latents)
            result.append(buckets)
        return result

    def _bucket_steps(self, steps_tokenized, num_latents: int):
        """
        Распределяет steps по buckets для latent токенов.

        ✅ ИСПРАВЛЕНО: Обрабатывает случай когда steps < latents
        используя циклическое повторение шагов.
        """
        if num_latents <= 0:
            return []
        if not steps_tokenized:
            return [[] for _ in range(num_latents)]

        num_steps = len(steps_tokenized)
        buckets = []

        # ✅ Если шагов меньше чем latents - циклически повторяем шаги
        if num_steps < num_latents:
            for idx in range(num_latents):
                step_idx = idx % num_steps  # Циклическое повторение
                merged: List[int] = list(steps_tokenized[step_idx])
                if self.config.data.max_step_tokens:
                    merged = merged[: self.config.data.max_step_tokens]
                buckets.append(merged)
        else:
            # Обычный bucketing: распределяем шаги по buckets
            for idx in range(num_latents):
                start = int(idx * num_steps / num_latents)
                end = int((idx + 1) * num_steps / num_latents)
                merged: List[int] = []
                for step in steps_tokenized[start:end]:
                    merged.extend(step)
                if self.config.data.max_step_tokens:
                    merged = merged[: self.config.data.max_step_tokens]
                buckets.append(merged)
        return buckets

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
