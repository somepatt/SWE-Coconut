import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from loguru import logger

class OptimizerManager:
    """Manage optimizer and learning rate scheduler"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.logger = logger
        
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer"""
        self.logger.info("Creating AdamW optimizer")
        
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_params,
            lr=self.config.optimizer.lr,
            betas=tuple(self.config.optimizer.betas),
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay,
        )
        
        self.logger.info(
            f"Optimizer created: lr={self.config.optimizer.lr}, "
            f"weight_decay={self.config.optimizer.weight_decay}"
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        self.logger.info("Creating learning rate scheduler")
        
        # Linear warmup
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=self.config.optimizer.warmup_steps,
        )
        
        # Linear decay after warmup
        decay_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=0.0,
            total_iters=self.config.training.num_training_steps - self.config.optimizer.warmup_steps,
        )
        
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[self.config.optimizer.warmup_steps],
        )
        
        return scheduler
    
    def step(self):
        """Perform optimization step"""
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.optimizer.max_grad_norm,
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return float(grad_norm)
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
