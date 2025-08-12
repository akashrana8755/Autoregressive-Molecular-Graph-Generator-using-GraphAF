"""
Optimization utilities for molecular generation models.

This module provides advanced optimization techniques including gradient clipping,
learning rate scheduling, and optimization strategies specifically designed
for training molecular generation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
import math
import warnings


class GradientClipper:
    """
    Advanced gradient clipping utilities.
    
    Provides various gradient clipping strategies to improve training stability
    for molecular generation models.
    """
    
    def __init__(self, 
                 clip_type: str = 'norm',
                 clip_value: float = 1.0,
                 adaptive: bool = False,
                 percentile: float = 95.0):
        """
        Initialize gradient clipper.
        
        Args:
            clip_type: Type of clipping ('norm', 'value', 'adaptive')
            clip_value: Clipping threshold
            adaptive: Whether to use adaptive clipping
            percentile: Percentile for adaptive clipping
        """
        self.clip_type = clip_type
        self.clip_value = clip_value
        self.adaptive = adaptive
        self.percentile = percentile
        
        # For adaptive clipping
        self.gradient_history = []
        self.history_size = 100
        
    def clip_gradients(self, model: nn.Module) -> Dict[str, float]:
        """
        Clip gradients of the model.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Dictionary with clipping statistics
        """
        # Collect gradients
        gradients = []
        for param in model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
                
        if not gradients:
            return {'grad_norm': 0.0, 'clipped': False}
            
        all_gradients = torch.cat(gradients)
        grad_norm = all_gradients.norm().item()
        
        # Update gradient history for adaptive clipping
        if self.adaptive:
            self.gradient_history.append(grad_norm)
            if len(self.gradient_history) > self.history_size:
                self.gradient_history.pop(0)
                
            # Compute adaptive threshold
            if len(self.gradient_history) >= 10:
                threshold = np.percentile(self.gradient_history, self.percentile)
                clip_value = max(threshold, self.clip_value)
            else:
                clip_value = self.clip_value
        else:
            clip_value = self.clip_value
            
        # Apply clipping
        clipped = False
        
        if self.clip_type == 'norm':
            if grad_norm > clip_value:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                clipped = True
                
        elif self.clip_type == 'value':
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
            clipped = grad_norm > clip_value
            
        elif self.clip_type == 'adaptive':
            # Adaptive clipping based on gradient statistics
            if grad_norm > clip_value:
                scale_factor = clip_value / grad_norm
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale_factor)
                clipped = True
                
        return {
            'grad_norm': grad_norm,
            'clip_value': clip_value,
            'clipped': clipped
        }


class WarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup period.
    
    Gradually increases learning rate from 0 to base_lr during warmup,
    then applies the main scheduler.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 main_scheduler: _LRScheduler,
                 warmup_steps: int,
                 warmup_factor: float = 0.1,
                 last_epoch: int = -1):
        """
        Initialize warmup scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            main_scheduler: Main scheduler to use after warmup
            warmup_steps: Number of warmup steps
            warmup_factor: Initial learning rate factor
            last_epoch: Last epoch index
        """
        self.main_scheduler = main_scheduler
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Warmup phase
            alpha = self.last_epoch / self.warmup_steps
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Main scheduler phase
            return self.main_scheduler.get_lr()
            
    def step(self, epoch=None):
        super().step(epoch)
        if self.last_epoch >= self.warmup_steps:
            self.main_scheduler.step(epoch)


class CosineAnnealingWarmRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts scheduler.
    
    Implements the SGDR (Stochastic Gradient Descent with Warm Restarts)
    learning rate schedule with cosine annealing.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 T_0: int,
                 T_mult: int = 1,
                 eta_min: float = 0,
                 last_epoch: int = -1):
        """
        Initialize cosine annealing with warm restarts.
        
        Args:
            optimizer: Optimizer to schedule
            T_0: Number of iterations for the first restart
            T_mult: Factor to increase T_i after each restart
            eta_min: Minimum learning rate
            last_epoch: Last epoch index
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs]
                
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            
        self.T_cur = self.T_cur + 1
        
        if self.T_cur >= self.T_i:
            # Restart
            self.T_cur = 0
            self.T_i = self.T_i * self.T_mult
            
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class AdaptiveLRScheduler(_LRScheduler):
    """
    Adaptive learning rate scheduler based on loss plateaus.
    
    Reduces learning rate when loss stops improving, with additional
    features for molecular generation training.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 mode: str = 'min',
                 factor: float = 0.5,
                 patience: int = 10,
                 threshold: float = 1e-4,
                 threshold_mode: str = 'rel',
                 cooldown: int = 0,
                 min_lr: float = 0,
                 eps: float = 1e-8,
                 last_epoch: int = -1):
        """
        Initialize adaptive learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            mode: 'min' or 'max' for loss monitoring
            factor: Factor to reduce learning rate
            patience: Number of epochs to wait before reducing
            threshold: Threshold for measuring improvement
            threshold_mode: 'rel' or 'abs' threshold mode
            cooldown: Number of epochs to wait after reduction
            min_lr: Minimum learning rate
            eps: Small value to avoid division by zero
            last_epoch: Last epoch index
        """
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        
        self.best = None
        self.num_bad_epochs = 0
        self.mode_worse = None
        self.cooldown_counter = 0
        
        if mode == 'min':
            self.mode_worse = float('inf')
        else:
            self.mode_worse = -float('inf')
            
        super().__init__(optimizer, last_epoch)
        
    def step(self, metrics, epoch=None):
        """
        Step the scheduler with current metrics.
        
        Args:
            metrics: Current loss/metric value
            epoch: Current epoch
        """
        current = float(metrics)
        
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.best is None:
            self.best = current
        elif self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
            
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
            
    def is_better(self, current, best):
        """Check if current metric is better than best."""
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return current < best * rel_epsilon
        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold
        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return current > best * rel_epsilon
        else:  # mode == 'max' and threshold_mode == 'abs':
            return current > best + self.threshold
            
    def _reduce_lr(self, epoch):
        """Reduce learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr


class OptimizerFactory:
    """
    Factory for creating optimizers with molecular generation-specific configurations.
    """
    
    @staticmethod
    def create_optimizer(model: nn.Module, 
                        config: Dict[str, Any]) -> optim.Optimizer:
        """
        Create optimizer from configuration.
        
        Args:
            model: Model to optimize
            config: Optimizer configuration
            
        Returns:
            Configured optimizer
        """
        optimizer_type = config.get('type', 'adam').lower()
        learning_rate = config.get('learning_rate', 1e-4)
        weight_decay = config.get('weight_decay', 1e-5)
        
        # Get model parameters with optional parameter grouping
        param_groups = OptimizerFactory._get_parameter_groups(model, config)
        
        if optimizer_type == 'adam':
            return optim.Adam(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=config.get('betas', (0.9, 0.999)),
                eps=config.get('eps', 1e-8)
            )
            
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=config.get('betas', (0.9, 0.999)),
                eps=config.get('eps', 1e-8)
            )
            
        elif optimizer_type == 'sgd':
            return optim.SGD(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=config.get('momentum', 0.9),
                nesterov=config.get('nesterov', False)
            )
            
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                alpha=config.get('alpha', 0.99),
                eps=config.get('eps', 1e-8)
            )
            
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                eps=config.get('eps', 1e-10)
            )
            
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    @staticmethod
    def _get_parameter_groups(model: nn.Module, 
                             config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create parameter groups with different learning rates.
        
        Args:
            model: Model to get parameters from
            config: Configuration with parameter group settings
            
        Returns:
            List of parameter groups
        """
        param_groups_config = config.get('parameter_groups', {})
        
        if not param_groups_config:
            # Single group with all parameters
            return [{'params': model.parameters()}]
            
        # Create parameter groups
        param_groups = []
        assigned_params = set()
        
        for group_name, group_config in param_groups_config.items():
            group_params = []
            
            # Get parameters matching the group criteria
            for name, param in model.named_parameters():
                if OptimizerFactory._param_matches_group(name, group_config):
                    group_params.append(param)
                    assigned_params.add(id(param))
                    
            if group_params:
                group_dict = {'params': group_params}
                
                # Add group-specific settings
                if 'learning_rate' in group_config:
                    group_dict['lr'] = group_config['learning_rate']
                if 'weight_decay' in group_config:
                    group_dict['weight_decay'] = group_config['weight_decay']
                    
                param_groups.append(group_dict)
                
        # Add remaining parameters to default group
        remaining_params = []
        for param in model.parameters():
            if id(param) not in assigned_params:
                remaining_params.append(param)
                
        if remaining_params:
            param_groups.append({'params': remaining_params})
            
        return param_groups
        
    @staticmethod
    def _param_matches_group(param_name: str, group_config: Dict[str, Any]) -> bool:
        """Check if parameter matches group criteria."""
        include_patterns = group_config.get('include', [])
        exclude_patterns = group_config.get('exclude', [])
        
        # Check include patterns
        if include_patterns:
            if not any(pattern in param_name for pattern in include_patterns):
                return False
                
        # Check exclude patterns
        if exclude_patterns:
            if any(pattern in param_name for pattern in exclude_patterns):
                return False
                
        return True


class SchedulerFactory:
    """
    Factory for creating learning rate schedulers.
    """
    
    @staticmethod
    def create_scheduler(optimizer: optim.Optimizer,
                        config: Dict[str, Any]) -> Optional[_LRScheduler]:
        """
        Create learning rate scheduler from configuration.
        
        Args:
            optimizer: Optimizer to schedule
            config: Scheduler configuration
            
        Returns:
            Configured scheduler or None
        """
        scheduler_type = config.get('type', 'none')
        
        if scheduler_type == 'none' or scheduler_type is None:
            return None
            
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.get('T_max', 100),
                eta_min=config.get('eta_min', 1e-6)
            )
            
        elif scheduler_type == 'cosine_restarts':
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.get('T_0', 10),
                T_mult=config.get('T_mult', 1),
                eta_min=config.get('eta_min', 1e-6)
            )
            
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get('mode', 'min'),
                factor=config.get('factor', 0.5),
                patience=config.get('patience', 10),
                threshold=config.get('threshold', 1e-4),
                min_lr=config.get('min_lr', 1e-6)
            )
            
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get('step_size', 30),
                gamma=config.get('gamma', 0.1)
            )
            
        elif scheduler_type == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=config.get('milestones', [30, 60, 90]),
                gamma=config.get('gamma', 0.1)
            )
            
        elif scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.get('gamma', 0.95)
            )
            
        elif scheduler_type == 'warmup':
            main_scheduler_config = config.get('main_scheduler', {'type': 'cosine'})
            main_scheduler = SchedulerFactory.create_scheduler(optimizer, main_scheduler_config)
            
            return WarmupScheduler(
                optimizer,
                main_scheduler,
                warmup_steps=config.get('warmup_steps', 1000),
                warmup_factor=config.get('warmup_factor', 0.1)
            )
            
        elif scheduler_type == 'adaptive':
            return AdaptiveLRScheduler(
                optimizer,
                mode=config.get('mode', 'min'),
                factor=config.get('factor', 0.5),
                patience=config.get('patience', 10),
                threshold=config.get('threshold', 1e-4),
                min_lr=config.get('min_lr', 1e-6)
            )
            
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def create_optimizer_and_scheduler(model: nn.Module,
                                  config: Dict[str, Any]) -> Tuple[optim.Optimizer, Optional[_LRScheduler]]:
    """
    Convenience function to create both optimizer and scheduler.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary with 'optimizer' and 'scheduler' sections
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer_config = config.get('optimizer', {})
    scheduler_config = config.get('scheduler', {})
    
    optimizer = OptimizerFactory.create_optimizer(model, optimizer_config)
    scheduler = SchedulerFactory.create_scheduler(optimizer, scheduler_config)
    
    return optimizer, scheduler


def get_learning_rate(optimizer: optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: Optimizer to get learning rate from
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer: optim.Optimizer, lr: float):
    """
    Set learning rate for all parameter groups.
    
    Args:
        optimizer: Optimizer to set learning rate for
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def freeze_parameters(model: nn.Module, patterns: List[str]):
    """
    Freeze parameters matching given patterns.
    
    Args:
        model: Model to freeze parameters in
        patterns: List of patterns to match parameter names
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = False


def unfreeze_parameters(model: nn.Module, patterns: List[str]):
    """
    Unfreeze parameters matching given patterns.
    
    Args:
        model: Model to unfreeze parameters in
        patterns: List of patterns to match parameter names
    """
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = True