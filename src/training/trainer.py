"""
Main training orchestrator for molecular generation models.

This module provides a unified training interface that supports both
GraphDiffusion and GraphAF models with configurable training loops,
proper data splits, and comprehensive experiment management.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch


from ..models.base_model import BaseGenerativeModel
from ..models.graph_diffusion import GraphDiffusion
from ..models.graph_af import GraphAF
from ..data.molecular_dataset import MolecularDataset, create_molecular_dataloader
from .config_manager import ExperimentConfig
from .experiment_logger import ExperimentLogger
from .losses import DiffusionLoss, FlowLoss
from .optimization import GradientClipper, OptimizerFactory, SchedulerFactory

logger = logging.getLogger(__name__)


class Trainer:
    """
    Main training orchestrator for molecular generation models.
    
    This trainer provides a unified interface for training both GraphDiffusion
    and GraphAF models with configurable training loops, data management,
    and experiment tracking.
    """
    
    def __init__(self,
                 config: Union[Dict[str, Any], ExperimentConfig],
                 output_dir: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration (dict or ExperimentConfig)
            output_dir: Directory to save experiment results (overrides config)
            device: Device to train on (auto-detected if None)
        """
        # Handle config types
        if isinstance(config, ExperimentConfig):
            self.config = config
            config_dict = {
                'name': config.name,
                'model': config.model.__dict__,
                'training': config.training.__dict__,
                'data': config.data.__dict__,
                'logging': config.logging.__dict__
            }
        else:
            self.config = config
            config_dict = config
            
        # Set output directory
        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(config_dict.get('output_dir', 'experiments'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Extract configuration sections
        if isinstance(self.config, ExperimentConfig):
            self.model_config = self.config.model.__dict__
            self.training_config = self.config.training.__dict__
            self.data_config = self.config.data.__dict__
            self.logging_config = self.config.logging.__dict__
        else:
            self.model_config = config.get('model', {})
            self.training_config = config.get('training', {})
            self.data_config = config.get('data', {})
            self.logging_config = config.get('logging', {})
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.loss_fn = None
        self.gradient_clipper = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': []
        }
        
        # Setup experiment logger
        self.experiment_name = self._generate_experiment_name()
        self.logger = ExperimentLogger(
            experiment_name=self.experiment_name,
            output_dir=self.output_dir,
            config=config_dict,
            use_wandb=self.logging_config.get('use_wandb', False),
            use_tensorboard=self.logging_config.get('use_tensorboard', False),
            wandb_project=self.logging_config.get('wandb_project', 'molecugen'),
            wandb_entity=self.logging_config.get('wandb_entity'),
            wandb_tags=self.logging_config.get('wandb_tags', [])
        )
        
    def _generate_experiment_name(self) -> str:
        """Generate a unique experiment name."""
        if isinstance(self.config, ExperimentConfig):
            base_name = self.config.name
        else:
            base_name = self.config.get('name', 'molecugen_experiment')
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.model_config.get('type', 'unknown')
        return f"{base_name}_{model_type}_{timestamp}"
            
    def setup_data(self, dataset: MolecularDataset):
        """
        Setup dataset and create train/validation/test splits.
        
        Args:
            dataset: Molecular dataset to use for training
        """
        logger.info("Setting up data splits...")
        
        # Get split ratios
        train_split = self.data_config.get('train_split', 0.8)
        val_split = self.data_config.get('val_split', 0.1)
        test_split = self.data_config.get('test_split', 0.1)
        
        # Validate splits
        total_split = train_split + val_split + test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
            
        # Calculate split sizes
        dataset_size = len(dataset)
        train_size = int(train_split * dataset_size)
        val_size = int(val_split * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        logger.info(f"Dataset size: {dataset_size}")
        logger.info(f"Train size: {train_size} ({train_split:.1%})")
        logger.info(f"Validation size: {val_size} ({val_split:.1%})")
        logger.info(f"Test size: {test_size} ({test_split:.1%})")
        
        # Create splits
        seed = self.training_config.get('seed', 42)
        generator = torch.Generator().manual_seed(seed)
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )
        
        # Create data loaders
        batch_size = self.training_config.get('batch_size', 32)
        num_workers = self.training_config.get('num_workers', 0)
        
        self.train_loader = create_molecular_dataloader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers
        )
        
        self.val_loader = create_molecular_dataloader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        self.test_loader = create_molecular_dataloader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers
        )
        
        # Log dataset info
        dataset_info = {
            'total_size': dataset_size,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'train_split': train_split,
            'val_split': val_split,
            'test_split': test_split
        }
        self.logger.log_dataset_info(dataset_info)
        
        logger.info("Data setup completed successfully")
        
    def _setup_loss_function(self):
        """Setup loss function based on model type."""
        model_type = self.model_config.get('type', 'diffusion')
        
        if model_type == 'diffusion':
            self.loss_fn = DiffusionLoss(
                loss_type=self.training_config.get('loss_type', 'mse'),
                node_weight=self.training_config.get('node_weight', 1.0),
                edge_weight=self.training_config.get('edge_weight', 1.0),
                timestep_weighting=self.training_config.get('timestep_weighting', 'uniform')
            )
        elif model_type == 'autoregressive_flow':
            self.loss_fn = FlowLoss(
                regularization_weight=self.training_config.get('regularization_weight', 0.0),
                entropy_regularization=self.training_config.get('entropy_regularization', 0.0),
                gradient_penalty_weight=self.training_config.get('gradient_penalty_weight', 0.0)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    def setup_model(self):
        """Setup the generative model based on configuration."""
        logger.info("Setting up model...")
        
        # Get model type
        model_type = self.model_config.get('type', 'diffusion')
        
        # Get feature dimensions from data loader
        if self.train_loader is None:
            raise ValueError("Data must be setup before model. Call setup_data() first.")
            
        # Get sample batch to determine feature dimensions
        sample_batch = next(iter(self.train_loader))
        if sample_batch is None or sample_batch.x.size(0) == 0:
            raise ValueError("No valid data found in training set")
            
        node_dim = sample_batch.x.size(1)
        edge_dim = sample_batch.edge_attr.size(1) if sample_batch.edge_attr is not None else 0
        
        logger.info(f"Feature dimensions: node_dim={node_dim}, edge_dim={edge_dim}")
        
        # Update model config with feature dimensions
        model_config = self.model_config.copy()
        model_config.update({
            'node_dim': node_dim,
            'edge_dim': edge_dim
        })
        
        # Create model
        if model_type == 'diffusion':
            self.model = GraphDiffusion(model_config)
        elif model_type == 'autoregressive_flow':
            self.model = GraphAF(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Move to device
        self.model.to(self.device)
        
        # Setup loss function
        self._setup_loss_function()
        
        # Setup gradient clipper
        if self.training_config.get('gradient_clip'):
            self.gradient_clipper = GradientClipper(
                clip_type='norm',
                clip_value=self.training_config['gradient_clip']
            )
        
        # Log model info
        self.logger.log_model_summary(self.model)
            
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        if self.model is None:
            raise ValueError("Model must be setup before optimizer. Call setup_model() first.")
            
        logger.info("Setting up optimizer and scheduler...")
        
        # Create optimizer using factory
        optimizer_config = self.training_config.get('optimizer', {})
        self.optimizer = OptimizerFactory.create_optimizer(self.model, optimizer_config)
        
        # Create scheduler using factory
        scheduler_config = self.training_config.get('scheduler', {})
        self.scheduler = SchedulerFactory.create_scheduler(self.optimizer, scheduler_config)
        
        logger.info(f"Setup {optimizer_config.get('type', 'adam')} optimizer")
        if self.scheduler is not None:
            logger.info(f"Setup {scheduler_config.get('type', 'cosine')} scheduler")
            
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Returns:
            Dictionary containing training results and metrics
        """
        if any(x is None for x in [self.model, self.optimizer, self.train_loader, self.val_loader]):
            raise ValueError("Model, optimizer, and data must be setup before training")
            
        logger.info("Starting training...")
        
        # Training configuration
        num_epochs = self.training_config.get('num_epochs', 100)
        patience = self.training_config.get('patience', None)
        save_every = self.training_config.get('save_every', 10)
        validate_every = self.training_config.get('validate_every', 1)
        
        # Early stopping
        patience_counter = 0
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training step
            train_metrics = self._train_epoch()
            
            # Validation step
            val_metrics = {}
            if epoch % validate_every == 0:
                val_metrics = self._validate_epoch()
                
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    if 'val_loss' in val_metrics:
                        self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
                    
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history['epochs'].append(epoch)
            self.training_history['train_losses'].append(train_metrics.get('train_loss', 0.0))
            self.training_history['learning_rates'].append(current_lr)
            
            if val_metrics:
                self.training_history['val_losses'].append(val_metrics.get('val_loss', 0.0))
                
            # Check for best model
            val_loss = val_metrics.get('val_loss', float('inf'))
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                logger.info(f"New best model with val_loss={val_loss:.6f}")
            else:
                patience_counter += 1
                
            # Periodic checkpoint saving
            if epoch % save_every == 0 or is_best:
                combined_metrics = {**train_metrics, **val_metrics}
                self.logger.save_checkpoint(
                    self.model, self.optimizer, self.scheduler, 
                    epoch, combined_metrics, is_best
                )
                
            # Logging
            log_metrics = {
                'epoch': epoch,
                'learning_rate': current_lr,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            self.logger.log_metrics(log_metrics, step=epoch)
            
            # Console logging
            log_msg = f"Epoch {epoch:3d}/{num_epochs}: "
            log_msg += f"train_loss={train_metrics.get('train_loss', 0.0):.6f}, "
            if val_metrics:
                log_msg += f"val_loss={val_loss:.6f}, "
            log_msg += f"lr={current_lr:.2e}"
            
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(log_msg)
                
            # Early stopping
            if patience is not None and patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                break
                
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        
        # Save final checkpoint
        final_metrics = {**train_metrics, **val_metrics}
        self.logger.save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            self.current_epoch, final_metrics, is_best=False
        )
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time,
            'training_history': self.training_history
        }
        
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            if batch is None or batch.x.size(0) == 0:
                continue
                
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            loss = self.model.training_step(batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clipper is not None:
                clip_stats = self.gradient_clipper.clip_gradients(self.model)
                # Could log gradient statistics here if needed
                
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)
        
        return {'train_loss': avg_loss}
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None or batch.x.size(0) == 0:
                    continue
                    
                batch = batch.to(self.device)
                
                # Forward pass
                val_metrics = self.model.validation_step(batch)
                loss = val_metrics.get('val_loss', self.model.training_step(batch))
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / max(num_batches, 1)
        
        return {'val_loss': avg_loss}
    def save_config(self):
        """Save training configuration."""
        if isinstance(self.config, ExperimentConfig):
            from .config_manager import ConfigManager
            manager = ConfigManager()
            manager.save_config(self.config, self.output_dir / "config.yaml")
        else:
            config_file = self.output_dir / "config.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load model from checkpoint and resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint metadata
        """
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint_data = BaseGenerativeModel.load_checkpoint(checkpoint_path, self.device)
        
        self.model = checkpoint_data['model']
        self.current_epoch = checkpoint_data.get('epoch', 0)
        
        # Load optimizer state if available
        if checkpoint_data.get('optimizer_state_dict') and self.optimizer:
            self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
            
        # Load scheduler state if available
        if checkpoint_data.get('scheduler_state_dict') and self.scheduler:
            self.scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
            
        logger.info(f"Resumed from epoch {self.current_epoch}")
        
        return checkpoint_data
        
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Test metrics
        """
        if self.test_loader is None:
            raise ValueError("Test data not available. Ensure test_split > 0 in data config.")
            
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None or batch.x.size(0) == 0:
                    continue
                    
                batch = batch.to(self.device)
                
                # Forward pass
                test_metrics = self.model.validation_step(batch)
                loss = test_metrics.get('val_loss', self.model.training_step(batch))
                
                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / max(num_batches, 1)
        
        test_results = {'test_loss': avg_loss}
        
        logger.info(f"Test loss: {avg_loss:.6f}")
        
        # Save test results
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
            
        # Log to experiment logger
        self.logger.log_metrics({f'test_{k}': v for k, v in test_results.items()})
            
        return test_results
        
    def finalize(self):
        """Finalize training and cleanup resources."""
        self.logger.finalize()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finalize()


def create_default_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'model': {
            'type': 'diffusion',  # or 'autoregressive_flow'
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.1,
            'max_nodes': 50,
            # Diffusion-specific
            'num_timesteps': 1000,
            'beta_schedule': 'cosine',
            # Flow-specific
            'num_flow_layers': 4
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'patience': 15,
            'seed': 42,
            'num_workers': 0,
            'save_every': 10,
            'validate_every': 1,
            'optimizer': {
                'type': 'adam',
                'betas': [0.9, 0.999],
                'eps': 1e-8
            },
            'scheduler': {
                'type': 'cosine',
                'eta_min': 1e-6
            }
        },
        'data': {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        },
        'logging': {
            'level': 'INFO',
            'wandb_project': 'molecugen'
        }
    }