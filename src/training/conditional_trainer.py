"""
Training utilities for conditional molecular generation models.

This module provides training loops and utilities for training
property-conditioned generative models with integrated property prediction.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import yaml
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from ..models.conditional_generation import (
    ConditionalGraphDiffusion, 
    ConditionalGraphAF,
    MultiObjectiveGenerator
)
from ..models.property_predictor import PropertyPredictor, MultiTaskPropertyPredictor
from ..data.molecular_dataset import MolecularDataset, create_molecular_dataloader
from ..data.property_calculator import PropertyCalculator

logger = logging.getLogger(__name__)


class ConditionalGenerationTrainer:
    """
    Trainer for property-conditioned molecular generation models.
    
    This trainer handles joint training of generative models with property
    conditioning, including integration with property predictors.
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 output_dir: str = "experiments/conditional_generation",
                 device: Optional[torch.device] = None):
        """
        Initialize the conditional generation trainer.
        
        Args:
            config: Training configuration
            output_dir: Directory to save results
            device: Device to train on
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Extract configuration
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.data_config = config.get('data', {})
        
        # Property configuration
        self.property_names = self.data_config.get('properties', ['logp', 'qed', 'molecular_weight'])
        self.property_dim = len(self.property_names)
        
        # Initialize components
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.generator = None
        self.property_predictor = None
        self.property_calculator = PropertyCalculator()
        
    def setup_data(self, dataset: MolecularDataset):
        """
        Setup dataset and data loaders.
        
        Args:
            dataset: Molecular dataset with property information
        """
        logger.info("Setting up conditional generation dataset...")
        
        self.dataset = dataset
        
        # Compute additional properties using RDKit if needed
        self._augment_dataset_properties()
        
        # Create data splits
        from torch.utils.data import random_split
        
        train_size = int(self.data_config.get('train_split', 0.8) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.training_config.get('seed', 42))
        )
        
        logger.info(f"Data splits: train={len(train_dataset)}, val={len(val_dataset)}")
        
        # Create data loaders
        batch_size = self.training_config.get('batch_size', 32)
        num_workers = self.training_config.get('num_workers', 0)
        
        self.train_loader = create_molecular_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.val_loader = create_molecular_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
    def _augment_dataset_properties(self):
        """Augment dataset with additional RDKit-calculated properties."""
        logger.info("Augmenting dataset with RDKit properties...")
        
        # Get SMILES from dataset
        smiles_list = []
        for data in self.dataset:
            if data is not None and hasattr(data, 'smiles'):
                smiles_list.append(data.smiles)
            else:
                smiles_list.append(None)
                
        # Calculate properties
        for i, smiles in enumerate(smiles_list):
            if smiles is not None:
                try:
                    props = self.property_calculator.calculate_properties(
                        smiles, self.property_names
                    )
                    
                    # Add properties to data object
                    data = self.dataset[i]
                    if data is not None:
                        for prop_name, value in props.items():
                            if not np.isnan(value):
                                setattr(data, prop_name, torch.tensor([value], dtype=torch.float))
                                
                except Exception as e:
                    logger.warning(f"Failed to calculate properties for molecule {i}: {e}")
                    continue
                    
    def setup_models(self):
        """Setup conditional generative model and property predictor."""
        logger.info("Setting up conditional generation models...")
        
        # Get feature dimensions from dataset
        sample_data = None
        for data in self.dataset:
            if data is not None:
                sample_data = data
                break
                
        if sample_data is None:
            raise ValueError("No valid data found in dataset")
            
        node_dim = sample_data.x.size(1)
        edge_dim = sample_data.edge_attr.size(1) if sample_data.edge_attr is not None else 0
        
        logger.info(f"Feature dimensions: node_dim={node_dim}, edge_dim={edge_dim}")
        
        # Setup generative model
        model_type = self.model_config.get('type', 'diffusion')
        
        # Add property conditioning configuration
        model_config = self.model_config.copy()
        model_config.update({
            'node_dim': node_dim,
            'edge_dim': edge_dim,
            'property_dim': self.property_dim,
            'property_names': self.property_names
        })
        
        if model_type == 'diffusion':
            self.generator = ConditionalGraphDiffusion(model_config)
        elif model_type == 'autoregressive_flow':
            self.generator = ConditionalGraphAF(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.generator.to(self.device)
        
        # Setup property predictor if requested
        if self.training_config.get('use_property_predictor', False):
            predictor_config = self.training_config.get('property_predictor', {})
            
            self.property_predictor = PropertyPredictor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_properties=self.property_dim,
                property_names=self.property_names,
                **predictor_config
            )
            self.property_predictor.to(self.device)
            
        logger.info(f"Created {model_type} generator with {sum(p.numel() for p in self.generator.parameters())} parameters")
        
    def train(self) -> Dict[str, Any]:
        """
        Train the conditional generation model.
        
        Returns:
            Training results and metrics
        """
        logger.info("Starting conditional generation training...")
        
        # Training configuration
        num_epochs = self.training_config.get('num_epochs', 100)
        learning_rate = self.training_config.get('learning_rate', 1e-4)
        weight_decay = self.training_config.get('weight_decay', 1e-5)
        
        # Setup optimizers
        gen_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        prop_optimizer = None
        if self.property_predictor is not None:
            prop_lr = self.training_config.get('property_predictor_lr', learning_rate)
            prop_optimizer = optim.Adam(
                self.property_predictor.parameters(),
                lr=prop_lr,
                weight_decay=weight_decay
            )
            
        # Setup schedulers
        gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=num_epochs)
        prop_scheduler = None
        if prop_optimizer is not None:
            prop_scheduler = optim.lr_scheduler.CosineAnnealingLR(prop_optimizer, T_max=num_epochs)
            
        # Training loop
        best_val_loss = float('inf')
        training_history = {
            'train_gen_losses': [],
            'train_prop_losses': [],
            'val_gen_losses': [],
            'val_prop_losses': []
        }
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self._train_epoch(gen_optimizer, prop_optimizer)
            
            # Validation
            val_metrics = self._validate_epoch()
            
            # Update schedulers
            gen_scheduler.step()
            if prop_scheduler is not None:
                prop_scheduler.step()
                
            # Record history
            training_history['train_gen_losses'].append(train_metrics['gen_loss'])
            training_history['val_gen_losses'].append(val_metrics['gen_loss'])
            
            if 'prop_loss' in train_metrics:
                training_history['train_prop_losses'].append(train_metrics['prop_loss'])
                training_history['val_prop_losses'].append(val_metrics['prop_loss'])
                
            # Save best model
            val_loss = val_metrics['total_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, gen_optimizer, prop_optimizer, val_metrics)
                
            # Logging
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs}: "
                    f"train_gen={train_metrics['gen_loss']:.4f}, "
                    f"val_gen={val_metrics['gen_loss']:.4f}, "
                    f"val_total={val_loss:.4f}"
                )
                
        # Save training history
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
            
        return {
            'best_val_loss': best_val_loss,
            'training_history': training_history
        }
        
    def _train_epoch(self, 
                    gen_optimizer: torch.optim.Optimizer,
                    prop_optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, float]:
        """Train for one epoch."""
        self.generator.train()
        if self.property_predictor is not None:
            self.property_predictor.train()
            
        gen_losses = []
        prop_losses = []
        
        for batch in self.train_loader:
            if batch is None:
                continue
                
            batch = batch.to(self.device)
            
            # Extract property targets
            property_targets = self._extract_properties(batch)
            
            # Train generator
            gen_optimizer.zero_grad()
            
            if isinstance(self.generator, ConditionalGraphDiffusion):
                gen_loss = self.generator.training_step(batch, properties=property_targets)
            else:  # ConditionalGraphAF
                gen_loss = self.generator.training_step(batch)
                
            gen_loss.backward()
            gen_optimizer.step()
            
            gen_losses.append(gen_loss.item())
            
            # Train property predictor if available
            if self.property_predictor is not None and prop_optimizer is not None:
                prop_optimizer.zero_grad()
                
                prop_predictions = self.property_predictor(batch)
                prop_targets = torch.cat([
                    getattr(batch, prop_name).view(-1, 1) 
                    for prop_name in self.property_names
                    if hasattr(batch, prop_name)
                ], dim=1)
                
                if prop_targets.size(0) > 0:
                    prop_loss = nn.MSELoss()(prop_predictions, prop_targets)
                    prop_loss.backward()
                    prop_optimizer.step()
                    
                    prop_losses.append(prop_loss.item())
                    
        metrics = {'gen_loss': np.mean(gen_losses)}
        if prop_losses:
            metrics['prop_loss'] = np.mean(prop_losses)
            
        return metrics
        
    def _validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.generator.eval()
        if self.property_predictor is not None:
            self.property_predictor.eval()
            
        gen_losses = []
        prop_losses = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue
                    
                batch = batch.to(self.device)
                
                # Extract property targets
                property_targets = self._extract_properties(batch)
                
                # Validate generator
                if isinstance(self.generator, ConditionalGraphDiffusion):
                    gen_loss = self.generator.training_step(batch, properties=property_targets)
                else:
                    gen_loss = self.generator.training_step(batch)
                    
                gen_losses.append(gen_loss.item())
                
                # Validate property predictor
                if self.property_predictor is not None:
                    prop_predictions = self.property_predictor(batch)
                    prop_targets = torch.cat([
                        getattr(batch, prop_name).view(-1, 1)
                        for prop_name in self.property_names
                        if hasattr(batch, prop_name)
                    ], dim=1)
                    
                    if prop_targets.size(0) > 0:
                        prop_loss = nn.MSELoss()(prop_predictions, prop_targets)
                        prop_losses.append(prop_loss.item())
                        
        metrics = {'gen_loss': np.mean(gen_losses)}
        if prop_losses:
            metrics['prop_loss'] = np.mean(prop_losses)
            
        # Total loss for model selection
        total_loss = metrics['gen_loss']
        if 'prop_loss' in metrics:
            prop_weight = self.training_config.get('property_loss_weight', 0.1)
            total_loss += prop_weight * metrics['prop_loss']
            
        metrics['total_loss'] = total_loss
        
        return metrics
        
    def _extract_properties(self, batch: Batch) -> Optional[torch.Tensor]:
        """Extract property targets from batch."""
        property_tensors = []
        
        for prop_name in self.property_names:
            if hasattr(batch, prop_name):
                prop_values = getattr(batch, prop_name)
                if prop_values.dim() == 1:
                    prop_values = prop_values.unsqueeze(-1)
                property_tensors.append(prop_values)
            else:
                # Use zero if property not available
                batch_size = batch.num_graphs
                zeros = torch.zeros(batch_size, 1, device=self.device)
                property_tensors.append(zeros)
                
        if property_tensors:
            return torch.cat(property_tensors, dim=1)
        else:
            return None
            
    def _save_checkpoint(self, 
                        epoch: int,
                        gen_optimizer: torch.optim.Optimizer,
                        prop_optimizer: Optional[torch.optim.Optimizer],
                        metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'gen_optimizer_state_dict': gen_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        if self.property_predictor is not None:
            checkpoint['property_predictor_state_dict'] = self.property_predictor.state_dict()
            
        if prop_optimizer is not None:
            checkpoint['prop_optimizer_state_dict'] = prop_optimizer.state_dict()
            
        checkpoint_path = self.output_dir / "best_model.pt"
        torch.save(checkpoint, checkpoint_path)
        
    def generate_molecules(self, 
                          num_samples: int,
                          property_targets: Optional[Dict[str, Tuple[float, float]]] = None,
                          **kwargs) -> List[Any]:
        """
        Generate molecules with optional property constraints.
        
        Args:
            num_samples: Number of molecules to generate
            property_targets: Dictionary of property constraints
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated molecules
        """
        if property_targets is not None and self.property_predictor is not None:
            # Use multi-objective generation
            multi_gen = MultiObjectiveGenerator(self.generator, self.property_predictor)
            return multi_gen.generate_with_constraints(
                num_samples=num_samples,
                property_targets=property_targets,
                **kwargs
            )
        else:
            # Standard conditional generation
            if property_targets is not None:
                # Convert targets to property vector
                device = self.generator.get_device()
                target_props = self._targets_to_conditioning(property_targets, num_samples, device)
                return self.generator.sample(num_samples, properties=target_props, **kwargs)
            else:
                return self.generator.sample(num_samples, **kwargs)
                
    def _targets_to_conditioning(self, 
                               targets: Dict[str, Tuple[float, float]],
                               num_samples: int,
                               device: torch.device) -> torch.Tensor:
        """Convert property targets to conditioning tensor."""
        target_values = []
        
        for prop_name in self.property_names:
            if prop_name in targets:
                min_val, max_val = targets[prop_name]
                # Use midpoint as target
                target_val = (min_val + max_val) / 2.0
            else:
                target_val = 0.0
                
            target_values.append(target_val)
            
        target_tensor = torch.tensor(target_values, device=device).unsqueeze(0)
        return target_tensor.repeat(num_samples, 1)
        
    def save_config(self):
        """Save training configuration."""
        config_file = self.output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)


def create_conditional_config() -> Dict[str, Any]:
    """Create default configuration for conditional generation."""
    return {
        'model': {
            'type': 'diffusion',  # or 'autoregressive_flow'
            'hidden_dim': 256,
            'num_layers': 4,
            'dropout': 0.1,
            'max_nodes': 50,
            'conditioning_type': 'concat',  # 'concat', 'cross_attention', 'film'
            'property_dropout': 0.1,
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
            'seed': 42,
            'num_workers': 0,
            'use_property_predictor': True,
            'property_predictor': {
                'hidden_dim': 128,
                'num_layers': 3,
                'dropout': 0.1
            },
            'property_predictor_lr': 1e-3,
            'property_loss_weight': 0.1
        },
        'data': {
            'properties': ['logp', 'qed', 'molecular_weight'],
            'train_split': 0.8,
            'val_split': 0.2
        }
    }