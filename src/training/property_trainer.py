"""
Training utilities for property prediction models.

This module provides training loops and utilities for training
property predictors on molecular datasets like QM9.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import yaml
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch_geometric.data import Batch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ..models.property_predictor import (
    PropertyPredictor, 
    MultiTaskPropertyPredictor,
    PropertyPredictionTrainer,
    compute_property_statistics
)
from ..data.molecular_dataset import QM9Dataset, create_molecular_dataloader

logger = logging.getLogger(__name__)


class QM9PropertyTrainer:
    """
    Specialized trainer for QM9 property prediction.
    
    Handles training property predictors on QM9 dataset with proper
    data splits, normalization, and evaluation metrics.
    """
    
    # QM9 properties that are commonly used for training
    DEFAULT_QM9_PROPERTIES = [
        'mu',      # Dipole moment
        'alpha',   # Isotropic polarizability  
        'homo',    # HOMO energy
        'lumo',    # LUMO energy
        'gap',     # HOMO-LUMO gap
        'r2',      # Electronic spatial extent
        'zpve',    # Zero point vibrational energy
        'u0',      # Internal energy at 0K
        'u298',    # Internal energy at 298.15K
        'h298',    # Enthalpy at 298.15K
        'g298',    # Free energy at 298.15K
        'cv'       # Heat capacity at 298.15K
    ]
    
    def __init__(self,
                 config: Dict[str, Any],
                 data_path: str,
                 output_dir: str = "experiments/property_prediction",
                 device: Optional[torch.device] = None):
        """
        Initialize the QM9 property trainer.
        
        Args:
            config: Training configuration
            data_path: Path to QM9 dataset
            output_dir: Directory to save results
            device: Device to train on
        """
        self.config = config
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Extract configuration
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.data_config = config.get('data', {})
        
        # Properties to predict
        self.properties = self.data_config.get('properties', self.DEFAULT_QM9_PROPERTIES[:4])
        logger.info(f"Training on properties: {self.properties}")
        
        # Initialize components
        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.model = None
        self.trainer = None
        self.property_stats = None
        
    def setup_data(self):
        """Setup QM9 dataset and data loaders."""
        logger.info("Setting up QM9 dataset...")
        
        # Load QM9 dataset
        self.dataset = QM9Dataset(
            data_path=self.data_path,
            properties=self.properties,
            cache_dir=self.output_dir / "cache"
        )
        
        logger.info(f"Loaded {len(self.dataset)} molecules")
        
        # Compute property statistics for normalization
        self.property_stats = compute_property_statistics(self.dataset, self.properties)
        logger.info("Property statistics:")
        for prop, (mean, std) in self.property_stats.items():
            logger.info(f"  {prop}: mean={mean:.4f}, std={std:.4f}")
            
        # Save statistics
        stats_file = self.output_dir / "property_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(self.property_stats, f, indent=2)
            
        # Create data splits
        train_size = int(self.data_config.get('train_split', 0.8) * len(self.dataset))
        val_size = int(self.data_config.get('val_split', 0.1) * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(self.training_config.get('seed', 42))
        )
        
        logger.info(f"Data splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        # Create data loaders
        batch_size = self.training_config.get('batch_size', 32)
        num_workers = self.training_config.get('num_workers', 0)
        
        self.train_loader = create_molecular_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.val_loader = create_molecular_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        self.test_loader = create_molecular_dataloader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
    def setup_model(self):
        """Setup property prediction model."""
        logger.info("Setting up property prediction model...")
        
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
        
        # Model configuration
        model_type = self.model_config.get('type', 'single_task')
        
        if model_type == 'multi_task':
            # Multi-task model with property-specific heads
            property_configs = {}
            for prop in self.properties:
                property_configs[prop] = self.model_config.get('property_configs', {}).get(prop, {})
                
            self.model = MultiTaskPropertyPredictor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                property_configs=property_configs,
                shared_hidden_dim=self.model_config.get('hidden_dim', 256),
                shared_num_layers=self.model_config.get('num_layers', 4),
                gnn_type=self.model_config.get('gnn_type', 'gcn'),
                pooling=self.model_config.get('pooling', 'mean'),
                dropout=self.model_config.get('dropout', 0.1),
                batch_norm=self.model_config.get('batch_norm', True),
                residual=self.model_config.get('residual', True),
                activation=self.model_config.get('activation', 'relu')
            )
        else:
            # Single-task model
            self.model = PropertyPredictor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                hidden_dim=self.model_config.get('hidden_dim', 256),
                num_layers=self.model_config.get('num_layers', 4),
                num_properties=len(self.properties),
                property_names=self.properties,
                gnn_type=self.model_config.get('gnn_type', 'gcn'),
                pooling=self.model_config.get('pooling', 'mean'),
                dropout=self.model_config.get('dropout', 0.1),
                batch_norm=self.model_config.get('batch_norm', True),
                residual=self.model_config.get('residual', True),
                activation=self.model_config.get('activation', 'relu')
            )
            
        logger.info(f"Created {model_type} model with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        # Setup trainer
        loss_fn = nn.MSELoss()
        property_weights = self.training_config.get('property_weights', {})
        
        self.trainer = PropertyPredictionTrainer(
            model=self.model,
            device=self.device,
            loss_fn=loss_fn,
            property_weights=property_weights,
            property_scalers=self.property_stats
        )
        
    def train(self) -> Dict[str, Any]:
        """
        Train the property prediction model.
        
        Returns:
            Training results and metrics
        """
        if self.trainer is None:
            raise ValueError("Model not setup. Call setup_model() first.")
            
        logger.info("Starting training...")
        
        # Training configuration
        num_epochs = self.training_config.get('num_epochs', 100)
        learning_rate = self.training_config.get('learning_rate', 1e-3)
        weight_decay = self.training_config.get('weight_decay', 1e-5)
        scheduler_type = self.training_config.get('scheduler', 'cosine')
        patience = self.training_config.get('patience', 10)
        
        # Setup optimizer
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup scheduler
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=patience//2
            )
        else:
            scheduler = None
            
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        for epoch in range(num_epochs):
            # Training
            train_losses = []
            for batch in self.train_loader:
                if batch is None:
                    continue
                    
                losses = self.trainer.train_step(batch, optimizer)
                train_losses.append(losses['total_loss'])
                
            avg_train_loss = np.mean(train_losses) if train_losses else 0.0
            
            # Validation
            val_metrics = self.trainer.validate(self.val_loader)
            val_loss = val_metrics.get('val_total_loss', 0.0)
            
            # Learning rate scheduling
            if scheduler is not None:
                if scheduler_type == 'plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                    
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record history
            training_history['train_losses'].append(avg_train_loss)
            training_history['val_losses'].append(val_loss)
            training_history['learning_rates'].append(current_lr)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                checkpoint_path = self.output_dir / "best_model.pt"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': self.config,
                    'property_stats': self.property_stats
                }, checkpoint_path)
                
            else:
                patience_counter += 1
                
            # Logging
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch:3d}/{num_epochs}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"lr={current_lr:.2e}"
                )
                
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        # Save training history
        history_file = self.output_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(training_history, f, indent=2)
            
        # Load best model for evaluation
        checkpoint = torch.load(self.output_dir / "best_model.pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("Training completed!")
        
        return {
            'best_val_loss': best_val_loss,
            'training_history': training_history,
            'num_epochs_trained': epoch + 1
        }
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the trained model on test set.
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        logger.info("Evaluating model on test set...")
        
        self.model.eval()
        
        all_predictions = {prop: [] for prop in self.properties}
        all_targets = {prop: [] for prop in self.properties}
        
        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None:
                    continue
                    
                batch = batch.to(self.device)
                
                if isinstance(self.model, MultiTaskPropertyPredictor):
                    predictions = self.model(batch)
                    
                    for prop in self.properties:
                        if hasattr(batch, prop):
                            pred = predictions[prop].cpu().numpy().flatten()
                            target = getattr(batch, prop).cpu().numpy().flatten()
                            
                            # Denormalize if needed
                            if prop in self.property_stats:
                                mean, std = self.property_stats[prop]
                                pred = pred * std + mean
                                # target is already in original scale
                                
                            all_predictions[prop].extend(pred)
                            all_targets[prop].extend(target)
                            
                else:
                    predictions = self.model(batch).cpu().numpy()
                    
                    for i, prop in enumerate(self.properties):
                        if hasattr(batch, prop):
                            pred = predictions[:, i]
                            target = getattr(batch, prop).cpu().numpy().flatten()
                            
                            # Denormalize if needed
                            if prop in self.property_stats:
                                mean, std = self.property_stats[prop]
                                pred = pred * std + mean
                                
                            all_predictions[prop].extend(pred)
                            all_targets[prop].extend(target)
                            
        # Compute metrics
        metrics = {}
        
        for prop in self.properties:
            if all_predictions[prop] and all_targets[prop]:
                pred = np.array(all_predictions[prop])
                target = np.array(all_targets[prop])
                
                mse = mean_squared_error(target, pred)
                mae = mean_absolute_error(target, pred)
                r2 = r2_score(target, pred)
                
                metrics[prop] = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(np.sqrt(mse)),
                    'r2': float(r2)
                }
                
                logger.info(f"{prop}: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
                
        # Save evaluation results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
            
        return metrics
        
    def save_config(self):
        """Save training configuration."""
        config_file = self.output_dir / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            
    def load_model(self, checkpoint_path: str) -> PropertyPredictor:
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Loaded model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Recreate model from config
        config = checkpoint['config']
        model_config = config.get('model', {})
        
        # Get feature dimensions (assuming they're saved in checkpoint)
        # In practice, you might want to save these in the checkpoint
        node_dim = model_config.get('node_dim', 128)  # Default fallback
        edge_dim = model_config.get('edge_dim', 64)   # Default fallback
        
        if model_config.get('type', 'single_task') == 'multi_task':
            property_configs = {}
            for prop in self.properties:
                property_configs[prop] = model_config.get('property_configs', {}).get(prop, {})
                
            model = MultiTaskPropertyPredictor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                property_configs=property_configs,
                **{k: v for k, v in model_config.items() if k not in ['type', 'property_configs']}
            )
        else:
            model = PropertyPredictor(
                node_dim=node_dim,
                edge_dim=edge_dim,
                num_properties=len(self.properties),
                property_names=self.properties,
                **{k: v for k, v in model_config.items() if k != 'type'}
            )
            
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for QM9 property prediction."""
    return {
        'model': {
            'type': 'single_task',  # or 'multi_task'
            'hidden_dim': 256,
            'num_layers': 4,
            'gnn_type': 'gcn',
            'pooling': 'mean',
            'dropout': 0.1,
            'batch_norm': True,
            'residual': True,
            'activation': 'relu'
        },
        'training': {
            'batch_size': 32,
            'num_epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'scheduler': 'cosine',
            'patience': 15,
            'seed': 42,
            'num_workers': 0
        },
        'data': {
            'properties': ['mu', 'alpha', 'homo', 'lumo'],
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1
        }
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train QM9 property predictor")
    parser.add_argument("--data_path", required=True, help="Path to QM9 dataset")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--output_dir", default="experiments/qm9_property_prediction", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = create_default_config()
        
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create trainer
    trainer = QM9PropertyTrainer(
        config=config,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    # Setup and train
    trainer.setup_data()
    trainer.setup_model()
    trainer.save_config()
    
    # Train model
    training_results = trainer.train()
    
    # Evaluate model
    evaluation_results = trainer.evaluate()
    
    print("Training completed!")
    print(f"Best validation loss: {training_results['best_val_loss']:.4f}")
    print("Test set results:")
    for prop, metrics in evaluation_results.items():
        print(f"  {prop}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")