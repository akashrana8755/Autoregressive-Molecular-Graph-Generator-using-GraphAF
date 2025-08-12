"""
Base generative model interface for molecular generation.

This module defines the abstract base class that all generative models
must implement, providing common functionality for training, sampling,
and checkpoint management.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
import os
import json
from pathlib import Path


class BaseGenerativeModel(nn.Module, ABC):
    """
    Abstract base class for generative models.
    
    This class defines the interface that all generative models must implement,
    including methods for training, sampling, and checkpoint management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base generative model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        self.config = config
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, batch: Batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            batch: Batch of molecular graphs
            **kwargs: Additional arguments specific to the model
            
        Returns:
            Dictionary containing model outputs
        """
        pass
    
    @abstractmethod
    def training_step(self, batch: Batch, **kwargs) -> torch.Tensor:
        """
        Compute loss for a training batch.
        
        Args:
            batch: Batch of molecular graphs
            **kwargs: Additional arguments for training
            
        Returns:
            Training loss tensor
        """
        pass
    
    @abstractmethod
    def sample(self, num_samples: int, **kwargs) -> List[Data]:
        """
        Generate molecular graphs by sampling from the model.
        
        Args:
            num_samples: Number of molecules to generate
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated molecular graphs
        """
        pass
    
    def validation_step(self, batch: Batch, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute validation metrics for a batch.
        
        Default implementation uses training_step. Override for custom validation.
        
        Args:
            batch: Batch of molecular graphs
            **kwargs: Additional arguments for validation
            
        Returns:
            Dictionary containing validation metrics
        """
        loss = self.training_step(batch, **kwargs)
        return {"val_loss": loss}
    
    def save_checkpoint(self, 
                       checkpoint_path: Union[str, Path], 
                       epoch: int,
                       optimizer_state: Optional[Dict] = None,
                       scheduler_state: Optional[Dict] = None,
                       metrics: Optional[Dict] = None) -> None:
        """
        Save model checkpoint with metadata.
        
        Args:
            checkpoint_path: Path to save the checkpoint
            epoch: Current training epoch
            optimizer_state: State dict of the optimizer
            scheduler_state: State dict of the learning rate scheduler
            metrics: Training/validation metrics
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_name": self.model_name,
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "metrics": metrics or {}
        }
        
        if optimizer_state is not None:
            checkpoint["optimizer_state_dict"] = optimizer_state
            
        if scheduler_state is not None:
            checkpoint["scheduler_state_dict"] = scheduler_state
            
        torch.save(checkpoint, checkpoint_path)
        
        # Save config as separate JSON file for easy inspection
        config_path = checkpoint_path.with_suffix('.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load_checkpoint(cls, 
                       checkpoint_path: Union[str, Path],
                       device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load model checkpoint and return model with metadata.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            device: Device to load the model on
            
        Returns:
            Dictionary containing loaded model and metadata
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model instance with saved config
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        return {
            "model": model,
            "epoch": checkpoint.get("epoch", 0),
            "optimizer_state_dict": checkpoint.get("optimizer_state_dict"),
            "scheduler_state_dict": checkpoint.get("scheduler_state_dict"),
            "metrics": checkpoint.get("metrics", {}),
            "config": checkpoint["config"]
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter count and configuration.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "config": self.config,
            "device": next(self.parameters()).device.type if len(list(self.parameters())) > 0 else "cpu"
        }
    
    def freeze_parameters(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze all model parameters.
        
        Args:
            freeze: Whether to freeze (True) or unfreeze (False) parameters
        """
        for param in self.parameters():
            param.requires_grad = not freeze
    
    def get_device(self) -> torch.device:
        """
        Get the device the model is currently on.
        
        Returns:
            Device of the model
        """
        return next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')