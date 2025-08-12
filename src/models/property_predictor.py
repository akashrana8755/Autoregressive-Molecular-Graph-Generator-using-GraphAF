"""
Property prediction models for molecular graphs.

This module implements GNN-based property predictors that can predict
molecular properties like logP, QED, molecular weight, and other
quantum mechanical properties from molecular graph representations.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, GraphConv, 
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm, LayerNorm
)

from .base_model import BaseGenerativeModel

logger = logging.getLogger(__name__)


class PropertyPredictor(nn.Module):
    """
    GNN-based property prediction model for molecular graphs.
    
    This model uses graph neural networks to predict molecular properties
    from graph representations, supporting both single and multi-property
    prediction tasks.
    """
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 4,
                 num_properties: int = 1,
                 property_names: Optional[List[str]] = None,
                 gnn_type: str = "gcn",
                 pooling: str = "mean",
                 dropout: float = 0.1,
                 batch_norm: bool = True,
                 residual: bool = True,
                 activation: str = "relu"):
        """
        Initialize the property predictor.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            num_properties: Number of properties to predict
            property_names: Names of properties being predicted
            gnn_type: Type of GNN ("gcn", "gat", "gin", "graph")
            pooling: Graph pooling method ("mean", "max", "add", "attention")
            dropout: Dropout rate
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
            activation: Activation function ("relu", "gelu", "swish")
        """
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_properties = num_properties
        self.property_names = property_names or [f"property_{i}" for i in range(num_properties)]
        self.gnn_type = gnn_type.lower()
        self.pooling = pooling.lower()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
            
        # Input projection
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        if edge_dim > 0:
            self.edge_embedding = nn.Linear(edge_dim, hidden_dim)
        else:
            self.edge_embedding = None
            
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        
        for i in range(num_layers):
            if self.gnn_type == "gcn":
                layer = GCNConv(hidden_dim, hidden_dim)
            elif self.gnn_type == "gat":
                layer = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            elif self.gnn_type == "gin":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation,
                    nn.Linear(hidden_dim, hidden_dim)
                )
                layer = GINConv(mlp)
            elif self.gnn_type == "graph":
                layer = GraphConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown GNN type: {self.gnn_type}")
                
            self.gnn_layers.append(layer)
            
            if batch_norm:
                self.batch_norms.append(BatchNorm(hidden_dim))
                
        # Attention pooling
        if self.pooling == "attention":
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.activation,
                nn.Linear(hidden_dim, 1)
            )
            
        # Output layers
        self.output_layers = nn.ModuleList()
        for _ in range(num_properties):
            output_layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                self.activation,
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )
            self.output_layers.append(output_layer)
            
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, batch: Batch) -> torch.Tensor:
        """
        Forward pass of the property predictor.
        
        Args:
            batch: Batch of molecular graphs
            
        Returns:
            Predicted properties tensor [batch_size, num_properties]
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_idx = batch.batch
        
        # Input embeddings
        x = self.node_embedding(x)
        x = self.activation(x)
        
        # GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_residual = x if self.residual else None
            
            # Apply GNN layer
            if self.gnn_type in ["gcn", "graph"]:
                x = gnn_layer(x, edge_index)
            elif self.gnn_type == "gat":
                x = gnn_layer(x, edge_index)
            elif self.gnn_type == "gin":
                x = gnn_layer(x, edge_index)
                
            # Batch normalization
            if self.batch_norm:
                x = self.batch_norms[i](x)
                
            # Activation
            x = self.activation(x)
            
            # Residual connection
            if self.residual and x_residual is not None:
                x = x + x_residual
                
            # Dropout
            x = self.dropout_layer(x)
            
        # Graph pooling
        if self.pooling == "mean":
            graph_repr = global_mean_pool(x, batch_idx)
        elif self.pooling == "max":
            graph_repr = global_max_pool(x, batch_idx)
        elif self.pooling == "add":
            graph_repr = global_add_pool(x, batch_idx)
        elif self.pooling == "attention":
            # Attention-based pooling
            attention_weights = self.attention_pool(x)
            attention_weights = F.softmax(attention_weights, dim=0)
            graph_repr = global_add_pool(x * attention_weights, batch_idx)
        else:
            graph_repr = global_mean_pool(x, batch_idx)
            
        # Property prediction
        predictions = []
        for output_layer in self.output_layers:
            pred = output_layer(graph_repr)
            predictions.append(pred)
            
        # Concatenate predictions
        predictions = torch.cat(predictions, dim=1)
        
        return predictions
        
    def predict_properties(self, graphs: List[Data]) -> Dict[str, List[float]]:
        """
        Predict properties for a list of molecular graphs.
        
        Args:
            graphs: List of molecular graph Data objects
            
        Returns:
            Dictionary mapping property names to predicted values
        """
        self.eval()
        
        with torch.no_grad():
            # Create batch
            batch = Batch.from_data_list(graphs)
            batch = batch.to(next(self.parameters()).device)
            
            # Predict
            predictions = self.forward(batch)
            predictions = predictions.cpu().numpy()
            
            # Organize results
            results = {}
            for i, prop_name in enumerate(self.property_names):
                results[prop_name] = predictions[:, i].tolist()
                
        return results
        
    def predict_single_property(self, graphs: List[Data], property_idx: int) -> List[float]:
        """
        Predict a single property for a list of graphs.
        
        Args:
            graphs: List of molecular graph Data objects
            property_idx: Index of the property to predict
            
        Returns:
            List of predicted values
        """
        predictions = self.predict_properties(graphs)
        prop_name = self.property_names[property_idx]
        return predictions[prop_name]
        
    def get_embeddings(self, batch: Batch) -> torch.Tensor:
        """
        Get graph embeddings (before final prediction layers).
        
        Args:
            batch: Batch of molecular graphs
            
        Returns:
            Graph embeddings tensor [batch_size, hidden_dim]
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_idx = batch.batch
        
        # Input embeddings
        x = self.node_embedding(x)
        x = self.activation(x)
        
        # GNN layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x_residual = x if self.residual else None
            
            # Apply GNN layer
            if self.gnn_type in ["gcn", "graph"]:
                x = gnn_layer(x, edge_index)
            elif self.gnn_type == "gat":
                x = gnn_layer(x, edge_index)
            elif self.gnn_type == "gin":
                x = gnn_layer(x, edge_index)
                
            # Batch normalization
            if self.batch_norm:
                x = self.batch_norms[i](x)
                
            # Activation
            x = self.activation(x)
            
            # Residual connection
            if self.residual and x_residual is not None:
                x = x + x_residual
                
            # Dropout (only during training)
            if self.training:
                x = self.dropout_layer(x)
                
        # Graph pooling
        if self.pooling == "mean":
            graph_repr = global_mean_pool(x, batch_idx)
        elif self.pooling == "max":
            graph_repr = global_max_pool(x, batch_idx)
        elif self.pooling == "add":
            graph_repr = global_add_pool(x, batch_idx)
        elif self.pooling == "attention":
            attention_weights = self.attention_pool(x)
            attention_weights = F.softmax(attention_weights, dim=0)
            graph_repr = global_add_pool(x * attention_weights, batch_idx)
        else:
            graph_repr = global_mean_pool(x, batch_idx)
            
        return graph_repr


class MultiTaskPropertyPredictor(PropertyPredictor):
    """
    Multi-task property predictor with task-specific heads.
    
    This variant allows for different architectures for different properties
    and can handle properties with different scales and distributions.
    """
    
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 property_configs: Dict[str, Dict[str, Any]],
                 shared_hidden_dim: int = 256,
                 shared_num_layers: int = 3,
                 **kwargs):
        """
        Initialize multi-task property predictor.
        
        Args:
            node_dim: Dimension of node features
            edge_dim: Dimension of edge features
            property_configs: Configuration for each property
            shared_hidden_dim: Hidden dimension for shared layers
            shared_num_layers: Number of shared GNN layers
            **kwargs: Additional arguments
        """
        self.property_configs = property_configs
        self.shared_hidden_dim = shared_hidden_dim
        self.shared_num_layers = shared_num_layers
        
        # Initialize with shared parameters
        super().__init__(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=shared_hidden_dim,
            num_layers=shared_num_layers,
            num_properties=len(property_configs),
            property_names=list(property_configs.keys()),
            **kwargs
        )
        
        # Replace output layers with task-specific heads
        self.output_layers = nn.ModuleDict()
        
        for prop_name, config in property_configs.items():
            head_hidden_dim = config.get('head_hidden_dim', shared_hidden_dim // 2)
            head_num_layers = config.get('head_num_layers', 2)
            head_dropout = config.get('head_dropout', self.dropout)
            
            # Build task-specific head
            layers = []
            input_dim = shared_hidden_dim
            
            for i in range(head_num_layers):
                if i == head_num_layers - 1:
                    # Final layer
                    layers.append(nn.Linear(input_dim, 1))
                else:
                    layers.extend([
                        nn.Linear(input_dim, head_hidden_dim),
                        self.activation,
                        nn.Dropout(head_dropout)
                    ])
                    input_dim = head_hidden_dim
                    
            self.output_layers[prop_name] = nn.Sequential(*layers)
            
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning property-specific predictions.
        
        Args:
            batch: Batch of molecular graphs
            
        Returns:
            Dictionary mapping property names to predictions
        """
        # Get shared graph embeddings
        graph_repr = self.get_embeddings(batch)
        
        # Task-specific predictions
        predictions = {}
        for prop_name, output_layer in self.output_layers.items():
            pred = output_layer(graph_repr)
            predictions[prop_name] = pred
            
        return predictions


class PropertyPredictionTrainer:
    """
    Trainer class for property prediction models.
    
    Handles training, validation, and evaluation of property predictors
    with support for multiple properties and loss functions.
    """
    
    def __init__(self,
                 model: PropertyPredictor,
                 device: torch.device,
                 loss_fn: Optional[nn.Module] = None,
                 property_weights: Optional[Dict[str, float]] = None,
                 property_scalers: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize the trainer.
        
        Args:
            model: Property predictor model
            device: Device to train on
            loss_fn: Loss function (default: MSE)
            property_weights: Weights for different properties
            property_scalers: Mean and std for property normalization
        """
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn or nn.MSELoss()
        self.property_weights = property_weights or {}
        self.property_scalers = property_scalers or {}
        
    def train_step(self, batch: Batch, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            optimizer: Optimizer
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        optimizer.zero_grad()
        
        batch = batch.to(self.device)
        
        # Forward pass
        if isinstance(self.model, MultiTaskPropertyPredictor):
            predictions = self.model(batch)
            
            # Compute loss for each property
            total_loss = 0
            losses = {}
            
            for prop_name in self.model.property_names:
                if hasattr(batch, prop_name):
                    target = getattr(batch, prop_name).view(-1, 1)
                    pred = predictions[prop_name]
                    
                    # Apply scaling if available
                    if prop_name in self.property_scalers:
                        mean, std = self.property_scalers[prop_name]
                        target = (target - mean) / std
                        
                    loss = self.loss_fn(pred, target)
                    weight = self.property_weights.get(prop_name, 1.0)
                    weighted_loss = weight * loss
                    
                    total_loss += weighted_loss
                    losses[f'{prop_name}_loss'] = loss.item()
                    
        else:
            predictions = self.model(batch)
            
            # Collect targets
            targets = []
            for i, prop_name in enumerate(self.model.property_names):
                if hasattr(batch, prop_name):
                    target = getattr(batch, prop_name).view(-1, 1)
                    
                    # Apply scaling if available
                    if prop_name in self.property_scalers:
                        mean, std = self.property_scalers[prop_name]
                        target = (target - mean) / std
                        
                    targets.append(target)
                    
            if targets:
                targets = torch.cat(targets, dim=1)
                total_loss = self.loss_fn(predictions, targets)
                losses = {'total_loss': total_loss.item()}
            else:
                total_loss = torch.tensor(0.0, device=self.device)
                losses = {'total_loss': 0.0}
                
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        losses['total_loss'] = total_loss.item()
        return losses
        
    def validate(self, val_loader) -> Dict[str, float]:
        """
        Validate the model on validation data.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                    
                batch = batch.to(self.device)
                
                # Forward pass
                if isinstance(self.model, MultiTaskPropertyPredictor):
                    predictions = self.model(batch)
                    
                    for prop_name in self.model.property_names:
                        if hasattr(batch, prop_name):
                            target = getattr(batch, prop_name).view(-1, 1)
                            pred = predictions[prop_name]
                            
                            # Apply scaling if available
                            if prop_name in self.property_scalers:
                                mean, std = self.property_scalers[prop_name]
                                target = (target - mean) / std
                                
                            loss = self.loss_fn(pred, target)
                            
                            if f'val_{prop_name}_loss' not in total_losses:
                                total_losses[f'val_{prop_name}_loss'] = 0
                            total_losses[f'val_{prop_name}_loss'] += loss.item()
                            
                else:
                    predictions = self.model(batch)
                    
                    # Collect targets
                    targets = []
                    for prop_name in self.model.property_names:
                        if hasattr(batch, prop_name):
                            target = getattr(batch, prop_name).view(-1, 1)
                            
                            # Apply scaling if available
                            if prop_name in self.property_scalers:
                                mean, std = self.property_scalers[prop_name]
                                target = (target - mean) / std
                                
                            targets.append(target)
                            
                    if targets:
                        targets = torch.cat(targets, dim=1)
                        loss = self.loss_fn(predictions, targets)
                        
                        if 'val_total_loss' not in total_losses:
                            total_losses['val_total_loss'] = 0
                        total_losses['val_total_loss'] += loss.item()
                        
                num_batches += 1
                
        # Average losses
        if num_batches > 0:
            for key in total_losses:
                total_losses[key] /= num_batches
                
        return total_losses


def compute_property_statistics(dataset, property_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Compute mean and standard deviation for properties in a dataset.
    
    Args:
        dataset: Molecular dataset
        property_names: List of property names
        
    Returns:
        Dictionary mapping property names to (mean, std) tuples
    """
    statistics = {}
    
    for prop_name in property_names:
        values = []
        
        for data in dataset:
            if data is not None and hasattr(data, prop_name):
                value = getattr(data, prop_name)
                if isinstance(value, torch.Tensor):
                    value = value.item()
                values.append(value)
                
        if values:
            mean = float(torch.tensor(values).mean())
            std = float(torch.tensor(values).std())
            statistics[prop_name] = (mean, std)
        else:
            logger.warning(f"No values found for property {prop_name}")
            statistics[prop_name] = (0.0, 1.0)
            
    return statistics